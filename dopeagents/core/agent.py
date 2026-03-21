"""Agent base class and core abstractions."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import (
    TypeVar,
    Generic,
    ClassVar,
    Any,
    Optional,
    Type,
    get_args,
    get_origin,
    Callable,
    cast,
)
from pydantic import BaseModel, Field

from dopeagents.core.context import AgentContext
from dopeagents.core.metadata import AgentMetadata
from dopeagents.core.types import AgentResult
from dopeagents.errors import TypeResolutionError, FrameworkNotInstalledError

# Load .env if present (python-dotenv is a declared dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class DebugInfo(BaseModel):
    """Complete transparency into what an agent will do with a given input.
    
    Returned by agent.debug(input) without making any LLM calls.
    Contains the graph structure, step prompts, and response schemas.
    """

    is_multi_step: bool = Field(
        default=False, description="Whether this agent has multiple steps"
    )
    graph_topology: Optional[dict[str, Any]] = Field(
        default=None,
        description="Step names and edges (for multi-step agents)",
    )
    step_prompts: dict[str, str] = Field(
        default_factory=dict, description="All prompts that will be sent"
    )
    step_schemas: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Pydantic schemas for each step's output",
    )
    system_prompt: str = Field(default="", description="Agent system prompt")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="Input schema (JSON Schema)"
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict, description="Output schema (JSON Schema)"
    )


class AgentDescription(BaseModel):
    """Structured description of an agent — its steps, loop structure, and model assignments.
    
    Returned by agent.describe() and used in discovery/composition.
    """

    name: str
    version: str
    description: str
    is_multi_step: bool = Field(
        default=False, description="Whether this agent has multiple steps"
    )
    steps: list[str] = Field(
        default_factory=list, description="List of step names in order"
    )
    has_loops: bool = Field(
        default=False, description="Whether this agent has refinement loops"
    )
    capabilities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    requires_llm: bool = Field(default=True)
    default_model: Optional[str] = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)


class Agent(ABC, Generic[InputT, OutputT]):
    """Base class for all DopeAgents.
    
    Agents are generic over InputT and OutputT (Pydantic models).
    Concrete agents specialize this base class with specific input/output types.
    
    Design principle: Agents contain ONLY workflow logic. Infrastructure concerns
    (cost tracking, observability, retry) are handled by the Lifecycle Layer.
    """

    # ── Class-level metadata (required by every concrete agent) ──────

    name: ClassVar[str]
    version: ClassVar[str] = "0.0.1"
    description: ClassVar[str] = ""
    capabilities: ClassVar[list[str]] = []
    tags: ClassVar[list[str]] = []
    requires_llm: ClassVar[bool] = True
    default_model: ClassVar[Optional[str]] = None
    system_prompt: ClassVar[str] = ""
    step_prompts: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        model: Optional[str] = None,
        step_models: Optional[dict[str, str]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize agent instance.
        
        Args:
            model: Model to use for all steps (can be overridden by step_models)
            step_models: Per-step model overrides
        """
        # Note: system_prompt is a ClassVar, so we don't override it at instance level
        # Remove this kwarg if passed to avoid unexpected behavior
        kwargs.pop("system_prompt", None)
        
        # Store instance-level model configuration
        # Prefer groq when key is present, fall back to openai model otherwise
        import os
        default_fallback = (
            "groq/llama-3.1-8b-instant"
            if os.environ.get("GROQ_API_KEY")
            else "openai/gpt-4o-mini"
        )
        self._model = model or self.default_model or default_fallback
        self._step_models = step_models or {}
        
        # Initialize cached graph (lazy-built by _get_graph)
        self._graph = None

    # ── Type introspection (for accessing InputT and OutputT) ───────

    @classmethod
    def input_type(cls) -> Type[InputT]:
        """Return the InputT type for this agent.
        
        Resolves generics via __orig_bases__ introspection.
        """
        return cls._resolve_type("input")

    @classmethod
    def output_type(cls) -> Type[OutputT]:
        """Return the OutputT type for this agent.
        
        Resolves generics via __orig_bases__ introspection.
        """
        return cls._resolve_type("output")

    @classmethod
    def _resolve_type(cls, position: str) -> Type[Any]:
        """Resolve InputT (position=0) or OutputT (position=1) from __orig_bases__."""
        idx = 0 if position == "input" else 1

        # Walk the MRO to find the Agent specialization
        if not hasattr(cls, "__orig_bases__"):
            raise ValueError(
                f"Agent {cls.__name__} has no __orig_bases__"
            )
        
        for base in cls.__orig_bases__:
            if get_origin(base) is Agent or (
                hasattr(base, "__origin__")
                and get_origin(base).__name__ == "Agent"
            ):
                args = get_args(base)
                if len(args) > idx:
                    return cast(Type[Any], args[idx])

        raise ValueError(
            f"Could not resolve {position}_type for {cls.__name__}"
        )

    # ── Extraction primitive (the ONLY place agents call LLMs) ───────

    def _extract(
        self,
        response_model: Type[BaseModel],
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Extract structured output from an LLM.
        
        This is the ONLY method agents call to interact with LLMs.
        All validation, routing, cost tracking, and retry happen here.
        
        Args:
            response_model: Pydantic model for validation
            messages: List of {"role", "content"} dicts
            model: Override model for this call (falls back to self.default_model)
            **kwargs: Additional args for Instructor/LiteLLM
            
        Returns:
            Instance of response_model with validated output
            
        Raises:
            ExtractionValidationError: Schema validation failed
            ExtractionProviderError: Provider error (rate limit, etc.)
        """
        import re
        import time
        import instructor
        import litellm
        from dopeagents.errors import ExtractionProviderError

        # Determine model: explicit arg > instance default
        resolved_model = model or self._model

        # Groq requires JSON mode (tool_call mode has array schema issues)
        if "groq" in resolved_model.lower():
            mode = instructor.Mode.MD_JSON
        else:
            mode = instructor.Mode.TOOLS

        litellm_client = instructor.from_litellm(litellm.completion, mode=mode)

        # Pacing delay for groq free-tier TPM limits (~2 calls/sec)
        if "groq" in resolved_model.lower():
            time.sleep(0.5)

        for attempt in range(4):
            try:
                result = litellm_client.chat.completions.create(
                    model=resolved_model,
                    response_model=response_model,
                    messages=messages,
                    **kwargs,
                )
                return cast(BaseModel, result)
            except Exception as exc:
                exc_str = str(exc).lower()
                if "rate_limit" not in exc_str and "429" not in exc_str:
                    raise

                # Parse retry-after from error message (e.g. "try again in 5m51s")
                retry_after: Optional[int] = None
                m = re.search(r"try again in (\d+)m([\d.]+)s", str(exc))
                if m:
                    retry_after = int(m.group(1)) * 60 + int(float(m.group(2)))

                # TPD (tokens per day) limit requires a very long wait — fail fast
                # rather than hanging; let the caller surface a clean error.
                if retry_after is not None and retry_after > 60:
                    provider = "groq" if "groq" in resolved_model.lower() else "unknown"
                    raise ExtractionProviderError(
                        message=(
                            f"Daily token quota reached for {provider}. "
                            f"Retry after {retry_after // 60}m{retry_after % 60}s."
                        ),
                        provider=provider,
                        status_code=429,
                        retry_after=retry_after,
                    ) from exc

                # Short TPM rate limit — exponential backoff (max 3 retries)
                if attempt < 3:
                    wait = 2 ** attempt  # 1, 2, 4 seconds
                    time.sleep(wait)
                    continue

                provider = "groq" if "groq" in resolved_model.lower() else "unknown"
                raise ExtractionProviderError(
                    message=f"Rate limit exceeded after {attempt + 1} attempts.",
                    provider=provider,
                    status_code=429,
                    retry_after=retry_after,
                ) from exc
        raise RuntimeError("_extract: exhausted retries")  # unreachable

    def _extract_partial(
        self,
        response_model: Type[BaseModel],
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Extract partial structured output (streaming).
        
        Used for LLMs that support incremental completion.
        Same semantics as _extract() but returns as streaming chunks.
        """
        raise NotImplementedError(
            "_extract_partial() is implemented when streaming support is added"
        )

    # ── Public interface ──────────────────────────────────────────

    @abstractmethod
    def run(self, input_data: InputT, context: Optional[AgentContext] = None) -> AgentResult[OutputT]:
        """Run the agent on input and return structured output.
        
        Args:
            input_data: Instance of InputT (validated at compile time)
            context: Optional execution context with run ID and metadata
            
        Returns:
            AgentResult[OutputT] with output and execution metadata
            
        Raises:
            ExtractionError: LLM extraction failed
            GraphExecutionError: Internal graph execution failed
            TypeResolutionError: Type introspection failed
        """
        pass

    def describe(self) -> AgentDescription:
        """Return structured description of this agent.
        
        Used for discovery, composition validation, and MCP schema generation.
        Does NOT execute the agent.
        """
        steps = list(self.step_prompts.keys())
        return AgentDescription(
            name=self.name,
            version=self.version,
            description=self.description,
            is_multi_step=bool(steps),
            steps=steps,
            has_loops=self._has_loops(),
            capabilities=self.capabilities,
            tags=self.tags,
            requires_llm=self.requires_llm,
            default_model=self.default_model,
            input_schema=self.input_type().model_json_schema(),
            output_schema=self.output_type().model_json_schema(),
        )

    def debug(self, input_data: InputT) -> DebugInfo:
        """Return complete debug info without executing the agent.
        
        Shows graph structure, prompts, and schemas that would be used
        if run() were called with this input. No LLM calls, no cost.
        """
        return DebugInfo(
            is_multi_step=self._has_loops() or bool(self.step_prompts),
            step_prompts=self.step_prompts,
            system_prompt=self.system_prompt,
            input_schema=self.input_type().model_json_schema(),
            output_schema=self.output_type().model_json_schema(),
        )

    def metadata(self) -> AgentMetadata:
        """Return structured metadata about this agent.
        
        Used by registry, composition checker, and MCP exposure.
        """
        return AgentMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            capabilities=self.capabilities,
            tags=self.tags,
            requires_llm=self.requires_llm,
            default_model=self.default_model,
            system_prompt=self.system_prompt,
            step_prompts=self.step_prompts,
            input_schema=self.input_type().model_json_schema(),
            output_schema=self.output_type().model_json_schema(),
        )

    # ── Framework adapter stubs (concrete impls in adapters/ modules) ───

    def as_callable(self) -> Callable[[InputT], OutputT]:
        """Wrap agent as a plain Python callable.
        
        Usage:
            fn = agent.as_callable()
            result = fn(input_data)
        """
        def wrapper(input_data: InputT) -> OutputT:
            result = self.run(input_data)
            if result.error:
                raise RuntimeError(result.error)
            if result.output is None:
                raise RuntimeError("Agent returned None output")
            return result.output

        return wrapper

    def as_langchain_runnable(self) -> Any:
        """Wrap agent as a LangChain Runnable.
        
        Requires: pip install dopeagents[langchain]
        """
        try:
            from dopeagents.adapters.langchain_adapter import (
                agent_to_langchain_runnable,
            )

            return agent_to_langchain_runnable(self)
        except ImportError as e:
            raise ImportError(
                "LangChain adapter requires: pip install langchain-core"
            ) from e

    def as_langgraph_node(self) -> Any:
        """Wrap agent as a LangGraph node.
        
        Requires: pip install langgraph (already core dep)
        """
        try:
            from dopeagents.adapters.langgraph_adapter import agent_to_langgraph_node

            return agent_to_langgraph_node(self)
        except ImportError as e:
            raise ImportError(
                "LangGraph adapter requires: pip install langgraph"
            ) from e

    def as_crewai_tool(self) -> Any:
        """Wrap agent as a CrewAI tool.
        
        Requires: pip install dopeagents[crewai]
        """
        try:
            from dopeagents.adapters.crewai_adapter import agent_to_crewai_tool

            return agent_to_crewai_tool(self)
        except ImportError as e:
            raise ImportError(
                "CrewAI adapter requires: pip install crewai"
            ) from e

    def as_autogen_function(self) -> Any:
        """Wrap agent as an AutoGen function.
        
        Requires: pip install dopeagents[autogen]
        """
        try:
            from dopeagents.adapters.autogen_adapter import agent_to_autogen_function

            return agent_to_autogen_function(self)
        except ImportError as e:
            raise ImportError(
                "AutoGen adapter requires: pip install autogen-agentchat"
            ) from e

    def as_openai_function(self) -> dict[str, Any]:
        """Wrap agent as an OpenAI function definition.
        
        Returns a dict compatible with OpenAI's functions API.
        """
        input_schema = self.input_type().model_json_schema()
        return {
            "name": self.name.lower().replace(" ", "_"),
            "description": self.description,
            "parameters": input_schema,
        }

    def as_mcp_tool(self, server: Optional[Any] = None) -> Any:
        """Register agent as MCP tool on a FastMCP server.
        
        Args:
            server: FastMCP server instance. If None, creates a new one.
            
        Returns:
            The MCP tool registration (or server if created).
        """
        try:
            from dopeagents.mcp_server.server import register_agent_as_mcp_tool

            return register_agent_as_mcp_tool(self, server)
        except ImportError as e:
            raise ImportError(
                "MCP support requires: pip install dopeagents[mcp]"
            ) from e

    def as_mcp_server(self) -> Any:
        """Create a standalone MCP server exposing this agent as a tool.
        
        Returns a FastMCP server ready to run.
        """
        try:
            from dopeagents.mcp_server.server import create_single_agent_mcp_server

            return create_single_agent_mcp_server(self)
        except ImportError as e:
            raise ImportError(
                "MCP support requires: pip install dopeagents[mcp]"
            ) from e

    # ── Introspection helpers ──────────────────────────────────────

    def _has_loops(self) -> bool:
        """Whether this agent has internal refinement loops.
        
        Override in subclasses that have conditional edges.
        """
        return False

    def _build_graph(self) -> Any:
        """Build the internal LangGraph (for multi-step agents).
        
        Override in subclasses that use LangGraph.
        Single-step agents return None.
        """
        return None

    def _get_graph(self) -> Any:
        """Lazily build and cache the internal LangGraph.
        
        Multi-step agents implement _build_graph() to return a compiled graph.
        Single-step agents don't override this — _get_graph() returns None.
        
        The compiled graph is a private implementation detail — callers use run().
        """
        if self._graph is None and hasattr(self, "_build_graph"):
            self._graph = self._build_graph()
        return self._graph

    def _model_for_step(self, step_name: str) -> str:
        """Returns the model to use for a given step.
        
        Respects step-level overrides via _step_models, falling back to _model.
        
        Args:
            step_name: Name of the step (e.g., "analyze", "summarize")
        
        Returns:
            Model name to use for this step
        """
        return self._step_models.get(step_name, self._model)

