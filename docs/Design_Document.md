# DopeAgents — Complete Design Document

> **Document Version**: 3.0.0
> **Software Version**: 0.1.0-dev (pre-release)
> **Purpose**: Authoritative **design specification** for all architecture, interfaces, conventions, and implementation targets. Every line of code written traces back to a section in this document.

> [!IMPORTANT]
> **Document Status — Design Specification, Not Implementation Record**
>
> This document describes the **target architecture** for DopeAgents v0.1.0. It is a normative design spec — code that gets written must conform to what is described here.
>
> As of the current repository state, the codebase contains **scaffold stubs** (module files with placeholder comments). No subsystem is production-complete yet. The [Development Roadmap (§22)](#22-development-roadmap) defines the phased implementation plan.
>
> Code examples in this document are **reference implementations** — they show exactly what should be built, not what exists today. Do not assume a working runtime from reading this document alone.

---

## Table of Contents

1. [Project Philosophy &amp; Principles](#1-project-philosophy--principles)
2. [System Architecture](#2-system-architecture)
3. [Core Type System &amp; Agent Interface](#3-core-type-system--agent-interface)
4. [Contract Verification Engine](#4-contract-verification-engine)
5. [Agent Lifecycle &amp; Hooks](#5-agent-lifecycle--hooks)
6. [Agent Sandbox](#6-agent-sandbox)
7. [Observability System](#7-observability-system)
8. [Cost Management &amp; Budget Guards](#8-cost-management--budget-guards)
9. [Resilience Layer — Retry, Fallback, Degradation](#9-resilience-layer--retry-fallback-degradation)
10. [Caching Layer](#10-caching-layer)
11. [Cross-Framework Adapters](#11-cross-framework-adapters)
12. [Wrapping External Agents](#12-wrapping-external-agents)
13. [Agent Spec (Open Agent Specification)](#13-agent-spec-open-agent-specification)
14. [Registry &amp; Discovery System](#14-registry--discovery-system)
15. [Benchmark &amp; Evaluation Framework](#15-benchmark--evaluation-framework)
16. [Tool Integration Layer](#16-tool-integration-layer)
17. [Core Agents](#17-core-agents)
18. [Package Structure](#18-package-structure)
19. [Error Taxonomy](#19-error-taxonomy)
20. [Configuration System](#20-configuration-system)
21. [Security &amp; PII Handling](#21-security--pii-handling)
22. [Development Roadmap](#22-development-roadmap)
23. [API Reference Summary](#23-api-reference-summary)
24. [Design Decisions Log](#24-design-decisions-log)

---

## 1. Project Philosophy & Principles

### 1.1 Core Philosophy

DopeAgents ships **production-grade AI agents** — internally sophisticated multi-step workflows with self-evaluation, iterative refinement, and intelligent branching — externally exposed as simple typed interfaces with built-in cost tracking, step-level observability, and protocol-native exposure via MCP.

**The core thesis:** Every team building AI applications rebuilds the same complex workflows from scratch. A summarizer that actually handles long documents needs chunking, multi-pass synthesis, and quality checks — not a single prompt. A research agent needs multiple search queries, source credibility evaluation, and cross-referencing — not a thin wrapper that dumps raw results into a context window. Each of these takes days to build correctly and weeks to harden for production. DopeAgents ships these workflows already built, tested, and production-hardened behind a clean `run(input) → output` interface.

**Two interfaces — nothing else to know:**

```python
# Plain Python — one line
output = DeepSummarizer().run(DeepSummarizerInput(text="...", style="bullets"))

# MCP — expose to Claude Desktop, Cursor, any MCP client
mcp = FastMCP("My App")
DeepSummarizer().as_mcp_tool(mcp)
mcp.run()
```

### 1.2 Design Principles

Every design decision must pass through these principles in order of priority:

```text
P1: Internally sophisticated, externally simple
    Every agent is a multi-step workflow — analyze, chunk, synthesize,
    self-evaluate, refine, format. Every agent is one line to use.
    Internal complexity is the value. Simple interface is the contract.

P2: Contracts are enforceable, not decorative
    If two agents are incompatible, the developer knows at composition
    time, not at runtime after burning tokens across multiple LLM calls.

P3: Every agent is debuggable at the step level
    A developer can inspect the graph structure, every step's prompt,
    every step's expected output schema, and every step's cost — without
    making a single API call. No magic. No hidden state.

P4: The graph engine is a private implementation detail
    Users never need to learn LangGraph's API. Agents expose
    run(input) → output. If a better orchestration primitive emerges,
    agents migrate internally without breaking a single line of user code.

P5: Step-level everything
    Cost tracking, budget guards, observability, and retry operate at the
    individual step level, not just the agent level. You see exactly which
    step is burning money and where quality is gained or lost.

P6: Production concerns are first-class
    Step-level cost tracking, budget guards for refinement loops,
    retry logic, fallback chains, observability, and PII redaction are
    part of every agent from day one — not bolted on afterward.

P7: Self-evaluating agents
    Quality scoring and iterative refinement are first-class patterns.
    Agents don't just produce output — they evaluate whether the output
    is good enough and improve it if it isn't.

P8: Stateless agents, external state
    Agents are pure functions at the external interface: input → output.
    Internal LangGraph state is a private implementation detail that never
    leaks out of the agent boundary. Agents are testable, scalable,
    and safe for concurrent use.

P9: LLM-optional
    Some agents use LLMs. Some don't. Some mix deterministic and
    LLM-powered steps. The interface doesn't assume either.

P10: Transparency over convenience
     If a developer asks "what prompt was sent to the evaluate step?",
     the answer is one function call away, always.

P11: Build on proven foundations
     Solved problems stay solved. Instructor handles structured LLM
     extraction and validation. LiteLLM handles provider routing, token
     accounting, and cost calculation. LangGraph handles internal graph
     orchestration. FastMCP handles protocol transport.
     DopeAgents does not reimplement these. Dependencies are chosen for
     focus, not avoided for purity.

P12: Protocol-native by default
     Every agent is exposable as an MCP tool with one method call.
     A single MCP tool call triggers the entire multi-step workflow.
     The client gets high-quality results without knowing how many
     internal steps executed.
```

### 1.3 What DopeAgents Is NOT

| DopeAgents is NOT                                     | It IS                                                                          |
| ----------------------------------------------------- | ------------------------------------------------------------------------------ |
| An orchestration framework                            | A library of production-grade agents built ON orchestration primitives         |
| A SaaS product                                        | A pip-installable OSS package                                                  |
| Fifty thin prompt wrappers                            | A small set of deeply-built multi-step agents with self-evaluation             |
| A graph API users need to learn                       | Agents that hide internal complexity behind `run(input) → output`           |
| Tied to one LLM provider                              | LLM-agnostic via Instructor + LiteLLM (100+ providers)                         |
| A state management system users interact with         | Internal graph state is a private implementation detail                        |
| Another LangGraph / LangChain                         | Uses LangGraph internally; never surfaces it to callers                        |
| Reimplementing structured extraction, routing, graphs | Built on Instructor + LiteLLM + LangGraph — proven, battle-tested foundations |
| A proprietary protocol                                | MCP-native — agents discoverable by any compliant client                      |

---

## 2. System Architecture

### 2.1 Layer Diagram

```text
┌──────────────────────────────────────────────────────────────┐
│                      User / MCP Client                        │
│   (Your Application  │  Claude Desktop  │  Cursor  │  etc.)  │
├──────────────────────────────────────────────────────────────┤
│                   External Interfaces                          │
│           Plain Python (.run())  │  MCP Protocol              │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌────────────────────────────────────────────────────────┐  │
│   │              Agent Lifecycle Layer                      │  │
│   │  ┌────────────┐ ┌──────────┐ ┌───────────────────────┐ │  │
│   │  │Observability│ │  Cost    │ │   Resilience          │ │  │
│   │  │  (per-step) │ │Step-level│ │ Retry/Fallback        │ │  │
│   │  └────────────┘ └──────────┘ └───────────────────────┘ │  │
│   │  ┌────────────┐ ┌──────────┐ ┌───────────────────────┐ │  │
│   │  │  Caching   │ │  Budget  │ │  Contract Checker     │ │  │
│   │  │  Layer     │ │  Guards  │ │  & Pipeline Validator  │ │  │
│   │  └────────────┘ └──────────┘ └───────────────────────┘ │  │
│   └───────────────────────────┬────────────────────────────┘  │
│                               │                                │
│   ┌───────────────────────────┴────────────────────────────┐  │
│   │              Agent External Interface                    │  │
│   │  Agent[InputT, OutputT]                                  │  │
│   │  • run(input) → output          (public)                 │  │
│   │  • debug(input) → DebugInfo     (graph, prompts, schemas)│  │
│   │  • describe() → AgentDescription (steps, has_loops)     │  │
│   │  • as_mcp_tool() / as_mcp_server()                       │  │
│   └───────────────────────────┬────────────────────────────┘  │
│                               │                                │
│   ┌───────────────────────────┴────────────────────────────┐  │
│   │              Agent Internal Workflow (private)           │  │
│   │  LangGraph StateGraph — hidden implementation detail     │  │
│   │                                                          │  │
│   │  analyze → chunk → summarize → synthesize               │  │
│   │                                    │                     │  │
│   │                                evaluate                  │  │
│   │                               ╱         ╲               │  │
│   │                        score < 0.8   score ≥ 0.8        │  │
│   │                            │              │              │  │
│   │                         refine       format_output        │  │
│   │                            │                             │  │
│   │                    (loop back to synthesize)             │  │
│   │                                                          │  │
│   │  Each step calls self._extract() → Instructor+LiteLLM   │  │
│   └───────────────────────────┬────────────────────────────┘  │
│                               │                                │
│   ┌───────────────────────────┴────────────────────────────┐  │
│   │              Structured Extraction Layer                  │  │
│   │  Instructor (validation, retries, schema enforcement)    │  │
│   └───────────────────────────┬────────────────────────────┘  │
│                               │                                │
│   ┌───────────────────────────┴────────────────────────────┐  │
│   │              Provider Routing & Cost Layer               │  │
│   │  LiteLLM (routing, token counting, pricing, budgets)    │  │
│   └───────────────────────────┬────────────────────────────┘  │
│                               │                                │
│   ┌───────────────────────────┴────────────────────────────┐  │
│   │              LLM Provider (any of 100+)                  │  │
│   │  OpenAI, Anthropic, Google, Ollama, DeepSeek, etc.      │  │
│   └────────────────────────────────────────────────────────┘  │
│                                                                │
│                         DopeAgents                             │
└──────────────────────────────────────────────────────────────┘
```

**Key insight**: Users see `run(input) → output`. The internal LangGraph graph with conditional branches, self-evaluation loops, and multi-step cost accumulation is a private implementation detail. Users never interact with LangGraph directly.

### 2.2 Dependency Graph

```text
dopeagents (core) — always installed
├── pydantic >= 2.11
├── instructor >= 1.14         # structured extraction + validation retries
├── litellm >= 1.56            # provider routing + token/cost accounting
├── langgraph >= 0.2           # internal graph orchestration (core dep)
├── typing-extensions >= 4.12
├── click >= 8.1
├── jsonschema >= 4.23
└── httpx >= 0.27

dopeagents[mcp]                # MCP exposure
└── fastmcp >= 3.0

dopeagents[cache]              # optional
└── diskcache >= 5.6

dopeagents[otel]               # optional
├── opentelemetry-api >= 1.28
└── opentelemetry-sdk >= 1.28

dopeagents[langchain]          # optional — for .as_langchain_runnable()
└── langchain-core >= 1.0

dopeagents[crewai]             # optional — for .as_crewai_tool()
└── crewai >= 1.0

dopeagents[autogen]            # optional — for .as_autogen_function()
└── autogen-agentchat >= 0.4
```

**Note:** `langgraph` is a **core dependency** — it powers the internal graphs of every multi-step agent. This is different from framework adapters (LangChain, CrewAI, AutoGen), which remain optional. LangGraph is used internally, not surfaced to callers.

### 2.3 Design Invariants

These must never be violated:

- **INV-1:** Every `Agent` subclass must define explicit `InputType` and `OutputType` using Pydantic models. No raw dicts cross the agent boundary.
- **INV-2:** `Agent.run()` must never mutate instance state (`self`) or external state. Side effects happen only through the Tool interface. Runtime configuration (e.g., model override) is passed as a parameter to `_extract()`, not stored on the instance. This ensures agents are safe for concurrent use.
- **INV-3:** Every LLM call must go through the `_extract()` primitive (Instructor over LiteLLM). Token count and request metadata are captured through Instructor hooks, and cost is read from LiteLLM response metadata (`_hidden_params.response_cost`) when present. No raw provider SDK calls inside agent step methods.
- **INV-4:** An agent that fails must fail with a typed error from the error taxonomy. No bare exceptions.
- **INV-5:** All observability, cost tracking, and resilience logic runs in the lifecycle layer, not inside the agent's `run()` method or any step method.
- **INV-6:** Framework adapters never modify agent behavior. They only translate input/output formats. The same `agent.run()` executes regardless of which framework calls it.
- **INV-7:** Framework dependencies (LangChain, CrewAI, AutoGen) are always optional and lazy-imported. Installing `dopeagents` core never pulls them in. `langgraph` is the only graph-related core dependency, and it is used internally by agents, not surfaced to users.
- **INV-8:** The internal LangGraph graph is a private implementation detail. It must never be exposed through public interfaces. Callers interact with `run(input) → output`, debug info, and step metadata only.
- **INV-9:** MCP adapters never modify agent behavior. They only translate the agent's existing typed interface (Pydantic input/output) into MCP tool schemas and handle transport. The same `agent.run()` executes whether called via MCP, a framework adapter, or direct Python.
- **INV-10:** Instructor is an internal implementation detail. Agent authors interact with `self._extract()`. Users interact with `agent.run()`. Neither agent authors nor users need to import or configure Instructor directly.

---

## 3. Core Type System & Agent Interface

### 3.1 The Agent Base Class

> 📄 **File:** `dopeagents/core/agent.py` | **Status:**   Implemented | **Role:** Generic base class with extraction primitives, multi-step graph support, type introspection, debug/describe interface, and framework adapter methods

```python
# dopeagents/core/agent.py

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, ClassVar, get_args, get_origin
from pydantic import BaseModel
from dopeagents.core.context import AgentContext
from dopeagents.core.metadata import AgentMetadata

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class Agent(ABC, Generic[InputT, OutputT]):
    """
    Base class for all DopeAgents agents.

    Every agent is a typed, stateless function: InputT → OutputT

    Single-step agents implement run() directly using self._extract().
    Multi-step agents implement _build_graph() to define an internal LangGraph
    workflow — and let the compiled graph drive run() automatically.

    The lifecycle layer handles observability, cost, retry, and caching.
    Framework and protocol adapters are inherited — never written per agent.
    LangGraph is an internal engine; callers never interact with it.
    """

    # -- Class-level metadata --
    name: ClassVar[str]
    version: ClassVar[str]
    description: ClassVar[str]
    capabilities: ClassVar[list[str]]  # Functional declarations surfaced to MCP clients for tool discovery
    tags: ClassVar[list[str]] = []     # Free-form labels used by the Registry (e.g. ["text", "reasoning"])
    requires_llm: ClassVar[bool] = True
    default_model: ClassVar[str] = "openai/gpt-4o-mini"

    # -- Prompt declarations --
    # Declared at class level so they're inspectable without execution.
    # system_prompt: used by single-step agents or as the base prompt for multi-step agents.
    # step_prompts: per-step prompt templates for multi-step agents. Keys are step names.
    # Both are ClassVars so describe() and debug() can expose them without running the agent.
    system_prompt: ClassVar[str] = ""
    step_prompts: ClassVar[dict[str, str]] = {}  # {"analyze": "...", "evaluate": "...", ...}

    # -- Instructor client backed by LiteLLM (lazy-initialized) --
    _client: object | None = None
    _graph: object | None = None  # Compiled LangGraph; built once per instance

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,     # Runtime override; shadows class-level default
        step_models: dict[str, str] | None = None,  # Per-step model overrides
        **kwargs,
    ):
        self._model = model or self.default_model
        self._step_models: dict[str, str] = step_models or {}
        self._client = None  # Lazy init on first _extract() call
        self._graph = None   # Lazy init on first run() call (multi-step agents)
        if system_prompt is not None:
            self.system_prompt = system_prompt

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        sp = cls.__dict__.get("system_prompt")
        if sp is not None and not isinstance(sp, property) and not isinstance(sp, str):
            raise TypeError(
                f"{cls.__name__}.system_prompt must be a str, got {type(sp).__name__}"
            )
        if (
            getattr(cls, "requires_llm", False)
            and not getattr(cls, "system_prompt", "")
            and not cls.step_prompts
            and not isinstance(cls.__dict__.get("system_prompt"), property)
        ):
            import warnings
            warnings.warn(
                f"{cls.__name__} has requires_llm=True but neither system_prompt "
                "nor step_prompts declared. Consider adding prompts for discoverability.",
                stacklevel=2,
            )

    def _get_client(self):
        """Lazily initialize the Instructor client over LiteLLM."""
        if self._client is None:
            import instructor
            from litellm import completion
            self._client = instructor.from_litellm(completion)
        return self._client

    def _get_graph(self):
        """
        Lazily build and compile the internal LangGraph graph.
        Multi-step agents implement _build_graph() to return a compiled graph.
        Single-step agents don't override this — _get_graph() returns None.
        """
        if self._graph is None and hasattr(self, "_build_graph"):
            self._graph = self._build_graph()
        return self._graph

    # -- Multi-step graph support (override in multi-step agents) --

    def _build_graph(self):
        """
        Override to define the internal multi-step LangGraph workflow.
        Returns a compiled LangGraph StateGraph.

        This is a private method. The compiled graph is a hidden implementation
        detail — callers use run(), not the graph directly.

        Example:
            def _build_graph(self):
                from langgraph.graph import StateGraph, END
                graph = StateGraph(MyState)
                graph.add_node("analyze", self._step_analyze)
                graph.add_node("synthesize", self._step_synthesize)
                graph.add_node("evaluate", self._step_evaluate)
                graph.add_edge("analyze", "synthesize")
                graph.add_conditional_edges(
                    "evaluate",
                    lambda s: "refine" if s.quality_score < 0.8 else END,
                )
                graph.set_entry_point("analyze")
                return graph.compile()
        """
        return None  # Default: agent is single-step, no graph needed

    # -- Instructor extraction primitives --

    def _extract(
        self,
        response_model: type[BaseModel],
        messages: list[dict[str, str]],
        max_retries: int = 3,
        model: str | None = None,
        **kwargs,
    ) -> BaseModel:
        """
        Structured extraction via Instructor + LiteLLM. The core primitive every
        step in a DopeAgents workflow uses.

        - Calls the LLM via Instructor's LiteLLM-backed client
        - Validates response against response_model (Pydantic)
        - Automatically retries with validation error feedback on schema mismatch
        - Uses LiteLLM's model routing + pricing metadata for provider/cost coverage

        Args:
            model: Optional model override. When provided (e.g., from a
                   step_models entry or context.model_override), takes precedence
                   over self._model. This avoids instance mutation in run() (INV-2).

        Agent authors call this inside step methods instead of raw LLM APIs.
        The lifecycle layer hooks into Instructor's event system to capture
        cost and observability per call (per step).
        """
        client = self._get_client()
        return client.chat.completions.create(
            model=model or self._model,
            response_model=response_model,
            messages=messages,
            max_retries=max_retries,
            **kwargs,
        )

    def _extract_partial(
        self,
        response_model: type[BaseModel],
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs,
    ):
        """Streaming extraction via Instructor. Returns partial results as they arrive."""
        client = self._get_client()
        return client.chat.completions.create_partial(
            model=model or self._model,
            response_model=response_model,
            messages=messages,
            **kwargs,
        )

    def _model_for_step(self, step_name: str) -> str:
        """Returns the model to use for a given step, respecting step_models overrides."""
        return self._step_models.get(step_name, self._model)

    # -- Type introspection --

    @classmethod
    def input_type(cls) -> type[BaseModel]:
        """Returns the Pydantic model class for this agent's input."""
        for base in cls.__orig_bases__:
            origin = get_origin(base)
            if origin is Agent or (isinstance(origin, type) and issubclass(origin, Agent)):
                args = get_args(base)
                if args and len(args) >= 1:
                    return args[0]
        raise TypeError(
            f"Agent {cls.__name__} must specify Generic types: Agent[InputT, OutputT]"
        )

    @classmethod
    def output_type(cls) -> type[BaseModel]:
        """Returns the Pydantic model class for this agent's output."""
        for base in cls.__orig_bases__:
            origin = get_origin(base)
            if origin is Agent or (isinstance(origin, type) and issubclass(origin, Agent)):
                args = get_args(base)
                if args and len(args) >= 2:
                    return args[1]
        raise TypeError(
            f"Agent {cls.__name__} must specify Generic types: Agent[InputT, OutputT]"
        )

    @classmethod
    def metadata(cls) -> AgentMetadata:
        """Returns structured metadata about this agent."""
        return AgentMetadata(
            name=cls.name,
            version=cls.version,
            description=cls.description,
            capabilities=cls.capabilities,
            tags=cls.tags,
            requires_llm=cls.requires_llm,
            default_model=cls.default_model,
            system_prompt=cls.system_prompt,
            step_prompts=cls.step_prompts,
            input_schema=cls.input_type().model_json_schema(),
            output_schema=cls.output_type().model_json_schema(),
        )

    def to_metadata(self) -> dict:
        """Returns agent metadata as a dict capturing instance-level state."""
        is_overridden = self.system_prompt != self.__class__.system_prompt
        steps = list(self.step_prompts.keys()) if self.step_prompts else []
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "model": self._model,
            "capabilities": self.capabilities,
            "requires_llm": self.requires_llm,
            "system_prompt": self.system_prompt,
            "system_prompt_overridden": is_overridden,
            "step_prompts": self.step_prompts,
            "steps": steps,
            "is_multi_step": bool(steps),
            "input_schema": self.input_type().model_json_schema(),
            "output_schema": self.output_type().model_json_schema(),
        }

    # -- Core interface --

    @abstractmethod
    def run(self, input: InputT, context: AgentContext | None = None) -> OutputT:
        """
        Execute the agent's core logic.

        Single-step agents implement this directly using self._extract().
        Multi-step agents implement _build_graph() instead; the base class
        can drive the graph from run() via self._get_graph().invoke(state).

        This method must be:
        - Stateless: no mutation of self or external state
        - Typed: input and output match the declared schemas
        - Focused: only core logic, no retry/caching/observability
        """
        ...

    # -- Debug and describe interfaces --

    def debug(self, input: InputT) -> "DebugInfo":
        """
        Returns everything needed to understand what this agent will do
        with the given input, WITHOUT executing it.

        For multi-step agents, this includes step_prompts, step_schemas,
        and a description of the internal graph structure.
        """
        step_schemas = {}
        for step_name in self.step_prompts:
            method = getattr(self, f"_step_{step_name}", None)
            if method and hasattr(method, "__annotations__"):
                step_schemas[step_name] = method.__annotations__
        return DebugInfo(
            agent_name=self.name,
            model=self._model if self.requires_llm else None,
            input_data=input.model_dump(),
            input_schema=self.input_type().model_json_schema(),
            output_schema=self.output_type().model_json_schema(),
            requires_llm=self.requires_llm,
            prompt=self._render_prompt(input) if self.requires_llm else None,
            model_config_data=self._get_model_config() if self.requires_llm else None,
            extraction_mode=self._get_extraction_mode() if self.requires_llm else None,
            step_prompts=self.step_prompts if self.step_prompts else None,
            step_schemas=step_schemas if step_schemas else None,
            is_multi_step=bool(self.step_prompts),
        )

    def describe(self) -> "AgentDescription":
        """
        Returns a structured description of the agent — steps, loop structure,
        model assignments — without executing it.

        Used by: CLI `dopeagents describe`, MCP resource endpoints,
                 Agent Registry catalog, benchmarking tools.
        """
        steps = list(self.step_prompts.keys()) if self.step_prompts else []
        return AgentDescription(
            name=self.name,
            version=self.version,
            description=self.description,
            is_multi_step=bool(steps),
            steps=steps,
            has_loops=self._has_loops(),
            default_model=self._model,
            step_models=self._step_models,
            capabilities=list(self.capabilities),
            tags=list(self.tags),
            input_schema=self.input_type().model_json_schema(),
            output_schema=self.output_type().model_json_schema(),
        )

    def _has_loops(self) -> bool:
        """Override in multi-step agents that contain refinement loops."""
        return False

    def _render_prompt(self, input: InputT) -> str | None:
        """Override in LLM-based agents to expose the rendered prompt for debugging."""
        return None

    def _get_model_config(self) -> dict | None:
        """Override in LLM-based agents to expose model configuration."""
        return None

    def _get_extraction_mode(self) -> str | None:
        """Returns the Instructor extraction mode for the current provider."""
        try:
            client = self._get_client()
            return str(getattr(client, 'mode', 'auto'))
        except Exception:
            return "auto"

    # -- Framework Adapters --
    # All adapters use lazy imports. Framework dependencies are optional.
    # Every agent inherits these — no per-agent adapter code needed.

    def as_langchain_runnable(self, **kwargs):
        """Use this agent as a LangChain Runnable in LCEL chains."""
        from dopeagents.adapters.langchain import to_langchain_runnable
        return to_langchain_runnable(self, **kwargs)

    def as_langchain_tool(self, **kwargs):
        """Use this agent as a LangChain Tool for agents."""
        from dopeagents.adapters.langchain import to_langchain_tool
        return to_langchain_tool(self, **kwargs)

    def as_crewai_tool(self, **kwargs):
        """Use this agent as a CrewAI tool."""
        from dopeagents.adapters.crewai import to_crewai_tool
        return to_crewai_tool(self, **kwargs)

    def as_autogen_function(self, **kwargs):
        """Use this agent as an AutoGen callable function."""
        from dopeagents.adapters.autogen import to_autogen_function
        return to_autogen_function(self, **kwargs)

    def as_openai_function(self, **kwargs):
        """Get OpenAI function calling schema + callable."""
        from dopeagents.adapters.openai_functions import to_openai_function
        return to_openai_function(self, **kwargs)

    def as_callable(self, **kwargs):
        """Get a plain callable (fallback for any framework)."""
        from dopeagents.adapters.generic import to_callable
        return to_callable(self, **kwargs)

    # -- MCP Adapters --

    def as_mcp_tool(self, mcp_server, **kwargs):
        """Register this agent as a tool on a FastMCP server instance."""
        from dopeagents.adapters.mcp import register_agent_as_mcp_tool
        return register_agent_as_mcp_tool(self, mcp_server, **kwargs)

    def as_mcp_server(self, name: str | None = None, **kwargs):
        """Create a standalone MCP server exposing just this agent."""
        from dopeagents.adapters.mcp import create_single_agent_mcp_server
        return create_single_agent_mcp_server(self, name=name, **kwargs)


class DebugInfo(BaseModel):
    """Complete transparency into what an agent will do with a given input."""
    agent_name: str
    model: str | None = None
    input_data: dict
    input_schema: dict
    output_schema: dict
    requires_llm: bool
    prompt: str | None = None
    model_config_data: dict | None = None
    extraction_mode: str | None = None
    # Multi-step specific
    step_prompts: dict[str, str] | None = None    # Per-step prompt templates
    step_schemas: dict[str, dict] | None = None   # Per-step output schemas
    is_multi_step: bool = False


class AgentDescription(BaseModel):
    """Structured description of an agent — its steps, loop structure, and model assignments."""
    name: str
    version: str
    description: str
    is_multi_step: bool
    steps: list[str]                    # Ordered list of step names (empty for single-step)
    has_loops: bool                     # True if the graph contains refinement loops
    default_model: str
    step_models: dict[str, str]         # Per-step model overrides (empty dict if none)
    capabilities: list[str]
    tags: list[str]
    input_schema: dict
    output_schema: dict
```

### 3.2 Agent Metadata

> 📄 **File:** `dopeagents/core/metadata.py` | **Status:**   Implemented | **Role:** Structured metadata model returned by `Agent.metadata()`

```python
# dopeagents/core/metadata.py

from pydantic import BaseModel
from typing import Any


class AgentMetadata(BaseModel):
    """Structured metadata about an agent, returned by Agent.metadata()."""
    name: str
    version: str
    description: str
    capabilities: list[str]
    tags: list[str] = []
    requires_llm: bool
    default_model: str | None = None
    system_prompt: str = ""
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}
```

### 3.3 Agent Context

> 📄 **File:** `dopeagents/core/context.py` | **Status:**   Implemented | **Role:** Execution context with run metadata, budget limits, and model overrides

```python
# dopeagents/core/context.py

from pydantic import BaseModel, Field
from typing import Any
from uuid import UUID, uuid4
from datetime import datetime, timezone


class AgentContext(BaseModel):
    """
    Execution context passed to agents.
    Contains metadata about the current execution but NO application state.
    """
    run_id: UUID = Field(default_factory=uuid4)
    trace_id: UUID | None = None
    parent_agent: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    environment: str = "development"
    max_cost_usd: float | None = None
    max_tokens: int | None = None
    model_override: str | None = None
    mcp_request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### 3.4 Agent Result Wrapper

> 📄 **File:** `dopeagents/core/types.py` | **Status:**   Implemented | **Role:** `AgentResult`, `ExecutionMetrics`, and `StepMetrics` — output wrapping with lifecycle metrics including per-step cost and latency

```python
# dopeagents/core/types.py

from pydantic import BaseModel, Field
from typing import TypeVar, Generic
from uuid import UUID
from datetime import datetime

OutputT = TypeVar("OutputT", bound=BaseModel)


class StepMetrics(BaseModel):
    """Cost and performance metrics for a single step in a multi-step agent."""
    name: str                          # Step name (e.g. "analyze", "evaluate")
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    metadata: dict = Field(default_factory=dict)  # Step-specific extras


class ExecutionMetrics(BaseModel):
    """Metrics collected by the lifecycle layer, NOT by the agent."""
    latency_ms: float
    token_count_in: int = 0
    token_count_out: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False
    retry_count: int = 0
    fallback_used: str | None = None
    # Multi-step agents: per-step breakdown
    steps: list[StepMetrics] = Field(default_factory=list)
    # Budget degradation tracking
    degraded: bool = False
    degradation_reason: str | None = None


class AgentResult(BaseModel, Generic[OutputT]):
    """Wraps agent output with execution metadata."""
    run_id: UUID
    agent_name: str
    agent_version: str
    timestamp: datetime
    output: OutputT
    metrics: ExecutionMetrics
    success: bool = True
    error: str | None = None
```

### 3.5 Concrete Agent Examples

#### Pattern A: Single-Step Agent

A single-step agent overrides `run()` directly and calls `self._extract()` once. This is the appropriate pattern for tasks that can be completed in a single LLM call:

```python
# dopeagents/agents/example_classifier.py
from typing import ClassVar, Literal
from pydantic import BaseModel, Field
from dopeagents.core.agent import Agent


class ClassifierInput(BaseModel):
    text: str
    categories: list[str]

class ClassifierOutput(BaseModel):
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

class Classifier(Agent[ClassifierInput, ClassifierOutput]):
    name: ClassVar[str] = "Classifier"
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = "Categorizes text into predefined labels"
    capabilities: ClassVar[list[str]] = ["classification", "categorization"]
    requires_llm: ClassVar[bool] = True
    default_model: ClassVar[str] = "openai/gpt-4o-mini"
    system_prompt: ClassVar[str] = (
        "You are a text classification agent. Choose the single best category."
    )

    def run(self, input: ClassifierInput, context=None) -> ClassifierOutput:
        cats = ", ".join(input.categories)
        return self._extract(
            response_model=ClassifierOutput,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Classify into [{cats}]:\n\n{input.text}"},
            ],
        )
```

Key design points:

- **Explicit prompt reference:** `self.system_prompt` appears in `messages` — what you read is what gets sent (INV-10, P8).
- **Internal extraction schema:** Where the LLM output schema differs from the public output, use a separate `_LLMOutput` class and post-process.
- **Thread-safe model override:** Resolve model from context inside `run()` and pass to `_extract()` — never mutate `self._model` (INV-2).

#### Pattern B: Multi-Step Agent

A multi-step agent implements `_build_graph()` to define an internal LangGraph workflow. The graph is built once per instance (lazy), stays private, and drives execution through multiple `_extract()` calls — one per step.

```python
# dopeagents/agents/example_multi_step.py
from __future__ import annotations
from typing import Annotated, Any, ClassVar, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from dopeagents.core.agent import Agent


# ── State ─────────────────────────────────────────────────────────────

class AnalysisState(TypedDict):
    text: str
    chunks: list[str]
    draft: str
    quality_score: float
    refined: str
    refinement_rounds: int

# ── Input / Output ─────────────────────────────────────────────────────

class AnalyzeInput(BaseModel):
    text: str

class AnalyzeOutput(BaseModel):
    result: str
    quality_score: float
    refinement_rounds: int

# ── Step output schemas (internal, used only in step methods) ──────────

class _AnalyzeStepOut(BaseModel):
    chunks: list[str]

class _DraftStepOut(BaseModel):
    draft: str

class _EvalStepOut(BaseModel):
    quality_score: float = Field(ge=0.0, le=1.0)
    feedback: str

class _RefineStepOut(BaseModel):
    refined: str

# ── Agent ──────────────────────────────────────────────────────────────

class AnalyzingAgent(Agent[AnalyzeInput, AnalyzeOutput]):
    name: ClassVar[str] = "AnalyzingAgent"
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = "Multi-step analysis with self-evaluation"
    capabilities: ClassVar[list[str]] = ["analysis", "self-evaluation"]
    requires_llm: ClassVar[bool] = True
    default_model: ClassVar[str] = "openai/gpt-4o-mini"
    step_prompts: ClassVar[dict[str, str]] = {
        "analyze":  "Split the text into logical chunks for analysis.",
        "draft":    "Draft an analysis based on the provided chunks.",
        "evaluate": "Score the draft quality (0.0–1.0). Provide actionable feedback.",
        "refine":   "Improve the draft using the evaluation feedback.",
    }

    def _build_graph(self):
        graph = StateGraph(AnalysisState)
        graph.add_node("analyze",  self._step_analyze)
        graph.add_node("draft",    self._step_draft)
        graph.add_node("evaluate", self._step_evaluate)
        graph.add_node("refine",   self._step_refine)
        graph.add_edge("analyze",  "draft")
        graph.add_edge("draft",    "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            lambda s: "refine" if s["quality_score"] < 0.8 else END,
        )
        graph.add_edge("refine",   "evaluate")
        graph.set_entry_point("analyze")
        return graph.compile()

    def _has_loops(self) -> bool:
        return True  # The evaluate→refine→evaluate cycle is a loop

    def run(self, input: AnalyzeInput, context=None) -> AnalyzeOutput:
        graph = self._get_graph()
        final_state = graph.invoke({"text": input.text, "refinement_rounds": 0})
        result = final_state.get("refined") or final_state["draft"]
        return AnalyzeOutput(
            result=result,
            quality_score=final_state["quality_score"],
            refinement_rounds=final_state["refinement_rounds"],
        )

    def _step_analyze(self, state: AnalysisState) -> dict[str, Any]:
        out: _AnalyzeStepOut = self._extract(
            response_model=_AnalyzeStepOut,
            messages=[
                {"role": "system", "content": self.step_prompts["analyze"]},
                {"role": "user",   "content": state["text"]},
            ],
            model=self._model_for_step("analyze"),
        )
        return {"chunks": out.chunks}

    def _step_draft(self, state: AnalysisState) -> dict[str, Any]:
        chunks_text = "\n\n".join(state["chunks"])
        out: _DraftStepOut = self._extract(
            response_model=_DraftStepOut,
            messages=[
                {"role": "system", "content": self.step_prompts["draft"]},
                {"role": "user",   "content": chunks_text},
            ],
            model=self._model_for_step("draft"),
        )
        return {"draft": out.draft}

    def _step_evaluate(self, state: AnalysisState) -> dict[str, Any]:
        out: _EvalStepOut = self._extract(
            response_model=_EvalStepOut,
            messages=[
                {"role": "system", "content": self.step_prompts["evaluate"]},
                {"role": "user",   "content": state.get("refined") or state["draft"]},
            ],
            model=self._model_for_step("evaluate"),
        )
        return {"quality_score": out.quality_score}

    def _step_refine(self, state: AnalysisState) -> dict[str, Any]:
        current = state.get("refined") or state["draft"]
        out: _RefineStepOut = self._extract(
            response_model=_RefineStepOut,
            messages=[
                {"role": "system", "content": self.step_prompts["refine"]},
                {"role": "user",   "content": current},
            ],
            model=self._model_for_step("refine"),
        )
        return {
            "refined": out.refined,
            "refinement_rounds": state["refinement_rounds"] + 1,
        }
```

Key design points for multi-step agents:

- **`step_prompts` as ClassVar:** All step prompts are declared at class level — inspectable by `describe()`, CLI, and observability layer without running the agent.
- **`_build_graph()` is private:** Callers use `run()`. The graph is never exposed.
- **Each step calls `self._extract()` once:** The lifecycle layer captures cost/latency per `_extract()` call, which maps to one step. Total metrics aggregate across all steps.
- **`_model_for_step(step_name)`:** Returns the per-step model override when set in `step_models`, or falls back to `self._model`.
- **`_has_loops() → True`:** Tells `describe()` and the CLI that this agent has a conditional refinement cycle.

> **Full reference implementation:** See §17 Core Agents — `DeepSummarizer` (7-step with refinement loop) and `ResearchAgent` (6-step).

---

### 3.6 `system_prompt` and `step_prompts` as First-Class Agent Attributes

#### Design Decision: Explicit Reference, Not Auto-Injection

Both `system_prompt` and `step_prompts` are declared as `ClassVar` on every agent. This makes them inspectable — by `describe()`, by the MCP Prompt primitive, by the observability layer, by benchmarking tools — without executing the agent.

There are two ways to wire the system prompt:

| Approach                     | How `_extract()` works                                                  | Trade-off                     |
| ---------------------------- | ------------------------------------------------------------------------- | ----------------------------- |
| **Auto-injection**     | `_extract()` silently prepends `self.system_prompt`                   | Less boilerplate, but magic   |
| **Explicit reference** | Agent writes `{"role": "system", "content": self.step_prompts["step"]}` | One extra line; fully visible |

**Decision: Explicit reference.** Matches P2 and P8: what you read in the step method is exactly what gets sent.

#### Pattern: Agent Without a System Prompt

Non-LLM agents keep the default empty string. Nothing special needed:

```python
class SchemaValidator(Agent[SchemaValidatorInput, SchemaValidatorOutput]):
    system_prompt = ""  # explicit: no prompt
    requires_llm = False

    def run(self, input, context=None):
        # Pure Python — no _extract(), no messages
        ...
```

#### Pattern: Dynamic System Prompt (Construction-Time Configuration)

Use a `@property` when the prompt depends on instance configuration, not per-call input:

```python
class Extractor(Agent[ExtractorInput, ExtractorOutput]):
    _default_system_prompt = "You are a structured data extraction agent."

    @property
    def system_prompt(self) -> str:
        base = self._default_system_prompt
        if self.strict_mode:
            base += " Never infer values that aren't explicitly stated in the text."
        return base
```

**Rule:** `system_prompt` captures the _identity_ of the agent (stable across calls). Per-input variation is _behavior_ and stays in `run()`.

#### Runtime Override

The `__init__` signature accepts `system_prompt` as an explicit keyword argument:

```python
# Default prompt
s = Summarizer()
s.system_prompt  # "You are a summarization agent..."

# Runtime override — instance attribute shadows class attribute
s2 = Summarizer(system_prompt="Summarize for legal professionals. Preserve all citations.")
s2.system_prompt  # "Summarize for legal professionals..."
```

This is standard Python: instance attribute shadowing a class attribute. No descriptors, no metaclass magic.

#### What This Unlocks

| Surface                    | Before                                                            | After                                                                  |
| -------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `.describe()` in Sandbox | No prompt shown                                                   | Prompt displayed; "(overridden)" suffix when runtime-overridden        |
| `.debug()`               | System prompt captured only if `_render_prompt()` is overridden | `self.system_prompt` readable before constructing input              |
| MCP Prompt primitive       | Hollow stub (no content without executing)                        | Real content via `_register_mcp_prompt()`                            |
| Observability / tracing    | Requires Instructor hook interception                             | `agent.system_prompt` readable directly in span attributes           |
| Benchmarking               | Prompt variation requires subclassing                             | Pass `system_prompt=` to constructor — prompt is a first-class axis |

#### What NOT to Build

| Temptation                            | Why not                                                                                                                             |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `user_prompt_template` attribute    | User messages depend on input, which varies per call. Imperative construction in `run()` is more expressive than template syntax. |
| Auto-injection in `_extract()`      | Violates P2/P8. What you see in `run()` must be exactly what gets sent.                                                           |
| Prompt versioning system              | Agent `version` already covers this. Bump the agent version when the prompt changes meaningfully.                                 |
| `system_prompt` as a required field | Breaks `SchemaValidator` and any deterministic agent. Default stays `""`.                                                       |

---

### 3.7 Per-Agent Input & Output Schemas

Each agent declares its own `InputT` and `OutputT` by specialising the generic base class. There is no shared input shape — the schema is entirely per-agent.

```python
class Summarizer(Agent[SummarizerInput, SummarizerOutput]): ...
class Classifier(Agent[ClassifierInput, ClassifierOutput]): ...
class SchemaValidator(Agent[SchemaValidatorInput, SchemaValidatorOutput]): ...
```

| Agent                     | Unique input fields                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------- |
| **Summarizer**      | `text`, `max_length`, `style`, `focus`, `max_key_points`                          |
| **Classifier**      | `text`, `categories`, `multi_label`, `multi_label_threshold`, `include_reasoning` |
| **Extractor**       | `text`, `extract_fields`, `output_format`                                             |
| **SchemaValidator** | `data`, `schema`, `strict`                                                            |

`InputT` is resolved at subclass definition time via `__orig_bases__`, making it available to `input_type()`, `output_type()`, contract checking, MCP schema generation, and all framework adapters — with no per-agent code.

---

## 4. Contract Verification Engine

### 4.1 Purpose

Validates that agents can be composed together **before execution**. Works for DopeAgents-to-DopeAgents composition. When agents are used inside external frameworks (LangGraph, LangChain), those frameworks handle their own data flow — but developers can still use `ContractChecker` as a pre-flight validation tool.

### 4.2 Compatibility Rules

**Rule 1 — Field Overlap:** At least one output field of agent A must match (by name and type) an input field of agent B.

**Rule 2 — Required Field Coverage:** All required fields of B's input must be satisfiable from A's output fields or from explicit field mappings.

**Rule 3 — Type Compatibility:**

| From          | To        | Result                      |
| ------------- | --------- | --------------------------- |
| `str`       | `str`   | ✓ Exact match              |
| `int`       | `float` | ✓ Coercion allowed         |
| `list[str]` | `list`  | ✓ Subtype allowed          |
| `str`       | `int`   | ✗ Incompatible — rejected |

### 4.3 Contract Types

> 📄 **File:** `dopeagents/contracts/types.py` | **Status:**   Implemented | **Role:** `FieldMapping`, `CompatibilityResult`, and `PipelineValidationError` models

```python
# dopeagents/contracts/types.py

from pydantic import BaseModel
from dopeagents.errors import ContractError


class FieldMapping(BaseModel):
    """Describes how one field maps between two agents."""
    source_field: str
    target_field: str
    type_coercion: str | None = None


class CompatibilityResult(BaseModel):
    """Result of checking whether two agents can be composed."""
    compatible: bool
    source_agent: str
    target_agent: str
    mappings: list[FieldMapping] = []
    errors: list[str] = []
    warnings: list[str] = []


class PipelineValidationError(ContractError):
    """Raised when a pipeline's agent chain has incompatible contracts."""
    pass
```

### 4.4 Contract Checker

> 📄 **File:** `dopeagents/contracts/checker.py` | **Status:**   Implemented | **Role:** Static composition validation with type compatibility checking and field mapping

```python
# dopeagents/contracts/checker.py

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from dopeagents.core.agent import Agent
from dopeagents.contracts.types import CompatibilityResult, FieldMapping


class ContractChecker:
    """
    Validates that two agents can be composed: Agent_A → Agent_B

    Works for:
    - DopeAgents Pipeline composition (full enforcement)
    - Pre-flight check before using agents in LangGraph/LangChain
      (advisory — the framework handles actual data flow)
    """

    @staticmethod
    def check(
        source: type[Agent],
        target: type[Agent],
        field_mappings: dict[str, str] | None = None,
    ) -> CompatibilityResult:
        source_output = source.output_type()
        target_input = target.input_type()
        source_fields = source_output.model_fields
        target_fields = target_input.model_fields
        mappings = field_mappings or {}
        errors: list[str] = []
        warnings: list[str] = []
        resolved_mappings: list[FieldMapping] = []

        for field_name, field_info in target_fields.items():
            if field_name in mappings:
                source_field_name = mappings[field_name]
                if source_field_name not in source_fields:
                    errors.append(
                        f"Mapped field '{source_field_name}' not found in "
                        f"{source.name} output"
                    )
                    continue
                type_compat = ContractChecker._check_type_compatibility(
                    source_fields[source_field_name], field_info
                )
                if not type_compat.compatible:
                    errors.append(
                        f"Type mismatch for mapping {source_field_name} → "
                        f"{field_name}: {type_compat.reason}"
                    )
                else:
                    resolved_mappings.append(FieldMapping(
                        source_field=source_field_name,
                        target_field=field_name,
                        type_coercion=type_compat.coercion,
                    ))
                continue

            if field_name in source_fields:
                type_compat = ContractChecker._check_type_compatibility(
                    source_fields[field_name], field_info
                )
                if not type_compat.compatible:
                    errors.append(
                        f"Field '{field_name}' exists in both schemas but types "
                        f"are incompatible: {type_compat.reason}"
                    )
                else:
                    resolved_mappings.append(FieldMapping(
                        source_field=field_name,
                        target_field=field_name,
                        type_coercion=type_compat.coercion,
                    ))
                continue

            if field_info.is_required():
                errors.append(
                    f"Required field '{field_name}' in {target.name} input "
                    f"has no match in {source.name} output and no explicit mapping"
                )
            else:
                warnings.append(
                    f"Optional field '{field_name}' in {target.name} input "
                    f"will use default value (not provided by {source.name})"
                )

        return CompatibilityResult(
            compatible=len(errors) == 0,
            source_agent=source.name,
            target_agent=target.name,
            mappings=resolved_mappings,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def _check_type_compatibility(
        source_field: FieldInfo, target_field: FieldInfo
    ) -> "TypeCompatibility":
        source_type = source_field.annotation
        target_type = target_field.annotation
        if source_type == target_type:
            return TypeCompatibility(compatible=True)
        if source_type is int and target_type is float:
            return TypeCompatibility(compatible=True, coercion="int→float")
        if target_type is str and source_type in (int, float, bool):
            return TypeCompatibility(
                compatible=True, coercion=f"{source_type.__name__}→str"
            )
        return TypeCompatibility(
            compatible=False,
            reason=f"{source_type} is not assignable to {target_type}",
        )


class TypeCompatibility(BaseModel):
    compatible: bool
    coercion: str | None = None
    reason: str | None = None
```

### 4.5 Pipeline Composition Validator

> 📄 **File:** `dopeagents/contracts/pipeline.py` | **Status:**   Implemented | **Role:** Multi-step agent pipeline with construction-time contract validation

```python
# dopeagents/contracts/pipeline.py

from dopeagents.core.agent import Agent
from dopeagents.contracts.checker import ContractChecker
from dopeagents.contracts.types import PipelineValidationError


class Pipeline:
    """
    Validates composition of multiple agents at construction time.

    Not an execution engine. Produces a validated sequence that can be
    executed by any runtime, or used as a pre-flight check before
    setting up a LangGraph/LangChain pipeline.
    """

    def __init__(
        self,
        steps: list[type[Agent]],
        field_mappings: dict[int, dict[str, str]] | None = None,
    ):
        self.steps = steps
        self.field_mappings = field_mappings or {}
        self._validate()

    def _validate(self):
        errors = []
        for i in range(len(self.steps) - 1):
            source = self.steps[i]
            target = self.steps[i + 1]
            step_mappings = self.field_mappings.get(i + 1)
            result = ContractChecker.check(source, target, step_mappings)
            if not result.compatible:
                errors.append(
                    f"Step {i} ({source.name}) → Step {i+1} ({target.name}): "
                    f"{'; '.join(result.errors)}"
                )
            for warning in result.warnings:
                import warnings as _warnings
                _warnings.warn(f"Step {i} → {i+1}: {warning}", UserWarning)
        if errors:
            raise PipelineValidationError(
                f"Pipeline validation failed:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

    def describe(self) -> str:
        lines = [f"Pipeline ({len(self.steps)} steps):"]
        for i, step in enumerate(self.steps):
            in_t = step.input_type().__name__
            out_t = step.output_type().__name__
            lines.append(f"  Step {i}: {step.name} ({in_t} → {out_t})")
        return "\n".join(lines)

    @property
    def input_type(self) -> type:
        return self.steps[0].input_type()

    @property
    def output_type(self) -> type:
        return self.steps[-1].output_type()
```

### 4.6 Using ContractChecker with External Frameworks

```python
# Pre-flight check before building a LangGraph pipeline
from dopeagents.contracts import ContractChecker

result = ContractChecker.check(
    source=Summarizer,
    target=Classifier,
    field_mappings={"text": "summary"},
)

if not result.compatible:
    print("⚠️ These agents may not compose cleanly in your LangGraph pipeline:")
    for error in result.errors:
        print(f"  • {error}")
    print("Consider adding appropriate state keys or field mappings.")
else:
    print("  Compatible. Safe to use in your pipeline.")

# Then build the LangGraph pipeline with confidence
graph.add_node("summarize", Summarizer().as_langgraph_node())
graph.add_node("classify", Classifier().as_langgraph_node(
    input_mapping={"text": "summary"}
))
```

---

## 5. Agent Lifecycle & Hooks

### 5.1 Lifecycle Execution Order

```text
User calls: executor.run(agent, input, context)
                    │
                    ▼
        ┌─ [1] Pre-execution ───────────────────┐
        │   • Input validation (Pydantic)         │
        │   • Budget check                        │
        │   • Cache lookup                         │
        │   • Trace span opened                    │
        │   • Instructor hooks attached            │
        └─────────────────────────────────────────┘
                    │
                    ▼ (cache miss)
        ┌─ [2] Agent execution ──────────────────┐
        │   • agent.run(input, context)            │
        │   │  └→ self._extract()                  │
        │   │      └→ Instructor client            │
        │   │          • LLM call                  │
        │   │          • Schema validation         │
        │   │          • Auto-retry on mismatch    │
        │   • Wrapped in retry logic               │
        │   • On failure: fallback chain            │
        └─────────────────────────────────────────┘
                    │
                    ▼
        ┌─ [3] Post-execution ──────────────────┐
        │   • Output validation (Pydantic)        │
        │   • Metrics collection from Instructor  │
        │     hooks + LiteLLM metadata (tokens,   │
        │     cost, latency)                       │
        │   • Cost recording                       │
        │   • Cache storage                        │
        │   • Trace span closed                    │
        │   • PII redaction for logs               │
        └─────────────────────────────────────────┘
                    │
                    ▼
            AgentResult[OutputT] returned
```

### 5.2 Agent Executor

> 📄 **File:** `dopeagents/lifecycle/executor.py` | **Status:**   Implemented | **Role:** Lifecycle wrapper — input/output validation, hook integration, metrics collection, error wrapping

```python
# dopeagents/lifecycle/executor.py

import time
from uuid import uuid4
from datetime import datetime, timezone
from dopeagents.core.agent import Agent, InputT, OutputT
from dopeagents.core.context import AgentContext
from dopeagents.core.types import AgentResult, ExecutionMetrics
from dopeagents.lifecycle.hooks import LifecycleHooks
from dopeagents.observability.tracer import Tracer
from dopeagents.cost.tracker import CostTracker
from dopeagents.cost.guard import BudgetGuard, BudgetConfig
from dopeagents.resilience.retry import RetryPolicy
from dopeagents.resilience.fallback import FallbackChain
from dopeagents.cache.manager import CacheManager
from dopeagents.errors import (
    AgentExecutionError,
    AllFallbacksFailedError,
    BudgetExceededError,
    OutputValidationError,
)


class AgentExecutor:
    """
    Executes agents with full lifecycle management.
    Wraps agent.run() with validation, cost tracking, retry, caching,
    and observability.
    """

    def __init__(
        self,
        tracer: Tracer | None = None,
        cost_tracker: CostTracker | None = None,
        cache_manager: CacheManager | None = None,
        budget: BudgetConfig | None = None,
        hooks: LifecycleHooks | None = None,
    ):
        self.tracer = tracer or Tracer.noop()
        self.cost_tracker = cost_tracker or CostTracker.noop()
        self.cache_manager = cache_manager
        self.budget = budget
        self.hooks = hooks or LifecycleHooks()

    def run(
        self,
        agent: Agent[InputT, OutputT],
        input: InputT,
        context: AgentContext | None = None,
        retry_policy: RetryPolicy | None = None,
        fallback_chain: FallbackChain | None = None,
    ) -> AgentResult[OutputT]:

        context = context or AgentContext()
        run_id = context.run_id
        start_time = time.monotonic()

        with self.tracer.span(
            name=f"agent.{agent.name}",
            run_id=run_id,
            trace_id=context.trace_id,
        ) as span:

            # [1] Pre-execution
            validated_input = agent.input_type().model_validate(
                input.model_dump() if hasattr(input, "model_dump") else input
            )

            BudgetGuard.check_pre_execution(
                agent=agent,
                context=context,
                cost_tracker=self.cost_tracker,
                budget=self.budget,
            )

            if self.cache_manager:
                cached = self.cache_manager.get(agent, validated_input)
                if cached is not None:
                    elapsed = (time.monotonic() - start_time) * 1000
                    span.set_attribute("cache_hit", True)
                    return AgentResult(
                        run_id=run_id,
                        agent_name=agent.name,
                        agent_version=agent.version,
                        timestamp=datetime.now(timezone.utc),
                        output=cached,
                        metrics=ExecutionMetrics(
                            latency_ms=elapsed, cache_hit=True
                        ),
                    )

            # Attach Instructor hooks for observability/cost capture
            self._attach_instructor_hooks(agent, span)

            # [2] Execution
            output = None
            retry_count = 0
            fallback_used = None

            try:
                output, retry_count = self._execute_with_retry(
                    agent, validated_input, context, retry_policy
                )
            except Exception as exec_error:
                if fallback_chain:
                    output, fallback_used = self._execute_fallback(
                        fallback_chain, validated_input, context
                    )
                else:
                    raise AgentExecutionError(
                        agent_name=agent.name, original_error=exec_error
                    ) from exec_error

            # [3] Post-execution
            validated_output = agent.output_type().model_validate(
                output.model_dump() if hasattr(output, "model_dump") else output
            )

            metrics = self._build_metrics_from_hooks(
                start_time, retry_count, fallback_used, span
            )

            self.cost_tracker.record(agent, context, metrics)

            if self.cache_manager:
                self.cache_manager.set(agent, validated_input, validated_output)

            span.set_attribute("agent.name", agent.name)
            span.set_attribute("agent.version", agent.version)
            span.set_attribute("metrics.latency_ms", metrics.latency_ms)
            span.set_attribute("metrics.cost_usd", metrics.cost_usd)

            return AgentResult(
                run_id=run_id,
                agent_name=agent.name,
                agent_version=agent.version,
                timestamp=datetime.now(timezone.utc),
                output=validated_output,
                metrics=metrics,
            )

    def _execute_with_retry(self, agent, input, context, retry_policy):
        if retry_policy is None:
            return agent.run(input, context), 0
        last_error = None
        for attempt in range(retry_policy.max_attempts):
            try:
                return agent.run(input, context), attempt
            except tuple(retry_policy.retryable_errors) as e:
                last_error = e
                if attempt < retry_policy.max_attempts - 1:
                    time.sleep(
                        retry_policy.delay_seconds
                        * (retry_policy.backoff_factor**attempt)
                    )
        raise last_error

    def _execute_fallback(self, fallback_chain, input, context):
        for fallback_agent in fallback_chain.agents:
            try:
                adapted_input = self._adapt_input(input, fallback_agent)
                output = fallback_agent.run(adapted_input, context)
                return output, fallback_agent.name
            except Exception:
                continue
        raise AllFallbacksFailedError(
            chain_agents=[a.name for a in fallback_chain.agents],
        )

    def _adapt_input(self, input, target_agent):
        """Attempt to adapt input for a fallback agent with a different schema."""
        target_type = type(target_agent).input_type()
        input_dict = input.model_dump()
        return target_type.model_validate(input_dict)

    def _attach_instructor_hooks(self, agent, span):
        """
        Wire lifecycle observability into the Instructor client's hook system.
        This captures token counts, cost, and the exact request/response
        without any code inside the agent's run() method.

        Also delegates to LifecycleHooks extraction methods so user-provided
        hooks can observe extraction events.
        """
        if not agent.requires_llm:
            return

        try:
            client = agent._get_client()
            context = None  # resolved per-call via executor.run()

            # Capture the exact request being sent (for debug/observability)
            client.on("completion:kwargs", lambda kwargs: (
                span.set_attribute("llm.model", kwargs.get("model", "unknown")),
                span.add_event("llm_request", {"messages_count": len(kwargs.get("messages", []))}),
                self.hooks.on_extraction_request(agent, kwargs.get("messages", []), kwargs.get("response_model"), context),
            ))

            # Capture token counts and cost from the response
            client.on("completion:response", lambda response: (
                span.set_attribute("llm.tokens_in", getattr(response.usage, "prompt_tokens", 0)),
                span.set_attribute("llm.tokens_out", getattr(response.usage, "completion_tokens", 0)),
                span.set_attribute(
                    "llm.cost_usd",
                    float((getattr(response, "_hidden_params", {}) or {}).get("response_cost", 0.0) or 0.0),
                ),
                self.hooks.on_extraction_response(agent, response, getattr(response, "usage", None), context),
            ))

            # Capture parse errors (Instructor validation failures before auto-retry)
            client.on("parse:error", lambda error: (
                span.add_event("instructor_parse_error", {"error": str(error)}),
                self.hooks.on_extraction_validation_error(agent, error, None, context),
            ))

        except Exception:
            # If hooks can't be attached (e.g., non-LLM agent), continue silently
            pass

    def _build_metrics_from_hooks(self, start_time, retry_count, fallback_used, span):
        """Build ExecutionMetrics from data captured via Instructor hooks."""
        elapsed = (time.monotonic() - start_time) * 1000
        return ExecutionMetrics(
            latency_ms=elapsed,
            token_count_in=span.attributes.get("llm.tokens_in", 0),
            token_count_out=span.attributes.get("llm.tokens_out", 0),
            llm_calls=1 if span.attributes.get("llm.model") else 0,
            cost_usd=span.attributes.get("llm.cost_usd", 0.0),
            retry_count=retry_count,
            fallback_used=fallback_used,
        )
```

### 5.3 Lifecycle Hooks

> 📄 **File:** `dopeagents/lifecycle/hooks.py` | **Status:**   Implemented | **Role:** Extension points for pre/post execution, error, retry, fallback, and Instructor extraction events

```python
# dopeagents/lifecycle/hooks.py

from dopeagents.core.agent import Agent
from dopeagents.core.context import AgentContext
from pydantic import BaseModel


class LifecycleHooks:
    """Extension points for custom logic at each lifecycle stage."""

    def pre_execution(self, agent: Agent, input: BaseModel, context: AgentContext) -> None:
        pass

    def post_execution(self, agent: Agent, input: BaseModel, output: BaseModel, context: AgentContext) -> None:
        pass

    def on_error(self, agent: Agent, input: BaseModel, error: Exception, context: AgentContext) -> None:
        pass

    def on_retry(self, agent: Agent, input: BaseModel, attempt: int, error: Exception, context: AgentContext) -> None:
        pass

    def on_fallback(self, original_agent: Agent, fallback_agent: Agent, context: AgentContext) -> None:
        pass

    # NEW: Instructor-level hooks
    def on_extraction_request(self, agent, messages, response_model, context):
        pass

    def on_extraction_response(self, agent, response, usage, context):
        pass

    def on_extraction_validation_error(self, agent, error, attempt, context):
        pass
```

---

## 6. Agent Sandbox

### 6.1 Purpose

The Agent Sandbox is a development-time tool for interactively running,
inspecting, and comparing agents without writing application code. It
exists to solve four problems:

1. During library development, contributors need to manually test agents
   as they build them without writing throwaway scripts.
2. During application development, users need to poke at agents to
   understand their behavior before wiring them into pipelines.
3. When evaluating DopeAgents, new users need a zero-friction way to
   see what agents do.
4. When exposing agents via MCP, developers need to verify tool schemas
   and test MCP registration before deploying to Claude Desktop or Cursor.

The sandbox is NOT an orchestration tool, NOT a notebook server, and
NOT a hosted service. It is a local CLI subcommand and Python module
that ships with the core package.

### 6.2 Architecture

The sandbox has three layers, each usable independently:

```text
┌─────────────────────────────────────────┐
│          CLI (`dopeagents`)       │
│  Parses args, calls SandboxRunner,       │
│  formats output via SandboxDisplay       │
├─────────────────────────────────────────┤
│          REPL (`dopeagents`)     │
│  Interactive Python console with         │
│  pre-loaded sandbox functions            │
├─────────────────────────────────────────┤
│          SandboxRunner (Python API)      │
│  load(), describe(), dry_run(),          │
│  run(), compare()                        │
│  Usable from notebooks, scripts, tests   │
├─────────────────────────────────────────┤
│     AgentExecutor    │    Agent.debug()   │
│     (lifecycle)      │    (introspection) │
└─────────────────────────────────────────┘
```

Key constraint: The REPL and CLI are thin shells. All logic lives in
`SandboxRunner`. This means the sandbox is fully usable as a Python
import without touching the CLI:

```python
from dopeagents.sandbox import SandboxRunner

runner = SandboxRunner()
agent = runner.load("Summarizer")
debug = runner.dry_run(agent, text="some text", style="bullets")
result = runner.run(agent, text="some text", style="bullets")
```

### 6.3 SandboxRunner

> 📄 **File:** `dopeagents/sandbox/runner.py` | **Status:** ○ Stub | **Role:** Core sandbox logic — load, describe, dry-run, run, compare, and MCP inspection

```python
# dopeagents/sandbox/runner.py

from dopeagents.core.agent import Agent
from dopeagents.core.context import AgentContext
from dopeagents.core.types import AgentResult
from dopeagents.lifecycle.executor import AgentExecutor
from dopeagents.errors import AgentNotFoundError
from pydantic import BaseModel


class ComparisonRow(BaseModel):
    agent_name: str
    cost_usd: float
    latency_ms: float
    output_preview: str
    success: bool


class ComparisonResult(BaseModel):
    rows: list[ComparisonRow]


class SandboxRunner:
    """
    Core sandbox logic. Usable programmatically from Python, notebooks,
    the CLI, or the REPL. No formatting — returns structured data.
    """

    # Built-in agent lookup. Simple dict, NOT the Phase 3 Registry.
    # Replaced by Registry.get() when Registry exists (Phase 3).
    _BUILTIN_AGENTS: dict[str, type[Agent]] = {}

    @classmethod
    def _register_builtins(cls):
        """Called once at import time to populate built-in agents."""
        if cls._BUILTIN_AGENTS:
            return
        # Lazy imports to avoid circular dependencies
        from dopeagents.agents.summarizer import Summarizer
        from dopeagents.agents.classifier import Classifier
        # Phase 1 agents
        for agent_cls in [Summarizer, Classifier]:
            cls._BUILTIN_AGENTS[agent_cls.name] = agent_cls
        # Phase 2 agents added when they ship:
        try:
            from dopeagents.agents.schema_validator import SchemaValidator
            cls._BUILTIN_AGENTS[SchemaValidator.name] = SchemaValidator
        except ImportError:
            pass
        # from dopeagents.agents.extractor import Extractor
        # from dopeagents.agents.web_searcher import WebSearcher

    def __init__(self, executor: AgentExecutor | None = None):
        self._executor = executor or AgentExecutor()
        self._register_builtins()

    def load(self, name_or_class) -> Agent:
        """
        Load an agent by name (string) or class.
        Returns an instantiated agent.
        """
        if isinstance(name_or_class, str):
            if name_or_class not in self._BUILTIN_AGENTS:
                available = ", ".join(sorted(self._BUILTIN_AGENTS.keys()))
                raise AgentNotFoundError(
                    agent_name=name_or_class,
                )
            return self._BUILTIN_AGENTS[name_or_class]()
        elif isinstance(name_or_class, type) and issubclass(name_or_class, Agent):
            return name_or_class()
        elif isinstance(name_or_class, Agent):
            return name_or_class
        else:
            raise TypeError(
                f"Expected agent name (str), Agent class, or Agent instance. "
                f"Got: {type(name_or_class)}"
            )

    def list_agents(self) -> list[dict]:
        """Returns metadata for all available agents."""
        return [
            {
                "name": cls.name,
                "version": cls.version,
                "requires_llm": cls.requires_llm,
                "capabilities": cls.capabilities,
            }
            for cls in self._BUILTIN_AGENTS.values()
        ]

    def describe(self, agent: Agent) -> dict:
        """Returns agent metadata including input/output schemas."""
        meta = type(agent).metadata()
        return meta.model_dump()

    def dry_run(self, agent: Agent, **kwargs) -> "DebugInfo":
        """
        Shows what the agent WOULD do without making any API calls.
        Builds input from kwargs, calls agent.debug().
        Raises InputValidationError if kwargs don't match input schema.
        """
        input_type = type(agent).input_type()
        validated_input = input_type.model_validate(kwargs)
        return agent.debug(validated_input)

    def run(
        self,
        agent: Agent,
        context: AgentContext | None = None,
        **kwargs,
    ) -> AgentResult:
        """
        Executes the agent through the full lifecycle (validation,
        cost tracking, metrics). Returns AgentResult with output
        and metrics.
        """
        input_type = type(agent).input_type()
        validated_input = input_type.model_validate(kwargs)
        if context is None:
            context = AgentContext()
        return self._executor.run(agent, validated_input, context)

    def compare(
        self,
        agents: list[Agent],
        **kwargs,
    ) -> ComparisonResult:
        """
        Runs the same input through multiple agents and returns
        a structured comparison. NOT a benchmark — single input only.
        """
        rows = []
        for agent in agents:
            try:
                result = self.run(agent, **kwargs)
                output_str = str(result.output.model_dump())
                preview = output_str[:80] + "..." if len(output_str) > 80 else output_str
                rows.append(ComparisonRow(
                    agent_name=agent.name,
                    cost_usd=result.metrics.cost_usd,
                    latency_ms=result.metrics.latency_ms,
                    output_preview=preview,
                    success=True,
                ))
            except Exception as e:
                rows.append(ComparisonRow(
                    agent_name=agent.name,
                    cost_usd=0.0,
                    latency_ms=0.0,
                    output_preview=f"ERROR: {e}",
                    success=False,
                ))
        return ComparisonResult(rows=rows)

    def inspect_mcp(self, agent: Agent) -> dict:
        """
        Shows how this agent would appear as an MCP tool — name, description,
        input schema, output schema — without starting an MCP server.
        Useful for verifying tool registration before deploying.
        """
        input_schema = type(agent).input_type().model_json_schema()
        output_schema = type(agent).output_type().model_json_schema()
        return {
            "mcp_tool_name": agent.name,
            "mcp_tool_description": agent.description,
            "mcp_input_schema": input_schema,
            "mcp_output_schema": output_schema,
            "mcp_capabilities": agent.capabilities,
        }
```

### 6.4 SandboxDisplay

> 📄 **File:** `dopeagents/sandbox/display.py` | **Status:** ○ Stub | **Role:** Terminal text formatting for sandbox output (describe, dry-run, result, comparison, MCP inspect)

```python
# dopeagents/sandbox/display.py

from dopeagents.sandbox.runner import ComparisonResult
from dopeagents.core.types import AgentResult


class SandboxDisplay:
    """
    Formats sandbox results as plain text for terminal output.
    Separate from SandboxRunner (SoC): runner returns data,
    display formats it. No external dependencies.
    """

    @staticmethod
    def format_describe(metadata: dict) -> str:
        lines = [
            f"{metadata['name']} v{metadata.get('version', '?')}",
            f"  {metadata.get('description', '')}",
            f"  LLM required: {metadata.get('requires_llm', '?')}",
        ]
        if metadata.get("default_model"):
            lines.append(f"  Model: {metadata['default_model']} (via Instructor)")
        prompt = metadata.get("system_prompt", "")
        if prompt:
            prompt_display = repr(prompt)
            if len(prompt_display) > 80:
                prompt_display = prompt_display[:77] + '..."'
            suffix = "  (overridden)" if metadata.get("system_prompt_overridden") else ""
            lines.append(f"  System: {prompt_display}{suffix}")
        lines.append(f"  Capabilities: {', '.join(metadata.get('capabilities', []))}")
        lines.append("")
        lines.append("  Input fields:")
        input_props = metadata.get("input_schema", {}).get("properties", {})
        input_required = set(metadata.get("input_schema", {}).get("required", []))
        for name, info in input_props.items():
            req = " (required)" if name in input_required else ""
            lines.append(f"    {name}: {info.get('type', '?')}{req}")
        lines.append("")
        lines.append("  Output fields:")
        output_props = metadata.get("output_schema", {}).get("properties", {})
        for name, info in output_props.items():
            lines.append(f"    {name}: {info.get('type', '?')}")
        return "\n".join(lines)

    @staticmethod
    def format_dry_run(debug_info) -> str:
        lines = [
            f"--- DRY RUN: {debug_info.agent_name} ---",
            "",
        ]
        if getattr(debug_info, "extraction_mode", None):
            lines.append(f"--- EXTRACTION MODE ---\n  {debug_info.extraction_mode}")
            lines.append("")
        if debug_info.prompt:
            lines.append("--- PROMPT ---")
            lines.append(debug_info.prompt)
            lines.append("")
        if debug_info.model_config_data:
            lines.append("--- MODEL CONFIG ---")
            for k, v in debug_info.model_config_data.items():
                lines.append(f"  {k}: {v}")
            lines.append("")
        lines.append("--- OUTPUT SCHEMA ---")
        output_props = debug_info.output_schema.get("properties", {})
        for name, info in output_props.items():
            lines.append(f"  {name}: {info.get('type', '?')}")
        lines.append("")
        lines.append("--- NO API CALL MADE ---")
        return "\n".join(lines)

    @staticmethod
    def format_result(result: AgentResult) -> str:
        lines = [
            f"--- RESULT: {result.agent_name} ---",
            "",
            "Output:",
        ]
        output_dict = result.output.model_dump()
        for k, v in output_dict.items():
            v_str = str(v)
            if len(v_str) > 200:
                v_str = v_str[:200] + "..."
            lines.append(f"  {k}: {v_str}")
        lines.append("")
        lines.append("Metrics:")
        lines.append(f"  Cost:    ${result.metrics.cost_usd:.4f}")
        lines.append(f"  Latency: {result.metrics.latency_ms:.0f}ms")
        lines.append(f"  Tokens:  {result.metrics.token_count_in} in / {result.metrics.token_count_out} out")
        if result.metrics.cache_hit:
            lines.append(f"  Cache:   Hit")
        return "\n".join(lines)

    @staticmethod
    def format_comparison(comparison: ComparisonResult) -> str:
        header = f"{'Agent':<25} {'Success':>8} {'Latency':>10} {'Cost':>10} {'Output':>30}"
        sep = "─" * len(header)
        rows = []
        sorted_rows = sorted(comparison.rows, key=lambda r: r.latency_ms)
        for r in sorted_rows:
            success = "✓" if r.success else "✗"
            preview = r.output_preview[:28] + ".." if len(r.output_preview) > 30 else r.output_preview
            rows.append(
                f"{r.agent_name:<25} {success:>8} {r.latency_ms:>8.0f}ms "
                f"${r.cost_usd:>8.4f} {preview:>30}"
            )
        return f"\n{header}\n{sep}\n" + "\n".join(rows) + "\n"

    @staticmethod
    def format_mcp_inspect(mcp_info: dict) -> str:
        lines = [
            f"--- MCP TOOL: {mcp_info['mcp_tool_name']} ---",
            f"  Description: {mcp_info['mcp_tool_description']}",
            "",
            "  Input Schema (JSON Schema):",
        ]
        input_props = mcp_info["mcp_input_schema"].get("properties", {})
        for name, info in input_props.items():
            lines.append(f"    {name}: {info.get('type', '?')}")
        lines.append("")
        lines.append("  Output Schema (JSON Schema):")
        output_props = mcp_info["mcp_output_schema"].get("properties", {})
        for name, info in output_props.items():
            lines.append(f"    {name}: {info.get('type', '?')}")
        return "\n".join(lines)
```

### 6.5 CLI Integration

The sandbox is exposed as a subcommand group on the `dopeagents` CLI entry point.

> 📄 **File:** `dopeagents/cli.py` | **Status:** ○ Stub | **Role:** CLI entry point with `sandbox` and `mcp` subcommand groups

```python
# dopeagents/cli.py (sandbox-relevant portion)

import click
import json
from dopeagents.sandbox.runner import SandboxRunner
from dopeagents.sandbox.display import SandboxDisplay


@click.group()
def main():
    """DopeAgents CLI."""
    pass


@main.group()
def mcp():
    """MCP server management."""
    pass


@main.group()
def sandbox():
    """Interactive agent testing and inspection."""
    pass


@sandbox.command("list")
def sandbox_list():
    """List all available agents."""
    runner = SandboxRunner()
    agents = runner.list_agents()
    for a in agents:
        llm = "LLM" if a["requires_llm"] else "no-LLM"
        click.echo(f"  {a['name']} v{a['version']} [{llm}]")


@sandbox.command("describe")
@click.argument("agent_name")
def sandbox_describe(agent_name):
    """Show agent metadata and schemas."""
    runner = SandboxRunner()
    agent = runner.load(agent_name)
    meta = runner.describe(agent)
    click.echo(SandboxDisplay.format_describe(meta))


@sandbox.command("dry-run")
@click.argument("agent_name")
@click.option("--input", "input_json", required=True, help="JSON input")
def sandbox_dry_run(agent_name, input_json):
    """Show what an agent would do without making API calls."""
    runner = SandboxRunner()
    agent = runner.load(agent_name)
    kwargs = json.loads(input_json)
    debug = runner.dry_run(agent, **kwargs)
    click.echo(SandboxDisplay.format_dry_run(debug))


@sandbox.command("run")
@click.argument("agent_name")
@click.option("--input", "input_json", required=True, help="JSON input")
@click.option("--model", default=None, help="Override model")
@click.option("--output-format", type=click.Choice(["text", "json"]), default="text")
def sandbox_run(agent_name, input_json, model, output_format):
    """Run an agent and show results with metrics."""
    runner = SandboxRunner()
    agent = runner.load(agent_name)
    kwargs = json.loads(input_json)
    context = None
    if model:
        from dopeagents.core.context import AgentContext
        context = AgentContext(model_override=model)
    result = runner.run(agent, context=context, **kwargs)
    if output_format == "json":
        click.echo(result.model_dump_json(indent=2))
    else:
        click.echo(SandboxDisplay.format_result(result))


@sandbox.command("compare")
@click.argument("agent_names", nargs=-1, required=True)
@click.option("--input", "input_json", required=True, help="JSON input")
def sandbox_compare(agent_names, input_json):
    """Compare multiple agents on the same input."""
    runner = SandboxRunner()
    agents = [runner.load(name) for name in agent_names]
    kwargs = json.loads(input_json)
    comparison = runner.compare(agents, **kwargs)
    click.echo(SandboxDisplay.format_comparison(comparison))


@sandbox.command("inspect-mcp")
@click.argument("agent_name")
def sandbox_inspect_mcp(agent_name):
    """Show how an agent would appear as an MCP tool."""
    runner = SandboxRunner()
    agent = runner.load(agent_name)
    mcp_info = runner.inspect_mcp(agent)
    click.echo(SandboxDisplay.format_mcp_inspect(mcp_info))


@sandbox.command("repl")
def sandbox_repl():
    """Launch interactive sandbox REPL."""
    from dopeagents.sandbox.repl import launch_repl
    launch_repl()


@mcp.command("serve")
@click.option("--agents", default=None, help="Comma-separated agent names (default: all)")
@click.option("--transport", type=click.Choice(["stdio", "streamable-http"]), default="stdio")
@click.option("--port", default=8000, help="Port for HTTP transport")
@click.option("--import", "import_module", default=None, help="Module to import for custom agents")
def mcp_serve(agents, transport, port, import_module):
    """Serve DopeAgents agents as MCP tools."""
    try:
        from dopeagents.adapters.mcp import create_mcp_server
    except ImportError:
        click.echo("FastMCP is required for MCP support: pip install dopeagents[mcp]")
        raise SystemExit(1)

    if import_module:
        import importlib
        importlib.import_module(import_module)

    agent_names = agents.split(",") if agents else None
    runner = SandboxRunner()

    if agent_names:
        agent_instances = [runner.load(name) for name in agent_names]
    else:
        agent_instances = None  # create_mcp_server uses all built-ins

    server = create_mcp_server(agents=agent_instances)

    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport="streamable-http", port=port)
```

### 6.6 REPL

> 📄 **File:** `dopeagents/sandbox/repl.py` | **Status:** ○ Stub | **Role:** Interactive Python console with pre-loaded sandbox functions

```python
# dopeagents/sandbox/repl.py

import code
from dopeagents.sandbox.runner import SandboxRunner
from dopeagents.sandbox.display import SandboxDisplay


def launch_repl():
    """
    Launches an interactive Python console with sandbox functions
    pre-loaded into the namespace.

    Uses stdlib code.InteractiveConsole. No IPython dependency.
    """
    runner = SandboxRunner()
    display = SandboxDisplay()

    def load(name_or_class):
        """Load an agent by name or class."""
        return runner.load(name_or_class)

    def describe(agent):
        """Show agent metadata and schemas."""
        meta = runner.describe(agent)
        print(display.format_describe(meta))

    def dry_run(agent, **kwargs):
        """Show what an agent would do without API calls."""
        debug = runner.dry_run(agent, **kwargs)
        print(display.format_dry_run(debug))
        return debug

    def run(agent, **kwargs):
        """Run an agent. Prints result AND returns AgentResult."""
        result = runner.run(agent, **kwargs)
        print(display.format_result(result))
        return result

    def compare(agents, **kwargs):
        """Compare agents on same input. Prints table AND returns result."""
        comparison = runner.compare(agents, **kwargs)
        print(display.format_comparison(comparison))
        return comparison

    def inspect_mcp(agent):
        """Show how an agent would appear as an MCP tool."""
        mcp_info = runner.inspect_mcp(agent)
        print(display.format_mcp_inspect(mcp_info))
        return mcp_info

    def list_agents():
        """List all available agents."""
        for a in runner.list_agents():
            llm = "LLM" if a["requires_llm"] else "no-LLM"
            print(f"  {a['name']} v{a['version']} [{llm}]")

    def help():
        """Show available sandbox commands."""
        print("""
Available commands:
  load(name)              Load an agent by name or class
  describe(agent)         Show agent metadata and schemas
  dry_run(agent, **kw)    Show prompt without API call
  run(agent, **kw)        Execute agent, print result
  compare([a1, a2], **kw) Compare agents on same input
  inspect_mcp(agent)      Show MCP tool schema for an agent
  list_agents()           List available agents
  help()                  This message

Examples:
  s = load("Summarizer")
  describe(s)
  dry_run(s, text="Hello world", style="bullets")
  r = run(s, text="Hello world", style="bullets")
  r.metrics.cost_usd
  compare([load("Summarizer"), load("Classifier")], text="Hello")
  inspect_mcp(s)
""")
        return ""  # Suppress None output in REPL

    namespace = {
        "load": load,
        "describe": describe,
        "dry_run": dry_run,
        "run": run,
        "compare": compare,
        "inspect_mcp": inspect_mcp,
        "list_agents": list_agents,
        "help": help,
    }

    agent_count = len(runner.list_agents())
    banner = (
        f"\nDopeAgents Sandbox\n"
        f"Available agents: {agent_count}\n"
        f"Type help() for commands.\n"
    )

    console = code.InteractiveConsole(locals=namespace)
    console.interact(banner=banner, exitmsg="")
```

### 6.7 Design Decisions

| ID     | Decision                                       | Rationale                                                                                                        |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| DD-SB1 | SandboxRunner is a plain Python class          | Usable from notebooks, scripts, tests, CLI, and REPL without any CLI dependency                                  |
| DD-SB2 | Display is separate from Runner                | Runner returns structured data (Pydantic models). Display formats for terminal. Different concerns.              |
| DD-SB3 | Built-in agent dict, not a mini-registry       | Simple dict is replaced by real Registry in Phase 3. No throwaway infrastructure.                                |
| DD-SB4 | REPL uses stdlib code.InteractiveConsole       | Zero additional dependencies. Users who want richer REPL use the Python API in IPython/Jupyter directly.         |
| DD-SB5 | click is a core dependency                     | CLI ships with core package. Sandbox must work on `pip install dopeagents` with no extras.                     |
| DD-SB6 | compare() is single-input, not a benchmark     | Quick side-by-side for development. Systematic evaluation uses BenchmarkRunner. Different tools, clear boundary. |
| DD-SB7 | Sandbox never modifies agent behavior (INV-8)  | Same agent.run() executes. Sandbox is a caller, not a wrapper.                                                   |
| DD-SB8 | `inspect_mcp()` does not start an MCP server | Read-only schema inspection. Starting a server is `dopeagents mcp serve`. Different concerns.                  |
| DD-SB9 | Dry run shows Instructor extraction mode       | Developers need to know whether the provider uses JSON mode, tool calling, etc.                                  |

### 6.8 What the Sandbox Is NOT

- **Not a hosted playground.** No web server, no auth, no infrastructure. Contradicts "pip-installable library."
- **Not a notebook integration.** `SandboxRunner` works in Jupyter natively because it's Python. No `%magic` commands needed.
- **Not a pipeline executor.** Runs single agents or compares agents on single inputs. Pipeline execution is the orchestration layer's job.
- **Not an MCP server.** The sandbox can inspect MCP schemas but does not serve MCP. Use `dopeagents mcp serve` for that. Different tools, different concerns.
- **Not session-persistent.** No save/replay. If needed later, serialize REPL history — 2-hour feature, not worth planning now.

---

## 7. Observability System

### 7.1 Tracer Interface

> 📄 **File:** `dopeagents/observability/tracer.py` | **Status:** ○ Stub | **Role:** `Tracer` ABC, `Span`, `NoopTracer`, and `ConsoleTracer`

```python
# dopeagents/observability/tracer.py

from abc import ABC, abstractmethod
from contextlib import contextmanager
from uuid import UUID
from typing import Any, Generator


class Span:
    def __init__(self, name: str, run_id: UUID, trace_id: UUID | None = None):
        self.name = name
        self.run_id = run_id
        self.trace_id = trace_id
        self.attributes: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append({"name": name, "attributes": attributes or {}})


class Tracer(ABC):
    @abstractmethod
    @contextmanager
    def span(self, name: str, run_id: UUID, trace_id: UUID | None = None) -> Generator[Span, None, None]:
        ...

    @classmethod
    def noop(cls) -> "Tracer":
        return NoopTracer()


class NoopTracer(Tracer):
    @contextmanager
    def span(self, name, run_id, trace_id=None):
        yield Span(name=name, run_id=run_id, trace_id=trace_id)


class ConsoleTracer(Tracer):
    @contextmanager
    def span(self, name, run_id, trace_id=None):
        span = Span(name=name, run_id=run_id, trace_id=trace_id)
        print(f"[TRACE] ▶ {name} (run={run_id})")
        try:
            yield span
        finally:
            print(f"[TRACE] ◀ {name} | {span.attributes}")
```

> 📄 **File:** `dopeagents/observability/otel.py` | **Status:** ○ Stub | **Role:** OpenTelemetry-backed tracer bridging DopeAgents spans to OTel

```python
# dopeagents/observability/otel.py

from contextlib import contextmanager
from uuid import UUID
from typing import Generator
from dopeagents.observability.tracer import Tracer, Span


class OTelTracer(Tracer):
    """
    OpenTelemetry-backed tracer. Bridges DopeAgents spans to OTel spans
    so they appear in Jaeger, Honeycomb, Grafana Tempo, etc.

    Requires: pip install dopeagents[otel]
    (opentelemetry-api + opentelemetry-sdk)
    """

    def __init__(self, service_name: str = "dopeagents"):
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource
        except ImportError:
            raise ImportError(
                "OpenTelemetry is required for OTelTracer.\n"
                "Install with: pip install dopeagents[otel]"
            )
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        otel_trace.set_tracer_provider(provider)
        self._tracer = otel_trace.get_tracer("dopeagents")

    @contextmanager
    def span(self, name: str, run_id: UUID, trace_id: UUID | None = None) -> Generator[Span, None, None]:
        dopeagents_span = Span(name=name, run_id=run_id, trace_id=trace_id)
        with self._tracer.start_as_current_span(name) as otel_span:
            otel_span.set_attribute("dopeagents.run_id", str(run_id))
            if trace_id:
                otel_span.set_attribute("dopeagents.trace_id", str(trace_id))
            try:
                yield dopeagents_span
            finally:
                for key, value in dopeagents_span.attributes.items():
                    otel_span.set_attribute(key, value)
                for event in dopeagents_span.events:
                    otel_span.add_event(event["name"], event.get("attributes", {}))
```

### 7.2 Instructor Hooks Callback

> 📄 **File:** `dopeagents/observability/instructor_hooks.py` | **Status:** ○ Not yet created | **Role:** Hooks into Instructor event system for LLM call observability

```python
# dopeagents/observability/instructor_hooks.py

from dopeagents.observability.tracer import Span


class InstructorObservabilityHooks:
    """
    Hooks into Instructor's event system to capture LLM call details
    for observability in the Instructor + LiteLLM stack.

    Instructor emits events at each stage of the extraction lifecycle:
    - completion:kwargs   → before the LLM call (capture prompt, model)
    - completion:response → after the LLM call (capture tokens, LiteLLM cost)
    - completion:error    → on LLM call failure
    - parse:error         → on output validation failure (before Instructor auto-retry)
    """

    def __init__(self, span: Span):
        self.span = span

    def on_completion_kwargs(self, kwargs: dict) -> None:
        """Capture the exact request being sent to the LLM."""
        self.span.set_attribute("llm.model", kwargs.get("model", "unknown"))
        self.span.set_attribute("llm.messages_count", len(kwargs.get("messages", [])))
        self.span.add_event("llm_request_sent", {
            "model": kwargs.get("model", "unknown"),
        })

    def on_completion_response(self, response) -> None:
        """Capture token counts + LiteLLM-computed cost from the LLM response."""
        usage = getattr(response, "usage", None)
        if usage:
            self.span.set_attribute("llm.tokens_in", getattr(usage, "prompt_tokens", 0))
            self.span.set_attribute("llm.tokens_out", getattr(usage, "completion_tokens", 0))
        hidden = getattr(response, "_hidden_params", {}) or {}
        if isinstance(hidden, dict):
            self.span.set_attribute("llm.cost_usd", float(hidden.get("response_cost", 0.0) or 0.0))

    def on_completion_error(self, error: Exception) -> None:
        """Capture LLM call failures."""
        self.span.add_event("llm_call_error", {"error": str(error), "type": type(error).__name__})

    def on_parse_error(self, error: Exception) -> None:
        """Capture Instructor validation failures (before auto-retry with error feedback)."""
        self.span.add_event("instructor_validation_error", {
            "error": str(error),
            "note": "Instructor will auto-retry with validation error in prompt",
        })

    def attach(self, client) -> None:
        """Attach all hooks to an Instructor client."""
        client.on("completion:kwargs", self.on_completion_kwargs)
        client.on("completion:response", self.on_completion_response)
        client.on("completion:error", self.on_completion_error)
        client.on("parse:error", self.on_parse_error)
```

### 7.3 Observability Contract

> 📄 **File:** `dopeagents/observability/contract.py` | **Status:** ○ Stub | **Role:** PII field declarations and custom metrics for observability

```python
# dopeagents/observability/contract.py

from pydantic import BaseModel


class ObservabilityContract(BaseModel):
    pii_fields: list[str] = []
    custom_metrics: list[str] = []
```

---

## 8. Cost Management & Budget Guards

### 8.1 Cost Tracker

> 📄 **File:** `dopeagents/cost/tracker.py` | **Status:** ○ Stub | **Role:** Cumulative cost tracking per-agent and globally, thread-safe

```python
# dopeagents/cost/tracker.py

from collections import defaultdict
from threading import Lock
from dopeagents.core.agent import Agent
from dopeagents.core.context import AgentContext
from dopeagents.core.types import ExecutionMetrics


class CostTracker:
    """
    Tracks cumulative cost across agent executions.
    Token and request metadata are captured via Instructor hooks.
    Cost is read from LiteLLM's response metadata (`response._hidden_params["response_cost"]`)
    when available, then recorded via lifecycle metrics.
    """

    def __init__(self):
        self._costs: dict[str, float] = defaultdict(float)
        self._global_cost: float = 0.0
        self._call_counts: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def record(self, agent: Agent, context: AgentContext, metrics: ExecutionMetrics) -> None:
        with self._lock:
            self._costs[agent.name] += metrics.cost_usd
            self._global_cost += metrics.cost_usd
            self._call_counts[agent.name] += 1

    def get_agent_cost(self, agent_name: str) -> float:
        return self._costs[agent_name]

    def get_total_cost(self) -> float:
        return self._global_cost

    def get_summary(self) -> dict:
        return {
            "total_cost_usd": self._global_cost,
            "by_agent": {
                name: {
                    "cost_usd": self._costs[name],
                    "calls": self._call_counts[name],
                    "avg_cost": (
                        self._costs[name] / self._call_counts[name]
                        if self._call_counts[name] > 0
                        else 0
                    ),
                }
                for name in self._costs
            },
        }

    @classmethod
    def noop(cls) -> "CostTracker":
        return cls()
```

### 8.2 Budget Guard

> 📄 **File:** `dopeagents/cost/guard.py` | **Status:** ○ Stub | **Role:** Pre-execution budget checks with configurable per-call, per-step, per-agent, and global limits; supports degradation mode for multi-step agents

```python
# dopeagents/cost/guard.py

from pydantic import BaseModel
from dopeagents.errors import BudgetExceededError


class BudgetConfig(BaseModel):
    max_cost_per_call: float | None = None   # Total cost cap per agent.run() call
    max_cost_per_step: float | None = None   # Cost cap for each step in a multi-step agent
    max_cost_per_agent: float | None = None  # Cumulative cost cap across all calls to one agent
    max_cost_global: float | None = None     # Cumulative global cap across all agents
    max_refinement_loops: int | None = None  # Max iterations of self-evaluation loops
    on_exceeded: str = "error"               # "error" | "warn" | "degrade"
    # "degrade": instead of raising, return the best result produced so far.
    #            Only meaningful for multi-step agents with partial results.
    #            Falls back to "error" for single-step agents.


class BudgetGuard:
    @staticmethod
    def check_pre_execution(agent, context, cost_tracker, budget=None):
        if budget is None and (context is None or context.max_cost_usd is None):
            return
        if budget and budget.max_cost_per_agent is not None:
            current = cost_tracker.get_agent_cost(agent.name)
            if current >= budget.max_cost_per_agent:
                BudgetGuard._handle_exceeded(
                    f"Agent '{agent.name}' budget exceeded: "
                    f"${current:.4f} >= ${budget.max_cost_per_agent:.4f}",
                    budget.on_exceeded,
                )
        if budget and budget.max_cost_global is not None:
            total = cost_tracker.get_total_cost()
            if total >= budget.max_cost_global:
                BudgetGuard._handle_exceeded(
                    f"Global budget exceeded: ${total:.4f} >= ${budget.max_cost_global:.4f}",
                    budget.on_exceeded,
                )

    @staticmethod
    def check_step_budget(step_name: str, step_cost: float, budget: "BudgetConfig | None"):
        """Called after each step to enforce per-step cost limits."""
        if budget is None or budget.max_cost_per_step is None:
            return
        if step_cost >= budget.max_cost_per_step:
            BudgetGuard._handle_exceeded(
                f"Step '{step_name}' cost ${step_cost:.4f} exceeded "
                f"max_cost_per_step ${budget.max_cost_per_step:.4f}",
                budget.on_exceeded,
            )

    @staticmethod
    def _handle_exceeded(message, action):
        if action == "error":
            raise BudgetExceededError(message)
        elif action == "warn":
            import warnings
            warnings.warn(message, ResourceWarning)
        elif action == "degrade":
            from dopeagents.errors import BudgetDegradedError
            raise BudgetDegradedError(message)
        # Unknown action: treat as "error"
        else:
            raise BudgetExceededError(message)
```

---

## 9. Resilience Layer — Retry, Fallback, Degradation

### 9.0 Three Levels of Resilience

DopeAgents handles resilience at three distinct levels:

**Level 1 — Extraction-level (Instructor):** When an LLM returns output that doesn't match the Pydantic schema, Instructor automatically re-asks the LLM with the validation error message. This is transparent to the agent author and handles the most common failure mode: malformed LLM output. Configured via the `max_retries` parameter on `_extract()`.

**Level 2 — Step-level (DopeAgents):** For multi-step agents, timeouts, rate limits, and network errors that occur within a single step are handled by the DopeAgents resilience layer via `RetryPolicy`. This wraps a single `_extract()` call inside a step. The resulting per-step metrics (including retry count) appear in `ExecutionMetrics.steps`.

**Level 3 — Agent-level (DegradationChain):** Complete agent failures are handled by `FallbackChain` / `DegradationChain`. This wraps the entire `agent.run()` call. When an agent in the chain succeeds, the result includes `agent_used` and `degradation_reason` so callers know which agent actually produced the output.

These three levels compose naturally:

- Instructor retries (Level 1) handle schema validation failures _within_ a single `_extract()` call
- DopeAgents step retries (Level 2) handle infrastructure failures _across_ step execution attempts
- DegradationChain (Level 3) handles complete agent failure by switching to the next agent in the chain

### 9.1 Retry Policy

Note: RetryPolicy handles infrastructure-level errors (timeouts, rate limits,
connection failures) at the step level. Schema validation retries are handled
automatically by Instructor within `_extract()` and do not count toward
RetryPolicy attempts.

> 📄 **File:** `dopeagents/resilience/retry.py` | **Status:** ○ Stub | **Role:** Configurable retry with backoff and retryable error types

```python
# dopeagents/resilience/retry.py

from pydantic import BaseModel, Field


class RetryPolicy(BaseModel):
    max_attempts: int = Field(default=3, ge=1, le=10)
    delay_seconds: float = Field(default=1.0, ge=0)
    backoff_factor: float = Field(default=2.0, ge=1.0)
    retryable_errors: list[type[Exception]] = Field(
        default_factory=lambda: [TimeoutError, ConnectionError]
    )

    class Config:
        arbitrary_types_allowed = True
```

### 9.2 Fallback & Degradation Chains

> 📄 **File:** `dopeagents/resilience/fallback.py` | **Status:** ○ Stub | **Role:** Ordered agent fallback with output compatibility validation, result includes `agent_used` and `degradation_reason`

```python
# dopeagents/resilience/fallback.py

from dopeagents.core.agent import Agent


class FallbackChain:
    def __init__(self, agents: list[Agent]):
        self.agents = agents
        self._validate_output_compatibility()

    def _validate_output_compatibility(self):
        if len(self.agents) < 2:
            return
        primary_output = type(self.agents[0]).output_type()
        for agent in self.agents[1:]:
            fallback_output = type(agent).output_type()
            primary_fields = set(primary_output.model_fields.keys())
            fallback_fields = set(fallback_output.model_fields.keys())
            missing = primary_fields - fallback_fields
            if missing:
                import warnings
                warnings.warn(
                    f"Fallback agent '{agent.name}' missing output fields: {missing}",
                    UserWarning,
                )


```

> 📄 **File:** `dopeagents/resilience/degradation.py` | **Status:** ○ Stub | **Role:** Specialized fallback ordering from most-capable to cheapest/most-reliable; executor result includes `agent_used` and `degradation_reason`

```python
# dopeagents/resilience/degradation.py

from typing import Any
from dopeagents.resilience.fallback import FallbackChain
from pydantic import BaseModel


class DegradationResult(BaseModel):
    """Wraps the output with context about which agent produced it."""
    output: Any
    agent_used: str          # Name of the agent that actually produced the output
    degradation_reason: str | None = None  # Why earlier agents were skipped (None = primary succeeded)


class DegradationChain(FallbackChain):
    """Ordered chain from most-capable to cheapest/most-reliable."""

    def __init__(self, agents: list):
        super().__init__(agents)
        last_agent_class = type(agents[-1])
        if last_agent_class.requires_llm:
            import warnings
            warnings.warn(
                f"Last agent in degradation chain ('{agents[-1].name}') requires LLM. "
                f"Consider adding a deterministic fallback as final option.",
                UserWarning,
            )

    def run_with_degradation(self, input, context=None) -> DegradationResult:
        """
        Try each agent in order. Return on first success, including
        which agent was used and why earlier agents were skipped.

        The lifecycle layer wraps this method — callers always get a
        DegradationResult, never a bare exception (unless all agents fail).
        """
        errors: list[str] = []
        for agent in self.agents:
            try:
                output = agent.run(input, context)
                reason = (
                    "; ".join(errors)
                    if errors else None
                )
                return DegradationResult(
                    output=output,
                    agent_used=agent.name,
                    degradation_reason=reason,
                )
            except Exception as e:
                errors.append(f"{agent.name}: {type(e).__name__}: {e}")
        # All agents failed
        from dopeagents.errors import AgentExecutionError
        raise AgentExecutionError(
            f"All agents in degradation chain failed: {'; '.join(errors)}"
        )
```

---

## 10. Caching Layer

> 📄 **File:** `dopeagents/cache/manager.py` | **Status:** ○ Stub | **Role:** `CacheManager` ABC and `InMemoryCache` with TTL support

```python
# dopeagents/cache/manager.py

from abc import ABC, abstractmethod
from pydantic import BaseModel
from dopeagents.core.agent import Agent
import hashlib
import json
import time


class CacheManager(ABC):
    @abstractmethod
    def get(self, agent: Agent, input: BaseModel) -> BaseModel | None: ...

    @abstractmethod
    def set(self, agent: Agent, input: BaseModel, output: BaseModel, ttl: int | None = None) -> None: ...

    @abstractmethod
    def invalidate(self, agent: Agent, input: BaseModel | None = None) -> None: ...

    @staticmethod
    def _build_key(agent: Agent, input: BaseModel) -> str:
        key_data = {
            "agent": agent.name,
            "version": agent.version,
            "input": input.model_dump(mode="json"),
        }
        key_json = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_json.encode()).hexdigest()


class InMemoryCache(CacheManager):
    def __init__(self):
        self._store: dict[str, tuple[BaseModel, float | None]] = {}

    def get(self, agent, input):
        key = self._build_key(agent, input)
        if key in self._store:
            value, expiry = self._store[key]
            if expiry is None or expiry > time.time():
                return value
            del self._store[key]
        return None

    def set(self, agent, input, output, ttl=None):
        key = self._build_key(agent, input)
        expiry = (time.time() + ttl) if ttl else None
        self._store[key] = (output, expiry)

    def invalidate(self, agent, input=None):
        if input:
            key = self._build_key(agent, input)
            self._store.pop(key, None)
        else:
            self._store.clear()
```

> 📄 **File:** `dopeagents/cache/disk.py` | **Status:** ○ Stub | **Role:** Persistent disk-based cache via `diskcache` (optional dependency)

```python
# dopeagents/cache/disk.py

from dopeagents.cache.manager import CacheManager
from pydantic import BaseModel
from dopeagents.core.agent import Agent


def _check_installed():
    try:
        import diskcache  # noqa: F401
    except ImportError:
        raise ImportError(
            "diskcache is required for DiskCache.\n"
            "Install with: pip install dopeagents[cache]"
        )


class DiskCache(CacheManager):
    def __init__(self, directory: str = ".dopeagents_cache"):
        _check_installed()
        import diskcache
        self._cache = diskcache.Cache(directory)

    def get(self, agent: Agent, input: BaseModel) -> BaseModel | None:
        key = self._build_key(agent, input)
        return self._cache.get(key)

    def set(self, agent: Agent, input: BaseModel, output: BaseModel, ttl: int | None = None) -> None:
        key = self._build_key(agent, input)
        self._cache.set(key, output, expire=ttl)

    def invalidate(self, agent: Agent, input: BaseModel | None = None) -> None:
        if input:
            key = self._build_key(agent, input)
            self._cache.delete(key)
        else:
            self._cache.clear()
```

---

## 11. Cross-Framework Adapters

### 11.1 Design Philosophy

1. **Thin** — adapters contain no business logic; they only convert data shapes.
2. **Non-modifying** — adapters never change agent behavior.
3. **Translation only** — adapters only translate input/output formats.
4. **Optional and lazy-imported** — framework dependencies are never mandatory at install time.
5. **Inherited** — adapter methods live on the `Agent` base class; every subclass gets them for free.
6. **Universal** — every agent gets every adapter automatically, with no per-agent code.
7. **Clear errors** — a missing framework dependency raises a helpful `ImportError` that includes the install command.
8. **Protocol-native** — MCP adapters expose agents as network services using the standard MCP protocol. The adapter handles transport, schema generation, and request/response mapping. The agent knows nothing about MCP.
9. **Schema reuse** — MCP tool schemas are generated directly from the same Pydantic models used for contracts, validation, and framework adapters. No separate schema definition needed.

### 11.2 LangGraph Adapter

> 📄 **File:** `dopeagents/adapters/langgraph.py` | **Status:** ○ Stub | **Role:** Converts agent to LangGraph `StateGraph` node function with state key mapping

```python
# dopeagents/adapters/langgraph.py

from typing import Any
from dopeagents.core.agent import Agent
from dopeagents.core.context import AgentContext


def _check_installed():
    try:
        import langgraph  # noqa: F401
    except ImportError:
        raise ImportError(
            "LangGraph is required for this adapter.\n"
            "Install with: pip install dopeagents[langgraph]"
        )


def to_langgraph_node(
    agent: Agent,
    input_mapping: dict[str, str] | None = None,
    output_mapping: dict[str, str] | None = None,
    context_factory: callable | None = None,
) -> callable:
    """
    Converts a DopeAgents agent into a LangGraph node function.
    LangGraph nodes: state (dict) → state updates (dict)

    Mapping directions:
      input_mapping:  {agent_field: state_key}  — agent field ← state key
      output_mapping: {state_key: agent_field}  — state key  ← agent field
    """
    _check_installed()

    input_type = type(agent).input_type()
    input_map = input_mapping or {}
    output_map = output_mapping or {}

    def node_function(state: dict[str, Any]) -> dict[str, Any]:
        input_fields = {}
        for field_name in input_type.model_fields:
            state_key = input_map.get(field_name, field_name)
            if state_key in state:
                input_fields[field_name] = state[state_key]

        agent_input = input_type.model_validate(input_fields)
        context = context_factory(state) if context_factory else None
        output = agent.run(agent_input, context)

        output_dict = output.model_dump()
        if output_map:
            return {
                state_key: output_dict[agent_field]
                for state_key, agent_field in output_map.items()
                if agent_field in output_dict
            }
        return output_dict

    node_function.__name__ = f"dopeagents_{agent.name}"
    node_function.__doc__ = agent.description
    node_function._dopeagents_agent = agent
    return node_function
```

### 11.3 LangChain Adapter

> 📄 **File:** `dopeagents/adapters/langchain.py` | **Status:** ○ Stub | **Role:** Converts agent to LangChain `Runnable` or `BaseTool`

```python
# dopeagents/adapters/langchain.py

from typing import Any
from dopeagents.core.agent import Agent


def _check_installed():
    try:
        from langchain_core.runnables import RunnableSerializable  # noqa: F401
        from langchain_core.tools import BaseTool  # noqa: F401
    except ImportError:
        raise ImportError(
            "LangChain is required for this adapter.\n"
            "Install with: pip install dopeagents[langchain]"
        )


def to_langchain_runnable(
    agent: Agent,
    input_key: str | None = None,
    output_key: str | None = None,
) -> Any:
    """Converts a DopeAgents agent into a LangChain Runnable."""
    _check_installed()
    from langchain_core.runnables import RunnableSerializable

    input_type = type(agent).input_type()

    class DopeAgentsRunnable(RunnableSerializable):
        name: str = agent.name

        class Config:
            arbitrary_types_allowed = True

        def invoke(self, input: Any, config: Any = None, **kwargs) -> Any:
            if isinstance(input, dict):
                agent_input = input_type.model_validate(input)
            elif isinstance(input, str) and input_key:
                agent_input = input_type.model_validate({input_key: input})
            elif isinstance(input, input_type):
                agent_input = input
            else:
                agent_input = input_type.model_validate(
                    input if isinstance(input, dict) else {"text": str(input)}
                )

            output = agent.run(agent_input)
            output_dict = output.model_dump()

            if output_key and output_key in output_dict:
                return output_dict[output_key]
            return output_dict

        async def ainvoke(self, input: Any, config: Any = None, **kwargs) -> Any:
            return self.invoke(input, config, **kwargs)

    return DopeAgentsRunnable()


def to_langchain_tool(agent: Agent, description_override: str | None = None) -> Any:
    """Converts a DopeAgents agent into a LangChain Tool."""
    _check_installed()
    from langchain_core.tools import BaseTool

    input_type = type(agent).input_type()

    class DopeAgentsTool(BaseTool):
        name: str = agent.name
        description: str = description_override or agent.description
        args_schema: type = input_type

        def _run(self, **kwargs) -> str:
            agent_input = input_type.model_validate(kwargs)
            output = agent.run(agent_input)
            return output.model_dump_json()

        async def _arun(self, **kwargs) -> str:
            return self._run(**kwargs)

    return DopeAgentsTool()
```

### 11.4 CrewAI Adapter

> 📄 **File:** `dopeagents/adapters/crewai.py` | **Status:** ○ Stub | **Role:** Converts agent to CrewAI `BaseTool`

```python
# dopeagents/adapters/crewai.py

from typing import Any
from dopeagents.core.agent import Agent


def _check_installed():
    try:
        from crewai.tools import BaseTool  # noqa: F401
    except ImportError:
        raise ImportError(
            "CrewAI is required for this adapter.\n"
            "Install with: pip install dopeagents[crewai]"
        )


def to_crewai_tool(agent: Agent, description_override: str | None = None) -> Any:
    """Converts a DopeAgents agent into a CrewAI tool."""
    _check_installed()
    from crewai.tools import BaseTool as CrewAIBaseTool

    input_type = type(agent).input_type()

    class DopeAgentsCrewTool(CrewAIBaseTool):
        name: str = agent.name
        description: str = description_override or agent.description

        def _run(self, **kwargs) -> str:
            agent_input = input_type.model_validate(kwargs)
            output = agent.run(agent_input)
            return output.model_dump_json()

    return DopeAgentsCrewTool()
```

### 11.5 AutoGen Adapter

> 📄 **File:** `dopeagents/adapters/autogen.py` | **Status:** ○ Stub | **Role:** Converts agent to AutoGen function calling format

```python
# dopeagents/adapters/autogen.py

from typing import Any
from dopeagents.core.agent import Agent


def _check_installed():
    try:
        import autogen_agentchat  # noqa: F401
    except ImportError:
        raise ImportError(
            "AutoGen is required for this adapter.\n"
            "Install with: pip install dopeagents[autogen]"
        )


def to_autogen_function(agent: Agent, description_override: str | None = None) -> dict[str, Any]:
    """Converts a DopeAgents agent into AutoGen function format."""
    _check_installed()

    input_type = type(agent).input_type()

    def agent_function(**kwargs) -> str:
        agent_input = input_type.model_validate(kwargs)
        output = agent.run(agent_input)
        return output.model_dump_json()

    agent_function.__name__ = agent.name
    agent_function.__doc__ = description_override or agent.description

    return {
        "function": agent_function,
        "name": agent.name,
        "description": description_override or agent.description,
        "parameters": input_type.model_json_schema(),
    }
```

### 11.6 OpenAI Function Calling Adapter

> 📄 **File:** `dopeagents/adapters/openai_functions.py` | **Status:** ○ Stub | **Role:** Generates OpenAI function calling schema + callable

```python
# dopeagents/adapters/openai_functions.py

from typing import Any
from dopeagents.core.agent import Agent


def to_openai_function(agent: Agent, description_override: str | None = None) -> dict[str, Any]:
    """Converts a DopeAgents agent into OpenAI function calling format."""
    input_type = type(agent).input_type()
    input_schema = input_type.model_json_schema()

    parameters = {
        "type": "object",
        "properties": input_schema.get("properties", {}),
        "required": input_schema.get("required", []),
    }

    def callable_function(**kwargs) -> str:
        agent_input = input_type.model_validate(kwargs)
        output = agent.run(agent_input)
        return output.model_dump_json()

    return {
        "schema": {
            "name": agent.name,
            "description": description_override or agent.description,
            "parameters": parameters,
        },
        "callable": callable_function,
    }
```

### 11.7 Generic Callable Adapter

> 📄 **File:** `dopeagents/adapters/generic.py` | **Status:** ○ Stub | **Role:** Plain callable fallback for any framework

```python
# dopeagents/adapters/generic.py

from typing import Any
from dopeagents.core.agent import Agent


def to_callable(
    agent: Agent,
    input_format: str = "dict",
    output_format: str = "dict",
) -> callable:
    """Converts a DopeAgents agent into a plain callable. Fallback for any framework."""
    input_type = type(agent).input_type()

    def agent_callable(input_data: Any = None, **kwargs) -> Any:
        if input_data is not None:
            if isinstance(input_data, str):
                agent_input = input_type.model_validate_json(input_data)
            elif isinstance(input_data, dict):
                agent_input = input_type.model_validate(input_data)
            else:
                agent_input = input_data
        else:
            agent_input = input_type.model_validate(kwargs)

        output = agent.run(agent_input)

        if output_format == "json":
            return output.model_dump_json()
        elif output_format == "dict":
            return output.model_dump()
        return output

    agent_callable.__name__ = agent.name
    agent_callable.__doc__ = agent.description
    return agent_callable
```

### 11.8 SimpleRunner

> 📄 **File:** `dopeagents/adapters/simple.py` | **Status:**   Implemented | **Role:** Minimal convenience wrapper around `AgentExecutor`

```python
# dopeagents/adapters/simple.py

from dopeagents.core.agent import Agent
from dopeagents.lifecycle.executor import AgentExecutor
from dopeagents.core.context import AgentContext
from dopeagents.core.types import AgentResult


class SimpleRunner:
    """Simplest way to run an agent with full lifecycle support."""

    def __init__(self, executor: AgentExecutor | None = None):
        self.executor = executor or AgentExecutor()

    def run(self, agent: Agent, input, context: AgentContext | None = None, **kwargs) -> AgentResult:
        return self.executor.run(agent, input, context, **kwargs)
```

### 11.9 MCP Adapter

> 📄 **File:** `dopeagents/adapters/mcp.py` | **Status:** ○ Not yet created | **Role:** FastMCP tool/prompt/resource registration and MCP server factory

```python
# dopeagents/adapters/mcp.py

from typing import Any
from dopeagents.core.agent import Agent


def _check_installed():
    try:
        from fastmcp import FastMCP  # noqa: F401
    except ImportError:
        raise ImportError(
            "FastMCP is required for MCP support.\n"
            "Install with: pip install dopeagents[mcp]"
        )


def register_agent_as_mcp_tool(
    agent: Agent,
    mcp_server: Any,
    description_override: str | None = None,
) -> None:
    """
    Register a DopeAgents agent as a tool on a FastMCP server.

    Maps directly:
    - Agent name → MCP tool name
    - Agent description → MCP tool description
    - Agent input Pydantic model → MCP tool inputSchema (auto JSON Schema)
    - Agent output Pydantic model → MCP tool outputSchema
    - agent.run() → MCP tool handler

    The Pydantic input model's fields become the tool's parameters.
    FastMCP generates the JSON Schema automatically from the type annotations.
    """
    _check_installed()

    input_type = type(agent).input_type()
    output_type = type(agent).output_type()
    tool_name = agent.name
    tool_description = description_override or agent.description

    @mcp_server.tool(
        name=tool_name,
        description=tool_description,
    )
    async def agent_tool(**kwargs) -> dict:
        """MCP tool handler — validates input, runs agent, returns output."""
        agent_input = input_type.model_validate(kwargs)
        result = agent.run(agent_input)
        return result.model_dump()


def register_agent_prompt(agent: Agent, mcp_server: Any) -> None:
    """
    Register an agent's debug prompt template as an MCP prompt.
    Allows MCP clients to inspect the agent's prompt before calling it.
    """
    _check_installed()

    input_type = type(agent).input_type()

    @mcp_server.prompt(name=f"{agent.name}_prompt")
    def agent_prompt(**kwargs):
        agent_input = input_type.model_validate(kwargs)
        debug = agent.debug(agent_input)
        return debug.prompt or f"Agent {agent.name} does not expose a prompt template."


def register_agent_catalog_resource(agents: list[Agent], mcp_server: Any) -> None:
    """
    Register a read-only resource listing all exposed agents and their metadata.
    Accessible at dopeagents://catalog
    """
    _check_installed()

    @mcp_server.resource("dopeagents://catalog")
    def agent_catalog():
        return {
            agent.name: {
                "version": agent.version,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "requires_llm": agent.requires_llm,
                "input_schema": type(agent).input_type().model_json_schema(),
                "output_schema": type(agent).output_type().model_json_schema(),
            }
            for agent in agents
        }


def create_single_agent_mcp_server(agent: Agent, name: str | None = None, **kwargs) -> Any:
    """
    Create a standalone MCP server for a single agent.
    Useful for microservice-style deployments.
    """
    _check_installed()
    from fastmcp import FastMCP

    server = FastMCP(name or agent.name)
    register_agent_as_mcp_tool(agent, server)
    register_agent_prompt(agent, server)
    register_agent_catalog_resource([agent], server)
    return server


def create_mcp_server(
    agents: list[Agent] | None = None,
    name: str = "DopeAgents",
) -> Any:
    """
    Create an MCP server exposing multiple DopeAgents agents.

    If agents is None, loads all built-in agents.
    Each agent is registered as:
    - An MCP Tool (callable)
    - An MCP Prompt (debug template)
    Plus a shared catalog Resource.
    """
    _check_installed()
    from fastmcp import FastMCP

    if agents is None:
        from dopeagents.sandbox.runner import SandboxRunner
        runner = SandboxRunner()
        agents = [runner.load(name) for name in runner._BUILTIN_AGENTS]

    server = FastMCP(name)
    for agent in agents:
        register_agent_as_mcp_tool(agent, server)
        register_agent_prompt(agent, server)
    register_agent_catalog_resource(agents, server)
    return server
```

### 11.10 MCP Adapter — How Agents Map to MCP Primitives

| DopeAgents Concept            | MCP Primitive     | Mapping Details                          |
| ----------------------------- | ----------------- | ---------------------------------------- |
| Agent (Summarizer, etc.)      | Tool              | agent.run() becomes the tool handler     |
| Input Pydantic model          | Tool inputSchema  | Auto-generated JSON Schema from Pydantic |
| Output Pydantic model         | Tool outputSchema | Auto-generated JSON Schema from Pydantic |
| Agent.name                    | Tool name         | Direct mapping                           |
| Agent.description             | Tool description  | Direct mapping, overridable              |
| Agent.debug() prompt template | Prompt            | Exposed as {agent_name}\_prompt          |
| Agent catalog + metadata      | Resource          | Read-only at dopeagents://catalog        |

---

## 12. Wrapping External Agents

### 12.1 Wrapping Functions

> 📄 **File:** `dopeagents/adapters/wrap.py` | **Status:** ○ Stub | **Role:** `wrap_function()` and `wrap_class()` — convert any Python function/class to a full Agent

```python
# dopeagents/adapters/wrap.py

from dopeagents.core.agent import Agent
from dopeagents.core.context import AgentContext
from pydantic import BaseModel
from typing import Callable, ClassVar


def wrap_function(
    func: Callable,
    name: str,
    version: str,
    input_type: type[BaseModel],
    output_type: type[BaseModel],
    description: str = "",
    capabilities: list[str] | None = None,
    requires_llm: bool = False,
) -> type[Agent]:
    """Wraps any Python function as a full DopeAgents agent."""

    class WrappedAgent(Agent[input_type, output_type]):
        name: ClassVar[str] = name
        version: ClassVar[str] = version
        description: ClassVar[str] = description or f"Wrapped: {func.__name__}"
        capabilities: ClassVar[list[str]] = capabilities or []
        requires_llm: ClassVar[bool] = requires_llm

        def run(self, input: input_type, context: AgentContext | None = None) -> output_type:
            input_dict = input.model_dump()
            result = func(**input_dict)
            if isinstance(result, output_type):
                return result
            elif isinstance(result, dict):
                return output_type.model_validate(result)
            elif isinstance(result, BaseModel):
                return output_type.model_validate(result.model_dump())
            else:
                first_field = list(output_type.model_fields.keys())[0]
                return output_type.model_validate({first_field: result})

    WrappedAgent.__name__ = name
    WrappedAgent.__qualname__ = name
    return WrappedAgent


def wrap_class(
    cls: type,
    method: str,
    name: str,
    version: str,
    input_type: type[BaseModel],
    output_type: type[BaseModel],
    description: str = "",
    capabilities: list[str] | None = None,
    requires_llm: bool = False,
) -> type[Agent]:
    """Wraps any class with a callable method as a DopeAgents agent."""
    instance = cls()
    func = getattr(instance, method)
    return wrap_function(
        func=func,
        name=name,
        version=version,
        input_type=input_type,
        output_type=output_type,
        description=description,
        capabilities=capabilities,
        requires_llm=requires_llm,
    )
```

### 12.2 How Wrapped Agents Get Framework Compatibility

Wrapped agents inherit from `Agent`, so they get all adapters automatically:

```python
MySummarizer = wrap_function(
    func=my_legacy_function,
    name="MySummarizer",
    version="1.0.0",
    input_type=MyInput,
    output_type=MyOutput,
)

# ALL of these work automatically:
MySummarizer().run(MyInput(...))                    # direct
MySummarizer().as_langgraph_node()                  # LangGraph
MySummarizer().as_langchain_tool()                  # LangChain
MySummarizer().as_crewai_tool()                     # CrewAI
MySummarizer().as_mcp_tool(mcp)                     # MCP
MySummarizer().as_mcp_server()                      # MCP server
ContractChecker.check(MySummarizer, Classifier)     # composition
Pipeline([MySummarizer, Classifier])                # pipeline
```

---

## 13. Agent Spec (Open Agent Specification)

### 13.1 Purpose

Machine-readable, machine-enforceable specification of an agent. Always **generated from the agent class**, never hand-written. Used for discovery and documentation, never for composition (composition uses Pydantic types directly).

```text
Agent Class (code)
     │
     ├──→ Pydantic types ──→ Contract Checking (composition)
     │                        [RUNTIME ENFORCEMENT]
     │
     └──→ AgentSpec ──→ Registry (discovery)
                        [HUMAN-FACING]
```

### 13.2 Spec Schema

> 📄 **File:** `dopeagents/spec/schema.py` | **Status:** ○ Stub | **Role:** `AgentSpec` Pydantic model — machine-readable agent specification

```python
# dopeagents/spec/schema.py

from pydantic import BaseModel
from typing import Any


class AgentSpec(BaseModel):
    name: str
    version: str
    description: str
    capabilities: list[str]
    tags: list[str] = []
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    requires_llm: bool
    default_model: str | None = None
    tools: list[str] = []
    pii_fields: list[str] = []
    custom_metrics: list[str] = []
    recommended_retry_policy: dict | None = None
    recommended_fallbacks: list[str] = []
    benchmark_results: dict[str, Any] | None = None
```

### 13.3 Spec Generator

> 📄 **File:** `dopeagents/spec/generator.py` | **Status:** ○ Stub | **Role:** Generates `AgentSpec` from agent class, with JSON/YAML export

```python
# dopeagents/spec/generator.py

from dopeagents.core.agent import Agent
from dopeagents.spec.schema import AgentSpec


class SpecGenerator:
    @staticmethod
    def generate(agent_class: type[Agent]) -> AgentSpec:
        return AgentSpec(
            name=agent_class.name,
            version=agent_class.version,
            description=agent_class.description,
            capabilities=agent_class.capabilities,
            tags=agent_class.tags,
            input_schema=agent_class.input_type().model_json_schema(),
            output_schema=agent_class.output_type().model_json_schema(),
            requires_llm=agent_class.requires_llm,
            default_model=getattr(agent_class, "default_model", None),
        )

    @staticmethod
    def to_yaml(spec: AgentSpec) -> str:
        """Requires PyYAML: pip install dopeagents[spec]"""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML export.\n"
                "Install with: pip install dopeagents[spec]"
            )
        return yaml.dump(spec.model_dump(exclude_none=True), default_flow_style=False)

    @staticmethod
    def to_json(spec: AgentSpec) -> str:
        return spec.model_dump_json(indent=2, exclude_none=True)
```

### 13.4 Spec Validator

> 📄 **File:** `dopeagents/spec/validator.py` | **Status:** ○ Stub | **Role:** Validates agent class has all required attributes and correct signatures

```python
# dopeagents/spec/validator.py

import re
import inspect
from dopeagents.core.agent import Agent


class SpecValidator:
    @staticmethod
    def validate(agent_class: type[Agent]) -> list[str]:
        errors = []
        for attr in ["name", "version", "description", "capabilities"]:
            if not hasattr(agent_class, attr) or not getattr(agent_class, attr):
                errors.append(f"Missing required class attribute: {attr}")
        try:
            agent_class.input_type()
        except TypeError as e:
            errors.append(f"Cannot resolve input type: {e}")
        try:
            agent_class.output_type()
        except TypeError as e:
            errors.append(f"Cannot resolve output type: {e}")
        sig = inspect.signature(agent_class.run)
        if "input" not in sig.parameters:
            errors.append("run() method must have 'input' parameter")
        if hasattr(agent_class, "version"):
            if not re.match(r"^\d+\.\d+\.\d+", agent_class.version):
                errors.append(f"Version '{agent_class.version}' is not semver format")
        return errors
```

---

## 14. Registry & Discovery System

> 📄 **File:** `dopeagents/registry/registry.py` | **Status:** ○ Stub | **Role:** Agent registration, versioning, capability/tag indexing, and `@register` decorator

```python
# dopeagents/registry/registry.py

from dopeagents.core.agent import Agent
from dopeagents.spec.schema import AgentSpec
from dopeagents.spec.generator import SpecGenerator
from dopeagents.spec.validator import SpecValidator


class Registry:
    _agents: dict[str, dict[str, type[Agent]]] = {}
    _specs: dict[str, AgentSpec] = {}
    _capability_index: dict[str, list[str]] = {}
    _tag_index: dict[str, list[str]] = {}

    @classmethod
    def register(cls, agent_class: type[Agent]) -> None:
        errors = SpecValidator.validate(agent_class)
        if errors:
            raise ValueError(
                f"Agent '{agent_class.__name__}' failed validation:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )
        spec = SpecGenerator.generate(agent_class)
        key = f"{spec.name}:{spec.version}"
        if spec.name not in cls._agents:
            cls._agents[spec.name] = {}
        cls._agents[spec.name][spec.version] = agent_class
        cls._specs[key] = spec
        for cap in spec.capabilities:
            cls._capability_index.setdefault(cap, []).append(key)
        for tag in spec.tags:
            cls._tag_index.setdefault(tag, []).append(key)

    @classmethod
    def find(cls, capability: str) -> list[AgentSpec]:
        keys = cls._capability_index.get(capability, [])
        return [cls._specs[k] for k in keys]

    @classmethod
    def search(cls, tags: list[str] | None = None, requires_llm: bool | None = None) -> list[AgentSpec]:
        results = list(cls._specs.values())
        if tags:
            results = [s for s in results if any(t in s.tags for t in tags)]
        if requires_llm is not None:
            results = [s for s in results if s.requires_llm == requires_llm]
        return results

    @classmethod
    def get(cls, name: str, version: str | None = None) -> type[Agent]:
        if name not in cls._agents:
            raise KeyError(f"Agent '{name}' not found in registry")
        versions = cls._agents[name]
        if version:
            if version not in versions:
                raise KeyError(f"Agent '{name}' version '{version}' not found")
            return versions[version]
        from packaging.version import Version
        latest = max(versions.keys(), key=Version)
        return versions[latest]

    @classmethod
    def list_all(cls) -> list[AgentSpec]:
        return list(cls._specs.values())

    @classmethod
    def clear(cls) -> None:
        cls._agents.clear()
        cls._specs.clear()
        cls._capability_index.clear()
        cls._tag_index.clear()


# Decorator
def register(cls: type[Agent]) -> type[Agent]:
    Registry.register(cls)
    return cls
```

---

## 15. Benchmark & Evaluation Framework

> 📄 **File:** `dopeagents/benchmark/suite.py` | **Status:** ○ Stub | **Role:** `BenchmarkCase` and `BenchmarkSuite` models for test case definition

```python
# dopeagents/benchmark/suite.py

from pydantic import BaseModel, Field
from typing import Any


class BenchmarkCase(BaseModel):
    input_data: dict[str, Any]
    expected_output: dict[str, Any] | None = None


class BenchmarkSuite(BaseModel):
    name: str
    version: str
    capability: str
    description: str
    cases: list[BenchmarkCase]
    quality_metrics: list[str] = Field(
        default_factory=lambda: ["accuracy", "completeness", "relevance"]
    )
```

> 📄 **File:** `dopeagents/benchmark/runner.py` | **Status:** ○ Stub | **Role:** Runs benchmark suites against agents, bypasses lifecycle for raw performance measurement

```python
# dopeagents/benchmark/runner.py

import time
from dopeagents.core.agent import Agent
from dopeagents.core.context import AgentContext
from dopeagents.benchmark.suite import BenchmarkSuite
from dopeagents.benchmark.results import AgentBenchmarkResult, BenchmarkResult


class BenchmarkRunner:
    """
    Runs benchmark suites against agents.

    NOTE: BenchmarkRunner calls agent.run() directly — intentionally
    bypassing AgentExecutor. Benchmarks measure raw agent performance
    (latency, accuracy) without lifecycle overhead (retry, caching,
    cost tracking). Use AgentExecutor for production execution.
    """
    def run_single(self, agent: Agent, suite: BenchmarkSuite, model: str | None = None) -> AgentBenchmarkResult:
        results = []
        for case in suite.cases:
            input_model = type(agent).input_type().model_validate(case.input_data)
            context = AgentContext(model_override=model)
            start = time.monotonic()
            try:
                result = agent.run(input_model, context)
                elapsed = (time.monotonic() - start) * 1000
                results.append({"success": True, "latency_ms": elapsed, "output": result.model_dump()})
            except Exception as e:
                results.append({"success": False, "error": str(e), "latency_ms": (time.monotonic() - start) * 1000})

        successful = [r for r in results if r["success"]]
        return AgentBenchmarkResult(
            agent_name=agent.name,
            agent_version=agent.version,
            suite_name=suite.name,
            total_cases=len(results),
            successful_cases=len(successful),
            avg_latency_ms=sum(r["latency_ms"] for r in successful) / len(successful) if successful else 0,
            results=results,
        )

    def compare(self, agents: list[Agent], suite: BenchmarkSuite, model: str | None = None) -> BenchmarkResult:
        return BenchmarkResult(
            suite_name=suite.name,
            suite_version=suite.version,
            agent_results=[self.run_single(a, suite, model) for a in agents],
        )
```

> 📄 **File:** `dopeagents/benchmark/results.py` | **Status:** ○ Stub | **Role:** `AgentBenchmarkResult` and `BenchmarkResult` with success rate and table formatting

```python
# dopeagents/benchmark/results.py

from pydantic import BaseModel
from typing import Any


class AgentBenchmarkResult(BaseModel):
    agent_name: str
    agent_version: str
    suite_name: str
    total_cases: int
    successful_cases: int
    avg_latency_ms: float
    results: list[dict[str, Any]]

    @property
    def success_rate(self) -> float:
        return self.successful_cases / self.total_cases if self.total_cases > 0 else 0.0


class BenchmarkResult(BaseModel):
    suite_name: str
    suite_version: str
    agent_results: list[AgentBenchmarkResult]

    def to_table(self) -> str:
        header = f"{'Agent':} {'Success':>8} {'Latency':>10}"
        separator = "─" * len(header)
        rows = []
        for r in sorted(self.agent_results, key=lambda x: x.avg_latency_ms):
            rows.append(f"{r.agent_name:} {r.success_rate:>7.0%} {r.avg_latency_ms:>8.0f}ms")
        return f"\n{header}\n{separator}\n" + "\n".join(rows) + "\n"
```

---

## 16. Tool Integration Layer

### 16.0 MCP: Two Directions

MCP appears in DopeAgents in two distinct roles:

1. **Agents AS MCP tools (Section 11.9):** DopeAgents agents are exposed
   as MCP tools via .as_mcp_tool(). This is an OUTPUT adapter — DopeAgents
   serves tools to external MCP clients.
2. **MCP tools IN agents (this section):** DopeAgents agents can USE external
   MCP tools as part of their logic. This is an INPUT integration — agents
   consume tools from MCP servers.

These are separate concerns in separate code paths.

> 📄 **File:** `dopeagents/tools/base.py` | **Status:** ○ Stub | **Role:** `Tool` ABC with `ToolInput`/`ToolOutput` base models and LLM function schema generation

```python
# dopeagents/tools/base.py

from abc import ABC, abstractmethod
from pydantic import BaseModel


class ToolInput(BaseModel):
    pass


class ToolOutput(BaseModel):
    success: bool = True
    error: str | None = None


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def call(self, input: ToolInput) -> ToolOutput: ...

    @abstractmethod
    def input_type(self) -> type[ToolInput]: ...

    def to_llm_function_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_type().model_json_schema(),
        }
```

> 📄 **File:** `dopeagents/tools/mcp.py` | **Status:** ○ Stub | **Role:** Wraps external MCP server tools for consumption within agents

```python
# dopeagents/tools/mcp.py

from dopeagents.tools.base import Tool, ToolInput, ToolOutput


class MCPTool(Tool):
    """
    Wraps an external MCP tool so an agent can call it like any other Tool.

    Usage:
        # Connect to an MCP server and discover tools
        tools = MCPTool.from_server("npx -y @modelcontextprotocol/server-github")
    """

    def __init__(self, name: str, description: str, input_schema: dict, session):
        self.name = name
        self.description = description
        self._input_schema = input_schema
        self._session = session

    def input_type(self) -> type[ToolInput]:
        # Dynamic Pydantic model generated from JSON Schema at init
        from pydantic import create_model
        fields = {
            k: (v.get("type", "string"), ...)
            for k, v in self._input_schema.get("properties", {}).items()
        }
        return create_model(f"{self.name}Input", __base__=ToolInput, **fields)

    def call(self, input: ToolInput) -> ToolOutput:
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            self._session.call_tool(self.name, arguments=input.model_dump())
        )
        return ToolOutput(success=True, **(result if isinstance(result, dict) else {"result": result}))

    @classmethod
    def from_server(cls, command: str) -> list["MCPTool"]:
        """
        Connect to an MCP server via stdio and return its tools
        as MCPTool instances.
        """
        try:
            from fastmcp import Client  # noqa: F401
        except ImportError:
            raise ImportError(
                "FastMCP is required for MCP tool consumption.\n"
                "Install with: pip install dopeagents[mcp]"
            )
        # Implementation deferred to Phase 2. Returns list of MCPTool
        # wrapping each tool the server exposes.
        raise NotImplementedError("MCPTool.from_server() ships in Phase 2")
```

> 📄 **File:** `dopeagents/tools/function.py` | **Status:** ○ Stub | **Role:** Wraps plain Python callables as `Tool` instances with automatic schema generation

```python
# dopeagents/tools/function.py

from __future__ import annotations
import inspect
from typing import Any, Callable

from dopeagents.tools.base import Tool, ToolInput, ToolOutput


class FunctionTool(Tool):
    """Wraps a plain Python function as a Tool.

    Introspects the function signature to generate the LLM-facing schema
    automatically. Supports both sync and async callables.
    """

    def __init__(self, fn: Callable[..., Any], *, name: str | None = None, description: str | None = None):
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description or (inspect.getdoc(fn) or f"Tool wrapping {fn.__name__}")

    def call(self, input: ToolInput) -> ToolOutput:
        result = self.fn(**input.model_dump())
        if isinstance(result, ToolOutput):
            return result
        return ToolOutput(success=True, data=result)

    def function_schema(self) -> dict:
        """Generate JSON Schema from the function's type hints."""
        sig = inspect.signature(self.fn)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            prop: dict[str, str] = {"type": "string"}  # default
            if param.annotation != inspect.Parameter.empty:
                prop = _annotation_to_schema(param.annotation)
            properties[param_name] = prop
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


def _annotation_to_schema(annotation: Any) -> dict[str, str]:
    """Map Python type annotations to JSON Schema primitives."""
    mapping = {str: "string", int: "integer", float: "number", bool: "boolean"}
    return {"type": mapping.get(annotation, "string")}
```

> 📄 **File:** `dopeagents/tools/rest.py` | **Status:** ○ Stub | **Role:** REST API tool with configurable URL, method, headers, and timeout

```python
# dopeagents/tools/rest.py

from pydantic import BaseModel
from dopeagents.tools.base import Tool, ToolInput, ToolOutput
import httpx


class RESTToolConfig(BaseModel):
    url: str
    method: str = "POST"
    headers: dict[str, str] = {}
    timeout: float = 30.0


class RESTTool(Tool):
    def __init__(self, config: RESTToolConfig):
        self.config = config
        self.name = f"rest_{config.url}"
        self.description = f"REST API call to {config.url}"

    def call(self, input: ToolInput) -> ToolOutput:
        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.request(
                method=self.config.method,
                url=self.config.url,
                json=input.model_dump(),
                headers=self.config.headers,
            )
            response.raise_for_status()
            return ToolOutput(success=True, **response.json())
```

---

## 17. Core Agents

DopeAgents ships production-grade multi-step agents. The agent list is shaped by what teams actually rebuild from scratch. Phase 1 ships the two highest-value agents.

### Phase 1 — Reference Agents

| Agent                    | Type           | Input                          | Output                                                                  |
| ------------------------ | -------------- | ------------------------------ | ----------------------------------------------------------------------- |
| **DeepSummarizer** | Multi-step LLM | text, max_length, style, focus | summary, quality_score, refinement_rounds, chunks_processed, key_points |
| **ResearchAgent**  | Multi-step LLM | query, max_sources, depth      | report, sources, confidence_scores, citations, fact_check_notes         |

---

### 17.1 DeepSummarizer

> 📄 **File:** `dopeagents/agents/deep_summarizer.py` | **Status:** ○ Stub | **Role:** 7-step summarization workflow with chunking, synthesis, self-evaluation, and iterative refinement

**Steps (in order):**

| Step           | Purpose                                                               | Model strategy               |
| -------------- | --------------------------------------------------------------------- | ---------------------------- |
| `analyze`    | Analyze text structure; decide on chunking strategy                   | Fast/cheap (gpt-4o-mini)     |
| `chunk`      | Split text into semantically coherent chunks                          | Fast/cheap (gpt-4o-mini)     |
| `summarize`  | Summarize each chunk independently                                    | Per-chunk (gpt-4o-mini)      |
| `synthesize` | Combine chunk summaries into a coherent whole                         | Smart (gpt-4o or equivalent) |
| `evaluate`   | Score the synthesis (0.0–1.0) and identify weaknesses                | Smart (gpt-4o or equivalent) |
| `refine`     | Improve synthesis using evaluation feedback (loops back to evaluate)  | Smart (gpt-4o or equivalent) |
| `format`     | Apply style constraints (paragraph / bullets / tldr), compute metrics | Fast/cheap (gpt-4o-mini)     |

The evaluate→refine→evaluate cycle loops until `quality_score >= 0.8` or `max_refinement_loops` is hit. When `BudgetConfig.on_exceeded = "degrade"`, the best synthesis produced so far is returned instead of raising.

**Interfaces:**

```python
class DeepSummarizerInput(BaseModel):
    text: str = Field(min_length=1)
    max_length: int = Field(default=500, ge=50, le=10000)
    style: Literal["paragraph", "bullets", "tldr"] = "paragraph"
    focus: str | None = None

class DeepSummarizerOutput(BaseModel):
    summary: str
    key_points: list[str]
    quality_score: float = Field(ge=0.0, le=1.0)   # final evaluate score
    refinement_rounds: int                           # how many refine loops ran
    chunks_processed: int                            # number of chunks in the pipeline
    word_count: int
    truncated: bool
```

**Usage:**

```python
from dopeagents.agents import DeepSummarizer, DeepSummarizerInput

agent = DeepSummarizer()
output = agent.run(DeepSummarizerInput(text="...", max_length=400, style="bullets"))
print(output.summary)
print(f"Quality: {output.quality_score:.2f}, Refined {output.refinement_rounds}x")
```

---

### 17.2 ResearchAgent

> 📄 **File:** `dopeagents/agents/research_agent.py` | **Status:** ○ Stub | **Role:** 6-step research workflow with query planning, search, source evaluation, synthesis, and fact-checking

**Steps (in order):**

| Step           | Purpose                                                  | Model strategy           |
| -------------- | -------------------------------------------------------- | ------------------------ |
| `formulate`  | Break the research question into sub-queries             | Smart (gpt-4o)           |
| `search`     | Execute searches and collect candidate sources           | Fast/cheap + Tool        |
| `evaluate`   | Score source credibility and relevance                   | Smart (gpt-4o)           |
| `synthesize` | Write a draft research report from the top sources       | Smart (gpt-4o)           |
| `fact_check` | Cross-reference key claims against multiple sources      | Smart (gpt-4o)           |
| `compose`    | Format final report with citations and confidence scores | Fast/cheap (gpt-4o-mini) |

**Interfaces:**

```python
class ResearchInput(BaseModel):
    query: str = Field(min_length=5)
    max_sources: int = Field(default=10, ge=2, le=50)
    depth: Literal["quick", "standard", "deep"] = "standard"

class ResearchOutput(BaseModel):
    report: str
    sources: list[str]
    confidence_scores: dict[str, float]  # Per source
    citations: list[str]
    fact_check_notes: list[str]
    sub_queries_used: list[str]
```

**Usage:**

```python
from dopeagents.agents import ResearchAgent, ResearchInput

agent = ResearchAgent()
output = agent.run(ResearchInput(query="Impact of LLMs on software development", depth="deep"))
print(output.report)
```

### Phase 2 — Additional Agents

| Agent                     | Type           | Input                       | Output                                |
| ------------------------- | -------------- | --------------------------- | ------------------------------------- |
| **DocumentAnalyst** | Multi-step LLM | document, questions, format | answers, evidence, confidence         |
| **CodeReviewer**    | Multi-step LLM | code, language, focus_areas | issues, suggestions, severity_map     |
| **DataExtractor**   | Multi-step LLM | text, schema, strict        | extracted, confidence, missing_fields |

---

## 18. Package Structure

```text
dopeagents/
├── __init__.py                   # Re-exports (__version__, Agent, core agents)
├── py.typed                       # PEP 561 marker
├── cli.py                         # CLI entry point (list/describe/run/dry-run/mcp serve)
├── core/
│   ├── __init__.py            
│   ├── agent.py                   # Agent base class + _extract() + adapter methods
│   ├── context.py                 # AgentContext dataclass
│   ├── types.py                   # AgentResult, ExtractionResult, ToolCall
│   └── metadata.py                # AgentCard, Capability, CostPolicy
├── sandbox/
│   ├── __init__.py             ○
│   ├── runner.py               ○  # SandboxRunner + inspect_mcp()
│   ├── display.py              ○  # SandboxDisplay (text formatting)
│   └── repl.py                 ○  # Interactive REPL
├── contracts/
│   ├── __init__.py             ○
│   ├── checker.py                 # ContractChecker (pre/post-conditions)
│   ├── pipeline.py                # PipelineValidator (type compatibility)
│   └── types.py                   # PreCondition, PostCondition, ContractResult
├── lifecycle/
│   ├── __init__.py             ○
│   ├── executor.py                # LifecycleExecutor (Instructor hooks)
│   └── hooks.py                   # LifecycleHooks protocol
├── observability/
│   ├── __init__.py             ○
│   ├── tracer.py               ○  # Tracer ABC + NoopTracer + ConsoleTracer
│   ├── otel.py                 ○  # OTelTracer (OpenTelemetry integration)
│   ├── instructor_hooks.py     ✧  # InstructorObservabilityHooks (not yet created)
│   └── contract.py             ○  # ObservabilityContract model
├── cost/
│   ├── __init__.py             ○
│   ├── tracker.py              ○  # CostTracker (per-call + cumulative)
│   └── guard.py                ○  # CostGuard (budget enforcement)
├── resilience/
│   ├── __init__.py             ○
│   ├── retry.py                ○  # RetryPolicy (exponential backoff)
│   ├── fallback.py             ○  # FallbackChain (model cascade)
│   └── degradation.py          ○  # GracefulDegradation (partial results)
├── cache/
│   ├── __init__.py             ○
│   ├── manager.py              ○  # CacheManager (get/set/invalidate)
│   └── disk.py                 ○  # DiskCache (SQLite-backed TTL cache)
├── adapters/
│   ├── __init__.py             ○
│   ├── langchain.py            ○  # LangChain BaseTool wrapper
│   ├── crewai.py               ○  # CrewAI agent adapter
│   ├── autogen.py              ○  # AutoGen conversable-agent wrapper
│   ├── openai_functions.py     ○  # OpenAI function-calling schema export
│   ├── generic.py              ○  # GenericAdapter (arbitrary callables)
│   ├── mcp.py                     # FastMCP adapter + server factory
│   ├── wrap.py                 ○  # Universal wrap() helper
│   └── simple.py                  # SimpleAdapter (dict I/O)
├── spec/
│   ├── __init__.py             ○
│   ├── schema.py               ○  # SpecSchema (Pydantic → JSON Schema)
│   ├── generator.py            ○  # SpecGenerator (Markdown/HTML output)
│   └── validator.py            ○  # SpecValidator (contract verification)
├── registry/
│   ├── __init__.py             ○
│   └── registry.py             ○  # AgentRegistry (singleton discovery)
├── benchmark/
│   ├── __init__.py             ○
│   ├── suite.py                ○  # BenchmarkSuite (test case collection)
│   ├── runner.py               ○  # BenchmarkRunner (execute + measure)
│   └── results.py              ○  # BenchmarkResult (metrics aggregation)
├── tools/
│   ├── __init__.py             ○
│   ├── base.py                 ○  # ToolSpec ABC
│   ├── function.py             ○  # FunctionTool (wrap Python callables)
│   ├── mcp.py                  ○  # MCPTool (consume external MCP servers)
│   └── rest.py                 ○  # RESTTool (HTTP endpoint wrapper)
├── agents/
│   ├── __init__.py             ○
│   ├── deep_summarizer.py         # 7-step summarization agent (Phase 1)
│   ├── research_agent.py          # 6-step research agent (Phase 1)
│   ├── document_analyst.py     ✧  # Multi-step document Q&A (Phase 2)
│   ├── code_reviewer.py        ✧  # Multi-step code review (Phase 2)
│   └── data_extractor.py       ✧  # Multi-step structured extraction (Phase 2)
├── security/
│   ├── __init__.py             ○
│   └── redaction.py            ○  # PII pattern detection + field redaction
├── errors.py                      # Full typed error hierarchy
└── config.py                      # DopeAgentsConfig + env/TOML loading
```

### 18.1 Root `__init__.py` Re-exports

The root package re-exports the most commonly used symbols so users can write
`from dopeagents import Agent, DeepSummarizer, ...` as shown in §23 API Reference.

> 📄 **File:** `dopeagents/__init__.py` | **Status:**   Implemented

```python
# dopeagents/__init__.py

__version__ = "3.0.0"

from dopeagents.core.agent import Agent
from dopeagents.core.context import AgentContext
from dopeagents.core.types import AgentResult, ExecutionMetrics, StepMetrics
from dopeagents.agents.deep_summarizer import DeepSummarizer
from dopeagents.agents.research_agent import ResearchAgent
from dopeagents.contracts.checker import ContractChecker
from dopeagents.contracts.pipeline import Pipeline
from dopeagents.adapters.simple import SimpleRunner
from dopeagents.registry.registry import Registry, register

__all__ = [
    "Agent",
    "AgentContext",
    "AgentResult",
    "ExecutionMetrics",
    "StepMetrics",
    "DeepSummarizer",
    "ResearchAgent",
    "ContractChecker",
    "Pipeline",
    "SimpleRunner",
    "Registry",
    "register",
]
```

---

## 19. Error Taxonomy

> 📄 **File:** `dopeagents/errors.py` | **Status:**   Implemented | **Role:** Complete typed error hierarchy — contract, execution, extraction, cost, registry, tool, adapter, and MCP errors

```python
# dopeagents/errors.py

class DopeAgentsError(Exception):
    """Base exception for all DopeAgents errors."""
    pass

# Contract errors
class ContractError(DopeAgentsError): pass
class PipelineValidationError(ContractError): pass
class InputValidationError(ContractError):
    def __init__(self, agent_name, validation_error):
        self.agent_name = agent_name
        self.validation_error = validation_error
        super().__init__(f"Input validation failed for '{agent_name}': {validation_error}")

class OutputValidationError(ContractError):
    def __init__(self, agent_name, output_data, validation_error):
        self.agent_name = agent_name
        self.output_data = output_data
        self.validation_error = validation_error
        super().__init__(f"Output validation failed for '{agent_name}': {validation_error}")

# Execution errors
class ExecutionError(DopeAgentsError): pass
class AgentExecutionError(ExecutionError):
    def __init__(self, agent_name, original_error):
        self.agent_name = agent_name
        self.original_error = original_error
        super().__init__(f"Agent '{agent_name}' failed: {original_error}")

class AllFallbacksFailedError(ExecutionError):
    def __init__(self, chain_agents):
        self.chain_agents = chain_agents
        super().__init__(f"All fallback agents failed: {chain_agents}")

# Cost errors
class CostError(DopeAgentsError): pass
class BudgetExceededError(CostError): pass

# Registry errors
class RegistryError(DopeAgentsError): pass
class AgentNotFoundError(RegistryError):
    def __init__(self, agent_name, version=None):
        msg = f"Agent '{agent_name}'"
        if version:
            msg += f" version '{version}'"
        super().__init__(f"{msg} not found in registry")

class AgentValidationError(RegistryError): pass

# Tool errors
class ToolError(DopeAgentsError): pass
class ToolExecutionError(ToolError):
    def __init__(self, tool_name, original_error):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' failed: {original_error}")

# Adapter errors
class AdapterError(DopeAgentsError): pass
class FrameworkNotInstalledError(AdapterError):
    def __init__(self, framework, install_cmd):
        self.framework = framework
        self.install_cmd = install_cmd
        super().__init__(f"{framework} required. Install: {install_cmd}")

# MCP errors (NEW)
class MCPError(DopeAgentsError): pass
class MCPNotInstalledError(MCPError, AdapterError):
    def __init__(self):
        super().__init__(
            "FastMCP is required for MCP support. Install with: pip install dopeagents[mcp]"
        )

class MCPRegistrationError(MCPError):
    def __init__(self, agent_name, reason):
        self.agent_name = agent_name
        self.reason = reason
        super().__init__(
            f"Failed to register agent '{agent_name}' as MCP tool: {reason}"
        )

class MCPServerError(MCPError):
    def __init__(self, reason):
        super().__init__(f"MCP server error: {reason}")

# Instructor errors (NEW)
class ExtractionError(DopeAgentsError): pass
class ExtractionValidationError(ExtractionError):
    """Instructor exhausted max_retries and still couldn't get valid output."""
    def __init__(self, agent_name, response_model, original_error):
        self.agent_name = agent_name
        self.response_model = response_model
        self.original_error = original_error
        super().__init__(
            f"Agent '{agent_name}' extraction failed after max retries. "
            f"Expected schema: {response_model.__name__}. Error: {original_error}"
        )

class ExtractionProviderError(ExtractionError):
    """The LLM provider returned an error (auth, rate limit, etc.)."""
    def __init__(self, agent_name, provider, original_error):
        self.agent_name = agent_name
        self.provider = provider
        self.original_error = original_error
        super().__init__(
            f"Agent '{agent_name}' provider '{provider}' error: {original_error}"
        )
```

---

## 20. Configuration System

> 📄 **File:** `dopeagents/config.py` | **Status:**   Implemented | **Role:** `DopeAgentsConfig` with env var and TOML loading, global singleton via `get_config()`/`set_config()`

```python
# dopeagents/config.py

from pydantic import BaseModel, Field
from typing import Literal


class DopeAgentsConfig(BaseModel):
    # Model / Extraction
    default_model: str = "openai/gpt-4o-mini"
    default_extraction_max_retries: int = 3
    default_budget: float | None = None
    tracer_type: Literal["noop", "console", "otel"] = "noop"
    cache_enabled: bool = False
    cache_type: Literal["memory", "disk"] = "memory"
    cache_ttl: int = 3600
    cache_directory: str = ".dopeagents_cache"
    default_retry_max_attempts: int = 3
    default_retry_delay: float = 1.0
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    redact_pii_in_logs: bool = True

    # Sandbox
    sandbox_default_model: str | None = None
    sandbox_output_format: Literal["text", "json"] = "text"

    # MCP
    mcp_server_name: str = "DopeAgents"
    mcp_default_transport: Literal["stdio", "streamable-http"] = "stdio"
    mcp_http_port: int = 8000

    @classmethod
    def from_env(cls) -> "DopeAgentsConfig":
        import os
        kwargs = {}
        env_mapping = {
            "DOPEAGENTS_DEFAULT_MODEL": "default_model",
            "DOPEAGENTS_EXTRACTION_MAX_RETRIES": "default_extraction_max_retries",
            "DOPEAGENTS_DEFAULT_BUDGET": "default_budget",
            "DOPEAGENTS_TRACER": "tracer_type",
            "DOPEAGENTS_CACHE_ENABLED": "cache_enabled",
            "DOPEAGENTS_LOG_LEVEL": "log_level",
            "DOPEAGENTS_SANDBOX_MODEL": "sandbox_default_model",
            "DOPEAGENTS_SANDBOX_OUTPUT": "sandbox_output_format",
            "DOPEAGENTS_MCP_SERVER_NAME": "mcp_server_name",
            "DOPEAGENTS_MCP_TRANSPORT": "mcp_default_transport",
            "DOPEAGENTS_MCP_PORT": "mcp_http_port",
        }
        for env_var, field_name in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                field_info = cls.model_fields[field_name]
                if field_info.annotation is bool:
                    kwargs[field_name] = value.lower() in ("true", "1", "yes")
                elif field_info.annotation is float or field_info.annotation == float | None:
                    kwargs[field_name] = float(value)
                elif field_info.annotation is int:
                    kwargs[field_name] = int(value)
                else:
                    kwargs[field_name] = value
        return cls(**kwargs)

    @classmethod
    def from_toml(cls, path: str = "dopeagents.toml") -> "DopeAgentsConfig":
        import tomllib
        from pathlib import Path
        config_path = Path(path)
        if config_path.exists():
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
            return cls(**data.get("dopeagents", {}))
        return cls()


_config: DopeAgentsConfig | None = None

def get_config() -> DopeAgentsConfig:
    global _config
    if _config is None:
        _config = DopeAgentsConfig.from_env()
    return _config

def set_config(config: DopeAgentsConfig) -> None:
    global _config
    _config = config
```

---

## 21. Security & PII Handling

> 📄 **File:** `dopeagents/security/redaction.py` | **Status:** ○ Stub | **Role:** Regex-based PII detection and field-level redaction for logs and observability

```python
# dopeagents/security/redaction.py

import re
from typing import Any


class PIIRedactor:
    PATTERNS = {
        "email": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
        "phone": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "credit_card": re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
    }

    @staticmethod
    def redact_fields(data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        redacted = data.copy()
        for field_path in fields:
            parts = field_path.split(".")
            PIIRedactor._redact_nested(redacted, parts)
        return redacted

    @staticmethod
    def _redact_nested(data: dict, path_parts: list[str]) -> None:
        if len(path_parts) == 1:
            if path_parts[0] in data:
                value = data[path_parts[0]]
                data[path_parts[0]] = (
                    f"[REDACTED: {len(value)} chars]"
                    if isinstance(value, str)
                    else "[REDACTED]"
                )
        elif path_parts[0] in data and isinstance(data[path_parts[0]], dict):
            PIIRedactor._redact_nested(data[path_parts[0]], path_parts[1:])

    @staticmethod
    def redact_patterns(text: str) -> str:
        for pii_type, pattern in PIIRedactor.PATTERNS.items():
            text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", text)
        return text
```

---

## 22. Development Roadmap

```text
Status key:  ○ = not started   ◐ = scaffold only   ● = implemented & tested

Phase 1: Core Thesis (ship this first)                        [◐ in progress]
  ● Agent base class + _build_graph() (internal LangGraph engine)
  ● _extract() per step (Instructor + LiteLLM)
  ● Pipeline with typed composition checks
  ● Step-level cost tracking + budget guards (on_exceeded="degrade")
  ● MCP exposure (FastMCP ≥ 3.0)
  ◐ DeepSummarizer (7-step) + ResearchAgent (6-step)
  ◐ Adapters shipped: plain Python + MCP only

Phase 2: Production Breadth                                    [○ not started]
  ○ Additional agents (DocumentAnalyst, CodeReviewer, DataExtractor)
  ○ Benchmark runner + evaluation suite
  ○ Agent Sandbox (runner + REPL + CLI dry-run)
  ○ Observability (tracer, Instructor hooks, PII redaction)
  ○ Retry, fallback, DegradationChain, caching
  ○ Additional framework adapters (LangChain, CrewAI, AutoGen)
  ○ wrap_function / wrap_class for external agents

Phase 3: Ecosystem                                             [○ not started]
  ○ Agent Spec, registry, @register decorator
  ○ Plugin system, CLI extensions
  ○ MCP prompt + resource support, Claude Desktop integration docs
  ○ Documentation, examples, PyPI release (pip install dopeagents)
```

---

## 23. API Reference Summary

```python
# Core
from dopeagents import Agent, AgentContext, AgentResult, SimpleRunner
from dopeagents import Pipeline, ContractChecker
from dopeagents.core.types import ExecutionMetrics, StepMetrics

# Agents (Phase 1)
from dopeagents.agents import DeepSummarizer, DeepSummarizerInput, DeepSummarizerOutput
from dopeagents.agents import ResearchAgent, ResearchInput, ResearchOutput

# Agents (Phase 2 — coming soon)
# from dopeagents.agents import DocumentAnalyst, CodeReviewer, DataExtractor

# Lifecycle
from dopeagents.lifecycle import AgentExecutor, LifecycleHooks

# Observability
from dopeagents.observability import Tracer, ConsoleTracer, OTelTracer
from dopeagents.observability import InstructorObservabilityHooks

# Cost
from dopeagents.cost import CostTracker, BudgetGuard, BudgetConfig

# Resilience
from dopeagents.resilience import RetryPolicy, FallbackChain, DegradationChain
from dopeagents.resilience import DegradationResult

# Cache
from dopeagents.cache import CacheManager, InMemoryCache, DiskCache

# Tools
from dopeagents.tools import Tool, RESTTool, MCPTool

# MCP Adapter
from dopeagents.adapters.mcp import (
    create_mcp_server,
    create_single_agent_mcp_server,
    register_agent_as_mcp_tool,
)

# Registry
from dopeagents import Registry, register

# Wrapping
from dopeagents.adapters.wrap import wrap_function, wrap_class

# Sandbox
from dopeagents.sandbox import SandboxRunner, SandboxDisplay
from dopeagents.sandbox.runner import ComparisonResult, ComparisonRow

# Benchmark
from dopeagents.benchmark import BenchmarkSuite, BenchmarkRunner, BenchmarkResult

# Config
from dopeagents.config import DopeAgentsConfig, get_config, set_config

# Errors
from dopeagents.errors import (
    DopeAgentsError, ContractError, PipelineValidationError,
    InputValidationError, OutputValidationError,
    ExecutionError, AgentExecutionError, AllFallbacksFailedError,
    BudgetExceededError, BudgetDegradedError, AgentNotFoundError,
    FrameworkNotInstalledError,
    MCPError, MCPNotInstalledError, MCPRegistrationError, MCPServerError,
    ExtractionError, ExtractionValidationError, ExtractionProviderError,
    ToolError, ToolExecutionError, AdapterError,
)
```

---

## 24. Design Decisions Log

| ID     | Decision                                                                              | Rationale                                                                                                                                                                                                                                                                                                               |
| ------ | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DD-001 | Pydantic for all contracts                                                            | Type safety, validation, JSON Schema, IDE support                                                                                                                                                                                                                                                                       |
| DD-002 | Agents are Generic[InputT, OutputT] classes                                           | Enables contract verification and type introspection                                                                                                                                                                                                                                                                    |
| DD-003 | Lifecycle wraps agents                                                                | Agents stay simple, production concerns consistent                                                                                                                                                                                                                                                                      |
| DD-004 | Spec generated from code                                                              | Eliminates drift between spec and implementation                                                                                                                                                                                                                                                                        |
| DD-005 | Framework adapters on base class via lazy imports                                     | Every agent gets every adapter. No per-agent code. No mandatory deps.                                                                                                                                                                                                                                                   |
| DD-006 | Adapters never modify agent behavior                                                  | Same run() executes regardless of framework                                                                                                                                                                                                                                                                             |
| DD-007 | Contract checking at Pipeline construction                                            | Fail fast — errors caught before execution                                                                                                                                                                                                                                                                             |
| DD-008 | Contract checker uses Pydantic types, not Agent Spec                                  | Spec is for humans/discovery. Types are for machines/composition.                                                                                                                                                                                                                                                       |
| DD-009 | LiteLLM for model abstraction                                                         | 100+ providers, cost calculation included                                                                                                                                                                                                                                                                               |
| DD-010 | OpenTelemetry for tracing                                                             | Industry standard, any backend                                                                                                                                                                                                                                                                                          |
| DD-011 | Two Phase 1 agents: DeepSummarizer (7-step) + ResearchAgent (6-step)                  | Ship multi-step thesis first using production-grade agents. Single-step agents are also supported as Pattern A for simple tasks.                                                                                                                                                                                        |
| DD-012 | wrap_function/wrap_class for external agents                                          | Users bring existing code into ecosystem with minimal effort                                                                                                                                                                                                                                                            |
| DD-013 | LangGraph is a**core** dependency; all other framework deps remain optional     | LangGraph is the internal orchestration engine inside every multi-step agent. It is hidden behind `_build_graph()` and never exposed to callers. All other integrations (LangChain, CrewAI, AutoGen) remain optional extras.                                                                                          |
| DD-014 | Clear ImportError messages with install commands                                      | Users never see cryptic ModuleNotFoundError                                                                                                                                                                                                                                                                             |
| DD-015 | Sandbox ships with core, not as optional extra                                        | Must be available on `pip install dopeagents`. Development tool, not an afterthought.                                                                                                                                                                                                                                 |
| DD-016 | click is a core dependency                                                            | CLI entry point is part of core package. ~80KB, no heavy transitive deps.                                                                                                                                                                                                                                               |
| DD-017 | SandboxRunner returns Pydantic models, not strings                                    | Programmatic API returns structured data. Formatting is display layer's job.                                                                                                                                                                                                                                            |
| DD-018 | REPL uses stdlib only                                                                 | code.InteractiveConsole. No IPython/ptprompt dependency. Users use Python API in richer REPLs.                                                                                                                                                                                                                          |
| DD-019 | Built-in agent lookup replaced by Registry later                                      | Simple dict in Phase 1. One-line change in Phase 3 when Registry exists. No premature abstraction.                                                                                                                                                                                                                      |
| DD-020 | compare() is NOT a benchmark                                                          | Single-input side-by-side for development. BenchmarkRunner handles systematic evaluation.                                                                                                                                                                                                                               |
| DD-021 | Instructor + LiteLLM compose the extraction stack                                     | Instructor enforces schemas and validation retries. LiteLLM provides routing, pricing, and token/cost infrastructure. DopeAgents builds on both layers.                                                                                                                                                                 |
| DD-022 | `_extract()` is a base class primitive, not a standalone util                       | Every LLM agent calls `self._extract()`. Keeps the extraction layer discoverable, mockable for tests, and hookable for observability.                                                                                                                                                                                 |
| DD-023 | Instructor is an internal detail, not a user-facing dependency                        | Agent authors call `_extract()`. Users call `agent.run()`. Neither needs to import or configure Instructor directly.                                                                                                                                                                                                |
| DD-024 | `from_litellm()` over `from_provider()` for runtime extraction                    | `instructor.from_litellm(completion)` keeps schema extraction in Instructor while delegating routing/cost to LiteLLM. Model is passed per call via `model=self._model`.                                                                                                                                             |
| DD-025 | Provider access is standardized through LiteLLM                                       | Provider SDKs remain optional and flow through LiteLLM's adapter surface; DopeAgents core does not require provider-specific extras per framework.                                                                                                                                                                      |
| DD-026 | MCP exposure via standalone FastMCP (`fastmcp>=3.0`)                                | FastMCP 3.x is the actively maintained standalone project (1M+ downloads/day, 3.1.1 as of March 2026). The `mcp` SDK embeds an older FastMCP 1.0 era; standalone `fastmcp` provides component versioning, authorization, OpenTelemetry, and provider types needed here. Import via `from fastmcp import FastMCP`. |
| DD-027 | `as_mcp_tool()` on the base class, like all other adapters                          | Consistent pattern: every adapter is inherited, lazy-imported, optional. MCP follows the same rules.                                                                                                                                                                                                                    |
| DD-028 | MCP schemas generated from existing Pydantic models                                   | No separate schema definition. The same Pydantic types drive contracts, validation, framework adapters, AND MCP tool schemas.                                                                                                                                                                                           |
| DD-029 | `dopeagents mcp serve` as a top-level CLI command                                   | MCP serving is a deployment concern, not a sandbox concern. Separate command group, not a sandbox subcommand.                                                                                                                                                                                                           |
| DD-030 | MCP adapter is optional (`dopeagents[mcp]`)                                         | Follows INV-7: no framework/protocol dependency is mandatory. Clear ImportError with install command.                                                                                                                                                                                                                   |
| DD-031 | Three-level resilience: Instructor retries + DopeAgents step-level + DegradationChain | Instructor handles schema validation retries (extraction-level). DopeAgents handles per-step infra retries (timeouts, rate limits). DegradationChain handles agent-level fallbacks within a workflow. All three compose independently.                                                                                  |
| DD-032 | Observability uses Instructor hooks plus LiteLLM metadata                             | Instructor hooks capture request/response events; cost is sourced from LiteLLM's `response_cost` metadata when present.                                                                                                                                                                                               |
| DD-033 | `system_prompt` is a `ClassVar[str]` on the base class                            | Makes the system prompt inspectable without execution:`.describe()`, MCP Prompt primitive, observability spans, and benchmarking all read `agent.system_prompt` directly. Default is `""`.                                                                                                                        |
| DD-034 | Explicit reference over auto-injection for `system_prompt`                          | Agent writes `{"role": "system", "content": self.system_prompt}` in `run()`. `_extract()` is unchanged. What you read in `run()` is exactly what gets sent. Preserves P2 and P8.                                                                                                                                |
| DD-035 | `system_prompt` accepts a runtime override in `__init__`                          | `Summarizer(system_prompt="...")` shadows the class-level default via normal Python instance attribute shadowing. No descriptors or metaclass magic. Enables prompt variation as a benchmarking axis without subclassing.                                                                                             |
| DD-036 | `__init_subclass__` validates `system_prompt` type and warns on missing prompts   | Type error raised early for non-string declarations. Warning (not error) fired when `requires_llm=True` and no `system_prompt` is set, to preserve compatibility with agents that use dynamic `@property` prompts.                                                                                                |
| DD-037 | LangGraph is a private implementation detail, never exposed to callers                | `_build_graph()` returns a compiled LangGraph `StateGraph`. Users call `agent.run(input)` — they never import or configure LangGraph directly. This preserves the clean single-method contract (INV-1) while enabling multi-step orchestration with cycles, conditional edges, and state-driven flow.            |
| DD-038 | `step_prompts: ClassVar[dict[str, str]]` for per-step prompt declarations           | Declaring step prompts as class-level attributes (not in `run()`) makes them inspectable without execution: `describe()` reads them, MCP Prompt primitives can surface them, and benchmarks can vary them independently. Class-level storage follows the same pattern as `system_prompt` (DD-033).                |
| DD-039 | `on_exceeded="degrade"` returns best-result-so-far on budget exhaustion             | Multi-step agents accumulate partial results at each step. When `BudgetConfig.on_exceeded="degrade"`, the agent returns whatever was computed before the budget was hit rather than raising an exception. `DegradationResult` records which step and why, enabling callers to make informed downstream decisions.   |
| DD-040 | `describe()` returns structured `AgentDescription`, not a formatted string        | `AgentDescription` is a Pydantic model with `name`, `steps`, `has_loops`, `model_per_step`. Callers can introspect the agent's workflow programmatically — CLI renders it as text, MCP servers expose it as a resource, and tests can assert on step count or model assignments without parsing strings.     |
