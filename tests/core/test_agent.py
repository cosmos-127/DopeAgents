"""Tests for Agent base class."""

from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, Field

from dopeagents.core.agent import Agent, DebugInfo
from dopeagents.core.context import AgentContext
from dopeagents.core.types import AgentResult

# ── Test Fixtures ─────────────────────────────────────────────────────


class SimpleInputModel(BaseModel):
    """Simple test input."""

    text: str = Field(min_length=1)


class SimpleOutputModel(BaseModel):
    """Simple test output."""

    result: str


class SimpleAgentImplementation(Agent[SimpleInputModel, SimpleOutputModel]):
    """Minimal test agent for validation."""

    name: ClassVar[str] = "SimpleAgentImplementation"
    version: ClassVar[str] = "0.1.0"
    description: ClassVar[str] = "A test agent"
    capabilities: ClassVar[list[str]] = ["testing"]
    tags: ClassVar[list[str]] = ["test"]

    def run(
        self,
        input_data: SimpleInputModel,
        context: AgentContext | None = None,
    ) -> AgentResult[SimpleOutputModel]:
        """Dummy implementation."""
        return AgentResult(
            output=SimpleOutputModel(result=f"Processed: {input_data.text}"),
            success=True,
        )


class MultiStepInputModel(BaseModel):
    """Multi-step test input."""

    text: str


class MultiStepOutputModel(BaseModel):
    """Multi-step test output."""

    summary: str
    score: float


class MultiStepAgentImplementation(Agent[MultiStepInputModel, MultiStepOutputModel]):
    """Multi-step agent with loops."""

    name: ClassVar[str] = "MultiStepAgentImplementation"
    version: ClassVar[str] = "0.1.0"
    description: ClassVar[str] = "A multi-step test agent"
    step_prompts: ClassVar[dict[str, str]] = {
        "analyze": "Analyze the text",
        "evaluate": "Evaluate quality",
        "refine": "Refine if needed",
    }

    def run(
        self,
        input_data: MultiStepInputModel,
        context: AgentContext | None = None,
    ) -> AgentResult[MultiStepOutputModel]:
        """Dummy multi-step implementation."""
        return AgentResult(
            output=MultiStepOutputModel(summary="Summary", score=0.9),
            success=True,
        )

    def _has_loops(self) -> bool:
        """This agent has refinement loops."""
        return True


class TestTypeResolution:
    """Test InputT/OutputT type introspection."""

    def test_input_type_resolution(self) -> None:
        """input_type() resolves InputT correctly."""
        input_type = SimpleAgentImplementation.input_type()
        assert input_type is SimpleInputModel

    def test_output_type_resolution(self) -> None:
        """output_type() resolves OutputT correctly."""
        output_type = SimpleAgentImplementation.output_type()
        assert output_type is SimpleOutputModel

    def test_type_resolution_with_inheritance(self) -> None:
        """Type resolution works for two-level inheritance."""

        # Create a subclass of SimpleAgentImplementation
        class DerivedAgent(SimpleAgentImplementation):
            pass

        input_type = DerivedAgent.input_type()
        output_type = DerivedAgent.output_type()

        assert input_type is SimpleInputModel
        assert output_type is SimpleOutputModel

    def test_type_mismatch_raises_error(self) -> None:
        """Missing type params raises error during type resolution."""

        # Create agent without type params
        class BadAgent(Agent):  # type: ignore[type-arg]
            name: ClassVar[str] = "BadAgent"

            def run(self, input_data: BaseModel, context: AgentContext | None = None) -> Any:
                pass

        # Attempting to resolve types should raise an error
        try:
            BadAgent.input_type()
            # If we get here without error, the agent doesn't define proper type params
            raise AssertionError(
                "Expected error when resolving types for agent without type params"
            )
        except (TypeError, ValueError, AttributeError):
            # Expected error when type resolution fails
            pass


class TestAgentContext:
    """Test AgentContext functionality."""

    def test_context_auto_generates_run_id(self) -> None:
        """AgentContext auto-generates run_id."""
        ctx1 = AgentContext()
        ctx2 = AgentContext()

        assert isinstance(ctx1.run_id, UUID)
        assert isinstance(ctx2.run_id, UUID)
        assert ctx1.run_id != ctx2.run_id

    def test_context_auto_generates_timestamp(self) -> None:
        """AgentContext auto-generates created_at."""
        ctx = AgentContext()
        assert isinstance(ctx.created_at, datetime)

    def test_context_metadata_dict(self) -> None:
        """AgentContext carries application metadata."""
        meta = {"user_id": "123", "session": "abc"}
        ctx = AgentContext(metadata=meta)
        assert ctx.metadata == meta

    def test_context_defaults(self) -> None:
        """AgentContext has sensible defaults."""
        ctx = AgentContext()
        assert ctx.run_id is not None
        assert ctx.created_at is not None
        assert ctx.metadata == {}


class TestAgentMetadata:
    """Test agent metadata and descriptions."""

    def test_describe_returns_agent_description(self) -> None:
        """describe() returns AgentDescription."""
        agent = SimpleAgentImplementation()
        desc = agent.describe()

        assert desc.name == "SimpleAgentImplementation"
        assert desc.version == "0.1.0"
        assert desc.description == "A test agent"
        assert "testing" in desc.capabilities
        assert "test" in desc.tags

    def test_describe_includes_steps_for_multistep(self) -> None:
        """describe() includes step names for multi-step agents."""
        agent = MultiStepAgentImplementation()
        desc = agent.describe()

        assert desc.steps == ["analyze", "evaluate", "refine"]
        assert desc.has_loops is True

    def test_describe_includes_schemas(self) -> None:
        """describe() includes input/output schemas."""
        agent = SimpleAgentImplementation()
        desc = agent.describe()

        assert "properties" in desc.input_schema
        assert "properties" in desc.output_schema

    def test_debug_returns_debug_info(self) -> None:
        """debug() returns DebugInfo without executing."""
        agent = SimpleAgentImplementation()
        input_data = SimpleInputModel(text="test")
        debug = agent.debug(input_data)

        assert isinstance(debug, DebugInfo)
        assert debug.system_prompt == ""
        assert "properties" in debug.input_schema
        assert "properties" in debug.output_schema

    def test_debug_includes_step_prompts(self) -> None:
        """debug() includes all step prompts for multi-step agents."""
        agent = MultiStepAgentImplementation()
        input_data = MultiStepInputModel(text="test")
        debug = agent.debug(input_data)

        assert debug.step_prompts == {
            "analyze": "Analyze the text",
            "evaluate": "Evaluate quality",
            "refine": "Refine if needed",
        }

    def test_metadata(self) -> None:
        """metadata() returns AgentMetadata model."""
        agent = SimpleAgentImplementation()
        meta = agent.metadata()

        assert meta.name == "SimpleAgentImplementation"
        assert meta.version == "0.1.0"
        assert meta.requires_llm is True


class TestFrameworkAdapters:
    """Test framework adapter stubs."""

    def test_as_callable(self) -> None:
        """as_callable() wraps agent as plain Python callable."""
        agent = SimpleAgentImplementation()
        fn = agent.as_callable()

        input_data = SimpleInputModel(text="hello")
        output = fn(input_data)

        assert output.result == "Processed: hello"

    def test_as_openai_function(self) -> None:
        """as_openai_function() returns OpenAI-compatible dict."""
        agent = SimpleAgentImplementation()
        func_def = agent.as_openai_function()

        assert func_def["name"] == "simpleagentimplementation"
        assert func_def["description"]
        assert "parameters" in func_def

    def test_mcp_adapter_import_error(self) -> None:
        """as_mcp_tool() returns tool definition or raises helpful error."""
        agent = SimpleAgentImplementation()

        # Try to call the adapter - it should return a tool definition or raise error
        try:
            result = agent.as_mcp_tool()
            # If mcp is installed, we get a tool definition
            assert isinstance(result, dict) or callable(result)
        except Exception as e:
            # If not installed, we expect an informative error message
            error_str = str(e).lower()
            assert "mcp" in error_str or "not installed" in error_str or "import" in error_str

    def test_langchain_adapter_import_error(self) -> None:
        """as_langchain_runnable() returns runnable or raises helpful error."""
        agent = SimpleAgentImplementation()

        # Try to call the adapter - it should return a runnable or raise error
        try:
            result = agent.as_langchain_runnable()
            # If langchain is installed, we get a runnable
            assert callable(result) or hasattr(result, "invoke")
        except Exception as e:
            # If not installed, we expect an informative error message
            error_str = str(e).lower()
            assert "not installed" in error_str or "import" in error_str or "langchain" in error_str


class TestAgentExecution:
    """Test basic agent execution flow."""

    def test_run_returns_agent_result(self) -> None:
        """run() returns AgentResult[OutputT]."""
        agent = SimpleAgentImplementation()
        input_data = SimpleInputModel(text="test input")

        result = agent.run(input_data)

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.output is not None
        assert result.error is None

    def test_run_with_context(self) -> None:
        """run() accepts optional AgentContext."""
        agent = SimpleAgentImplementation()
        input_data = SimpleInputModel(text="test")
        ctx = AgentContext(metadata={"user": "test"})

        result = agent.run(input_data, context=ctx)

        assert result.success is True

    def test_multistep_agent_execution(self) -> None:
        """Multi-step agents execute successfully."""
        agent = MultiStepAgentImplementation()
        input_data = MultiStepInputModel(text="test")

        result = agent.run(input_data)

        assert result.success is True
        assert result.output is not None
        assert result.output.score == 0.9


class TestTypeObjects:
    """Test that type objects are properly accessible."""

    def test_schemas_are_pydantic_schemas(self) -> None:
        """Returned schemas are valid Pydantic JSON schemas."""
        input_type = SimpleAgentImplementation.input_type()
        schema = input_type.model_json_schema()

        assert "properties" in schema
        assert "type" in schema

    def test_input_output_types_can_instantiate(self) -> None:
        """Types resolved from agents can be instantiated."""
        InputType = SimpleAgentImplementation.input_type()
        OutputType = SimpleAgentImplementation.output_type()

        input_instance = InputType(text="test")
        output_instance = OutputType(result="result")

        assert input_instance.text == "test"
        assert output_instance.result == "result"
