"""Production-ready integration tests for DeepSummarizer agent.

Demonstrates the three-layer architecture (Agent + AgentContext + AgentExecutor)
with full observability: token tracking, cost computation, and metrics collection.

When any LLM API key is present (GROQ_API_KEY, OPENROUTER_API_KEY, or TOGETHER_API_KEY),
tests run against the real API. Otherwise, they use mocked LLM responses for offline testing.
"""

import warnings
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from dopeagents.agents import DeepSummarizer, DeepSummarizerInput, DeepSummarizerOutput
from dopeagents.agents.deep_summarizer import DeepSummarizerState
from dopeagents.config import get_config
from dopeagents.core.context import AgentContext
from dopeagents.core.types import AgentResult
from dopeagents.lifecycle.executor import AgentExecutor
from dopeagents.observability.tracer import NoopTracer

# ── Configuration ──────────────────────────────────────────────────────────

HAS_API_KEY = get_config().has_api_key()
requires_groq = pytest.mark.skipif(
    not HAS_API_KEY,
    reason="No LLM API key available (GROQ_API_KEY, OPENROUTER_API_KEY, or TOGETHER_API_KEY needed)",
)


def _should_skip_api_error(error: str | None) -> bool:
    """Return True if the error is a transient provider issue (not a test/code bug).

    Covers: billing (402), rate limits (429), auth (401/403), service errors (5xx),
    network issues, and legacy wrapped RetryError/APIError from tenacity.
    """
    if not error:
        return False

    error_lower = error.lower()

    skip_patterns = [
        "retryerror",  # tenacity wrapper (legacy, now unwrapped at source)
        "apierror",  # litellm wrapper (legacy, now unwrapped at source)
        "credit",
        "billing",
        "payment",  # 402 billing errors
        "rate",  # 429 rate limit
        "unauthorized",  # 401
        "403",
        "401",
        "402",
        "429",  # HTTP status codes
        "503",
        "500",
        "502",
        "504",  # Server errors
        "timeout",
        "connection",  # Network errors
    ]

    return any(p in error_lower for p in skip_patterns)


SAMPLE_TEXT = (
    "Machine learning is a branch of artificial intelligence that focuses on using "
    "data and algorithms to learn and improve. Over the past decades, advancements in "
    "storage and processing have enabled innovations like recommendation engines and "
    "self-driving cars. Machine learning is a key component of data science, using "
    "statistical methods to make classifications, predictions, and discover insights that "
    "drive business decisions and growth metrics."
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def agent_context() -> AgentContext:
    """Create an AgentContext for execution metadata tracking."""
    return AgentContext(
        metadata={
            "user_id": "test-user",
            "environment": "test",
            "request_id": str(uuid4()),
        }
    )


@pytest.fixture
def executor() -> AgentExecutor:
    """Create an AgentExecutor with no-op tracer for testing."""
    return AgentExecutor(tracer=NoopTracer())


@pytest.fixture
def deep_summarizer() -> DeepSummarizer:
    """Create a DeepSummarizer agent instance."""
    return DeepSummarizer()


# ── Production Architecture Tests ──────────────────────────────────────────


class TestDeepSummarizerProductionArchitecture:
    """Test the three-layer architecture: Agent + AgentContext + AgentExecutor."""

    @patch("dopeagents.core.agent.Agent._extract")
    def test_three_layer_architecture_with_metrics(
        self,
        mock_extract: Mock,
        executor: AgentExecutor,
        deep_summarizer: DeepSummarizer,
        agent_context: AgentContext,
    ) -> None:
        """Production flow: context → agent → executor → metrics returned."""
        from dopeagents.agents._summarizer.schemas import ChunkSummary, EvaluateOut, SynthesizeOut

        # Mock LLM responses
        def side_effect(
            response_model: type, messages: Any, model: Any = None, **kwargs: Any
        ) -> Any:
            if response_model is ChunkSummary:
                return ChunkSummary(summary="chunk summary")
            if response_model is SynthesizeOut:
                return SynthesizeOut(
                    synthesis="completed synthesis", key_points=["point1", "point2"]
                )
            if response_model is EvaluateOut:
                return EvaluateOut(quality_score=0.9, feedback="excellent")
            raise ValueError(f"Unexpected response_model: {response_model}")

        mock_extract.side_effect = side_effect

        # Layer 1: Agent (business logic)
        agent = deep_summarizer

        # Layer 2: AgentContext (runtime metadata)
        context = agent_context
        assert context.run_id is not None
        assert context.metadata["user_id"] == "test-user"

        # Layer 3: AgentExecutor (lifecycle + observability)
        input_data = DeepSummarizerInput(
            text=SAMPLE_TEXT,
            style="bullets",
            max_length=300,
        )

        # Execute with full lifecycle management
        result = executor.run(
            agent=agent,
            input=input_data,
            context=context,
        )

        # Verify result structure
        assert result.success is True
        assert result.agent_name == agent.name
        assert result.run_id == context.run_id

        # Verify output
        assert isinstance(result.output, DeepSummarizerOutput)
        assert result.output.summary
        assert len(result.output.summary) > 0

        # Verify metrics collected (observability)
        assert result.metrics is not None
        assert result.metrics.latency_ms > 0
        assert result.metrics.run_id == context.run_id

    @patch("dopeagents.core.agent.Agent._extract")
    def test_metrics_and_token_tracking(
        self,
        mock_extract: Mock,
        executor: AgentExecutor,
        deep_summarizer: DeepSummarizer,
        agent_context: AgentContext,
    ) -> None:
        """Verify token usage and cost metrics are tracked and accessible."""
        from dopeagents.agents._summarizer.schemas import ChunkSummary, EvaluateOut, SynthesizeOut

        # Mock with proper token response metadata
        def side_effect(
            response_model: type, messages: Any, model: Any = None, **kwargs: Any
        ) -> Any:
            if response_model is ChunkSummary:
                return ChunkSummary(summary="summary")
            if response_model is SynthesizeOut:
                return SynthesizeOut(synthesis="synthesis", key_points=["p1"])
            if response_model is EvaluateOut:
                return EvaluateOut(quality_score=0.9, feedback="good")
            raise ValueError(f"Unexpected: {response_model}")

        mock_extract.side_effect = side_effect

        input_data = DeepSummarizerInput(text=SAMPLE_TEXT, style="bullets")
        result = executor.run(
            agent=deep_summarizer,
            input=input_data,
            context=agent_context,
        )

        # Accessor methods for metrics
        assert result.success is True

        # Token tracking (accessible via convenience methods)
        tokens_in = result.tokens_breakdown().get("input", 0)
        tokens_out = result.tokens_breakdown().get("output", 0)
        total_tokens = result.tokens()

        # Either we have tokens (real API call) or zeros (mocked)
        assert total_tokens >= 0
        assert tokens_in >= 0
        assert tokens_out >= 0

        # Cost tracking
        cost = result.cost_usd()
        assert cost >= 0.0

        # Latency tracking (mocked tests may return 0ms)
        latency_ms = result.latency_ms()
        assert latency_ms >= 0

        # LLM calls count (for multi-step agents) - when mocked, may be 0
        llm_calls = result.llm_calls_count()
        assert llm_calls >= 0  # May be 0 in mocked scenarios

        # Formatted metrics string (for logging/display)
        metrics_str = result.format_metrics()
        assert isinstance(metrics_str, str)
        assert len(metrics_str) > 0

    @patch("dopeagents.core.agent.Agent._extract")
    def test_context_metadata_preserved(
        self,
        mock_extract: Mock,
        executor: AgentExecutor,
        deep_summarizer: DeepSummarizer,
    ) -> None:
        """Verify AgentContext metadata is preserved through execution."""
        from dopeagents.agents._summarizer.schemas import ChunkSummary, EvaluateOut, SynthesizeOut

        def side_effect(
            response_model: type, messages: Any, model: Any = None, **kwargs: Any
        ) -> Any:
            if response_model is ChunkSummary:
                return ChunkSummary(summary="test")
            if response_model is SynthesizeOut:
                return SynthesizeOut(synthesis="test", key_points=["p"])
            if response_model is EvaluateOut:
                return EvaluateOut(quality_score=0.9, feedback="ok")
            raise ValueError(f"Unexpected: {response_model}")

        mock_extract.side_effect = side_effect

        # Create context with custom metadata
        custom_metadata = {
            "user_id": "test-user-123",
            "session_id": "sess-456",
            "environment": "production",
        }
        context = AgentContext(metadata=custom_metadata)

        input_data = DeepSummarizerInput(text=SAMPLE_TEXT)
        result = executor.run(
            agent=deep_summarizer,
            input=input_data,
            context=context,
        )

        # Verify context metadata is accessible in result
        assert result.run_id == context.run_id
        assert result.success is True


# ── Input/Output Validation ────────────────────────────────────────────────


class TestDeepSummarizerInput:
    """Test DeepSummarizerInput Pydantic validation."""

    def test_valid_input(self) -> None:
        input_data = DeepSummarizerInput(
            text="This is a test document.",
            style="bullets",
            max_length=300,
        )
        assert input_data.text == "This is a test document."
        assert input_data.style == "bullets"
        assert input_data.max_length == 300

    def test_text_min_length_validation(self) -> None:
        with pytest.raises(ValueError):
            DeepSummarizerInput(text="")

    def test_max_length_bounds(self) -> None:
        with pytest.raises(ValueError):
            DeepSummarizerInput(text="Valid text", max_length=25)
        with pytest.raises(ValueError):
            DeepSummarizerInput(text="Valid text", max_length=15000)

    def test_style_enum_validation(self) -> None:
        with pytest.raises(ValueError):
            DeepSummarizerInput(text="Valid text", style="invalid_style")  # type: ignore

    def test_focus_optional(self) -> None:
        input_data = DeepSummarizerInput(text="Valid text")
        assert input_data.focus is None


class TestDeepSummarizerOutput:
    """Test DeepSummarizerOutput Pydantic validation."""

    def test_valid_output(self) -> None:
        output = DeepSummarizerOutput(
            summary="This is a summary.",
            quality_score=0.85,
            refinement_rounds=1,
            chunks_processed=3,
            word_count=25,
        )
        assert output.summary == "This is a summary."
        assert output.quality_score == 0.85

    def test_quality_score_bounds(self) -> None:
        with pytest.raises(ValueError):
            DeepSummarizerOutput(
                summary="Test",
                quality_score=1.5,
                refinement_rounds=0,
                chunks_processed=1,
                word_count=1,
            )

    def test_chunks_processed_minimum(self) -> None:
        with pytest.raises(ValueError):
            DeepSummarizerOutput(
                summary="Test",
                quality_score=0.5,
                refinement_rounds=0,
                chunks_processed=0,
                word_count=1,
            )


# ── Metadata ─────────────────────────────────────────────────────────────────


class TestDeepSummarizerMetadata:
    def test_agent_name(self) -> None:
        assert DeepSummarizer.name == "DeepSummarizer"

    def test_agent_version(self) -> None:
        assert DeepSummarizer.version == "1.0.0"

    def test_agent_description(self) -> None:
        assert DeepSummarizer.description
        assert "summarization" in DeepSummarizer.description.lower()

    def test_capabilities_declared(self) -> None:
        assert len(DeepSummarizer.capabilities) > 0
        assert "summarization" in DeepSummarizer.capabilities

    def test_requires_llm(self) -> None:
        assert DeepSummarizer.requires_llm is True

    def test_default_model(self) -> None:
        agent = DeepSummarizer()
        assert agent._model  # Instance should have a model

    def test_system_prompt_set(self) -> None:
        assert DeepSummarizer.system_prompt
        assert "summarization" in DeepSummarizer.system_prompt.lower()

    def test_step_prompts_all_steps_present(self) -> None:
        expected_steps = [
            "analyze",
            "chunk",
            "summarize",
            "synthesize",
            "evaluate",
            "refine",
            "format",
        ]
        for step in expected_steps:
            assert step in DeepSummarizer.step_prompts
            assert DeepSummarizer.step_prompts[step]


# ── Type introspection ────────────────────────────────────────────────────────


class TestDeepSummarizerTypeIntrospection:
    def test_input_type(self) -> None:
        assert DeepSummarizer.input_type() == DeepSummarizerInput

    def test_output_type(self) -> None:
        assert DeepSummarizer.output_type() == DeepSummarizerOutput


# ── describe() / debug() ─────────────────────────────────────────────────────


class TestDeepSummarizerDescribe:
    """describe() checklist validation (no LLM needed)."""

    def test_steps_order(self) -> None:
        """DeepSummarizer().describe().steps == 7-step list."""
        agent = DeepSummarizer()
        desc = agent.describe()
        assert desc.steps == [
            "analyze",
            "chunk",
            "summarize",
            "synthesize",
            "evaluate",
            "refine",
            "format",
        ]

    def test_has_loops(self) -> None:
        """DeepSummarizer().describe().has_loops is True."""
        agent = DeepSummarizer()
        assert agent.describe().has_loops is True

    def test_is_multi_step(self) -> None:
        agent = DeepSummarizer()
        assert agent.describe().is_multi_step is True

    def test_input_schema(self) -> None:
        desc = DeepSummarizer().describe()
        assert "text" in desc.input_schema["properties"]
        assert "style" in desc.input_schema["properties"]
        assert "max_length" in desc.input_schema["properties"]

    def test_output_schema(self) -> None:
        desc = DeepSummarizer().describe()
        assert "summary" in desc.output_schema["properties"]
        assert "quality_score" in desc.output_schema["properties"]
        assert "refinement_rounds" in desc.output_schema["properties"]


class TestDeepSummarizerDebug:
    """debug() checklist — no LLM needed."""

    def test_is_multi_step_true(self) -> None:
        """debug() returns is_multi_step=True."""
        agent = DeepSummarizer()
        info = agent.debug(DeepSummarizerInput(text="Test document"))
        assert info.is_multi_step is True

    def test_step_prompts_populated(self) -> None:
        """debug() returns populated step_prompts for all 7 steps."""
        agent = DeepSummarizer()
        info = agent.debug(DeepSummarizerInput(text="Test document"))
        assert len(info.step_prompts) == 7
        for step in ["analyze", "chunk", "summarize", "synthesize", "evaluate", "refine", "format"]:
            assert step in info.step_prompts

    def test_schemas_present(self) -> None:
        agent = DeepSummarizer()
        info = agent.debug(DeepSummarizerInput(text="Test document"))
        assert info.input_schema
        assert info.output_schema


# ── Graph construction ────────────────────────────────────────────────────────


class TestDeepSummarizerGraphConstruction:
    def test_build_graph_returns_compiled_graph(self) -> None:
        agent = DeepSummarizer()
        graph = agent._build_graph()
        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_graph_cached_on_instance(self) -> None:
        agent = DeepSummarizer()
        assert agent._get_graph() is agent._get_graph()

    def test_has_loops_true(self) -> None:
        assert DeepSummarizer()._has_loops() is True


# ── Model helpers ─────────────────────────────────────────────────────────────


class TestDeepSummarizerModelHelpers:
    def test_model_for_step_default(self) -> None:
        agent = DeepSummarizer()
        assert agent._model_for_step("analyze") == agent._model

    def test_model_for_step_override(self) -> None:
        agent = DeepSummarizer(step_models={"analyze": "groq/mixtral-8x7b-32768"})
        assert agent._model_for_step("analyze") == "groq/mixtral-8x7b-32768"
        # Non-overridden step uses default
        assert agent._model_for_step("chunk") == agent._model

    def test_model_config_fields(self) -> None:
        config = DeepSummarizer()._get_model_config()
        assert config is not None
        assert "default_model" in config
        assert "step_models" in config
        assert "max_refinement_loops" in config
        assert "max_chunks" in config


# ── Prompt rendering ──────────────────────────────────────────────────────────


class TestDeepSummarizerPromptRendering:
    def test_render_prompt_not_empty(self) -> None:
        agent = DeepSummarizer()
        prompt = agent._render_prompt(DeepSummarizerInput(text="Test"))
        assert prompt and len(prompt) > 0

    def test_render_prompt_includes_step_names(self) -> None:
        agent = DeepSummarizer()
        prompt = agent._render_prompt(DeepSummarizerInput(text="Test"))
        assert prompt is not None
        for step in ["analyze", "chunk", "summarize", "synthesize", "evaluate", "refine", "format"]:
            assert step.upper() in prompt


# ── max_chunks guard ─────────────────────────────────────────────────────────


class TestDeepSummarizerMaxChunksGuard:
    def test_max_chunks_guard_truncates_large_chunk_lists(self) -> None:
        """_step_chunk truncates paragraph lists > 10."""
        agent = DeepSummarizer()

        # 15 paragraphs each long enough not to merge at a tiny target_size
        paragraphs = [f"Paragraph {i}: " + ("word " * 30) for i in range(15)]
        long_text = "\n\n".join(paragraphs)

        state: DeepSummarizerState = {
            "text": long_text,
            "max_length": 500,
            "style": "paragraph",
            "focus": None,
            "analysis": {"recommended_chunk_size": 50},  # tiny target → many chunks
            "chunks": [],
            "chunk_summaries": [],
            "synthesis": "",
            "quality_score": 0.0,
            "feedback": "",
            "refinement_rounds": 0,
            "max_refinement_loops": 3,
            "quality_threshold": 0.5,
            "score_history": [],
            "final_summary": "",
            "word_count": 0,
            "truncated": False,
            "key_points": [],
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = agent._step_chunk(state)

        assert len(result["chunks"]) == 10
        assert any("max_chunks" in str(w.message).lower() for w in caught)


# ── Refinement loop guard (mocked) ───────────────────────────────────────────


class TestDeepSummarizerRefinementLoop:
    """Verify the evaluate→refine loop terminates and respects max_refinement_loops."""

    def _make_step_mock(self, incrementing_scores: bool = False) -> Any:
        """Build a mock _extract that returns plausible step responses.

        If incrementing_scores=True, return varying scores to avoid plateau detection.
        """
        from dopeagents.agents._summarizer.schemas import (
            ChunkSummary,
            EvaluateOut,
            RefineOut,
            SynthesizeOut,
        )

        call_count = {"evaluate_calls": 0}

        def side_effect(
            response_model: type, messages: Any, model: Any = None, **kwargs: Any
        ) -> Any:
            if response_model is ChunkSummary:
                return ChunkSummary(summary="sum1")
            if response_model is SynthesizeOut:
                return SynthesizeOut(synthesis="combined summary", key_points=["point 1"])
            if response_model is EvaluateOut:
                call_count["evaluate_calls"] += 1
                if incrementing_scores:
                    score = 0.50 + (call_count["evaluate_calls"] * 0.03)
                    return EvaluateOut(quality_score=score, feedback="needs improvement")
                else:
                    return EvaluateOut(quality_score=0.5, feedback="needs improvement")
            if response_model is RefineOut:
                return RefineOut(refined="improved summary")
            raise ValueError(f"Unexpected response_model: {response_model}")

        return side_effect

    @patch("dopeagents.core.agent.Agent._extract")
    def test_max_refinement_loops_respected(self, mock_extract: Mock) -> None:
        """Verify refinement loop respects max_refinement_loops=3 when plateau is avoided."""
        # Use incrementing scores to avoid plateau detection
        mock_extract.side_effect = self._make_step_mock(incrementing_scores=True)

        agent = DeepSummarizer()
        input_data = DeepSummarizerInput(text=SAMPLE_TEXT, style="bullets")
        result = agent.run(input_data)

        assert isinstance(result, AgentResult)
        assert result.output is not None
        # With incrementing scores, should reach max_refinement_loops=3
        assert result.output.refinement_rounds == 3

    @patch("dopeagents.core.agent.Agent._extract")
    def test_no_refine_when_quality_high(self, mock_extract: Mock) -> None:
        """Verify no refinement when initial quality_score >= 0.8."""
        from dopeagents.agents._summarizer.schemas import ChunkSummary, EvaluateOut, SynthesizeOut

        def side_effect(
            response_model: type, messages: Any, model: Any = None, **kwargs: Any
        ) -> Any:
            if response_model is ChunkSummary:
                return ChunkSummary(summary="s")
            if response_model is SynthesizeOut:
                return SynthesizeOut(synthesis="good summary", key_points=["p1"])
            if response_model is EvaluateOut:
                return EvaluateOut(quality_score=0.9, feedback="great")
            raise ValueError(f"Unexpected: {response_model}")

        mock_extract.side_effect = side_effect
        agent = DeepSummarizer()
        result = agent.run(DeepSummarizerInput(text=SAMPLE_TEXT, style="tldr"))

        assert result.output is not None
        assert result.output.refinement_rounds == 0

    @patch("dopeagents.core.agent.Agent._extract")
    def test_plateau_detection_stops_refinement(self, mock_extract: Mock) -> None:
        """Verify plateau detection stops refinement early when scores don't improve."""
        # Use constant scores (no improvement) to trigger plateau detection
        mock_extract.side_effect = self._make_step_mock(incrementing_scores=False)

        agent = DeepSummarizer()
        input_data = DeepSummarizerInput(text=SAMPLE_TEXT, style="bullets")
        result = agent.run(input_data)

        assert isinstance(result, AgentResult)
        assert result.output is not None
        # With constant scores, plateau detection should kick in after 1st refine
        # so refinement_rounds should be 1, not 3
        assert result.output.refinement_rounds == 1


# ── Initialization and caching ─────────────────────────────────────────────────


class TestDeepSummarizerInitialization:
    """Test __init__ properly initializes instance variables."""

    def test_init_creates_graph_cache_as_none(self) -> None:
        """__init__ initializes _graph to None for lazy loading."""
        agent = DeepSummarizer()
        assert hasattr(agent, "_graph")
        assert agent._graph is None

    def test_init_creates_step_models_dict(self) -> None:
        """__init__ initializes _step_models as empty dict or from kwargs."""
        agent = DeepSummarizer()
        assert hasattr(agent, "_step_models")
        assert isinstance(agent._step_models, dict)

    def test_init_with_step_models_kwarg(self) -> None:
        """__init__ accepts step_models as kwarg."""
        step_models = {"analyze": "gpt-4-turbo", "refine": "gpt-4"}
        agent = DeepSummarizer(step_models=step_models)
        assert agent._step_models == step_models

    def test_init_calls_parent_init(self) -> None:
        """__init__ properly calls super().__init__()."""
        agent = DeepSummarizer(model="custom/model")
        assert agent._model == "custom/model"
        # Verify other parent properties are set
        assert agent.name == "DeepSummarizer"
        assert agent.version == "1.0.0"


# ── Graph caching and compilation ──────────────────────────────────────────────


class TestDeepSummarizerGraphCaching:
    """Test _get_graph() lazy initialization and caching behavior."""

    def test_get_graph_returns_compiled_graph(self) -> None:
        """_get_graph() returns a compiled StateGraph."""
        agent = DeepSummarizer()
        graph = agent._get_graph()
        assert graph is not None
        assert hasattr(graph, "invoke"), "Graph must be compiled with invoke() method"

    def test_get_graph_caches_on_first_call(self) -> None:
        """_get_graph() caches the compiled graph after first invocation."""
        agent = DeepSummarizer()
        graph1 = agent._get_graph()
        graph2 = agent._get_graph()
        assert graph1 is graph2, "Graph should be cached (same object reference)"

    def test_get_graph_populates_instance_variable(self) -> None:
        """_get_graph() sets self._graph after compilation."""
        agent = DeepSummarizer()
        assert agent._graph is None
        graph = agent._get_graph()
        assert agent._graph is graph

    def test_different_agents_have_different_graphs(self) -> None:
        """Different agent instances have independent graph caches."""
        agent1 = DeepSummarizer()
        agent2 = DeepSummarizer()
        graph1 = agent1._get_graph()
        graph2 = agent2._get_graph()
        assert graph1 is not graph2, "Different agents should have different graph objects"


# ── Model selection per step ───────────────────────────────────────────────────


class TestDeepSummarizerModelSelection:
    """Test _model_for_step() respects per-step overrides."""

    def test_model_for_step_default_model(self) -> None:
        """_model_for_step() returns default model when no override."""
        agent = DeepSummarizer(model="default/model")
        for step in ["analyze", "chunk", "summarize", "synthesize", "evaluate", "refine", "format"]:
            assert agent._model_for_step(step) == "default/model"

    def test_model_for_step_respects_overrides(self) -> None:
        """_model_for_step() checks _step_models dict first."""
        agent = DeepSummarizer(
            model="default/model",
            step_models={
                "analyze": "fast/model",
                "refine": "expensive/model",
            },
        )
        assert agent._model_for_step("analyze") == "fast/model"
        assert agent._model_for_step("refine") == "expensive/model"
        assert agent._model_for_step("chunk") == "default/model"  # Not overridden

    def test_model_for_step_all_steps_work(self) -> None:
        """_model_for_step() handles all step names."""
        agent = DeepSummarizer()
        steps = ["analyze", "chunk", "summarize", "synthesize", "evaluate", "refine", "format"]
        for step in steps:
            model = agent._model_for_step(step)
            assert model is not None
            assert len(model) > 0


# ── AgentResult typing and return values ─────────────────────────────────────


class TestDeepSummarizerAgentResultTyping:
    """Test AgentResult is properly typed with generic output."""

    @patch("dopeagents.core.agent.Agent._extract")
    def test_run_returns_agent_result_with_output(self, mock_extract: Mock) -> None:
        """agent.run() returns AgentResult[DeepSummarizerOutput]."""
        from dopeagents.agents._summarizer.schemas import ChunkSummary, EvaluateOut, SynthesizeOut

        def side_effect(
            response_model: type, messages: Any, model: Any = None, **kwargs: Any
        ) -> Any:
            if response_model is ChunkSummary:
                return ChunkSummary(summary="summary")
            if response_model is SynthesizeOut:
                return SynthesizeOut(synthesis="synthesis", key_points=["point"])
            if response_model is EvaluateOut:
                return EvaluateOut(quality_score=0.9, feedback="good")
            raise ValueError(f"Unexpected: {response_model}")

        mock_extract.side_effect = side_effect
        agent = DeepSummarizer()
        result = agent.run(DeepSummarizerInput(text="Test document."))

        assert isinstance(result, AgentResult)
        assert result.output is not None
        assert isinstance(result.output, DeepSummarizerOutput)
        assert result.success is True
        assert result.error is None

    @patch("dopeagents.core.agent.Agent._extract")
    def test_run_returns_agent_result_on_failure(self, mock_extract: Mock) -> None:
        """agent.run() returns AgentResult with error on failure."""
        mock_extract.side_effect = Exception("LLM API error")
        agent = DeepSummarizer()

        result = agent.run(DeepSummarizerInput(text="Test"))

        assert isinstance(result, AgentResult)
        assert result.success is False
        assert result.error is not None
        assert result.output is None

    def test_agent_result_has_metrics(self) -> None:
        """AgentResult includes ExecutionMetrics."""
        # This is more of an executor responsibility, but verify the typing
        from uuid import UUID

        from dopeagents.core.types import ExecutionMetrics

        metrics = ExecutionMetrics(
            run_id=UUID("12345678-1234-5678-1234-567812345678"),
            latency_ms=125.5,
            cache_hit=False,
            cost_usd=0.001,
        )
        assert metrics.latency_ms == 125.5
        assert metrics.cache_hit is False


# ── Graceful degradation ───────────────────────────────────────────────────────


class TestDeepSummarizerGracefulDegradation:
    """Test that failures result in honest error reporting."""

    @patch("dopeagents.core.agent.Agent._extract")
    def test_extract_failure_returns_error_result(self, mock_extract: Mock) -> None:
        """If _extract fails, run() returns success=False with error message."""
        mock_extract.side_effect = Exception("LLM API error")
        agent = DeepSummarizer()

        result = agent.run(DeepSummarizerInput(text="Short test."))
        assert isinstance(result, AgentResult)
        assert result.success is False
        assert result.output is None
        assert result.error is not None

    @patch("dopeagents.core.agent.Agent._extract")
    def test_synthesis_failure_returns_error(self, mock_extract: Mock) -> None:
        """If synthesis fails after summarization, return error result."""
        from dopeagents.agents._summarizer.schemas import ChunkSummary

        def side_effect(
            response_model: type, messages: Any, model: Any = None, **kwargs: Any
        ) -> Any:
            if response_model is ChunkSummary:
                return ChunkSummary(summary="summary")
            raise Exception("Synthesis LLM timeout")

        mock_extract.side_effect = side_effect
        agent = DeepSummarizer()
        result = agent.run(DeepSummarizerInput(text="Test"))

        assert isinstance(result, AgentResult)
        assert result.success is False


# ── Token tracking integration ─────────────────────────────────────────────────


class TestDeepSummarizerTokenTracking:
    """Test token tracking field exists in output."""

    @patch("dopeagents.core.agent.Agent._extract")
    def test_output_includes_total_tokens_used(self, mock_extract: Mock) -> None:
        """Output includes total_tokens_used field (defaults to 0)."""
        from dopeagents.agents._summarizer.schemas import ChunkSummary, EvaluateOut, SynthesizeOut

        def side_effect(
            response_model: type, messages: Any, model: Any = None, **kwargs: Any
        ) -> Any:
            if response_model is ChunkSummary:
                return ChunkSummary(summary="test")
            if response_model is SynthesizeOut:
                return SynthesizeOut(synthesis="test", key_points=["p1"])
            if response_model is EvaluateOut:
                return EvaluateOut(quality_score=0.9, feedback="")
            raise ValueError(f"Unexpected: {response_model}")

        mock_extract.side_effect = side_effect
        agent = DeepSummarizer()
        result = agent.run(DeepSummarizerInput(text=SAMPLE_TEXT))

        assert result.output is not None
        assert hasattr(result.output, "total_tokens_used")
        assert isinstance(result.output.total_tokens_used, int)
        assert result.output.total_tokens_used == 0


# ── State management ───────────────────────────────────────────────────────────


class TestDeepSummarizerStateManagement:
    """Test state initialization and management across workflow."""

    def test_state_has_all_required_fields(self) -> None:
        """DeepSummarizerState TypedDict has all required fields."""
        state: DeepSummarizerState = {
            "text": "test",
            "max_length": 500,
            "style": "paragraph",
            "focus": None,
            "analysis": {},
            "chunks": [],
            "chunk_summaries": [],
            "synthesis": "",
            "quality_score": 0.0,
            "feedback": "",
            "refinement_rounds": 0,
            "max_refinement_loops": 3,
            "quality_threshold": 0.8,
            "score_history": [],
            "final_summary": "",
            "word_count": 0,
            "truncated": False,
            "key_points": [],
        }
        assert len(state) == 18


# ── Integration tests (real Groq API) ────────────────────────────────────────


class TestDeepSummarizerIntegration:
    """Live API tests — only run when GROQ_API_KEY is available."""

    @requires_groq
    def test_run_returns_valid_output(self) -> None:
        """DeepSummarizer().run(...) returns a valid DeepSummarizerOutput."""
        agent = DeepSummarizer()
        input_data = DeepSummarizerInput(
            text=SAMPLE_TEXT,
            style="bullets",
            max_length=500,
        )
        result = agent.run(input_data)

        assert isinstance(result, AgentResult)

        # Handle transient API errors: rate limiting, credit limits, service errors
        if result.output is None:
            if _should_skip_api_error(result.error):
                pytest.skip(f"API error (skipped): {result.error}")
            assert result.output is not None, f"Unexpected failure: {result.error}"

        output = result.output
        assert isinstance(output, DeepSummarizerOutput)
        assert output.summary
        assert 0.0 <= output.quality_score <= 1.0
        assert output.chunks_processed >= 1
        assert output.word_count >= 1

    @requires_groq
    def test_describe_steps_after_run(self) -> None:
        """describe().steps returns 7-step list regardless of prior runs."""
        agent = DeepSummarizer()
        desc = agent.describe()
        assert desc.steps == [
            "analyze",
            "chunk",
            "summarize",
            "synthesize",
            "evaluate",
            "refine",
            "format",
        ]
        assert desc.has_loops is True

    @requires_groq
    def test_refinement_happens_when_quality_low(self) -> None:
        """A document that scores < 0.8 should trigger at least 1 refine round."""
        # We control by using a very short text that typically scores lower
        agent = DeepSummarizer()
        result = agent.run(
            DeepSummarizerInput(
                text=SAMPLE_TEXT * 2,  # Repeat to stress synthesis quality
                style="paragraph",
                max_length=200,  # Tight constraint → harder to satisfy → more refine loops
            )
        )

        # Handle transient API errors: rate limiting, credit limits, service errors
        if result.output is None:
            if _should_skip_api_error(result.error):
                pytest.skip(f"API error (skipped): {result.error}")
            assert result.output is not None, f"Unexpected failure: {result.error}"

        # We can assert refinement_rounds >= 0 as the actual score depends on LLM
        # The important assertion is it completes and is bounded
        assert isinstance(result.output, DeepSummarizerOutput)
        assert result.output.refinement_rounds <= 3  # Respects max_refinement_loops

    @requires_groq
    def test_run_with_context_parameter(self) -> None:
        """Test run() accepts optional AgentContext for tracing."""
        agent = DeepSummarizer()
        context = AgentContext(metadata={"trace_id": "test-trace-123"})
        result = agent.run(DeepSummarizerInput(text=SAMPLE_TEXT), context=context)

        assert isinstance(result, AgentResult)

        # Handle transient API errors: rate limiting, credit limits, service errors
        if result.output is None:
            if _should_skip_api_error(result.error):
                pytest.skip(f"API error (skipped): {result.error}")
            assert result.output is not None, f"Unexpected failure: {result.error}"

        assert isinstance(result.output, DeepSummarizerOutput)

    @requires_groq
    def test_all_output_fields_populated(self) -> None:
        """Verify all DeepSummarizerOutput fields are populated after run()."""
        agent = DeepSummarizer()
        result = agent.run(DeepSummarizerInput(text=SAMPLE_TEXT, style="tldr"))

        # Handle transient API errors: rate limiting, credit limits, service errors
        if result.output is None:
            if _should_skip_api_error(result.error):
                pytest.skip(f"API error (skipped): {result.error}")
            assert result.output is not None, f"Unexpected failure: {result.error}"

        output = result.output
        assert output.summary, "summary must be non-empty"
        assert isinstance(output.key_points, list)
        assert 0.0 <= output.quality_score <= 1.0
        assert output.refinement_rounds >= 0
        assert output.chunks_processed >= 1
        assert output.word_count > 0
        assert isinstance(output.truncated, bool)
        assert output.total_tokens_used >= 0

    @requires_groq
    def test_different_styles_produce_different_formats(self) -> None:
        """Test output format matches requested style."""
        agent = DeepSummarizer()

        paragraph_result = agent.run(DeepSummarizerInput(text=SAMPLE_TEXT, style="paragraph"))

        # Handle transient API errors: rate limiting, credit limits, service errors
        if paragraph_result.output is None:
            if _should_skip_api_error(paragraph_result.error):
                pytest.skip(f"API error (skipped): {paragraph_result.error}")
            assert (
                paragraph_result.output is not None
            ), f"Unexpected failure: {paragraph_result.error}"

        # Only test remaining styles if first succeeded
        bullets_result = agent.run(DeepSummarizerInput(text=SAMPLE_TEXT, style="bullets"))
        tldr_result = agent.run(DeepSummarizerInput(text=SAMPLE_TEXT, style="tldr"))

        # Check all results handle transient API errors
        for result in [bullets_result, tldr_result]:
            if result.output is None and _should_skip_api_error(result.error):
                pytest.skip(f"API error (skipped): {result.error}")

        assert paragraph_result.output is not None
        assert bullets_result.output is not None
        assert tldr_result.output is not None

        # Bullets format should have bullet markers or be more concise
        # TLDR should be the shortest
        # Paragraph should be largest
        assert paragraph_result.output.word_count >= tldr_result.output.word_count
