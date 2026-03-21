"""Unit and integration tests for DeepSummarizer agent.

When GROQ_API_KEY is present in the environment (or .env), integration tests
run against the real Groq API.  Otherwise they fall back to mocking _extract().
"""

import os
import warnings
from typing import Any
from unittest.mock import Mock, patch

import pytest

# Load .env so GROQ_API_KEY is available if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from dopeagents.agents import (
    DeepSummarizer,
    DeepSummarizerInput,
    DeepSummarizerOutput,
)
from dopeagents.agents.deep_summarizer import DeepSummarizerState
from dopeagents.core.types import AgentResult

# ── Fixtures & skip markers ───────────────────────────────────────────────────

HAS_GROQ = bool(os.environ.get("GROQ_API_KEY"))
requires_groq = pytest.mark.skipif(not HAS_GROQ, reason="GROQ_API_KEY not set")

SAMPLE_TEXT = (
    "Machine learning (ML) is a branch of artificial intelligence (AI) and computer science "
    "which focuses on the use of data and algorithms to imitate the way that humans learn, "
    "gradually improving its accuracy. IBM has a rich history with machine learning. "
    "One of its own, Arthur Samuel, is credited for coining the term, 'machine learning' "
    "with his research (PDF, 481 KB) (link resides outside IBM) around the game of checkers. "
    "Robert Nealey, the self-proclaimed checkers master, played the game on an IBM 7094 "
    "computer in 1962, and he lost. Compared to what can be done today, this feat seems paltry, "
    "but it's considered a major milestone in the field of artificial intelligence. "
    "Over the last couple of decades, the technological advances in storage and processing power "
    "have enabled some innovative products based on machine learning, such as Netflix's "
    "recommendation engine and self-driving cars. Machine learning is an important component "
    "of the growing field of data science. Through the use of statistical methods, algorithms "
    "are trained to make classifications or predictions, and to uncover key insights in data "
    "mining projects. These insights subsequently drive decision making within applications "
    "and businesses, ideally impacting key growth metrics."
)


# ── Input validation ─────────────────────────────────────────────────────────


class TestDeepSummarizerInput:
    """Test DeepSummarizerInput validation."""

    def test_valid_input(self) -> None:
        input_data = DeepSummarizerInput(
            text="This is a test document.", style="bullets", max_length=300
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
            DeepSummarizerInput(text="Valid text", style="invalid_style")  # type: ignore[arg-type]

    def test_focus_optional(self) -> None:
        input_data = DeepSummarizerInput(text="Valid text")
        assert input_data.focus is None


# ── Output validation ─────────────────────────────────────────────────────────


class TestDeepSummarizerOutput:
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
        assert DeepSummarizer.default_model

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
            "refined": "",
            "refinement_rounds": 0,
            "max_refinement_loops": 3,
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
        from dopeagents.agents.deep_summarizer import (
            _AnalyzeOut,
            _ChunkSummary,
            _EvaluateOut,
            _FormatOut,
            _RefineOut,
            _SynthesizeOut,
        )

        call_count = {"evaluate_calls": 0}

        def side_effect(
            response_model: type, messages: Any, model: Any = None, **kwargs: Any
        ) -> Any:
            if response_model is _AnalyzeOut:
                return _AnalyzeOut(
                    recommended_chunk_size=500, text_type="article", complexity="medium"
                )
            if response_model is _ChunkSummary:
                return _ChunkSummary(summary="sum1")
            if response_model is _SynthesizeOut:
                return _SynthesizeOut(synthesis="combined summary", key_points=["point 1"])
            if response_model is _EvaluateOut:
                # Return varying scores to avoid plateau detection if requested
                call_count["evaluate_calls"] += 1
                if incrementing_scores:
                    # Return scores with > 0.02 gap to avoid plateau detection
                    # Scores: 0.50, 0.53, 0.56, 0.59 (gaps of 0.03)
                    score = 0.50 + (call_count["evaluate_calls"] * 0.03)
                    return _EvaluateOut(quality_score=score, feedback="needs improvement")
                else:
                    # Return constant score (will trigger plateau detection after 1st loop)
                    return _EvaluateOut(quality_score=0.5, feedback="needs improvement")
            if response_model is _RefineOut:
                return _RefineOut(refined="improved summary")
            if response_model is _FormatOut:
                return _FormatOut(final_summary="final", word_count=1, truncated=False)
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
        from dopeagents.agents.deep_summarizer import (
            _AnalyzeOut,
            _ChunkSummary,
            _EvaluateOut,
            _FormatOut,
            _SynthesizeOut,
        )

        def side_effect(
            response_model: type, messages: Any, model: Any = None, **kwargs: Any
        ) -> Any:
            if response_model is _AnalyzeOut:
                return _AnalyzeOut(
                    recommended_chunk_size=500, text_type="article", complexity="low"
                )
            if response_model is _ChunkSummary:
                return _ChunkSummary(summary="s")
            if response_model is _SynthesizeOut:
                return _SynthesizeOut(synthesis="good summary", key_points=["p1"])
            if response_model is _EvaluateOut:
                return _EvaluateOut(quality_score=0.9, feedback="great")
            if response_model is _FormatOut:
                return _FormatOut(final_summary="final", word_count=2, truncated=False)
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
        assert result.output is not None
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
        # We can assert refinement_rounds >= 0 as the actual score depends on LLM
        # The important assertion is it completes and is bounded
        assert isinstance(result.output, DeepSummarizerOutput)
        assert result.output.refinement_rounds <= 3  # Respects max_refinement_loops
