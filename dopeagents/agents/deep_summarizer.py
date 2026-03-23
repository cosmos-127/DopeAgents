"""DeepSummarizer agent: multi-step document summarization with self-evaluation and refinement.

7-step workflow with deterministic analysis, cost-bounded LLM summarization,
and evaluation-driven iterative refinement:
- Text structure analysis and chunk size heuristic (code)
- Split document into cost-bounded chunks (code)
- Summarize each chunk independently (LLM)
- Combine chunk summaries into a coherent whole (LLM)
- Score synthesis on faithfulness, completeness, coherence (LLM)
- Improve synthesis using targeted feedback (LLM)
- Apply output style and length truncation (code)
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from dopeagents.agents._summarizer.analyzer import TextAnalysis, analyze_text
from dopeagents.agents._summarizer.chunker import SummarizerChunker
from dopeagents.agents._summarizer.formatter import SummaryFormatter
from dopeagents.agents._summarizer.schemas import (
    ChunkSummary,
    EvaluateOut,
    RefineOut,
    SynthesizeOut,
)
from dopeagents.core.agent import Agent
from dopeagents.core.context import AgentContext
from dopeagents.core.types import AgentResult
from dopeagents.errors import ExtractionProviderError
from dopeagents.observability.logging import get_logger

if TYPE_CHECKING:
    from dopeagents.cost.guard import BudgetConfig
    from dopeagents.observability.tracer import Span

logger = get_logger(__name__)

# ── Internal State ─────────────────────────────────────────────────────────


class DeepSummarizerState(TypedDict):
    """Internal LangGraph state shared across all steps."""

    text: str
    max_length: int
    style: str
    focus: str | None
    analysis: dict[str, Any]
    chunks: list[str]
    chunk_summaries: list[str]
    synthesis: str
    quality_score: float
    feedback: str
    refinement_rounds: int
    max_refinement_loops: int
    quality_threshold: float
    score_history: list[float]
    final_summary: str
    word_count: int
    truncated: bool
    key_points: list[str]


# ── Public Input / Output ──────────────────────────────────────────────────────


class DeepSummarizerInput(BaseModel):
    """Input schema for DeepSummarizer agent."""

    text: str = Field(
        min_length=1, max_length=500_000, description="Text to summarize (max 500 000 chars)"
    )
    max_length: int = Field(default=500, ge=50, le=10000, description="Maximum length of summary")
    style: Literal["paragraph", "bullets", "tldr"] = Field(
        default="paragraph", description="Output format style"
    )
    focus: str | None = Field(default=None, description="Optional focus area for summarization")
    quality_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum quality score to accept without refinement",
    )
    max_refinement_loops: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of evaluate→refine cycles",
    )


class DeepSummarizerOutput(BaseModel):
    """Output schema for DeepSummarizer agent."""

    summary: str = Field(description="The generated summary")
    key_points: list[str] = Field(default_factory=list, description="Key points in the summary")
    quality_score: float = Field(ge=0.0, le=1.0, description="Final quality score")
    refinement_rounds: int = Field(ge=0, description="Number of refinement loops executed")
    chunks_processed: int = Field(ge=1, description="Number of text chunks")
    word_count: int = Field(ge=0, description="Word count of summary")
    truncated: bool = Field(default=False, description="Whether summary was truncated")
    total_tokens_used: int = Field(default=0, description="Total tokens used across all LLM calls")


# ── Internal helpers ─────────────────────────────────────────────────────────

_EVAL_EXCERPT_CHARS: int = 8000


def _sample_source(text: str, max_chars: int = _EVAL_EXCERPT_CHARS) -> str:
    """Sample start, middle, and end sections for evaluation/refinement grounding.

    Returns full text when it fits within max_chars; otherwise returns equal slices
    from the start, middle, and end so faithfulness scoring covers the whole document
    rather than only the opening section.
    """
    if len(text) <= max_chars:
        return text
    third = max_chars // 3
    mid = len(text) // 2
    return "\n\n[\u2026section omitted\u2026]\n\n".join(
        [
            text[:third],
            text[mid - third // 2 : mid + (third - third // 2)],
            text[-third:],
        ]
    )


# ── DeepSummarizer Agent ───────────────────────────────────────────────────────


class DeepSummarizer(Agent[DeepSummarizerInput, DeepSummarizerOutput]):
    """
    7-step summarization agent with self-evaluation and iterative refinement.

    Steps (in order):
      1. analyze    — Code: Text structure analysis and chunk size heuristic
      2. chunk      — Code: Split document into cost-bounded chunks
      3. summarize  — LLM:  Summarize each chunk independently
      4. synthesize — LLM:  Combine chunk summaries into a coherent whole
      5. evaluate   — LLM:  Score synthesis on faithfulness, completeness, coherence
      6. refine     — LLM:  Improve synthesis using targeted feedback (loops to 5)
      7. format     — Code: Apply output style and length truncation

    The evaluate→refine→evaluate cycle continues until quality_score >= threshold,
    max_refinement_loops is reached, or the score plateaus (< 0.02 change).

    Minimum LLM calls per run: 3 (single chunk, quality above threshold on first eval).
    """

    name: ClassVar[str] = "DeepSummarizer"
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = (
        "Multi-step document summarization with chunking, synthesis, "
        "self-evaluation, and iterative refinement"
    )
    capabilities: ClassVar[list[str]] = ["summarization", "text-analysis", "refinement"]
    tags: ClassVar[list[str]] = ["text", "llm-based", "multi-step"]
    requires_llm: ClassVar[bool] = True

    system_prompt: ClassVar[str] = (
        "You are an expert document summarization agent. Your task is to produce "
        "high-quality, concise summaries that capture the essence of documents."
    )

    step_prompts: ClassVar[dict[str, str]] = {
        "analyze": "Deterministic text analysis: structure detection and chunk size heuristic.",
        "chunk": (
            "Split the provided text into semantically coherent chunks. "
            "Each chunk should be complete and meaningful. "
            "Return a list of chunks respecting the recommended chunk size."
        ),
        "summarize": "Summarize the chunk concisely, capturing all key information.",
        "synthesize": (
            "Synthesize the chunk summaries into one coherent summary. Extract 3-5 key points."
        ),
        "evaluate": (
            "Score the summary 0-1 on faithfulness (all claims grounded in source), "
            "completeness (all key points covered), coherence (structure and readability), "
            "and overall quality. List any claims not supported by the source text."
        ),
        "refine": "Improve the summary by addressing all points in the evaluation feedback.",
        "format": "Deterministic formatting: style application and truncation.",
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initialize DeepSummarizer.

        ``Agent.__init__`` (the parent) sets ``self._model``, ``self._step_models``
        (per-step model overrides), and ``self._graph`` (lazily built on first run).
        ``self._extract(response_model, messages, model)`` is the sole LLM interface,
        defined on ``Agent`` and instrumented by the executor for cost tracking.

        Args:
            **kwargs: Passed to parent Agent.__init__ (model, hooks, cache, etc.) plus
                      ``step_prompts`` — a dict whose keys override the class-level defaults.
        """
        custom_step_prompts = kwargs.pop("step_prompts", None)
        super().__init__(**kwargs)

        # Always create an instance-level copy from the class constant so the ClassVar
        # is never mutated; other instances and class-level tooling see stable values.
        merged = dict(DeepSummarizer.step_prompts)
        if custom_step_prompts is not None:
            merged.update(custom_step_prompts)
        self.step_prompts: dict[str, str] = merged  # type: ignore[misc]

        # Lock for thread-safe lazy graph initialization
        self._graph_lock = threading.Lock()
        # Thread-local storage for per-run context (tracer + budget access)
        self._run_local = threading.local()

    def _model_for_step(self, step_name: str) -> str:
        """Get the model for a specific step, respecting per-step overrides.

        Args:
            step_name: Name of the step (analyze, chunk, summarize, etc.)

        Returns:
            Model string to use for this step
        """
        return self._step_models.get(step_name, self._model)

    def _get_graph(self) -> Any:
        """Get or lazily build the compiled LangGraph state machine.

        Caches the compiled graph in self._graph to avoid rebuilding on each run.
        Thread-safe via double-checked locking.

        Returns:
            Compiled StateGraph ready for invocation
        """
        if self._graph is None:
            with self._graph_lock:
                if self._graph is None:
                    self._graph = self._build_graph()
        return self._graph

    def _build_graph(self) -> Any:
        """Build the internal LangGraph StateGraph for the 7-step workflow.

        Steps 1 (analyze), 2 (chunk), and 7 (format) are deterministic.
        Steps 3-6 use LLM calls.

        Returns:
            Compiled and ready-to-invoke StateGraph
        """
        graph = StateGraph(DeepSummarizerState)

        # Add all step nodes
        graph.add_node("analyze", self._step_analyze)
        graph.add_node("chunk", self._step_chunk)
        graph.add_node("summarize", self._step_summarize)
        graph.add_node("synthesize", self._step_synthesize)
        graph.add_node("evaluate", self._step_evaluate)
        graph.add_node("refine", self._step_refine)
        graph.add_node("format", self._step_format)

        # Build edge sequence
        graph.add_edge(START, "analyze")
        graph.add_edge("analyze", "chunk")
        graph.add_edge("chunk", "summarize")
        graph.add_edge("summarize", "synthesize")
        graph.add_edge("synthesize", "evaluate")

        # Conditional edge from evaluate: loop if quality < threshold, under max loops, and no plateau
        def should_refine(state: DeepSummarizerState) -> str:
            quality_threshold = state["quality_threshold"]
            max_loops = state["max_refinement_loops"]
            refinement_rounds = state["refinement_rounds"]
            quality_score = state["quality_score"]
            score_history = state["score_history"]

            # Budget may further restrict refinement loops (set by AgentExecutor)
            budget = self._budget_config()
            if budget and budget.max_refinement_loops is not None:
                max_loops = min(max_loops, budget.max_refinement_loops)

            plateau = len(score_history) >= 2 and (score_history[-1] - score_history[-2]) < 0.02
            at_max = refinement_rounds >= max_loops

            if quality_score >= quality_threshold or at_max or plateau:
                return "format"
            return "refine"

        graph.add_conditional_edges("evaluate", should_refine)
        graph.add_edge("refine", "evaluate")  # Loop back to evaluate
        graph.add_edge("format", END)

        return graph.compile()

    def _has_loops(self) -> bool:
        """Override to indicate this agent has refinement loops."""
        return True

    # -- Helper Methods ─────────────────────────────────────────────────────────

    @contextmanager
    def _step_span(self, step_name: str) -> Generator[Span | None, None, None]:
        """Create a tracer span for a workflow step (if tracer is available via context)."""
        ctx = getattr(self._run_local, "context", None)
        tracer = ctx.metadata.get("tracer") if ctx else None
        if tracer and ctx is not None:
            run_id = cast(str, ctx.run_id)
            with tracer.span(f"step.{step_name}", run_id) as span:
                yield span
        else:
            yield None

    def _budget_config(self) -> BudgetConfig | None:
        """Access budget config from context metadata (set by AgentExecutor)."""
        ctx = getattr(self._run_local, "context", None)
        return ctx.metadata.get("budget") if ctx else None

    # -- Step Methods ───────────────────────────────────────────────────────────

    def _step_analyze(self, state: DeepSummarizerState) -> dict[str, Any]:
        """Code: Text structure analysis — determine chunk size and document type.

        No LLM call. Delegates to :func:`~dopeagents.agents._summarizer.analyzer.analyze_text`.
        """
        with self._step_span("analyze") as span:
            analysis: TextAnalysis = analyze_text(state["text"])
            if span:
                span.set_attribute("text_type", analysis.text_type)
                span.set_attribute("complexity", analysis.complexity)
                span.set_attribute("paragraph_count", analysis.paragraph_count)
            return {
                "analysis": {
                    "recommended_chunk_size": analysis.recommended_chunk_size,
                    "text_type": analysis.text_type,
                    "complexity": analysis.complexity,
                    "paragraph_count": analysis.paragraph_count,
                }
            }

    def _step_chunk(self, state: DeepSummarizerState) -> dict[str, Any]:
        """Code: Split document into cost-bounded chunks.

        No LLM call. Delegates to :class:`~dopeagents.agents._summarizer.chunker.SummarizerChunker`.
        """
        with self._step_span("chunk") as span:
            target_size = state["analysis"].get("recommended_chunk_size", 512)
            chunks = SummarizerChunker().chunk(state["text"], target_size)
            if span:
                span.set_attribute("chunk_count", len(chunks))
                span.set_attribute("target_size", target_size)
            return {"chunks": chunks}

    def _step_summarize(self, state: DeepSummarizerState) -> dict[str, Any]:
        """Summarize each chunk independently.

        Uses sequential processing — _extract thread safety is unverified,
        and rate limits make parallel calls unreliable with most providers.

        Args:
            state: Current LangGraph state with 'chunks' list

        Returns:
            State dict with 'chunk_summaries' list
        """
        with self._step_span("summarize") as span:
            chunks = state["chunks"]
            summaries: list[str] = []
            fallback_count = 0
            logger.info(f"[summarize] Summarizing {len(chunks)} chunk(s)")

            for i, chunk in enumerate(chunks):
                logger.debug(f"[summarize] chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
                try:
                    out = cast(
                        ChunkSummary,
                        self._extract(
                            response_model=ChunkSummary,
                            messages=[
                                {"role": "system", "content": self.step_prompts["summarize"]},
                                {
                                    "role": "user",
                                    "content": f"Chunk {i + 1} of {len(chunks)}:\n\n{chunk}",
                                },
                            ],
                            model=self._model_for_step("summarize"),
                        ),
                    )
                    summaries.append(out.summary)
                except ExtractionProviderError:
                    raise  # Auth/billing/quota — let executor handle retry or fallback
                except Exception as exc:
                    logger.warning(
                        f"[summarize] chunk {i + 1}/{len(chunks)} failed "
                        f"({type(exc).__name__}: {exc}); using truncated source as fallback"
                    )
                    summaries.append(chunk[:500])
                    fallback_count += 1

            if span:
                span.set_attribute("chunk_count", len(chunks))
                span.set_attribute("fallback_count", fallback_count)
            return {"chunk_summaries": summaries}

    def _step_synthesize(self, state: DeepSummarizerState) -> dict[str, Any]:
        """Combine chunk summaries into a coherent narrative.

        Merges overlapping summaries and elimates duplicates.
        Respects focus area if provided.
        Extracts 3-5 key points.

        Args:
            state: Current LangGraph state with 'chunk_summaries'

        Returns:
            State dict with 'synthesis' and 'key_points'
        """
        with self._step_span("synthesize") as span:
            logger.info(f"[synthesize] Combining {len(state['chunk_summaries'])} chunk summaries")
            summaries_text = "\n".join(state["chunk_summaries"])
            focus = state.get("focus")
            focus_instruction = (
                f"\n\nFOCUS AREA: Prioritize information related to '{focus}'." if focus else ""
            )
            out = cast(
                SynthesizeOut,
                self._extract(
                    response_model=SynthesizeOut,
                    messages=[
                        {"role": "system", "content": self.step_prompts["synthesize"]},
                        {"role": "user", "content": summaries_text + focus_instruction},
                    ],
                    model=self._model_for_step("synthesize"),
                ),
            )
            if span:
                span.set_attribute("key_points_count", len(out.key_points))
            return {"synthesis": out.synthesis, "key_points": out.key_points}

    def _step_evaluate(self, state: DeepSummarizerState) -> dict[str, Any]:
        """Score the synthesis on three independent criteria.

        Criteria:
        - Faithfulness: are all claims grounded in source?
        - Completeness: are all key points covered?
        - Coherence: is it well-structured and readable?

        Penalizes hallucinations and identifies weakest criterion for
        targeted refinement feedback.

        Args:
            state: Current LangGraph state

        Returns:
            State dict with quality_score, feedback, and score_history
        """
        with self._step_span("evaluate") as span:
            refinement_round = state["refinement_rounds"]
            logger.info(f"[evaluate] Scoring synthesis (round {refinement_round})")
            current_synthesis = state["synthesis"]
            # Sample start/middle/end so faithfulness scoring covers the full document
            source_excerpt = _sample_source(state["text"])
            focus = state.get("focus")
            focus_criterion = (
                f"\n- FOCUS ADHERENCE: does it emphasize '{focus}' as requested?" if focus else ""
            )
            out = cast(
                EvaluateOut,
                self._extract(
                    response_model=EvaluateOut,
                    messages=[
                        {
                            "role": "system",
                            "content": self.step_prompts["evaluate"] + focus_criterion,
                        },
                        {
                            "role": "user",
                            "content": (
                                f"ORIGINAL TEXT:\n{source_excerpt}\n\n"
                                f"---\n\n"
                                f"SUMMARY TO EVALUATE:\n{current_synthesis}"
                            ),
                        },
                    ],
                    model=self._model_for_step("evaluate"),
                ),
            )

            score = out.quality_score

            # Penalize if hallucinations detected; scale by claim count (each claim -0.05, floor 0.3)
            if out.unsupported_claims:
                n = len(out.unsupported_claims)
                penalty_ceiling = max(0.3, 0.75 - 0.05 * n)
                score = min(score, penalty_ceiling)
                logger.warning(
                    f"[evaluate] {n} unsupported claim(s) detected; "
                    f"capping score at {penalty_ceiling:.2f}"
                )
                feedback = (
                    f"UNSUPPORTED CLAIMS DETECTED — remove or correct these: "
                    f"{out.unsupported_claims}. {out.feedback}"
                )
            else:
                feedback = out.feedback

            # Identify weakest criterion for targeted refinement
            weakest = min(
                [
                    ("faithfulness", out.faithfulness_score),
                    ("completeness", out.completeness_score),
                    ("coherence", out.coherence_score),
                ],
                key=lambda x: x[1],
            )
            targeted_feedback = (
                f"PRIORITY: Improve {weakest[0]} (scored {weakest[1]:.2f}). {feedback}"
            )
            logger.info(
                f"[evaluate] quality_score={score:.3f} "
                f"(faithfulness={out.faithfulness_score:.2f}, "
                f"completeness={out.completeness_score:.2f}, "
                f"coherence={out.coherence_score:.2f})"
            )

            # Update score history for plateau detection
            score_history = [*state.get("score_history", []), score]

            if span:
                span.set_attribute("quality_score", score)
                span.set_attribute("faithfulness", out.faithfulness_score)
                span.set_attribute("completeness", out.completeness_score)
                span.set_attribute("coherence", out.coherence_score)
                span.set_attribute("refinement_round", refinement_round)

            return {
                "quality_score": score,
                "feedback": targeted_feedback,
                "score_history": score_history,
            }

    def _step_refine(self, state: DeepSummarizerState) -> dict[str, Any]:
        """Improve synthesis using targeted feedback from evaluation.

        Provides original text for grounding to prevent hallucinations.

        Args:
            state: Current LangGraph state

        Returns:
            State dict with updated 'synthesis' and refinement_rounds
        """
        with self._step_span("refine") as span:
            refinement_round = state["refinement_rounds"] + 1
            logger.info(f"[refine] Refining synthesis (round {refinement_round})")
            current_synthesis = state["synthesis"]
            feedback = state["feedback"]
            # Sample start/middle/end to ground refinement across the full document
            source_excerpt = _sample_source(state["text"])
            out = cast(
                RefineOut,
                self._extract(
                    response_model=RefineOut,
                    messages=[
                        {"role": "system", "content": self.step_prompts["refine"]},
                        {
                            "role": "user",
                            "content": (
                                f"ORIGINAL TEXT (for reference — ground all claims here):\n"
                                f"{source_excerpt}\n\n"
                                f"---\n\n"
                                f"CURRENT SUMMARY:\n{current_synthesis}\n\n"
                                f"FEEDBACK TO ADDRESS:\n{feedback}"
                            ),
                        },
                    ],
                    model=self._model_for_step("refine"),
                ),
            )

            if span:
                span.set_attribute("refinement_round", refinement_round)
            return {
                "synthesis": out.refined,
                "refinement_rounds": refinement_round,
            }

    def _step_format(self, state: DeepSummarizerState) -> dict[str, Any]:
        """Code: Apply output style and length truncation. No LLM call.

        Delegates to :class:`~dopeagents.agents._summarizer.formatter.SummaryFormatter`.
        """
        with self._step_span("format") as span:
            result = SummaryFormatter().format(
                synthesis=state["synthesis"],
                style=state["style"],
                max_length=state["max_length"],
                key_points=state.get("key_points") or [],
            )
            if span:
                span.set_attribute("word_count", result.word_count)
                span.set_attribute("truncated", result.truncated)
                span.set_attribute("style", state["style"])
            return {
                "final_summary": result.text,
                "word_count": result.word_count,
                "truncated": result.truncated,
            }

    # -- Public Interface ───────────────────────────────────────────────────────

    def run(
        self, input_data: DeepSummarizerInput, context: AgentContext | None = None
    ) -> AgentResult[DeepSummarizerOutput]:
        """Execute the 7-step DeepSummarizer workflow.

        Args:
            input_data: DeepSummarizerInput with text, max_length, style, focus
            context: Optional AgentContext (managed by executor)

        Returns:
            AgentResult[DeepSummarizerOutput] with summary, quality_score, metrics
        """
        context = context or AgentContext()
        self._run_local.context = context
        logger.info(
            f"[DeepSummarizer] run start — "
            f"text_len={len(input_data.text)}, style={input_data.style}, "
            f"max_length={input_data.max_length}, "
            f"quality_threshold={input_data.quality_threshold}"
        )
        graph = self._get_graph()

        initial_state: DeepSummarizerState = {
            "text": input_data.text,
            "max_length": input_data.max_length,
            "style": input_data.style,
            "focus": input_data.focus,
            "analysis": {},
            "chunks": [],
            "chunk_summaries": [],
            "synthesis": "",
            "quality_score": 0.0,
            "feedback": "",
            "refinement_rounds": 0,
            "max_refinement_loops": input_data.max_refinement_loops,
            "quality_threshold": input_data.quality_threshold,
            "score_history": [],
            "final_summary": "",
            "word_count": 0,
            "truncated": False,
            "key_points": [],
        }

        try:
            final_state = graph.invoke(initial_state)
        except ExtractionProviderError:
            raise  # Auth/billing/quota — let executor handle via retry or fallback
        except Exception as exc:
            error_msg = str(exc)
            cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
            if cause is not None:
                error_msg = str(cause)
            logger.error(f"[DeepSummarizer] run failed: {error_msg}")
            return AgentResult[DeepSummarizerOutput](output=None, success=False, error=error_msg)

        logger.info(
            f"[DeepSummarizer] run complete — "
            f"quality_score={final_state['quality_score']:.3f}, "
            f"refinement_rounds={final_state['refinement_rounds']}, "
            f"chunks={len(final_state['chunks'])}"
        )
        output = DeepSummarizerOutput(
            summary=final_state["final_summary"],
            key_points=final_state["key_points"],
            quality_score=final_state["quality_score"],
            refinement_rounds=final_state["refinement_rounds"],
            chunks_processed=len(final_state["chunks"]),
            word_count=final_state["word_count"],
            truncated=final_state["truncated"],
        )

        return AgentResult(output=output)

    def _render_prompt(self, input_data: DeepSummarizerInput) -> str | None:
        """Return a readable overview of all step prompts for debugging and introspection.

        Used by agent.debug(input) to show the active prompts and run configuration
        for the given input, including focus area and quality settings.

        Args:
            input_data: Agent input — focus area and settings are shown alongside prompts.

        Returns:
            Formatted string with active configuration and all step prompts.
        """
        lines = [
            "DeepSummarizer 7-Step Workflow Prompts:\n",
            f"style={input_data.style!r}  max_length={input_data.max_length}  "
            f"quality_threshold={input_data.quality_threshold}  "
            f"max_refinement_loops={input_data.max_refinement_loops}",
        ]
        if input_data.focus:
            lines.append(
                f"focus={input_data.focus!r}  \u2190 injected into synthesize and evaluate steps"
            )
        for step_name, prompt in self.step_prompts.items():
            lines.append(f"\n--- {step_name.upper()} ---")
            lines.append(prompt)
        return "\n".join(lines)

    def _get_model_config(self) -> dict[str, Any] | None:
        """Return model configuration for debugging and introspection.

        Shows which models are used for each step and key limits.
        Used by agent.describe() and observability tooling.

        Returns:
            Dict with default_model, step_models, and key hyperparameters
        """
        return {
            "default_model": self._model,
            "step_models": self._step_models,
            "max_refinement_loops": DeepSummarizerInput.model_fields[
                "max_refinement_loops"
            ].default,
            "max_chunks": 25,
        }
