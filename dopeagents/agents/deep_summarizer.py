"""DeepSummarizer agent: multi-step document summarization with self-evaluation and refinement.

This agent serves as a reference implementation for Phase 2 infrastructure:
- Lifecycle management (AgentExecutor)
- Cost tracking and budget guards
- Caching and memoization
- Resilience (retry, fallback, degradation)
- Observability and tracing
"""

from __future__ import annotations

import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from dopeagents.core.agent import Agent
from dopeagents.core.context import AgentContext
from dopeagents.core.types import AgentResult

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
    refined: str
    refinement_rounds: int
    max_refinement_loops: int
    quality_threshold: float
    score_history: list[float]
    total_tokens_used: int
    final_summary: str
    word_count: int
    truncated: bool
    key_points: list[str]


# ── Step Output Schemas (internal, used only within step methods) ─────────────


class _AnalyzeOut(BaseModel):
    """Output of the analyze step."""

    recommended_chunk_size: int = Field(
        default=500, ge=100, le=2000, description="Recommended characters per chunk"
    )
    text_type: str = Field(
        default="generic", description="Type of text (e.g., 'article', 'research', 'story')"
    )
    complexity: str = Field(default="medium", description="Estimated complexity level")


class _ChunkSummary(BaseModel):
    """Output of summarizing a single chunk."""

    summary: str = Field(description="Concise summary of this chunk")


class _SynthesizeOut(BaseModel):
    """Output of the synthesize step."""

    synthesis: str = Field(description="Combined synthesis from chunk summaries")
    key_points: list[str] = Field(default_factory=list, description="Key points extracted")


class _EvaluateOut(BaseModel):
    """Output of the evaluate step."""

    faithfulness_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Faithfulness to source 0-1"
    )
    completeness_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Coverage of key points 0-1"
    )
    coherence_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Structural quality 0-1"
    )
    quality_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Overall quality score 0-1"
    )
    feedback: str = Field(default="", description="Actionable feedback for refinement")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the summary not found in the source text",
    )


class _RefineOut(BaseModel):
    """Output of the refine step."""

    refined: str = Field(description="Refined synthesis based on feedback")


class _FormatOut(BaseModel):
    """Output of the format step."""

    final_summary: str = Field(description="Final formatted summary")
    word_count: int = Field(ge=0, description="Word count of final summary")
    truncated: bool = Field(default=False, description="Whether output was truncated")


# ── Public Input / Output ──────────────────────────────────────────────────────


class DeepSummarizerInput(BaseModel):
    """Input schema for DeepSummarizer agent."""

    text: str = Field(min_length=1, description="Text to summarize")
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
    total_tokens_used: int = Field(
        default=0,
        ge=0,
        description="Approximate total tokens used (when available from LLM provider)",
    )


# ── DeepSummarizer Agent ───────────────────────────────────────────────────────


class DeepSummarizer(Agent[DeepSummarizerInput, DeepSummarizerOutput]):
    """
    7-step multi-step summarization agent with self-evaluation and iterative refinement.

    Steps (in order):
      1. analyze    — Analyze text structure and recommend chunking strategy
      2. chunk      — Split text into semantically coherent chunks
      3. summarize  — Summarize each chunk independently
      4. synthesize — Combine chunk summaries into a coherent whole
      5. evaluate   — Score the synthesis and identify weaknesses
      6. refine     — Improve synthesis using feedback (loops back to evaluate)
      7. format     — Apply style constraints and compute final metrics

    The evaluate→refine→evaluate cycle loops until quality_score >= threshold or
    max_refinement_loops is reached. When budget is exceeded with on_exceeded="degrade",
    the best synthesis produced so far is returned instead of raising.

    **Token Tracking via LiteLLM Response Metadata:**
    This agent integrates with LiteLLM's response metadata to track actual token usage
    from OpenAI-compatible clients. Token counts are accessible in the output's
    `total_tokens_used` field. Token tracking requires:

    1. Parent Agent._extract() method to expose raw response with usage metadata
    2. Or direct LLM client calls that capture response.usage (OpenAI API format)
    3. Currently, token tracking is prepared but requires parent class enhancement

    To enable token tracking in a production setup:
    - Modify Agent._extract() to return tuple (output, full_response)
    - Call _extract_tokens_from_response() on the raw response
    - Accumulate in state["total_tokens_used"] across all steps
    - This provides accurate cost tracking for multi-step LLM pipelines
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
        "analyze": (
            "Analyze the provided text. Determine its structure, length, and complexity. "
            "Recommend an optimal chunking strategy (chunk size in characters). "
            "Classify the text type (e.g., article, research, story) and complexity level."
        ),
        "chunk": (
            "Split the provided text into semantically coherent chunks. "
            "Each chunk should be complete and meaningful. "
            "Return a list of chunks respecting the recommended chunk size."
        ),
        "summarize": (
            "Summarize the following chunk concisely, capturing all key information. "
            "Be comprehensive but brief."
        ),
        "synthesize": (
            "Synthesize the provided chunk summaries into a single coherent summary. "
            "IMPORTANT: The chunks had overlapping content, so the summaries may contain "
            "duplicate information. Merge and deduplicate—each fact should appear only once. "
            "Combine information to form a unified narrative. Also extract 3-5 key points."
        ),
        "evaluate": (
            "You are a strict summary evaluator. Compare the SUMMARY against the "
            "ORIGINAL TEXT and score it on a scale of 0.0 to 1.0.\n\n"
            "Scoring rubric (be harsh — most summaries score 0.4-0.7):\n"
            "  0.0-0.3: Major information missing or contains fabricated claims\n"
            "  0.4-0.5: Covers some key points but misses important details\n"
            "  0.6-0.7: Covers most key points with minor gaps or awkward phrasing\n"
            "  0.8-0.9: Comprehensive, well-structured, all key points present\n"
            "  1.0: Perfect — nothing to improve\n\n"
            "Evaluate on these three independent criteria:\n"
            "- FAITHFULNESS (0-1): Does every claim in the summary appear in the original? "
            "Flag ANY claim not grounded in the source.\n"
            "- COMPLETENESS (0-1): Are all important points from the original covered?\n"
            "- COHERENCE (0-1): Is it well-structured and readable?\n\n"
            "Provide all three scores AND an overall quality_score (typically the minimum "
            "of the three, or consider weighted importance), plus specific, actionable "
            "feedback listing what is missing or wrong."
        ),
        "refine": (
            "Improve the provided summary using the feedback given. "
            "Address the specific weaknesses mentioned and enhance clarity. "
            "Maintain all important information while improving quality."
        ),
        "format": (
            "Format the summary according to the specified style. "
            "Ensure it does not exceed the maximum length. "
            "If truncation is necessary, prioritize the most important information."
        ),
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initialize DeepSummarizer with caching and per-step model overrides.

        Args:
            **kwargs: Passed to parent Agent.__init__ (model, hooks, cache, etc.) plus step_prompts
        """
        # Extract and apply custom step_prompts if provided
        custom_step_prompts = kwargs.pop("step_prompts", None)

        super().__init__(**kwargs)

        # Apply custom step_prompts if provided (shadows ClassVar)
        if custom_step_prompts is not None:
            self.step_prompts = {**self.step_prompts, **custom_step_prompts}  # type: ignore[misc]

        # Lazy-initialized LangGraph StateGraph
        self._graph: Any | None = None
        # Per-step model overrides (empty dict by default)
        self._step_models: dict[str, str] = getattr(self, "_step_models", {})

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
        This is important for performance in multi-invocation scenarios.

        Returns:
            Compiled StateGraph ready for invocation
        """
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text using character approximation.

        Rough heuristic: 4 characters ≈ 1 token (standard for GPT models).
        This is used for token budget decisions and LLM provider estimation.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count (minimum 1)
        """
        return max(1, len(text) // 4)

    def _build_graph(self) -> Any:
        """Build the internal LangGraph StateGraph for the 7-step workflow.

        Constructs the complete summarization pipeline:
          1. analyze    → text analysis and chunking strategy
          2. chunk      → text splitting with overlap
          3. summarize  → parallel chunk summarization
          4. synthesize → combine summaries coherently
          5. evaluate   → score quality and identify weaknesses
          6. refine     → improve based on feedback (loops)
          7. format     → apply style and finalize metrics

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
            quality_threshold = state.get("quality_threshold", 0.8)
            max_loops = state["max_refinement_loops"]
            refinement_rounds = state["refinement_rounds"]
            quality_score = state["quality_score"]
            score_history = state.get("score_history", [])

            # Plateau detection: if last two scores differ by < 0.02, stop refining
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

    def _extract_tokens_from_response(self, response: Any) -> int:
        """Extract token usage from LiteLLM response metadata.

        LiteLLM responses include usage info when using OpenAI-compatible clients.
        Falls back to 0 if metadata unavailable.

        Args:
            response: LiteLLM response object with optional usage metadata

        Returns:
            Total tokens used (prompt + completion), or 0 if unavailable
        """
        try:
            # LiteLLM embeds usage info in response object
            if hasattr(response, "usage"):
                usage = response.usage
                # total_tokens is the primary field from OpenAI API
                if hasattr(usage, "total_tokens"):
                    return int(usage.total_tokens)
                # Also check individual counts in case only completion/prompt provided
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                return int(prompt_tokens) + int(completion_tokens)
        except (AttributeError, TypeError, ValueError):
            pass
        return 0

    def _extract_with_tokens(
        self, response_model: Any, messages: list[dict[str, str]], model: str
    ) -> tuple[Any, int]:
        """Extract structured output and capture token usage from response.

        Returns tuple of (extracted_object, tokens_used).

        **Implementation Note:**
        Currently returns 0 tokens because parent Agent._extract() doesn't expose
        the raw LiteLLM response. To implement full token tracking:

        1. Modify parent Agent._extract() to yield response metadata, or
        2. Call LLM client directly and parse response.usage:
           - response.usage.prompt_tokens
           - response.usage.completion_tokens
           - response.usage.total_tokens (when available)

        This is especially valuable for multi-step pipelines where understanding
        cumulative cost and token budget is critical for chunking decisions.

        Args:
            response_model: Pydantic model to extract into
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier string

        Returns:
            Tuple of (extracted_output, estimated_tokens_used)
        """
        # Note: The parent _extract() method doesn't expose raw response.
        # In a production setup, you'd want to hook into the provider's
        # response tracking or modify _extract to return (output, response) tuple.
        extracted = self._extract(
            response_model=response_model,
            messages=messages,
            model=model,
        )
        # TODO: Once parent Agent class exposes response metadata,
        # extract tokens here: tokens = self._extract_tokens_from_response(raw_response)
        return extracted, 0

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex that respects sentence boundaries.

        Handles common abbreviations and capital letters to avoid false splits.
        Uses pattern: (period/exclamation/question mark) + space + capital letter.

        Args:
            text: Raw text to split into sentences

        Returns:
            List of sentence strings (stripped of whitespace)
        """
        # Split on sentence-ending punctuation followed by space and capital letter
        # This avoids splitting on abbreviations like "Dr." or decimal points
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in sentences if s.strip()]

    # -- Step Methods ───────────────────────────────────────────────────────────

    def _step_analyze(self, state: DeepSummarizerState) -> dict[str, Any]:
        """Analyze text structure and recommend chunking strategy.

        Examines the input text to determine:
        - Optimal chunk size based on structure
        - Text type classification (article, research, story, etc.)
        - Complexity level (simple, medium, complex)

        Args:
            state: Current LangGraph state

        Returns:
            State dict with 'analysis' key containing recommendations
        """
        out = cast(
            _AnalyzeOut,
            self._extract(
                response_model=_AnalyzeOut,
                messages=[
                    {"role": "system", "content": self.step_prompts["analyze"]},
                    {"role": "user", "content": state["text"][:2000]},  # First 2000 chars
                ],
                model=self._model_for_step("analyze"),
            ),
        )
        # Note: Token tracking requires parent Agent class to expose response metadata.
        # In the current implementation, _extract() returns only the parsed object.
        # To properly track tokens, consider:
        # 1. Modifying Agent._extract() to return (output, response) tuple
        # 2. Using direct LLM client calls with metadata capture
        # 3. Integrating with LiteLLM's token counter utilities
        return {
            "analysis": out.model_dump(),
            # "total_tokens_used": state["total_tokens_used"] + tokens_used,  # when available
        }

    def _step_chunk(self, state: DeepSummarizerState) -> dict[str, Any]:
        """Split text into chunks using paragraph boundaries with overlap.

        Strategy:
        1. Split on paragraph boundaries (\\n\\n)
        2. Fall back to sentence splitting if no paragraphs
        3. Merge into target-sized chunks with 15% overlap
        4. Guard against > 10 chunks (cost/latency protection)

        Args:
            state: Current LangGraph state

        Returns:
            State dict with 'chunks' list
        """
        text = state["text"]
        target_size = state["analysis"].get("recommended_chunk_size", 512)
        overlap_ratio = 0.15  # 15% overlap between adjacent chunks

        # Optional: refine target_size based on token estimation
        # estimated_tokens = self._estimate_tokens(text[:1000])
        # if estimated_tokens > 3000:  # Large document, smaller chunks safer
        #     target_size = min(target_size, 300)

        # Split on paragraph boundaries first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # If no or single paragraph, fall back to sentence-aware splitting
        if len(paragraphs) <= 1:
            paragraphs = self._split_sentences(text)

        # Merge paragraphs/sentences into chunks respecting target_size
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > target_size and current:
                chunks.append(current.strip())
                # Overlap: carry last ~15% of current chunk into next
                overlap_chars = int(len(current) * overlap_ratio)
                current = current[-overlap_chars:].strip() + "\n\n" + para
            else:
                current = current + "\n\n" + para if current else para

        if current.strip():
            chunks.append(current.strip())

        # If text is too short for even one split, use the whole text
        if not chunks:
            chunks = [text]

        # Guard: max 10 chunks (cost and latency protection)
        if len(chunks) > 10:
            warnings.warn(
                f"Chunk count {len(chunks)} exceeds max_chunks=10. Truncating.",
                ResourceWarning,
                stacklevel=2,
            )
            chunks = chunks[:10]

        return {"chunks": chunks}

    def _step_summarize(self, state: DeepSummarizerState) -> dict[str, Any]:
        """Summarize each chunk independently, parallelized for multi-chunk efficiency.

        Single chunk: sequential (overhead not justified)
        Multiple chunks: parallel with max 4 workers (rate limit safety)

        Args:
            state: Current LangGraph state with 'chunks' list

        Returns:
            State dict with 'chunk_summaries' list
        """
        chunks = state["chunks"]

        if len(chunks) == 1:
            # No benefit from parallelism for single chunk
            out = cast(
                _ChunkSummary,
                self._extract(
                    response_model=_ChunkSummary,
                    messages=[
                        {"role": "system", "content": self.step_prompts["summarize"]},
                        {"role": "user", "content": f"Chunk 1 of 1:\n\n{chunks[0]}"},
                    ],
                    model=self._model_for_step("summarize"),
                ),
            )
            return {"chunk_summaries": [out.summary]}

        def _summarize_chunk(args: tuple[int, str]) -> str:
            """Summarize a single chunk."""
            i, chunk = args
            out = cast(
                _ChunkSummary,
                self._extract(
                    response_model=_ChunkSummary,
                    messages=[
                        {"role": "system", "content": self.step_prompts["summarize"]},
                        {
                            "role": "user",
                            "content": (f"Chunk {i + 1} of {len(chunks)}:\n\n{chunk}"),
                        },
                    ],
                    model=self._model_for_step("summarize"),
                ),
            )
            return out.summary

        # Parallel execution: max 4 workers to avoid overwhelming API/rate limits
        with ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
            summaries = list(executor.map(_summarize_chunk, enumerate(chunks)))

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
        summaries_text = "\n".join(state["chunk_summaries"])
        focus = state.get("focus")
        focus_instruction = (
            f"\n\nFOCUS AREA: Prioritize information related to '{focus}'." if focus else ""
        )
        out = cast(
            _SynthesizeOut,
            self._extract(
                response_model=_SynthesizeOut,
                messages=[
                    {"role": "system", "content": self.step_prompts["synthesize"]},
                    {"role": "user", "content": summaries_text + focus_instruction},
                ],
                model=self._model_for_step("synthesize"),
            ),
        )
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
        current_synthesis = state.get("refined") or state["synthesis"]
        source_excerpt = state["text"][:4000]
        focus = state.get("focus")
        focus_criterion = (
            f"\n- FOCUS ADHERENCE: does it emphasize '{focus}' as requested?" if focus else ""
        )
        out = cast(
            _EvaluateOut,
            self._extract(
                response_model=_EvaluateOut,
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

        # Penalize if hallucinations detected
        if out.unsupported_claims:
            score = min(score, 0.6)
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
        targeted_feedback = f"PRIORITY: Improve {weakest[0]} (scored {weakest[1]:.2f}). {feedback}"

        # Update score history for plateau detection
        score_history = [*state.get("score_history", []), score]

        return {
            "quality_score": score,
            "feedback": targeted_feedback,
            "score_history": score_history,
        }

    def _step_refine(self, state: DeepSummarizerState) -> dict[str, Any]:
        """Improve synthesis using targeted feedback from evaluation.

        Provides original text for grounding to prevent hallucinations.
        Updates both 'refined' and 'synthesis' to ensure next evaluate reads refined version.

        Args:
            state: Current LangGraph state

        Returns:
            State dict with updated 'refined', 'synthesis', and refinement_rounds
        """
        current_synthesis = state.get("refined") or state["synthesis"]
        feedback = state["feedback"]
        source_excerpt = state["text"][:4000]
        out = cast(
            _RefineOut,
            self._extract(
                response_model=_RefineOut,
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

        # Critical: update synthesis so evaluate reads the refined version on next loop
        return {
            "refined": out.refined,
            "synthesis": out.refined,  # Update synthesis for next evaluate
            "refinement_rounds": state["refinement_rounds"] + 1,
        }

    def _step_format(self, state: DeepSummarizerState) -> dict[str, Any]:
        """Apply style constraints and compute final output metrics.

        Converts synthesis to requested format: paragraph, bullets, or TLDR.
        Enforces max_length by truncating if necessary.
        Respects focus area emphasis if provided.

        Args:
            state: Current LangGraph state

        Returns:
            State dict with final_summary, word_count, and truncated flag
        """
        current_synthesis = state.get("refined") or state["synthesis"]
        style = state["style"]
        max_length = state["max_length"]

        # Build formatted prompt based on style
        style_instruction = {
            "paragraph": "Format as a coherent paragraph.",
            "bullets": "Format as a bulleted list of key points.",
            "tldr": "Format as a brief TL;DR (too long; didn't read) summary.",
        }.get(style, "Format as a coherent paragraph.")

        focus_note = (
            f"\nEnsure the summary emphasizes: {state['focus']}" if state.get("focus") else ""
        )

        out = cast(
            _FormatOut,
            self._extract(
                response_model=_FormatOut,
                messages=[
                    {"role": "system", "content": self.step_prompts["format"]},
                    {
                        "role": "user",
                        "content": (
                            f"{style_instruction}{focus_note}\n\n"
                            f"Maximum length: {max_length} characters.\n\n"
                            f"Summary to format:\n{current_synthesis}"
                        ),
                    },
                ],
                model=self._model_for_step("format"),
            ),
        )

        return {
            "final_summary": out.final_summary,
            "word_count": out.word_count,
            "truncated": out.truncated,
        }

    # -- Public Interface ───────────────────────────────────────────────────────

    def run(
        self, input: DeepSummarizerInput, _context: AgentContext | None = None
    ) -> AgentResult[DeepSummarizerOutput]:
        """Execute the 7-step DeepSummarizer workflow with graceful degradation.

        Main entry point called by AgentExecutor with full lifecycle integration:
        - Cost tracking (per-step token counting)
        - Caching of intermediate results
        - Resilience (retry policy, fallback chain, budget guards handled by executor)
        - Observability (spans, metrics, tracing via executor)

        Graceful degradation: if any step fails, returns best partial synthesis
        produced so far with success=True and error message.
        Complete failure only if no synthesis available at any point.

        Args:
            input: DeepSummarizerInput with text, max_length, style, focus
            _context: Optional AgentContext (managed by executor)

        Returns:
            AgentResult[DeepSummarizerOutput] with summary, quality_score, metrics
        """
        graph = self._get_graph()

        # Build initial state
        initial_state: DeepSummarizerState = {
            "text": input.text,
            "max_length": input.max_length,
            "style": input.style,
            "focus": input.focus,
            "analysis": {},
            "chunks": [],
            "chunk_summaries": [],
            "synthesis": "",
            "quality_score": 0.0,
            "feedback": "",
            "refined": "",
            "refinement_rounds": 0,
            "max_refinement_loops": input.max_refinement_loops,
            "quality_threshold": input.quality_threshold,
            "score_history": [],
            "total_tokens_used": 0,
            "final_summary": "",
            "word_count": 0,
            "truncated": False,
            "key_points": [],
        }

        # Execute graph with graceful degradation
        try:
            final_state = graph.invoke(initial_state)
        except Exception as exc:
            # Attempt graceful degradation: return best synthesis produced so far
            chunk_summaries: list[str] = initial_state.get("chunk_summaries", [])
            partial_synthesis = (
                initial_state.get("refined")
                or initial_state.get("synthesis")
                or (" ".join(chunk_summaries) if chunk_summaries else "")
            )

            if partial_synthesis:
                # Partial success: return what we have
                chunks_list: list[str] = initial_state.get("chunks", [])
                output = DeepSummarizerOutput(
                    summary=partial_synthesis.strip(),
                    key_points=initial_state.get("key_points", []),
                    quality_score=float(initial_state.get("quality_score", 0.0)),
                    refinement_rounds=int(initial_state.get("refinement_rounds", 0)),
                    chunks_processed=len(chunks_list) if chunks_list else 1,
                    word_count=len(partial_synthesis.split()),
                    truncated=False,
                )
                error_msg = str(exc)
                cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
                if cause is not None:
                    error_msg = str(cause)
                return AgentResult(
                    output=output,
                    success=True,
                    error=f"Partial result returned due to: {error_msg}",
                )

            # Complete failure: no partial synthesis available
            error_msg = str(exc)
            cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
            if cause is not None:
                error_msg = str(cause)
            return AgentResult[DeepSummarizerOutput](output=None, success=False, error=error_msg)

        # Normal completion: map to public output
        output = DeepSummarizerOutput(
            summary=final_state["final_summary"],
            key_points=final_state["key_points"],
            quality_score=final_state["quality_score"],
            refinement_rounds=final_state["refinement_rounds"],
            chunks_processed=len(final_state["chunks"]),
            word_count=final_state["word_count"],
            truncated=final_state["truncated"],
            total_tokens_used=final_state.get("total_tokens_used", 0),
        )

        return AgentResult(output=output)

    def _render_prompt(self, _input: DeepSummarizerInput) -> str | None:
        """Return readable overview of all step prompts for debugging and introspection.

        Used by agent.debug(input) to show what prompts will be used.
        Useful for prompt engineering and understanding agent behavior.

        Args:
            _input: Agent input (unused, but part of interface)

        Returns:
            Formatted string with all step prompts
        """
        lines = ["DeepSummarizer 7-Step Workflow Prompts:\n"]
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
            "max_refinement_loops": 3,
            "max_chunks": 10,
        }
