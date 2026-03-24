"""DeepResearcher: hybrid research workflow with coded pipeline + bounded LLM tool calling.

Hybrid 13-step research agent that uses real free APIs with optional
bounded tool calling in the analysis step:
- Load context from previous research sessions (code)
- Query expansion and strategy determination (LLM)
- Real search across Wikipedia, DuckDuckGo, Semantic Scholar, arXiv, CrossRef (code)
- Content extraction from source URLs (code)
- Credibility scoring via domain authority, citations, recency (code)
- Deep analysis: claim extraction + bounded tool calls for fact-check/search/citations (LLM+tools)
- Cross-referencing claims for agreement/contradiction (LLM)
- Evidence-based synthesis with real citations (LLM)
- Grounded confidence calculation from measurable signals (code)
- Quality evaluation and gap detection (LLM)
- Targeted gap-filling refinement (LLM + code)
- Structured report generation in multiple formats (LLM + code)
- Session persistence for follow-up queries (code)
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid

from collections.abc import Generator
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from dopeagents.agent_utils.chunking import RelevanceRanker, SemanticChunker
from dopeagents.agent_utils.content_extractor import ContentExtractor, ExtractedContent
from dopeagents.agent_utils.credibility import score_credibility
from dopeagents.agent_utils.fact_checker import FactChecker
from dopeagents.agent_utils.search_providers import SearchEngine, SearchResult
from dopeagents.agents._researcher.claim_analysis import CrossReferenceOutput
from dopeagents.agents._researcher.confidence import ConfidenceCalculator
from dopeagents.agents._researcher.hybrid_step import HybridStepResult, HybridStepRunner
from dopeagents.agents._researcher.memory import ResearchMemory, ResearchSession
from dopeagents.agents._researcher.model_capability import ModelCapability, detect_capability
from dopeagents.agents._researcher.progress import ResearchProgress
from dopeagents.agents._researcher.report_generator import ReportFormat, ReportGenerator
from dopeagents.agents._researcher.tools import ANALYSIS_TOOLS, ToolBudget, ToolExecutor
from dopeagents.core.agent import Agent
from dopeagents.core.context import AgentContext
from dopeagents.core.types import AgentResult
from dopeagents.errors import ExtractionProviderError
from dopeagents.observability.logging import get_logger

if TYPE_CHECKING:
    from dopeagents.cost.guard import BudgetConfig
    from dopeagents.observability.tracer import Span

logger = get_logger(__name__)


class _AsyncBridge:
    """Reusable async-to-sync bridge backed by a single persistent background event loop.

    Creates one event loop on a daemon thread and reuses it for all async calls,
    avoiding the overhead of spinning up a new loop per call and the fragility of
    asyncio.get_running_loop() which fires for any thread that has a loop, not just
    the calling coroutine's thread.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            with self._lock:
                if self._loop is None or self._loop.is_closed():
                    self._loop = asyncio.new_event_loop()
                    self._thread = threading.Thread(
                        target=self._loop.run_forever,
                        daemon=True,
                        name="deep-researcher-async",
                    )
                    self._thread.start()
        return self._loop

    def run(self, coro: Any) -> Any:
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    def shutdown(self) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)
            self._loop.close()


_async_bridge = _AsyncBridge()


def _run_async(coro: Any) -> Any:
    """Bridge async coroutines into sync LangGraph nodes via a persistent event loop."""
    return _async_bridge.run(coro)


# ── Internal State ─────────────────────────────────────────────────────────


class DeepResearcherState(TypedDict):
    """Internal LangGraph state for the 13-step research workflow."""

    # Input
    query: str
    research_focus: str | None
    quality_threshold: float
    max_refinement_loops: int

    # Step 1: Prior context from memory
    prior_context: dict[str, Any]
    known_sources: list[dict[str, Any]]

    # Step 2: Query expansion
    expanded_queries: list[str]
    search_strategy: str

    # Step 3: Real search results
    search_results: list[dict[str, Any]]

    # Step 4: Extracted content
    extracted_content: list[dict[str, Any]]

    # Step 5: Credibility scores
    credibility_scores: list[dict[str, Any]]

    # Step 6: Deep analysis - claim extraction
    claims: list[dict[str, Any]]
    verified_claims: list[dict[str, Any]]

    # Step 7: Cross-reference
    claim_clusters: list[dict[str, Any]]
    information_gaps: list[str]

    # Step 8: Synthesize
    synthesis: str
    key_findings: list[str]
    citations: list[dict[str, str]]

    # Step 9: Calculate confidence
    calculated_confidence: float
    confidence_breakdown: dict[str, Any]

    # Step 10: LLM evaluation
    quality_score: float
    evaluation_feedback: str
    previous_quality_score: float

    # Step 12: Report generation
    structured_report: dict[str, Any]
    markdown_report: str
    report_title: str

    # Step 13: Session persistence
    session_id: str

    # Refinement tracking
    refinement_rounds: int

    # Tool tracking (hybrid steps)
    tool_usage: dict[str, Any]
    tool_insights: list[str]
    additional_sources: list[dict[str, Any]]


# ── Step Output Schemas ────────────────────────────────────────────────────


class _ExpandQueryOut(BaseModel):
    """Output of query expansion step."""

    expanded_queries: list[str] = Field(max_length=5, description="3-5 refined search queries")
    search_strategy: str = Field(
        description="Recommended strategy: 'academic', 'general', 'news', or 'comprehensive'"
    )


class _SynthesizeOut(BaseModel):
    """Output of synthesis step."""

    synthesis: str = Field(description="Evidence-based synthesis")
    key_findings: list[str] = Field(description="Top 3-5 key findings")
    citations: list[dict[str, str]] = Field(
        default_factory=list,
        description="Citations: list of {claim, source_title, url}",
    )


class _EvaluateOut(BaseModel):
    """Output of evaluation step."""

    quality_score: float = Field(ge=0.0, le=1.0)
    coverage_score: float = Field(ge=0.0, le=1.0)
    credibility_score: float = Field(ge=0.0, le=1.0)
    coherence_score: float = Field(ge=0.0, le=1.0)
    feedback: str
    missing_aspects: list[str] = Field(default_factory=list)


class _GapAnalysisOut(BaseModel):
    """Output of gap analysis for refinement."""

    additional_queries: list[str] = Field(
        default_factory=list,
        description="New queries to fill information gaps",
    )


class _ClaimItem(BaseModel):
    """A single extracted claim with mandatory source attribution."""

    claim_text: str = Field(description="The specific factual claim")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_url: str = Field(default="", description="URL of the source this claim came from")
    source_title: str = Field(default="", description="Title of the source")
    supporting_quote: str = Field(
        default="", description="Direct quote from source supporting the claim"
    )


class _HybridAnalysisOut(BaseModel):
    """Output of the hybrid analysis step (claims + insights from tools)."""

    claims: list[_ClaimItem] = Field(
        default_factory=list,
        description="Extracted claims with mandatory source_url attribution",
    )
    tool_insights: list[str] = Field(
        default_factory=list,
        description="Key insights gained from tool usage (fact checks, additional sources)",
    )
    additional_sources_found: list[dict[str, str]] = Field(
        default_factory=list,
        description="New sources discovered via search_for_more tool",
    )
    verified_facts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Facts that were verified via fact_check tool",
    )
    confidence_notes: list[str] = Field(
        default_factory=list,
        description="Notes on confidence level based on verification results",
    )


# ── Public Input / Output ──────────────────────────────────────────────────


class DeepResearcherInput(BaseModel):
    """Input schema for DeepResearcher."""

    query: str = Field(min_length=5, description="Research query or topic")
    research_focus: str | None = Field(
        default=None,
        description="Focus: 'academic', 'practical', 'recent_news', 'comprehensive'",
    )
    quality_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum quality score to accept without refinement",
    )
    max_refinement_loops: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum number of refinement iterations",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.MARKDOWN,
        description="Output report format: markdown, html, json, executive_summary, academic",
    )
    enable_fact_check: bool = Field(
        default=True,
        description="Enable fact verification against Wikipedia/Wikidata",
    )
    enable_memory: bool = Field(
        default=True,
        description="Save session and check prior research for context",
    )
    enable_tool_calling: bool | None = Field(
        default=None,
        description=(
            "Enable LLM tool calling in analysis steps. "
            "None = auto-detect from model capability. "
            "True = force enable. False = force disable."
        ),
    )
    tool_budget: int = Field(
        default=5,
        ge=0,
        le=10,
        description="Max tool calls allowed per hybrid step",
    )


class DeepResearcherOutput(BaseModel):
    """Rich output with multiple report formats and grounded confidence."""

    # Core output
    synthesis: str = Field(description="Final research synthesis with citations")
    key_findings: list[str] = Field(default_factory=list, description="Top findings from research")

    # Structured report
    markdown_report: str = Field(default="", description="Full markdown report")
    structured_report: dict[str, Any] = Field(
        default_factory=dict, description="Structured report data"
    )
    report_title: str = Field(default="", description="Generated report title")

    # Confidence (grounded, not LLM self-eval)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Grounded confidence score")
    confidence_breakdown: dict[str, Any] = Field(
        default_factory=dict, description="Detailed confidence components"
    )
    llm_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM's self-evaluated quality (separate from grounded confidence)",
    )

    # Sources
    citations: list[dict[str, str]] = Field(default_factory=list)
    sources_analyzed: int = Field(default=0, ge=0, description="Number of sources analyzed")
    source_breakdown: dict[str, int] = Field(
        default_factory=dict, description="Count of sources by provider"
    )
    credibility_summary: dict[str, float] = Field(
        default_factory=dict, description="Average credibility scores"
    )

    # Analysis
    claim_clusters: list[dict[str, Any]] = Field(default_factory=list)
    verified_claims: list[dict[str, Any]] = Field(default_factory=list)
    information_gaps: list[str] = Field(default_factory=list)

    # Meta
    refinement_rounds: int = Field(default=0, ge=0, description="Number of refinement loops")
    session_id: str = Field(default="", description="Session ID for follow-up queries")
    total_duration_seconds: float = Field(default=0.0, description="Total workflow duration")

    # Tool usage reporting (hybrid)
    tool_usage: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool call budget usage summary",
    )
    tool_insights: list[str] = Field(
        default_factory=list,
        description="Insights gained from tool calls during analysis",
    )
    hybrid_mode: bool = Field(
        default=False,
        description="Whether tool calling was active",
    )
    model_tier: str = Field(
        default="unknown",
        description="Detected model capability tier: strong, medium, weak",
    )


# ── DeepResearcher ──────────────────────────────────────────────────────────


class DeepResearcher(Agent[DeepResearcherInput, DeepResearcherOutput]):
    """
    Hybrid 13-step research agent.

    Pipeline: CODE-CONTROLLED (fixed step order, guaranteed execution)
    Analysis: LLM WITH BOUNDED TOOLS (adaptive, but limited)

    The LLM can use tools ONLY in step 6 (deep_analysis). All other steps
    remain purely coded or purely LLM-driven without tool access.

    Steps:
      1.  load_context         — Code: Check memory for prior research
      2.  expand_query         — LLM: Expand query + determine strategy
      3.  real_search          — Code: Search 5 providers
      4.  extract_content      — Code: Fetch readable text
      5.  score_sources        — Code: Credibility scoring
      6.  deep_analysis        — LLM+TOOLS: Claim extraction with bounded tool access
      7.  cross_reference      — LLM: Find agreements/contradictions/gaps
      8.  synthesize           — LLM: Evidence-based synthesis
      9.  calculate_confidence — Code: Grounded confidence metrics
      10. evaluate             — LLM: Quality evaluation
      11. refine               — LLM: Gap-filling queries (loops to step 3)
      12. generate_report      — LLM+Code: Structured report
      13. save_session         — Code: Session persistence
    """

    name: ClassVar[str] = "DeepResearcher"
    version: ClassVar[str] = "3.1.0"
    description: ClassVar[str] = (
        "Hybrid research agent: coded pipeline with bounded LLM tool calling "
        "in analysis steps. Real search, fact verification, grounded confidence."
    )
    capabilities: ClassVar[list[str]] = [
        "research",
        "real-search",
        "content-extraction",
        "claim-analysis",
        "fact-verification",
        "credibility-scoring",
        "grounded-confidence",
        "structured-reports",
        "research-memory",
        "synthesis",
        "hybrid-tool-calling",
    ]
    tags: ClassVar[list[str]] = ["research", "multi-step", "llm-based", "tool-augmented", "hybrid"]
    requires_llm: ClassVar[bool] = True

    system_prompt: ClassVar[str] = (
        "You are an expert research analyst. You have access to real search results "
        "and extracted content from actual web sources. Your job is to analyze this "
        "real data critically, extract verifiable claims, and synthesize evidence-based "
        "findings with proper citations. Never fabricate sources or URLs."
    )

    step_prompts: ClassVar[dict[str, str]] = {
        "expand_query": (
            "Given a research query, generate 3-5 refined search queries that will find "
            "diverse, relevant sources. Consider:\n"
            "- Different phrasings and synonyms\n"
            "- Specific subtopics within the query\n"
            "- Academic vs practical angles\n"
            "Also determine the best search strategy: 'academic' (papers/journals), "
            "'general' (web + encyclopedias), 'news' (recent events), or "
            "'comprehensive' (all sources)."
        ),
        "deep_analysis": (
            "Analyze the research sources provided below. Your task:\n\n"
            "1. EXTRACT CLAIMS: Identify specific factual claims relevant to the research query.\n"
            "   For each claim: state it precisely, rate confidence (0-1), include the source URL,\n"
            "   source title, and a direct supporting quote from the text.\n\n"
            "2. USE TOOLS STRATEGICALLY:\n"
            "   - Use 'fact_check' for surprising or critical claims that could change conclusions\n"
            "   - Use 'search_for_more' ONLY if you identify a clear gap no existing source covers\n"
            "   - Use 'lookup_citation' for papers you want to cite with full details\n"
            "   - Use 'get_full_text' if a snippet looks highly relevant but is too short\n"
            "   - Use 'compare_claims' if two sources seem to contradict each other\n\n"
            "3. DO NOT use tools for:\n"
            "   - Obvious or well-established facts\n"
            "   - Information already well-covered by existing sources\n"
            "   - Tangential topics not central to the research query\n\n"
            "Focus on quality over quantity. A few well-verified critical claims "
            "are more valuable than many unverified ones."
        ),
        "cross_reference": (
            "Analyze the claims extracted from multiple sources. Group related claims "
            "and identify:\n"
            "- CONSENSUS: Claims supported by multiple sources\n"
            "- CONTRADICTIONS: Claims where sources disagree\n"
            "- UNIQUE FINDINGS: Claims from only one source\n"
            "- GAPS: Important aspects not covered by any source\n\n"
            "For each cluster, note the level of agreement and how to handle it in synthesis."
        ),
        "synthesize": (
            "Create an evidence-based synthesis using the analyzed claims and cross-references. "
            "Requirements:\n"
            "- Write at minimum 3 substantial paragraphs (aim for 500+ words)\n"
            "- Stay focused on the research query — do not drift to tangential topics\n"
            "- Cite specific sources by their title and URL\n"
            "- Clearly mark consensus vs controversial findings\n"
            "- Address identified information gaps honestly\n"
            "- Structure with clear sections and key findings\n"
            "- Never fabricate information — only use what's in the provided sources\n\n"
            "Format citations as [Source Title](URL) inline."
        ),
        "evaluate": (
            "Evaluate the synthesis rigorously:\n"
            "- COVERAGE (0-1): Does it address all aspects of the query?\n"
            "- CREDIBILITY (0-1): Are claims well-supported by credible sources?\n"
            "- COHERENCE (0-1): Is the narrative logical and well-structured?\n"
            "- Overall quality (0-1): Weighted average\n\n"
            "List specific missing aspects that additional research could address."
        ),
        "gap_analysis": (
            "Based on evaluation feedback, generate 1-3 targeted search queries to fill "
            "the identified information gaps. Focus on the weakest aspects."
        ),
    }

    def __init__(self, **kwargs: Any) -> None:
        progress_callback = kwargs.pop("progress_callback", None)
        google_factcheck_api_key = kwargs.pop("google_factcheck_api_key", None)
        memory_dir = kwargs.pop("memory_dir", ".research_memory")
        custom_step_prompts = kwargs.pop("step_prompts", None)

        # Hybrid config
        self._force_tool_calling: bool | None = kwargs.pop("enable_tool_calling", None)
        self._tool_budget_max: int = kwargs.pop("tool_budget", 5)

        super().__init__(**kwargs)
        if custom_step_prompts:
            self.step_prompts = {**self.step_prompts, **custom_step_prompts}  # type: ignore[misc]

        self._search_engine: SearchEngine | None = None
        self._content_extractor: ContentExtractor | None = None
        self._graph: Any | None = None

        # Thread-safety locks for double-checked lazy init
        self._search_engine_lock = threading.Lock()
        self._content_extractor_lock = threading.Lock()
        self._fact_checker_lock = threading.Lock()
        self._report_generator_lock = threading.Lock()
        self._tool_executor_lock = threading.Lock()

        # V3 components
        self._report_generator: ReportGenerator | None = None
        self._fact_checker: FactChecker | None = None
        self._chunker = SemanticChunker()
        self._ranker = RelevanceRanker()
        self._chunk_cache: dict[str, list[Any]] = {}
        self._confidence_calc = ConfidenceCalculator()
        self._memory = ResearchMemory(storage_dir=memory_dir)
        self._progress: ResearchProgress | None = None
        self._google_factcheck_api_key = google_factcheck_api_key
        self._run_local = threading.local()

        if progress_callback:
            self._progress = ResearchProgress(callback=progress_callback)

        # Hybrid mode: detect model capability and configure
        self._model_capability: ModelCapability = detect_capability(self._model)
        self._tool_executor_instance: ToolExecutor | None = None
        self._hybrid_runner: HybridStepRunner | None = None

    # ── Lifecycle / Observability helpers (mirrors DeepSummarizer) ────

    def _model_for_step(self, step_name: str) -> str:
        """Get the model for a specific step, respecting per-step overrides."""
        return self._step_models.get(step_name, self._model)

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

    @property
    def _tools_enabled(self) -> bool:
        """Whether tool calling is active for this run (per-run overrides instance default)."""
        per_run = cast(bool | None, getattr(self._run_local, "force_tool_calling", None))
        effective = per_run if per_run is not None else self._force_tool_calling
        if effective is not None:
            return effective
        return self._model_capability.supports_tool_calling

    def _get_search_engine(self) -> SearchEngine:
        if self._search_engine is None:
            with self._search_engine_lock:
                if self._search_engine is None:
                    self._search_engine = SearchEngine()
        return self._search_engine

    def _get_content_extractor(self) -> ContentExtractor:
        if self._content_extractor is None:
            with self._content_extractor_lock:
                if self._content_extractor is None:
                    self._content_extractor = ContentExtractor()
        return self._content_extractor

    def _get_fact_checker(self) -> FactChecker:
        if self._fact_checker is None:
            with self._fact_checker_lock:
                if self._fact_checker is None:
                    self._fact_checker = FactChecker(google_api_key=self._google_factcheck_api_key)
        return self._fact_checker

    def _get_report_generator(self) -> ReportGenerator:
        if self._report_generator is None:
            with self._report_generator_lock:
                if self._report_generator is None:
                    self._report_generator = ReportGenerator(extract_fn=self._extract)
        return self._report_generator

    def _get_tool_executor(self) -> ToolExecutor:
        if self._tool_executor_instance is None:
            with self._tool_executor_lock:
                if self._tool_executor_instance is None:
                    self._tool_executor_instance = ToolExecutor(
                        search_engine=self._get_search_engine(),
                        content_extractor=self._get_content_extractor(),
                        fact_checker=self._get_fact_checker(),
                    )
        return self._tool_executor_instance

    def _get_hybrid_runner(self, budget_override: int | None = None) -> HybridStepRunner:
        budget_value = getattr(self._run_local, "tool_budget", self._tool_budget_max)
        max_calls = budget_override if budget_override is not None else cast(int, budget_value)

        # Reduce tool count for medium-tier models
        if self._model_capability.tier == "medium":
            max_calls = min(max_calls, 3)

        budget = ToolBudget(max_calls=max_calls)

        return HybridStepRunner(
            extract_fn=self._extract,
            chat_fn=getattr(self, "_chat_completion", None),
            tool_executor=self._get_tool_executor() if self._tools_enabled else None,
            budget=budget,
            tools_enabled=self._tools_enabled,
        )

    def _get_graph(self) -> Any:
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self) -> Any:
        """Build research workflow graph.

        Steps 7+8 (extract_claims + verify_claims) are merged into
        a single 'deep_analysis' hybrid step.
        """
        graph = StateGraph(DeepResearcherState)

        graph.add_node("load_context", self._step_load_context)
        graph.add_node("expand_query", self._step_expand_query)
        graph.add_node("real_search", self._step_real_search)
        graph.add_node("extract_content", self._step_extract_content)
        graph.add_node("score_sources", self._step_score_sources)
        graph.add_node("deep_analysis", self._step_deep_analysis)
        graph.add_node("cross_reference", self._step_cross_reference)
        graph.add_node("synthesize", self._step_synthesize)
        graph.add_node("calculate_confidence", self._step_calculate_confidence)
        graph.add_node("evaluate", self._step_evaluate)
        graph.add_node("generate_report", self._step_generate_report)
        graph.add_node("save_session", self._step_save_session)
        graph.add_node("refine", self._step_refine)

        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "expand_query")
        graph.add_edge("expand_query", "real_search")
        graph.add_edge("real_search", "extract_content")
        graph.add_edge("extract_content", "score_sources")
        graph.add_edge("score_sources", "deep_analysis")
        graph.add_edge("deep_analysis", "cross_reference")
        graph.add_edge("cross_reference", "synthesize")
        graph.add_edge("synthesize", "calculate_confidence")
        graph.add_edge("calculate_confidence", "evaluate")

        def should_refine(state: DeepResearcherState) -> str:
            # Use the lower of grounded confidence and LLM quality score so that
            # a poor LLM evaluation can still trigger refinement even when the
            # grounded signal looks acceptable, and vice-versa.
            effective_score = min(
                state["calculated_confidence"],
                state.get("quality_score", 1.0),
            )
            max_loops = state["max_refinement_loops"]

            # Budget may further restrict refinement loops (set by AgentExecutor)
            budget = self._budget_config()
            if budget and budget.max_refinement_loops is not None:
                max_loops = min(max_loops, budget.max_refinement_loops)

            # Hard stop: max refinement loops reached
            if state["refinement_rounds"] >= max_loops:
                return "generate_report"

            # Skip refinement if previous round produced no quality improvement
            prev = state.get("previous_quality_score", 0.0)
            if state["refinement_rounds"] > 0 and effective_score <= prev + 0.02:
                logger.info(
                    "Refinement produced no quality improvement (%.3f -> %.3f), stopping",
                    prev,
                    effective_score,
                )
                return "generate_report"

            # Content-quality guard: force refinement if synthesis is too thin
            synthesis_len = len(state.get("synthesis", ""))
            findings_count = len(state.get("key_findings", []))
            if synthesis_len < 500 or findings_count < 3:
                logger.info(
                    "Synthesis too thin (%d chars, %d findings) — forcing refinement",
                    synthesis_len,
                    findings_count,
                )
                return "refine"

            if effective_score >= state["quality_threshold"]:
                return "generate_report"
            return "refine"

        graph.add_conditional_edges("evaluate", should_refine)
        graph.add_edge("refine", "real_search")
        graph.add_edge("generate_report", "save_session")
        graph.add_edge("save_session", END)

        return graph.compile()

    def _has_loops(self) -> bool:
        return True

    # ── Step Implementations ───────────────────────────────────────────

    def _step_load_context(self, state: DeepResearcherState) -> dict[str, Any]:
        """CODE: Check memory for related previous research."""
        with self._step_span("load_context") as span:
            if self._progress:
                self._progress.start_step("load_context", "Checking research memory...")

            if not getattr(self._run_local, "enable_memory", True):
                if self._progress:
                    self._progress.complete_step("load_context", "Memory disabled")
                if span:
                    span.set_attribute("memory_enabled", False)
                return {"prior_context": {}, "known_sources": []}

            context = self._memory.get_context_for_follow_up(state["query"])

            if self._progress:
                if context:
                    self._progress.complete_step(
                        "load_context",
                        f"Found {len(context['related_sessions'])} related sessions",
                        {"sessions": len(context["related_sessions"])},
                    )
                else:
                    self._progress.complete_step("load_context", "No prior research found")

            if span:
                span.set_attribute("memory_enabled", True)
                span.set_attribute(
                    "related_sessions",
                    len(context["related_sessions"]) if context else 0,
                )

            return {
                "prior_context": context or {},
                "known_sources": context["known_sources"] if context else [],
            }

    def _step_expand_query(self, state: DeepResearcherState) -> dict[str, Any]:
        """LLM: Expand and refine the research query."""
        with self._step_span("expand_query") as span:
            if self._progress:
                self._progress.start_step("expand_query", "Expanding research query...")
            focus_hint = (
                f"\nResearch focus: {state['research_focus']}"
                if state.get("research_focus")
                else ""
            )
            # Include prior context if available
            prior = state.get("prior_context", {})
            prior_hint = ""
            if prior.get("previous_findings"):
                prior_hint = (
                    "\n\nPrevious research found these findings — expand into new areas:\n"
                    + "\n".join(f"- {f}" for f in prior["previous_findings"][:5])
                )
            if prior.get("known_gaps"):
                prior_hint += "\n\nKnown information gaps to address:\n" + "\n".join(
                    f"- {g}" for g in prior["known_gaps"][:5]
                )

            out = cast(
                _ExpandQueryOut,
                self._extract(
                    response_model=_ExpandQueryOut,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": f"{self.step_prompts['expand_query']}\n\n{state['query']}{focus_hint}{prior_hint}",
                        },
                    ],
                    model=self._model_for_step("expand_query"),
                    allow_fallback=True,
                ),
            )

            # Post-truncate: LLMs may ignore the max_length constraint
            queries = out.expanded_queries[:5]

            if self._progress:
                self._progress.complete_step(
                    "expand_query",
                    f"Generated {len(queries)} queries ({out.search_strategy})",
                    {"queries": len(queries), "strategy": out.search_strategy},
                )

            if span:
                span.set_attribute("query_count", len(queries))
                span.set_attribute("strategy", out.search_strategy)

            return {
                "expanded_queries": queries,
                "search_strategy": out.search_strategy,
            }

    def _step_real_search(self, state: DeepResearcherState) -> dict[str, Any]:  # noqa: C901
        """CODE: Execute real searches across all providers."""
        with self._step_span("real_search") as span:
            if self._progress:
                self._progress.start_step("real_search", "Searching across providers...")
            engine = self._get_search_engine()

            strategy = state.get("search_strategy", "comprehensive")
            content_types: list[str] | None = None
            if strategy == "academic":
                content_types = ["academic", "encyclopedia"]
            elif strategy == "news":
                content_types = ["web", "news"]

            all_results: list[SearchResult] = []

            async def _do_search() -> None:
                async def _search_one(q: str) -> list[SearchResult]:
                    try:
                        return await asyncio.wait_for(
                            engine.search(
                                q,
                                max_results_per_provider=3,
                                content_types=content_types,
                            ),
                            timeout=15.0,
                        )
                    except TimeoutError:
                        logger.warning("Search timed out for query: %s", q[:80])
                        return []

                batches = await asyncio.gather(*[_search_one(q) for q in state["expanded_queries"]])
                for batch in batches:
                    all_results.extend(batch)

            _run_async(_do_search())

            # Deduplicate across all queries
            seen: set[str] = set()
            deduped: list[SearchResult] = []
            for r in all_results:
                url_key = r.url.rstrip("/").lower()
                if url_key not in seen:
                    seen.add(url_key)
                    deduped.append(r)

            # Merge with existing results from previous refinement rounds
            existing_urls = {r["url"].rstrip("/").lower() for r in state.get("search_results", [])}
            new_results = [r for r in deduped if r.url.rstrip("/").lower() not in existing_urls]

            serialized_new = [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source_provider": r.source_provider,
                    "domain": r.domain,
                    "authors": r.authors,
                    "doi": r.doi,
                    "citation_count": r.citation_count,
                    "content_type": r.content_type,
                    "published_date": r.published_date,
                }
                for r in new_results[:10]  # reduced from 15 to cap per-round accumulation
            ]

            # Log per-provider result counts and warn on missing providers
            provider_counts: dict[str, int] = {}
            for r in deduped:
                provider_counts[r.source_provider] = provider_counts.get(r.source_provider, 0) + 1
            if strategy == "comprehensive" and "duckduckgo" not in provider_counts:
                logger.warning(
                    "No web search results from DuckDuckGo — web coverage may be incomplete"
                )

            combined = state.get("search_results", []) + serialized_new
            logger.info("Found %d new results, %d total", len(serialized_new), len(combined))

            if self._progress:
                self._progress.complete_step(
                    "real_search",
                    f"Found {len(serialized_new)} new results, {len(combined)} total",
                    {"new": len(serialized_new), "total": len(combined)},
                )

            if span:
                span.set_attribute("new_results", len(serialized_new))
                span.set_attribute("total_results", len(combined))
                span.set_attribute("strategy", strategy)

            return {"search_results": combined}

    def _step_extract_content(self, state: DeepResearcherState) -> dict[str, Any]:
        """CODE: Fetch and extract readable text from source URLs."""
        with self._step_span("extract_content") as span:
            if self._progress:
                self._progress.start_step("extract_content", "Extracting content from sources...")
            extractor = self._get_content_extractor()

            already_extracted = {e["url"] for e in state.get("extracted_content", [])}
            urls_to_extract = [
                r["url"]
                for r in state["search_results"]
                if r["url"] not in already_extracted and r["url"].startswith("http")
            ][:10]

            extracted: list[ExtractedContent] = []
            if urls_to_extract:

                async def _do_extract() -> None:
                    nonlocal extracted
                    try:
                        extracted = await asyncio.wait_for(
                            extractor.extract_batch(urls_to_extract, max_concurrent=5),
                            timeout=60.0,
                        )
                    except ExtractionProviderError:
                        raise
                    except TimeoutError:
                        logger.warning("Content extraction timed out after 60s — partial results")
                        extracted = []

                _run_async(_do_extract())

            serialized_new = [
                {
                    "url": e.url,
                    "title": e.title,
                    "full_text": e.full_text[:3000],  # capped: 10 sources x 3k = 30k chars max
                    "word_count": e.word_count,
                    "success": e.success,
                }
                for e in extracted
                if e.success and e.word_count > 50
            ]

            combined = state.get("extracted_content", []) + serialized_new
            logger.info("Extracted %d new sources, %d total", len(serialized_new), len(combined))

            if self._progress:
                self._progress.complete_step(
                    "extract_content",
                    f"Extracted {len(serialized_new)} new sources, {len(combined)} total",
                    {"new": len(serialized_new), "total": len(combined)},
                )

            if span:
                span.set_attribute("new_extracted", len(serialized_new))
                span.set_attribute("total_extracted", len(combined))

            return {"extracted_content": combined}

    def _step_score_sources(self, state: DeepResearcherState) -> dict[str, Any]:
        """CODE: Score source credibility using heuristics."""
        with self._step_span("score_sources") as span:
            if self._progress:
                self._progress.start_step("score_sources", "Scoring source credibility...")
            scores = []
            for result in state["search_results"]:
                cred = score_credibility(
                    url=result["url"],
                    published_date=result.get("published_date"),
                    citation_count=result.get("citation_count"),
                    content_type=result.get("content_type", "web"),
                    has_author=bool(result.get("authors")),
                    word_count=next(
                        (
                            e["word_count"]
                            for e in state.get("extracted_content", [])
                            if e["url"] == result["url"]
                        ),
                        0,
                    ),
                )
                scores.append(
                    {
                        "url": result["url"],
                        "overall": cred.overall,
                        "domain_authority": cred.domain_authority,
                        "recency_score": cred.recency_score,
                        "citation_score": cred.citation_score,
                        "signals": cred.signals,
                    }
                )
            if self._progress:
                self._progress.complete_step(
                    "score_sources",
                    f"Scored {len(scores)} sources",
                    {"sources": len(scores)},
                )

            if span:
                span.set_attribute("source_count", len(scores))

            return {"credibility_scores": scores}

    # ── HYBRID DEEP ANALYSIS STEP ──────────────────────────────────────
    # Replaces the old _step_extract_claims + _step_verify_claims.
    # This is the ONLY step where the LLM gets tool access.

    def _step_deep_analysis(self, state: DeepResearcherState) -> dict[str, Any]:
        """HYBRID: LLM analyzes sources with bounded tool access.

        The LLM can optionally call tools (fact_check, search_for_more, etc.)
        within a hard budget. If the model is weak, tools are disabled and
        this runs as pure claim extraction.
        """
        with self._step_span("deep_analysis") as span:
            if self._progress:
                mode = "hybrid (tools enabled)" if self._tools_enabled else "extraction only"
                self._progress.start_step(
                    "deep_analysis",
                    f"Deep analysis — {mode}, model tier: {self._model_capability.tier}",
                )

            new_sources = self._get_unprocessed_sources(state)

            if not new_sources:
                if self._progress:
                    self._progress.complete_step("deep_analysis", "No new sources to analyze")
                if span:
                    span.set_attribute("sources_analyzed", 0)
                return {
                    "claims": state.get("claims", []),
                    "verified_claims": state.get("verified_claims", []),
                    "tool_usage": {},
                    "tool_insights": [],
                    "additional_sources": [],
                }

            try:
                result, analysis = self._run_hybrid_analysis(state, new_sources)
            except Exception as exc:
                logger.warning(
                    "Deep analysis failed; continuing with empty analysis payload: %s",
                    exc,
                )
                analysis = _HybridAnalysisOut()
                result = HybridStepResult(
                    structured_output=analysis,
                    tools_enabled=self._tools_enabled,
                    llm_rounds=1,
                )

            if span:
                span.set_attribute("sources_analyzed", len(new_sources))
                span.set_attribute("claims_found", len(analysis.claims))
                span.set_attribute("tools_enabled", self._tools_enabled)

            return self._merge_analysis_results(state, result, analysis)

    def _get_unprocessed_sources(
        self,
        state: DeepResearcherState,
    ) -> list[dict[str, Any]]:
        """Get sources not yet processed, sorted by credibility."""
        scored_urls = {s["url"]: s["overall"] for s in state.get("credibility_scores", [])}
        sources_with_content = [
            e
            for e in state.get("extracted_content", [])
            if e.get("success") and e.get("word_count", 0) > 50
        ]
        sources_with_content.sort(key=lambda x: scored_urls.get(x["url"], 0.5), reverse=True)
        already_processed = {c.get("source_url") for c in state.get("claims", [])}
        unprocessed = [s for s in sources_with_content if s["url"] not in already_processed]
        # Allow more sources in refinement rounds to justify the extra loop
        cap = 8 if state.get("refinement_rounds", 0) > 0 else 5
        if len(unprocessed) > cap:
            logger.info(
                "Limiting analysis to %d of %d unprocessed sources (sorted by credibility)",
                cap,
                len(unprocessed),
            )
        return unprocessed[:cap]

    def _get_relevant_text(
        self, full_text: str, query: str, url: str, title: str, max_chars: int = 2500
    ) -> str:
        """Chunk text and return the most query-relevant portions."""
        if url not in self._chunk_cache:
            self._chunk_cache[url] = self._chunker.chunk(
                full_text, source_url=url, source_title=title
            )
        chunks = self._chunk_cache[url]
        if not chunks:
            return full_text[:max_chars]
        top = self._ranker.top_k(query, chunks, k=5)
        parts: list[str] = []
        total = 0
        for c in top:
            if total + len(c.text) > max_chars:
                parts.append(c.text[: max_chars - total])
                break
            parts.append(c.text)
            total += len(c.text)
        return "\n...\n".join(parts)

    def _run_hybrid_analysis(
        self,
        state: DeepResearcherState,
        new_sources: list[dict[str, Any]],
    ) -> tuple[HybridStepResult, _HybridAnalysisOut]:
        """Execute the hybrid analysis step with optional tool calling."""
        scored_urls = {s["url"]: s["overall"] for s in state.get("credibility_scores", [])}

        source_blocks = []
        for i, source in enumerate(new_sources):
            cred = scored_urls.get(source["url"], 0.0)
            relevant_text = self._get_relevant_text(
                source["full_text"], state["query"], source["url"], source["title"]
            )
            source_blocks.append(
                f"### Source [{i + 1}]: {source['title']}\n"
                f"URL: {source['url']}\n"
                f"Credibility: {cred:.2f}\n\n"
                f"{relevant_text}\n"
                f"---"
            )

        user_prompt = (
            f"Research query: {state['query']}\n\n"
            f"## Sources to Analyze ({len(new_sources)} sources)\n\n" + "\n\n".join(source_blocks)
        )

        tools = ANALYSIS_TOOLS if self._tools_enabled else None
        if self._tools_enabled and self._model_capability.tier == "medium":
            simple_tool_names = {"fact_check", "search_for_more"}
            tools = [t for t in ANALYSIS_TOOLS if t["function"]["name"] in simple_tool_names]

        # Respect per-run enable_fact_check flag
        if tools and not getattr(self._run_local, "enable_fact_check", True):
            tools = [t for t in tools if t["function"]["name"] != "fact_check"]

        runner = self._get_hybrid_runner()
        # Combine agent identity with step-specific tool guidance
        combined_system_prompt = (
            f"{self.system_prompt}\n\n"
            f"## Task: Deep Analysis with Strategic Tool Use\n\n"
            f"{self.step_prompts['deep_analysis']}"
        )
        result: HybridStepResult = runner.run(
            system_prompt=combined_system_prompt,
            user_prompt=user_prompt,
            output_model=_HybridAnalysisOut,
            model=self._model,
            tools=tools,
        )
        return result, result.structured_output

    def _merge_analysis_results(
        self,
        state: DeepResearcherState,
        result: HybridStepResult,
        analysis: _HybridAnalysisOut,
    ) -> dict[str, Any]:
        """Merge hybrid analysis results into state."""
        all_claims = list(state.get("claims", []))
        for claim in analysis.claims:
            all_claims.append(
                {
                    "claim_text": claim.claim_text,
                    "confidence": claim.confidence,
                    "source_url": claim.source_url,
                    "source_title": claim.source_title,
                    "supporting_quote": claim.supporting_quote,
                }
            )

        verified = list(state.get("verified_claims", []))
        verified.extend(analysis.verified_facts)
        for tr in result.tool_calls:
            if tr.tool_name == "fact_check" and tr.success:
                verified.append(tr.result)

        additional_sources = list(state.get("additional_sources", []))
        for tr in result.tool_calls:
            if tr.tool_name == "search_for_more" and tr.success:
                additional_sources.extend(tr.result.get("results", []))

        new_search_results = self._merge_additional_sources(
            state.get("search_results", []), additional_sources
        )

        if result.tools_enabled and not result.tool_calls:
            logger.warning(
                "Tool calling was enabled but LLM made 0 tool calls — "
                "model may not support tool use despite capability detection"
            )

        if self._progress:
            self._progress.complete_step(
                "deep_analysis",
                (
                    f"Extracted {len(analysis.claims)} claims, "
                    f"verified {len(verified)} facts, "
                    f"found {len(additional_sources)} additional sources"
                    + (
                        f", used {result.budget_summary.get('total_calls', 0)} tool calls"
                        if result.tools_enabled
                        else ""
                    )
                ),
                {
                    "claims": len(all_claims),
                    "verified": len(verified),
                    "tool_calls": result.budget_summary.get("total_calls", 0),
                    "tools_enabled": result.tools_enabled,
                    "llm_rounds": result.llm_rounds,
                },
            )

        return {
            "claims": all_claims,
            "verified_claims": verified,
            "search_results": new_search_results,
            "tool_usage": result.budget_summary,
            "tool_insights": analysis.tool_insights,
            "additional_sources": additional_sources,
        }

    def _merge_additional_sources(
        self,
        existing_results: list[dict[str, Any]],
        additional_sources: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge tool-discovered sources into search results."""
        existing_urls = {r["url"].rstrip("/").lower() for r in existing_results}
        merged = list(existing_results)
        for src in additional_sources:
            url_key = src.get("url", "").rstrip("/").lower()
            if url_key and url_key not in existing_urls:
                existing_urls.add(url_key)
                merged.append(
                    {
                        "title": src.get("title", ""),
                        "url": src.get("url", ""),
                        "snippet": src.get("snippet", ""),
                        "source_provider": src.get("source_provider", "tool:search_for_more"),
                        "domain": "",
                        "authors": [],
                        "doi": None,
                        "citation_count": None,
                        "content_type": src.get("content_type", "web"),
                        "published_date": None,
                    }
                )
        return merged

    def _step_cross_reference(self, state: DeepResearcherState) -> dict[str, Any]:
        """LLM: Cross-reference claims across sources."""
        with self._step_span("cross_reference") as span:
            if self._progress:
                self._progress.start_step("cross_reference", "Cross-referencing claims...")
            claims = state.get("claims", [])
            if not claims:
                if self._progress:
                    self._progress.complete_step("cross_reference", "No claims to cross-reference")
                if span:
                    span.set_attribute("claim_count", 0)
                return {
                    "claim_clusters": [],
                    "information_gaps": ["No claims extracted from sources"],
                }

            # Cap at top 20 by confidence to avoid unbounded context growth across refinement rounds
            top_claims = sorted(claims, key=lambda c: c.get("confidence", 0.5), reverse=True)[:20]
            claims_text = "\n\n".join(
                f'Claim {i}: "{c["claim_text"]}" '
                f"(Source: {c.get('source_title', 'Unknown')}, "
                f"Confidence: {c['confidence']})"
                for i, c in enumerate(top_claims)
            )

            out = cast(
                CrossReferenceOutput,
                self._extract(
                    response_model=CrossReferenceOutput,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"{self.step_prompts['cross_reference']}\n\n"
                                f"Research query: {state['query']}\n\n"
                                f"Claims from {len(claims)} extractions:\n{claims_text}"
                                + (
                                    "\n\n## Verification Insights\n"
                                    + "\n".join(
                                        f"- {ins}" for ins in state.get("tool_insights", [])
                                    )
                                    if state.get("tool_insights")
                                    else ""
                                )
                            ),
                        },
                    ],
                    model=self._model_for_step("cross_reference"),
                    allow_fallback=True,
                ),
            )

            if self._progress:
                self._progress.complete_step(
                    "cross_reference",
                    f"Found {len(out.claim_clusters)} clusters, {len(out.information_gaps)} gaps",
                    {"clusters": len(out.claim_clusters), "gaps": len(out.information_gaps)},
                )

            if span:
                span.set_attribute("clusters", len(out.claim_clusters))
                span.set_attribute("gaps", len(out.information_gaps))

            return {
                "claim_clusters": [c.model_dump() for c in out.claim_clusters],
                "information_gaps": out.information_gaps,
            }

    def _step_synthesize(self, state: DeepResearcherState) -> dict[str, Any]:  # noqa: C901
        """LLM: Synthesize findings with real citations."""
        with self._step_span("synthesize") as span:
            if self._progress:
                self._progress.start_step("synthesize", "Synthesizing findings...")
            cred_map = {s["url"]: s for s in state.get("credibility_scores", [])}
            content_map = {
                e["url"]: e for e in state.get("extracted_content", []) if e.get("success")
            }

            # Source list with credibility
            sources_summary = []
            for r in state["search_results"][:12]:
                cred = cred_map.get(r["url"], {})
                sources_summary.append(
                    f"- [{r['title']}]({r['url']}) "
                    f"(credibility: {cred.get('overall', 'N/A')}, "
                    f"type: {r.get('content_type', 'web')})"
                )

            # Include actual source excerpts so the LLM can synthesize from content
            source_excerpts = []
            for r in state["search_results"][:8]:
                content = content_map.get(r["url"])
                if content:
                    text = self._get_relevant_text(
                        content["full_text"],
                        state["query"],
                        r["url"],
                        r["title"],
                        max_chars=800,
                    )
                    source_excerpts.append(f"### {r['title']}\nURL: {r['url']}\n\n{text}\n")

            # Include detailed claims with supporting quotes
            claims = state.get("claims", [])
            claims_text = ""
            if claims:
                top_claims = sorted(claims, key=lambda c: c.get("confidence", 0.5), reverse=True)[
                    :15
                ]
                claims_detail = []
                for i, c in enumerate(top_claims, 1):
                    detail = f'{i}. "{c["claim_text"]}" (confidence: {c["confidence"]}'
                    if c.get("source_title"):
                        detail += f", source: {c['source_title']}"
                    if c.get("supporting_quote"):
                        detail += f', evidence: "{c["supporting_quote"][:200]}"'
                    detail += ")"
                    claims_detail.append(detail)
                claims_text = "\n\n## Detailed Claims with Evidence\n" + "\n".join(claims_detail)

            clusters_text = ""
            for i, cluster in enumerate(state.get("claim_clusters", [])):
                clusters_text += (
                    f"\nCluster {i + 1}: {cluster['representative_claim']}\n"
                    f"  Agreement: {cluster['agreement_score']}\n"
                    f"  Note: {cluster.get('synthesis_note', '')}\n"
                )

            gaps_text = "\n".join(f"- {g}" for g in state.get("information_gaps", []))

            # Include verification results in synthesis context
            verified = state.get("verified_claims", [])
            verification_text = ""
            if verified:
                verification_text = "\n\n## Verified Facts\n" + "\n".join(
                    f"- {v.get('claim', v.get('claim_text', ''))}: "
                    f"{'\u2713 verified' if v.get('verified') else '\u2717 unverified' if v.get('verified') is False else '? inconclusive'}"
                    for v in verified
                )

            out = cast(
                _SynthesizeOut,
                self._extract(
                    response_model=_SynthesizeOut,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"{self.step_prompts['synthesize']}\n\n"
                                f"Research query: {state['query']}\n\n"
                                f"## Available Sources:\n" + "\n".join(sources_summary) + "\n\n"
                                "## Source Excerpts (key content):\n"
                                + "\n".join(source_excerpts)
                                + "\n\n"
                                f"## Claim Analysis:\n{clusters_text}"
                                f"{claims_text}\n\n"
                                f"## Information Gaps:\n{gaps_text}"
                                f"{verification_text}"
                            ),
                        },
                    ],
                    model=self._model_for_step("synthesize"),
                    allow_fallback=True,
                ),
            )

            if self._progress:
                self._progress.complete_step(
                    "synthesize",
                    f"Synthesized with {len(out.key_findings)} key findings",
                    {"findings": len(out.key_findings), "citations": len(out.citations)},
                )

            if span:
                span.set_attribute("findings", len(out.key_findings))
                span.set_attribute("citations", len(out.citations))

            return {
                "synthesis": out.synthesis,
                "key_findings": out.key_findings,
                "citations": out.citations,
            }

    def _step_calculate_confidence(self, state: DeepResearcherState) -> dict[str, Any]:
        """CODE: Calculate grounded confidence from measurable signals."""
        if self._progress:
            self._progress.start_step("calculate_confidence", "Calculating confidence...")

        breakdown = self._confidence_calc.calculate(
            search_results=state.get("search_results", []),
            extracted_content=state.get("extracted_content", []),
            credibility_scores=state.get("credibility_scores", []),
            claim_clusters=state.get("claim_clusters", []),
            information_gaps=state.get("information_gaps", []),
            claims=state.get("claims", []),
        )

        if self._progress:
            self._progress.complete_step(
                "calculate_confidence",
                f"Confidence: {breakdown.overall:.0%} — {breakdown.explanation}",
                {"score": breakdown.overall},
            )

        return {
            "calculated_confidence": breakdown.overall,
            "confidence_breakdown": {
                "overall": breakdown.overall,
                "source_diversity": breakdown.source_diversity,
                "source_credibility": breakdown.source_credibility,
                "claim_coverage": breakdown.claim_coverage,
                "claim_agreement": breakdown.claim_agreement,
                "extraction_success_rate": breakdown.extraction_success_rate,
                "explanation": breakdown.explanation,
                "components": breakdown.components,
            },
        }

    def _step_evaluate(self, state: DeepResearcherState) -> dict[str, Any]:
        """LLM: Evaluate synthesis quality."""
        with self._step_span("evaluate") as span:
            if self._progress:
                self._progress.start_step("evaluate", "Evaluating synthesis quality...")
            out = cast(
                _EvaluateOut,
                self._extract(
                    response_model=_EvaluateOut,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"{self.step_prompts['evaluate']}\n\n"
                                f"Original query: {state['query']}\n\n"
                                f"Synthesis:\n{state['synthesis']}\n\n"
                                f"Sources used: {len(state.get('search_results', []))}\n"
                                f"Claims analyzed: {len(state.get('claims', []))}\n"
                                f"Information gaps identified: {state.get('information_gaps', [])}"
                            ),
                        },
                    ],
                    model=self._model_for_step("evaluate"),
                    allow_fallback=True,
                ),
            )

            if self._progress:
                self._progress.complete_step(
                    "evaluate",
                    f"Quality: {out.quality_score:.0%}, missing: {len(out.missing_aspects)} aspects",
                    {"quality": out.quality_score},
                )

            if span:
                span.set_attribute("quality_score", out.quality_score)

            return {
                "quality_score": out.quality_score,
                "evaluation_feedback": out.feedback,
                "refinement_rounds": state["refinement_rounds"],
                "previous_quality_score": min(
                    state.get("calculated_confidence", 0.0),
                    state.get("quality_score", 1.0),
                ),
            }

    def _step_refine(self, state: DeepResearcherState) -> dict[str, Any]:
        """LLM+CODE: Generate targeted queries to fill gaps, then loop back."""
        with self._step_span("refine") as span:
            if self._progress:
                self._progress.start_step(
                    "refine", f"Refining (round {state['refinement_rounds'] + 1})..."
                )
            out = cast(
                _GapAnalysisOut,
                self._extract(
                    response_model=_GapAnalysisOut,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"{self.step_prompts['gap_analysis']}\n\n"
                                f"Query: {state['query']}\n\n"
                                f"Evaluation feedback: {state.get('evaluation_feedback', '')}\n"
                                f"Information gaps: {state.get('information_gaps', [])}"
                            ),
                        },
                    ],
                    model=self._model_for_step("refine"),
                    allow_fallback=True,
                ),
            )

            if self._progress:
                self._progress.complete_step(
                    "refine",
                    f"Generated {len(out.additional_queries or [])} follow-up queries",
                )

            if span:
                span.set_attribute("refinement_round", state["refinement_rounds"] + 1)
                span.set_attribute("new_queries", len(out.additional_queries or []))

            return {
                "expanded_queries": out.additional_queries or [state["query"]],
                "refinement_rounds": state["refinement_rounds"] + 1,
            }

    def _step_generate_report(self, state: DeepResearcherState) -> dict[str, Any]:
        """LLM+CODE: Generate structured report in multiple formats."""
        with self._step_span("generate_report") as span:
            if self._progress:
                self._progress.start_step("generate_report", "Generating report...")

            citations = self._get_report_generator().build_citations(
                state.get("search_results", []),
                state.get("credibility_scores", []),
            )

            report = self._get_report_generator().structure_report(
                synthesis=state["synthesis"],
                key_findings=state.get("key_findings", []),
                query=state["query"],
                citations=citations,
                claim_clusters=state.get("claim_clusters", []),
                information_gaps=state.get("information_gaps", []),
                model=self._model_for_step("generate_report"),
            )

            markdown_report = self._get_report_generator().render(report, ReportFormat.MARKDOWN)
            json_report = self._get_report_generator().render(report, ReportFormat.JSON_STRUCTURED)

            # Also render in the caller-requested format if it differs from markdown
            requested_format = getattr(self._run_local, "report_format", ReportFormat.MARKDOWN)
            if requested_format not in (ReportFormat.MARKDOWN, ReportFormat.JSON_STRUCTURED):
                markdown_report = self._get_report_generator().render(report, requested_format)

            if self._progress:
                self._progress.complete_step(
                    "generate_report",
                    f"Generated report: {report.title}",
                    {"sections": len(report.sections), "citations": len(report.citations)},
                )

            if span:
                span.set_attribute("sections", len(report.sections))
                span.set_attribute("citations", len(report.citations))

            return {
                "structured_report": json.loads(json_report),
                "markdown_report": markdown_report,
                "report_title": report.title,
            }

    def _step_save_session(self, state: DeepResearcherState) -> dict[str, Any]:
        """CODE: Save session to memory for future follow-ups."""
        with self._step_span("save_session") as span:
            if self._progress:
                self._progress.start_step("save_session", "Saving research session...")

            if not getattr(self._run_local, "enable_memory", True):
                if self._progress:
                    self._progress.complete_step("save_session", "Memory disabled")
                if span:
                    span.set_attribute("memory_enabled", False)
                return {"session_id": ""}

            session_id = str(uuid.uuid4())[:8]

            session = ResearchSession(
                session_id=session_id,
                query=state["query"],
                timestamp=time.time(),
                synthesis=state["synthesis"],
                key_findings=state.get("key_findings", []),
                sources=state.get("search_results", []),
                credibility_scores=state.get("credibility_scores", []),
                claims=state.get("claims", []),
                quality_score=state.get("calculated_confidence", 0.0),
                information_gaps=state.get("information_gaps", []),
            )

            try:
                self._memory.save_session(session)
            except Exception as e:
                logger.warning("Failed to save research session: %s", e)

            if self._progress:
                self._progress.complete_step(
                    "save_session",
                    f"Saved session {session_id}",
                    {"session_id": session_id},
                )

            if span:
                span.set_attribute("session_id", session_id)

            return {"session_id": session_id}

    # ── Public Interface ───────────────────────────────────────────────

    def run(
        self, input: DeepResearcherInput, context: AgentContext | None = None
    ) -> AgentResult[DeepResearcherOutput]:
        """Execute the hybrid research workflow."""
        workflow_start_time = time.monotonic()
        context = context or AgentContext()
        run_id = context.run_id

        # Store per-run overrides in thread-local storage so concurrent calls
        # on the same instance don't clobber each other's tool configuration.
        self._run_local.context = context
        self._run_local.force_tool_calling = input.enable_tool_calling
        self._run_local.tool_budget = input.tool_budget
        self._run_local.enable_memory = input.enable_memory
        self._run_local.enable_fact_check = input.enable_fact_check
        self._run_local.report_format = input.report_format
        if input.enable_tool_calling is True and not self._model_capability.supports_tool_calling:
            logger.warning(
                "Tool calling forced for model tier '%s' — may produce poor results",
                self._model_capability.tier,
            )

        logger.info(
            "Starting DeepResearcher: run_id=%s, model=%s, tier=%s, tools=%s, budget=%d",
            run_id,
            self._model,
            self._model_capability.tier,
            self._tools_enabled,
            self._run_local.tool_budget,
        )

        graph = self._get_graph()

        initial_state: DeepResearcherState = {
            "query": input.query,
            "research_focus": input.research_focus,
            "quality_threshold": input.quality_threshold,
            "max_refinement_loops": input.max_refinement_loops,
            "prior_context": {},
            "known_sources": [],
            "expanded_queries": [],
            "search_strategy": "comprehensive",
            "search_results": [],
            "extracted_content": [],
            "credibility_scores": [],
            "claims": [],
            "verified_claims": [],
            "claim_clusters": [],
            "information_gaps": [],
            "synthesis": "",
            "key_findings": [],
            "citations": [],
            "calculated_confidence": 0.0,
            "confidence_breakdown": {},
            "quality_score": 0.0,
            "evaluation_feedback": "",
            "previous_quality_score": 0.0,
            "structured_report": {},
            "markdown_report": "",
            "report_title": "",
            "session_id": "",
            "refinement_rounds": 0,
            "tool_usage": {},
            "tool_insights": [],
            "additional_sources": [],
        }

        try:
            final_state = graph.invoke(initial_state)
        except ExtractionProviderError:
            raise
        except Exception as exc:
            logger.exception("Research workflow failed")
            return AgentResult[DeepResearcherOutput](
                output=None,
                success=False,
                error=str(exc),
                run_id=run_id,
                agent_name=self.name,
                agent_version=self.version,
            )
        finally:
            # Always clear thread-local state so a subsequent run on the same
            # thread starts clean (prevents config bleed-over between calls).
            self._run_local.__dict__.clear()

        total_duration = time.monotonic() - workflow_start_time

        # Build source breakdown
        source_breakdown: dict[str, int] = {}
        for r in final_state.get("search_results", []):
            provider = r.get("source_provider", "unknown")
            source_breakdown[provider] = source_breakdown.get(provider, 0) + 1

        # Build credibility summary
        cred_scores = final_state.get("credibility_scores", [])
        cred_summary: dict[str, float] = {}
        if cred_scores:
            overall_scores = [s["overall"] for s in cred_scores]
            cred_summary = {
                "mean_credibility": round(sum(overall_scores) / len(overall_scores), 3),
                "max_credibility": round(max(overall_scores), 3),
                "min_credibility": round(min(overall_scores), 3),
            }

        output = DeepResearcherOutput(
            synthesis=final_state["synthesis"],
            key_findings=final_state.get("key_findings", []),
            markdown_report=final_state.get("markdown_report", ""),
            structured_report=final_state.get("structured_report", {}),
            report_title=final_state.get("report_title", ""),
            confidence=final_state.get("calculated_confidence", 0.0),
            confidence_breakdown=final_state.get("confidence_breakdown", {}),
            llm_quality_score=final_state.get("quality_score", 0.0),
            citations=final_state.get("citations", []),
            sources_analyzed=len(final_state.get("search_results", [])),
            source_breakdown=source_breakdown,
            credibility_summary=cred_summary,
            claim_clusters=final_state.get("claim_clusters", []),
            verified_claims=final_state.get("verified_claims", []),
            information_gaps=final_state.get("information_gaps", []),
            refinement_rounds=final_state["refinement_rounds"],
            session_id=final_state.get("session_id", ""),
            total_duration_seconds=round(total_duration, 2),
            tool_usage=final_state.get("tool_usage", {}),
            tool_insights=final_state.get("tool_insights", []),
            hybrid_mode=self._tools_enabled,
            model_tier=self._model_capability.tier,
        )

        return AgentResult(
            output=output,
            run_id=run_id,
            agent_name=self.name,
            agent_version=self.version,
        )

    def cleanup(self) -> None:
        """Clean up async resources synchronously."""

        async def _cleanup_async() -> None:
            if self._search_engine:
                await self._search_engine.close()
            if self._content_extractor:
                await self._content_extractor.close()
            if self._fact_checker:
                await self._fact_checker.close()

        _run_async(_cleanup_async())
        _async_bridge.shutdown()

    def __del__(self) -> None:
        with suppress(Exception):
            self.cleanup()
