"""ResearchAgent: multi-step research workflow with query expansion, search, analysis, and synthesis.

This agent serves as a reference implementation for Phase 2 multi-step workflows:
- Query expansion and refinement
- Parallel search across multiple queries
- Source analysis and credibility evaluation
- Evidence-based synthesis with citations
- Self-evaluation and refinement

This is the reference 6-step agent mentioned in Design_Document.md §17.2.
"""

from __future__ import annotations

from typing import Any, ClassVar, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from dopeagents.core.agent import Agent
from dopeagents.core.context import AgentContext
from dopeagents.core.types import AgentResult

# ── Internal State ─────────────────────────────────────────────────────────


class ResearchAgentState(TypedDict):
    """Internal LangGraph state for the 6-step research workflow."""

    query: str
    research_focus: str | None
    expanded_queries: list[str]
    search_results: list[dict[str, Any]]
    analyzed_sources: list[dict[str, Any]]
    synthesis: str
    quality_score: float
    refinement_rounds: int
    max_refinement_loops: int


# ── Step Output Schemas ────────────────────────────────────────────────────


class _ExpandQueryOut(BaseModel):
    """Output of query expansion step."""

    expanded_queries: list[str] = Field(  # type: ignore[call-overload]
        description="List of expanded and refined queries",
        min_items=1,
        max_items=5,
    )


class _SearchOut(BaseModel):
    """Output of search step."""

    results: list[dict[str, Any]] = Field(
        description="List of search results with title, snippet, url, domain"
    )


class _AnalyzeOut(BaseModel):
    """Output of source analysis step."""

    analyzed: list[dict[str, Any]] = Field(
        description="Analyzed sources with credibility score, key findings, quality assessment"
    )


class _SynthesizeOut(BaseModel):
    """Output of synthesis step."""

    synthesis: str = Field(description="Evidence-based synthesis with citations")
    key_findings: list[str] = Field(default_factory=list, description="Top 3-5 key findings")


class _EvaluateOut(BaseModel):
    """Output of evaluation step."""

    quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality score (0-1)")
    coverage_score: float = Field(
        ge=0.0, le=1.0, description="How comprehensively the query is addressed"
    )
    credibility_score: float = Field(ge=0.0, le=1.0, description="How credible the sources are")
    feedback: str = Field(description="Actionable feedback for refinement")


# ── Public Input / Output ──────────────────────────────────────────────────


class ResearchAgentInput(BaseModel):
    """Input schema for ResearchAgent."""

    query: str = Field(min_length=5, description="Research query or topic")
    research_focus: str | None = Field(
        default=None,
        description="Optional focus area (e.g., 'academic', 'recent news', 'practical')",
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


class ResearchAgentOutput(BaseModel):
    """Output schema for ResearchAgent."""

    synthesis: str = Field(description="Final research synthesis with citations")
    key_findings: list[str] = Field(default_factory=list, description="Top findings from research")
    quality_score: float = Field(ge=0.0, le=1.0, description="Final quality score")
    sources_analyzed: int = Field(ge=0, description="Number of sources analyzed")
    refinement_rounds: int = Field(ge=0, description="Number of refinement loops")


# ── ResearchAgent ─────────────────────────────────────────────────────────


class ResearchAgent(Agent[ResearchAgentInput, ResearchAgentOutput]):
    """
    6-step multi-step research agent with query expansion, search, analysis, synthesis, and evaluation.

    Steps (in order):
      1. expand_query  — Expand and refine the research query
      2. search        — Search for relevant sources
      3. analyze       — Analyze sources for credibility and key findings
      4. synthesize    — Create evidence-based synthesis with citations
      5. evaluate      — Score quality and identify gaps
      6. refine        — Improve synthesis based on feedback (loops back to search/synthesize)

    The evaluate→refine→search/synthesize cycle loops until quality_score >= threshold or
    max_refinement_loops is reached.
    """

    name: ClassVar[str] = "ResearchAgent"
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = (
        "Multi-step research workflow with query expansion, search, analysis, synthesis, "
        "and self-evaluation"
    )
    capabilities: ClassVar[list[str]] = ["research", "synthesis", "source-analysis"]
    tags: ClassVar[list[str]] = ["research", "multi-step", "llm-based"]
    requires_llm: ClassVar[bool] = True

    system_prompt: ClassVar[str] = (
        "You are an expert research agent. Your task is to conduct thorough research, "
        "analyze sources critically, and synthesize evidence-based findings."
    )

    step_prompts: ClassVar[dict[str, str]] = {
        "expand_query": (
            "Given the research query, expand it into 3-5 refined search queries that capture "
            "different aspects and perspectives. Return queries that will yield diverse, relevant results."
        ),
        "search": (
            "Search for information on the provided queries. Return results with title, snippet, URL, "
            "and domain for each source. Aim for 5-10 diverse sources per query."
        ),
        "analyze": (
            "Analyze each source for credibility, relevance, and key findings. Score credibility "
            "(0-1 based on source authority, recency, expert consensus). List 2-3 key findings per source."
        ),
        "synthesize": (
            "Synthesize the analyzed sources into a coherent narrative. Cite specific sources. "
            "Highlight agreements and disagreements. Extract 3-5 key findings. Be comprehensive but concise."
        ),
        "evaluate": (
            "Evaluate the synthesis on three criteria:\n"
            "- COVERAGE: Does it comprehensively address the original query?\n"
            "- CREDIBILITY: Are sources authoritative and well-cited?\n"
            "- COHERENCE: Is the narrative logical and well-structured?\n\n"
            "Provide a quality score (0-1) and specific feedback for improvement."
        ),
        "refine": (
            "Improve the synthesis based on feedback. Consider conducting additional searches "
            "if coverage is incomplete. Strengthen citations and address any identified gaps."
        ),
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ResearchAgent.

        Args:
            **kwargs: Passed to parent Agent.__init__ (model, hooks, cache, etc.) plus step_prompts
        """
        # Extract and apply custom step_prompts if provided
        custom_step_prompts = kwargs.pop("step_prompts", None)

        super().__init__(**kwargs)

        # Apply custom step_prompts if provided (shadows ClassVar)
        if custom_step_prompts is not None:
            self.step_prompts = {**self.step_prompts, **custom_step_prompts}  # type: ignore[misc]

        self._graph: Any | None = None

    def _get_graph(self) -> Any:
        """Get or lazily build the compiled LangGraph state machine."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self) -> Any:
        """Build the 6-step research workflow graph."""
        graph = StateGraph(ResearchAgentState)

        graph.add_node("expand_query", self._step_expand_query)
        graph.add_node("search", self._step_search)
        graph.add_node("analyze", self._step_analyze)
        graph.add_node("synthesize", self._step_synthesize)
        graph.add_node("evaluate", self._step_evaluate)
        graph.add_node("refine", self._step_refine)

        graph.add_edge(START, "expand_query")
        graph.add_edge("expand_query", "search")
        graph.add_edge("search", "analyze")
        graph.add_edge("analyze", "synthesize")
        graph.add_edge("synthesize", "evaluate")

        def should_refine(state: ResearchAgentState) -> str:
            threshold = 0.75
            max_loops = state["max_refinement_loops"]
            rounds = state["refinement_rounds"]
            score = state["quality_score"]

            if score >= threshold or rounds >= max_loops:
                return END
            return "refine"

        graph.add_conditional_edges("evaluate", should_refine)
        graph.add_edge("refine", "synthesize")

        return graph.compile()

    def _has_loops(self) -> bool:
        """This agent has refinement loops."""
        return True

    # ── Step Methods ───────────────────────────────────────────────────

    def _step_expand_query(self, state: ResearchAgentState) -> dict[str, Any]:
        """Expand the research query into multiple refined queries."""
        out = cast(
            _ExpandQueryOut,
            self._extract(
                response_model=_ExpandQueryOut,
                messages=[
                    {"role": "system", "content": self.step_prompts["expand_query"]},
                    {"role": "user", "content": state["query"]},
                ],
                model=self._model,
            ),
        )
        return {"expanded_queries": out.expanded_queries}

    def _step_search(self, state: ResearchAgentState) -> dict[str, Any]:
        """Search for sources using expanded queries."""
        queries_text = "\n".join(state["expanded_queries"])
        out = cast(
            _SearchOut,
            self._extract(
                response_model=_SearchOut,
                messages=[
                    {"role": "system", "content": self.step_prompts["search"]},
                    {"role": "user", "content": queries_text},
                ],
                model=self._model,
            ),
        )
        return {"search_results": out.results}

    def _step_analyze(self, state: ResearchAgentState) -> dict[str, Any]:
        """Analyze sources for credibility and key findings."""
        results_text = str(state["search_results"][:10])  # Limit to first 10
        out = cast(
            _AnalyzeOut,
            self._extract(
                response_model=_AnalyzeOut,
                messages=[
                    {"role": "system", "content": self.step_prompts["analyze"]},
                    {"role": "user", "content": results_text},
                ],
                model=self._model,
            ),
        )
        return {"analyzed_sources": out.analyzed}

    def _step_synthesize(self, state: ResearchAgentState) -> dict[str, Any]:
        """Synthesize analyzed sources into coherent narrative."""
        sources_text = str(state["analyzed_sources"])
        out = cast(
            _SynthesizeOut,
            self._extract(
                response_model=_SynthesizeOut,
                messages=[
                    {
                        "role": "system",
                        "content": self.step_prompts["synthesize"],
                    },
                    {"role": "user", "content": sources_text},
                ],
                model=self._model,
            ),
        )
        return {"synthesis": out.synthesis, "key_findings": out.key_findings}

    def _step_evaluate(self, state: ResearchAgentState) -> dict[str, Any]:
        """Evaluate synthesis quality."""
        out = cast(
            _EvaluateOut,
            self._extract(
                response_model=_EvaluateOut,
                messages=[
                    {
                        "role": "system",
                        "content": self.step_prompts["evaluate"],
                    },
                    {
                        "role": "user",
                        "content": f"Query: {state['query']}\n\nSynthesis: {state['synthesis']}",
                    },
                ],
                model=self._model,
            ),
        )
        return {
            "quality_score": out.quality_score,
            "refinement_rounds": state["refinement_rounds"],
        }

    def _step_refine(self, state: ResearchAgentState) -> dict[str, Any]:
        """Refine synthesis based on evaluation feedback."""
        out = cast(
            _SynthesizeOut,
            self._extract(
                response_model=_SynthesizeOut,
                messages=[
                    {"role": "system", "content": self.step_prompts["refine"]},
                    {
                        "role": "user",
                        "content": f"Current synthesis: {state['synthesis']}",
                    },
                ],
                model=self._model,
            ),
        )
        return {
            "synthesis": out.synthesis,
            "key_findings": out.key_findings,
            "refinement_rounds": state["refinement_rounds"] + 1,
        }

    # ── Public Interface ───────────────────────────────────────────────

    def run(
        self, input: ResearchAgentInput, _context: AgentContext | None = None
    ) -> AgentResult[ResearchAgentOutput]:
        """Execute the 6-step research workflow."""
        graph = self._get_graph()

        initial_state: ResearchAgentState = {
            "query": input.query,
            "research_focus": input.research_focus,
            "expanded_queries": [],
            "search_results": [],
            "analyzed_sources": [],
            "synthesis": "",
            "quality_score": 0.0,
            "refinement_rounds": 0,
            "max_refinement_loops": input.max_refinement_loops,
        }

        try:
            final_state = graph.invoke(initial_state)
        except Exception as exc:
            return AgentResult[ResearchAgentOutput](
                output=None,
                success=False,
                error=str(exc),
            )

        output = ResearchAgentOutput(
            synthesis=final_state["synthesis"],
            key_findings=final_state.get("key_findings", []),
            quality_score=final_state["quality_score"],
            sources_analyzed=len(final_state.get("analyzed_sources", [])),
            refinement_rounds=final_state["refinement_rounds"],
        )

        return AgentResult(output=output)
