"""
DeepResearcher — Production-Grade Real-World Test Case.

This comprehensive test case demonstrates production usage combining:
  • Custom step prompts for technical domain specialization (Infrastructure/DevOps)
  • Research focus with specific outcome targets
  • Hybrid mode with intelligent tool calling configuration
  • AgentExecutor with full observability, tracing, and budget controls
  • Advanced result inspection with confidence metrics and source analysis
  • Retry policies and error handling for robust execution
  • Memory persistence for session continuity

Use case: Comprehensive research on distributed systems design patterns, deployment
challenges, and emerging best practices for enterprise infrastructure teams.
"""

import sys

from dopeagents.agents import DeepResearcher, DeepResearcherInput
from dopeagents.agents._researcher.report_generator import ReportFormat
from dopeagents.config import get_config
from dopeagents.cost.guard import BudgetConfig
from dopeagents.lifecycle.executor import AgentExecutor
from dopeagents.observability.tracer import ConsoleTracer
from dopeagents.resilience.retry import RetryPolicy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_cfg = get_config()
if not _cfg.has_api_key():
    raise RuntimeError(
        "No API key found. Set NVIDIA_NIM_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, "
        "or TOGETHER_API_KEY in your .env file or environment before running these examples."
    )


# ===========================================================================
# REAL-WORLD SCENARIO: Enterprise Distributed Systems Architecture Research
#
# This test case demonstrates production usage combining:
#   • Custom step prompts for infrastructure domain specialization
#   • Hybrid mode with LLM-driven tool calling for deep analysis
#   • AgentExecutor with full observability, tracing, and cost controls
#   • Advanced result inspection with confidence breakdown & source analysis
#   • Retry policies for resilience & error recovery
#   • Memory persistence for session continuity
# ===========================================================================

print("=" * 80)
print("REAL-WORLD SCENARIO: Enterprise Distributed Systems Architecture Research")
print("=" * 80)
print(
    "\nContext: Technical research on scaling challenges, consensus patterns, and\n"
    "deployment best practices for mission-critical infrastructure systems.\n"
)

# Step 1: Configure infrastructure domain agent with custom specialization prompts
infrastructure_researcher = DeepResearcher(
    memory_dir=".research_memory",
    step_prompts={
        "expand_query": (
            "You are a senior infrastructure architect researching distributed systems. "
            "Expand this query to explore: (1) consensus and coordination mechanisms, "
            "(2) scaling patterns and bottleneck mitigation, (3) failure modes and resilience strategies, "
            "(4) operational complexity and monitoring requirements, and (5) production deployment lessons."
        ),
        "deep_analysis": (
            "Analyze sources from an infrastructure perspective. Focus on: "
            "(1) Architecture patterns (primary-backup, quorum, eventual consistency), "
            "(2) Performance & scalability trade-offs (consistency guarantees, latency, throughput), "
            "(3) Operational challenges (complexity, debugging, fault isolation), "
            "(4) Production examples and real failure stories, and "
            "(5) Evaluation of claimed benefits vs. actual deployment complexity. "
            "Use fact_check strategically for the most critical claims: system throughput benchmarks, "
            "failure recovery SLAs, and hard consistency guarantees. Skip obvious or well-established facts. "
            "Flag any unsubstantiated performance claims."
        ),
        "synthesize": (
            "Synthesize findings into an architecture guide for infrastructure teams. Structure as: "
            "1. DESIGN DECISIONS — key patterns with pro/con trade-offs (consistency, availability, partition handling) "
            "2. SCALABILITY ANALYSIS — throughput, latency, and cost characteristics; measured data preferred "
            "3. OPERATIONAL REALITY — complexity assessment, monitoring requirements, failure recovery procedures "
            "4. DEPLOYMENT GUIDANCE — lessons from production systems, anti-patterns, and maturity requirements "
            "5. RECOMMENDATIONS — when to adopt vs. avoid, suitable team sizes, and prerequisite expertise. "
            "Target: actionable for teams deciding whether a pattern fits their constraints."
        ),
        "evaluate": (
            "Evaluate this architecture guide on four dimensions:\n"
            "  • TECHNICAL_ACCURACY (0-1): Correctness of pattern descriptions and trade-off analysis\n"
            "  • COMPLETENESS (0-1): Coverage of scalability, operational, and deployment aspects\n"
            "  • PRAGMATISM (0-1): Realism of guidance; real-world applicability vs. theoretical\n"
            "  • ACTIONABILITY (0-1): Clarity for decision-making; specificity of recommendations\n"
            "Return weighted quality_score: 0.3*accuracy + 0.35*completeness + 0.2*pragmatism + 0.15*actionability"
        ),
    },
)

# Step 2: Configure AgentExecutor with full observability and budget controls
executor = AgentExecutor(
    tracer=ConsoleTracer(),  # prints detailed execution spans
    budget=BudgetConfig(
        max_cost_per_call=1.50,  # higher budget for comprehensive research
        max_refinement_loops=3,  # allow quality refinement
        on_exceeded="error",  # fail hard if budget exceeded
    ),
)

# Step 3: Execute comprehensive research with hybrid tool calling
print("\n→ Researching distributed systems architecture patterns and deployment reality...\n")

result = executor.run(
    agent=infrastructure_researcher,
    input=DeepResearcherInput(
        query=(
            """What are the reliability lessons from the 2023-2024 major infrastructure outages
            (AWS, Google Cloud, Azure, Cloudflare)? What specific architectural or operational decisions led to cascading failures?
            For each incident, what was the recovery time, root cause, and what design patterns would have prevented it?
            Give a detailed analysis of the failure modes and how they relate to consensus mechanisms, scaling patterns, and operational complexity."""
        ),
        research_focus="comprehensive",
        report_format=ReportFormat.ACADEMIC,
        # Hybrid mode: use tool calling intelligently for analysis
        enable_tool_calling=None,  # auto-detect from model capability
        tool_budget=5,  # 5 tool calls; medium-tier models are internally capped at 3
        # Quality targets
        quality_threshold=0.80,  # aim for high-quality synthesis
        max_refinement_loops=3,  # allow refinement if below threshold
        # Memory for continuity
        enable_memory=True,
    ),
    retry_policy=RetryPolicy(
        max_attempts=3,
        delay_seconds=2.0,
        backoff_factor=2.0,  # 2s → 4s → 8s
        retryable_errors=[TimeoutError, ConnectionError],
    ),
)

# Step 4: Display comprehensive results with quality and performance metrics
print("\n" + "=" * 80)
print("RESEARCH RESULTS: DISTRIBUTED SYSTEMS ARCHITECTURE")
print("=" * 80)

if result.success:
    out = result.output
    assert out is not None

    print("\n📋 RESEARCH OVERVIEW")
    print("-" * 80)
    print(f"Title:          {out.report_title}")
    print(f"Session ID:     {out.session_id}")
    print(f"Duration:       {out.total_duration_seconds:.1f} seconds")

    print("\n🔍 SYNTHESIS & KEY FINDINGS")
    print("-" * 80)
    print(out.synthesis)

    if out.key_findings:
        print("\n✓ KEY FINDINGS")
        print("-" * 80)
        for i, finding in enumerate(out.key_findings, 1):
            print(f"  {i}. {finding}")

    print("\n📊 CONFIDENCE & QUALITY METRICS")
    print("-" * 80)
    print(f"  Grounded Confidence     : {out.confidence:.2f}/1.00")
    print("    └─ Based on source count, agreement, and recency")
    print(f"  LLM Quality Score       : {out.llm_quality_score:.2f}/1.00")
    print("    └─ Self-evaluation across accuracy/completeness/pragmatism")
    if out.confidence_breakdown:
        print(f"  Breakdown: {out.confidence_breakdown}")

    print("\n📚 SOURCE ANALYSIS")
    print("-" * 80)
    print(f"  Sources Analyzed        : {out.sources_analyzed}")
    if out.source_breakdown:
        print(f"  Source Breakdown        : {out.source_breakdown}")
    if out.credibility_summary:
        print(f"  Credibility Summary     : {out.credibility_summary}")

    print("\n🔗 VERIFIED CLAIMS & CITATIONS")
    print("-" * 80)
    print(f"  Verified Claims         : {len(out.verified_claims)}")
    if out.claim_clusters:
        print(f"  Claim Clusters          : {len(out.claim_clusters)} clusters identified")
    if out.information_gaps:
        print(f"  Information Gaps        : {out.information_gaps}")

    print("\n⚙️ HYBRID MODE & TOOL USAGE")
    print("-" * 80)
    print(f"  Hybrid Mode Enabled     : {out.hybrid_mode}")
    print(f"  Model Tier              : {out.model_tier}")
    print(f"  Tool Calls Used         : {out.tool_usage}")
    if out.tool_insights:
        print(f"  Tool Insights           : {out.tool_insights}")

    print("\n🔄 REFINEMENT & MEMORY")
    print("-" * 80)
    print(f"  Refinement Loops        : {out.refinement_rounds}")

    if result.metrics:
        m = result.metrics
        print("\n⚡ EXECUTION PERFORMANCE")
        print("-" * 80)
        print(f"  Total Latency           : {m.latency_ms:.0f} ms")
        print(f"  Estimated Cost          : ${m.cost_usd:.6f} USD")
        print(f"  Cache Hit               : {m.cache_hit}")

    print("\n✅ Research complete. Ready for architecture decision-making & team planning.")

else:
    print(f"\n❌ ERROR: {result.error}")
