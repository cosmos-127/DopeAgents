"""
DeepSummarizer — Real-world usage example.

This comprehensive test case demonstrates production usage combining:
  • Custom step prompts for legal domain specialization
  • Focus-guided summarization to highlight key risks / compliance concerns
  • AgentExecutor with full observability, tracing, and budget controls
  • Advanced result inspection with quality metrics and execution metrics
  • Retry policies and error handling
"""

from dopeagents.agents import DeepSummarizer, DeepSummarizerInput
from dopeagents.config import get_config
from dopeagents.cost.guard import BudgetConfig
from dopeagents.lifecycle.executor import AgentExecutor
from dopeagents.observability.tracer import ConsoleTracer
from dopeagents.resilience.retry import RetryPolicy

_cfg = get_config()
if not _cfg.has_api_key():
    raise RuntimeError(
        "No API key found. Set NVIDIA_NIM_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, "
        "or TOGETHER_API_KEY in your .env file or environment before running these examples."
    )

# Sample document used in all examples
ARTICLE = """
            📄 Comprehensive Article: Legal, Operational, and Regulatory Implications of Large Language Models
            1. Introduction

            Large Language Models (LLMs) have rapidly evolved from research prototypes into critical infrastructure powering modern software systems. Organizations across industries are integrating LLMs into products for customer support, legal drafting, code generation, healthcare triage, and financial advisory workflows.

            However, this rapid adoption introduces a complex web of legal liabilities, compliance requirements, operational risks, and governance challenges that are still poorly understood and inconsistently regulated across jurisdictions.

            This document provides a detailed exploration of these concerns, focusing on real-world deployment risks, regulatory exposure, and organizational responsibilities.

            2. Training Data and Intellectual Property Risks
            2.1 Data Provenance Issues

            LLMs are trained on vast corpora of publicly available and proprietary data, often scraped from the internet without explicit consent.

            This creates several legal risks:

            Copyright infringement
            Training on copyrighted material without licenses may violate intellectual property laws.
            Lack of attribution
            Models may reproduce content similar to training data without proper attribution.
            Unclear ownership of outputs
            It is often ambiguous whether generated outputs are derivative works.
            2.2 Litigation Exposure

            Organizations deploying LLMs may face:

            Lawsuits from content creators
            Class-action claims related to unauthorized data usage
            Regulatory scrutiny over dataset transparency
            2.3 Compliance Obligations

            Emerging regulations (e.g., EU AI Act) may require:

            Disclosure of training data sources
            Documentation of data filtering processes
            Mechanisms for data removal (“right to be forgotten”)
            3. Model Behavior and Liability
            3.1 Hallucinations and Misinformation

            LLMs can generate:

            Factually incorrect information
            Fabricated citations
            Misleading or outdated advice

            Legal implications:

            Liability for damages caused by incorrect outputs
            Professional malpractice risks (e.g., legal or medical advice)
            3.2 Lack of Determinism

            Outputs vary across runs, making:

            Auditing difficult
            Reproducibility challenging
            Legal accountability अस्पष्ट (unclear)
            3.3 Responsibility Attribution

            Key question:

            Who is liable — the model provider, the deployer, or the end user?

            This remains unresolved in most jurisdictions.

            4. Safety, Alignment, and Misuse Risks
            4.1 Jailbreaks and Prompt Injection

            Users can manipulate models to:

            Bypass safety filters
            Generate harmful or restricted content
            Leak system prompts or sensitive data
            4.2 Malicious Use Cases

            LLMs can be exploited for:

            Phishing and social engineering
            Malware generation
            Disinformation campaigns
            4.3 Alignment Limitations

            Despite techniques like RLHF and constitutional AI:

            Models can still exhibit bias
            Safety constraints are not guaranteed
            Edge cases remain unpredictable
            5. Deployment and Operational Risks
            5.1 Cost and Resource Constraints

            Training and deploying LLMs involves:

            High compute costs (millions of dollars)
            Dependency on specialized hardware (GPUs/TPUs)
            Vendor lock-in risks
            5.2 Inference Risks

            Even after deployment:

            Latency variability affects user experience
            Scaling costs increase unpredictably
            Failures can cascade across systems
            5.3 Third-Party Dependencies

            Using API-based models introduces:

            Dependency on external providers
            Risk of outages or API changes
            Data-sharing concerns
            6. Data Privacy and Security
            6.1 Sensitive Data Leakage

            LLMs may unintentionally expose:

            Personally identifiable information (PII)
            Confidential business data
            Training data artifacts
            6.2 Regulatory Compliance

            Organizations must comply with:

            GDPR (Europe)
            HIPAA (Healthcare)
            CCPA (California)

            Key requirements include:

            Data minimization
            User consent
            Secure storage and processing
            6.3 Cross-Border Data Transfer

            Cloud-based LLMs may process data across jurisdictions, creating:

            Legal conflicts
            Data sovereignty issues
            7. Governance and Auditability
            7.1 Lack of Transparency

            LLMs are often:

            Black-box systems
            Difficult to interpret
            Hard to explain in legal contexts
            7.2 Audit Challenges

            Organizations struggle with:

            Tracking model decisions
            Logging inputs/outputs
            Demonstrating compliance
            7.3 Required Controls

            Best practices include:

            Model monitoring systems
            Human-in-the-loop review
            Audit trails and logging
            8. Regulatory Landscape
            8.1 Emerging Regulations

            Governments are introducing AI-specific laws:

            Risk-based classification systems
            Mandatory risk assessments
            Transparency requirements
            8.2 High-Risk Use Cases

            Applications in:

            Healthcare
            Finance
            Legal decision-making

            …are subject to stricter scrutiny.

            8.3 Penalties for Non-Compliance

            Organizations may face:

            Heavy fines
            Operational restrictions
            Reputational damage
            9. Risk Mitigation Strategies
            9.1 Technical Safeguards
            Output filtering
            Monitoring and anomaly detection
            Red-teaming and adversarial testing
            9.2 Organizational Measures
            Clear usage policies
            Employee training
            Incident response plans
            9.3 Legal Protections
            Terms of service disclaimers
            Liability limitations
            Insurance coverage
            10. Future Outlook

            As LLMs continue to evolve:

            Regulations will become stricter
            Liability frameworks will mature
            Standardization will increase

            However, organizations that fail to proactively address risks may face:

            Legal exposure
            Financial losses
            Loss of user trust
            11. Conclusion

            LLMs offer transformative potential but introduce significant legal and operational challenges. Responsible deployment requires:

            Rigorous risk assessment
            Strong governance frameworks
            Continuous monitoring and adaptation

            Organizations must treat LLMs not just as tools, but as high-risk systems requiring structured oversight and accountability.
        """


# ===========================================================================
# REAL-WORLD SCENARIO: Comprehensive Legal Document Analysis
#
# This test case demonstrates production usage combining:
#   • Custom step prompts for legal domain specialization
#   • Focus-guided summarization to highlight key risks
#   • AgentExecutor with full observability & budget controls
#   • Advanced result inspection with metrics & tracing
# ===========================================================================

print("=" * 70)
print("REAL-WORLD SCENARIO: Legal Document Analysis")
print("=" * 70)
print("\nContext: Analyzing LLM documentation for liability & compliance risks\n")

# Step 1: Configure legal domain agent with custom prompts
legal_summarizer = DeepSummarizer(
    step_prompts={
        "analyze": (
            "Analyze this document for legal and compliance concerns. "
            "Identify sections discussing liabilities, obligations, risks, and regulatory impacts."
        ),
        "summarize": (
            "Summarize each chunk focusing on: (1) liability exposure, (2) deployment risks, "
            "(3) compliance obligations, and (4) safety/alignment concerns."
        ),
        "synthesize": (
            "You are a legal analyst synthesizing liability assessment. Create a comprehensive "
            "legal brief that: (1) identifies key liabilities and obligations, (2) flags deployment "
            "and compliance risks, (3) summarizes safety considerations, and (4) highlights regulatory "
            "gaps. Use clear, actionable language for both legal and technical audiences."
        ),
        "evaluate": (
            "Evaluate this legal brief on three dimensions:\n"
            "  • LEGAL_ACCURACY (0-1): Correctness of liability & obligation interpretation\n"
            "  • RISK_COMPLETENESS (0-1): Coverage of compliance, safety, and deployment risks\n"
            "  • ACTIONABILITY (0-1): Clarity and usefulness for legal/compliance teams\n"
            "Return a weighted quality_score combining these three metrics (weights: 0.4, 0.4, 0.2)"
        ),
    }
)

# Step 2: Configure AgentExecutor with observability & budget controls
executor = AgentExecutor(
    tracer=ConsoleTracer(),
    budget=BudgetConfig(
        max_cost_per_call=0.50,
        max_refinement_loops=2,
        on_exceeded="error",
    ),
)

# Step 3: Execute with focus on liability & compliance risks
print("\n→ Analyzing document for liability, compliance, and safety risks...\n")

result = executor.run(
    agent=legal_summarizer,
    input=DeepSummarizerInput(
        text=ARTICLE,
        style="bullets",
        max_length=700,
        # Focus on legal/business-critical aspects
        focus="training costs, liability exposure, safety risks, deployment considerations, and regulatory concerns",
        quality_threshold=0.85,
    ),
    retry_policy=RetryPolicy(
        max_attempts=2,
        delay_seconds=1.0,
        backoff_factor=2.0,
        retryable_errors=[TimeoutError, ConnectionError],
    ),
)

# Step 4: Display comprehensive results
print("\n" + "=" * 70)
print("LEGAL ANALYSIS RESULTS")
print("=" * 70)

if result.success:
    out = result.output
    assert out is not None

    print("\n📋 EXECUTIVE SUMMARY")
    print("-" * 70)
    print(out.summary)

    if out.key_points:
        print("\n🔑 KEY RISK FACTORS")
        print("-" * 70)
        for i, point in enumerate(out.key_points, 1):
            print(f"  {i}. {point}")

    print("\n📊 QUALITY METRICS")
    print("-" * 70)
    print(f"  Quality Score:     {out.quality_score:.2f}/1.00   (target: ≥0.85)")
    print(f"  Word Count:        {out.word_count} words")
    print(f"  Chunks Processed:  {out.chunks_processed}")
    print(f"  Truncated:         {'Yes' if out.truncated else 'No'}")
    print(f"  Refinement Loops:  {out.refinement_rounds}")

    if result.metrics:
        m = result.metrics
        print("\n⚡ PERFORMANCE METRICS")
        print("-" * 70)
        print(f"  Total Latency:     {m.latency_ms:.0f} ms")
        print(f"  Cost:              ${m.cost_usd:.6f} USD")
        print(f"  Cache Hit:         {m.cache_hit}")

    print("\n✅ Analysis complete. Ready for legal review & compliance assessment.")

else:
    print(f"\n❌ ERROR: {result.error}")
