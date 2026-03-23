"""Evidence-based confidence scoring (not LLM self-evaluation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ConfidenceBreakdown:
    """Detailed confidence assessment grounded in measurable signals."""

    overall: float  # 0-1
    source_diversity: float
    source_credibility: float
    claim_coverage: float
    claim_agreement: float
    extraction_success_rate: float
    components: dict[str, float]
    explanation: str


class ConfidenceCalculator:
    """Calculate research confidence from measurable signals.

    Replaces LLM self-evaluation with ground-truth metrics.
    The LLM evaluation step is still useful for coherence/readability,
    but this gives an objective quality floor.
    """

    @staticmethod
    def _claim_agreement_score(
        claim_clusters: list[dict[str, Any]],
    ) -> float:
        """Compute claim agreement, penalising highly controversial sets."""
        if not claim_clusters:
            return 0.0
        scores = [c.get("agreement_score", 0.5) for c in claim_clusters]
        agreement = sum(scores) / len(scores)
        controversial = sum(1 for a in scores if a < 0.4)
        if controversial > len(scores) * 0.5:
            agreement *= 0.8
        return agreement  # type: ignore[no-any-return]

    @staticmethod
    def _build_explanation(
        overall: float,
        source_diversity: float,
        source_credibility: float,
        extraction_rate: float,
        claim_agreement: float,
        coverage: float,
        gap_count: int,
    ) -> str:
        """Create a human-readable explanation of confidence."""
        weak: list[str] = []
        if source_diversity < 0.5:
            weak.append("limited source diversity")
        if source_credibility < 0.5:
            weak.append("low average source credibility")
        if extraction_rate < 0.5:
            weak.append("many sources couldn't be fully extracted")
        if claim_agreement < 0.5:
            weak.append("significant disagreement between sources")
        if coverage < 0.5:
            weak.append(f"{gap_count} information gaps identified")
        if weak:
            return f"Confidence limited by: {'; '.join(weak)}"
        if overall >= 0.8:
            return "High confidence: diverse, credible sources with strong agreement"
        return "Moderate confidence: reasonable source coverage"

    def calculate(
        self,
        search_results: list[dict[str, Any]],
        extracted_content: list[dict[str, Any]],
        credibility_scores: list[dict[str, Any]],
        claim_clusters: list[dict[str, Any]],
        information_gaps: list[str],
        claims: list[dict[str, Any]] | None = None,
    ) -> ConfidenceBreakdown:
        """Calculate confidence from research artifacts."""
        components: dict[str, float] = {}

        # 1. Source diversity (different providers and domains)
        providers = {r.get("source_provider", "") for r in search_results}
        domains = {r.get("domain", "") for r in search_results}
        provider_diversity = min(len(providers) / 4.0, 1.0)
        domain_diversity = min(len(domains) / 8.0, 1.0)
        source_diversity = (provider_diversity + domain_diversity) / 2
        components["provider_diversity"] = round(provider_diversity, 3)
        components["domain_diversity"] = round(domain_diversity, 3)

        # 2. Source credibility
        if credibility_scores:
            cred_values = [s.get("overall", 0.5) for s in credibility_scores]
            avg_cred = sum(cred_values) / len(cred_values)
            high_cred_bonus = min(sum(1 for c in cred_values if c >= 0.7) / 3.0, 0.2)
            source_credibility = min(avg_cred + high_cred_bonus, 1.0)
        else:
            source_credibility = 0.0
        components["avg_credibility"] = round(source_credibility, 3)

        # 3. Extraction success rate
        total_results = len(search_results)
        successful = sum(1 for e in extracted_content if e.get("success", False))
        extraction_rate = successful / total_results if total_results > 0 else 0.0
        components["extraction_success"] = round(extraction_rate, 3)

        # 4. Claim agreement
        claim_agreement = self._claim_agreement_score(claim_clusters)
        components["claim_agreement"] = round(claim_agreement, 3)

        # 5. Coverage (inverse of gap count)
        gap_count = len(information_gaps)
        coverage = max(0.0, 1.0 - (gap_count * 0.15))
        components["coverage"] = round(coverage, 3)

        # 6. Volume adequacy (use sources that produced claims, not just fetched)
        if claims is not None:
            analyzed_urls = {c.get("source_url") for c in claims if c.get("source_url")}
            # Fall back to total search results if no claim has a source_url populated
            if analyzed_urls:
                volume_score = min(len(analyzed_urls) / 10.0, 1.0)
            else:
                volume_score = min(len(search_results) / 10.0, 1.0)
        else:
            volume_score = min(len(search_results) / 10.0, 1.0)
        components["source_volume"] = round(volume_score, 3)

        # Weighted overall
        overall = (
            source_diversity * 0.15
            + source_credibility * 0.25
            + extraction_rate * 0.10
            + claim_agreement * 0.20
            + coverage * 0.20
            + volume_score * 0.10
        )

        explanation = self._build_explanation(
            overall,
            source_diversity,
            source_credibility,
            extraction_rate,
            claim_agreement,
            coverage,
            gap_count,
        )

        return ConfidenceBreakdown(
            overall=round(overall, 3),
            source_diversity=round(source_diversity, 3),
            source_credibility=round(source_credibility, 3),
            claim_coverage=round(coverage, 3),
            claim_agreement=round(claim_agreement, 3),
            extraction_success_rate=round(extraction_rate, 3),
            components=components,
            explanation=explanation,
        )
