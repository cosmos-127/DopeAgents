"""Extract claims from sources and cross-reference them."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A factual claim extracted from a source."""

    claim_text: str = Field(description="The factual claim")
    confidence: float = Field(ge=0.0, le=1.0, description="How confidently stated")
    source_index: int = Field(description="Index of source this came from")
    supporting_quote: str = Field(default="", description="Direct quote supporting the claim")


class ClaimCluster(BaseModel):
    """A group of related claims across sources."""

    representative_claim: str
    supporting_sources: list[int] = Field(default_factory=list)
    contradicting_sources: list[int] = Field(default_factory=list)
    agreement_score: float = Field(
        ge=0.0,
        le=1.0,
        description="1.0 = all sources agree, 0.0 = complete contradiction",
    )
    synthesis_note: str = Field(
        default="",
        description="How to handle this in synthesis (consensus, controversy, etc.)",
    )


class CrossReferenceOutput(BaseModel):
    """Output of cross-referencing step."""

    claim_clusters: list[ClaimCluster] = Field(default_factory=list)
    overall_consensus: float = Field(ge=0.0, le=1.0)
    key_controversies: list[str] = Field(default_factory=list)
    information_gaps: list[str] = Field(default_factory=list)
