"""Domain-based and signal-based credibility scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlparse

# Domain authority tiers (curated list)
DOMAIN_TIERS: dict[str, float] = {
    # Tier 1: Primary sources, major institutions
    "nature.com": 0.95,
    "science.org": 0.95,
    "thelancet.com": 0.95,
    "nejm.org": 0.95,
    "ieee.org": 0.90,
    "acm.org": 0.90,
    "arxiv.org": 0.85,
    "nih.gov": 0.95,
    "cdc.gov": 0.90,
    "who.int": 0.90,
    "un.org": 0.85,
    # Tier 2: Quality journalism and reference
    "reuters.com": 0.85,
    "apnews.com": 0.85,
    "nytimes.com": 0.80,
    "bbc.com": 0.80,
    "bbc.co.uk": 0.80,
    "theguardian.com": 0.75,
    "washingtonpost.com": 0.75,
    "en.wikipedia.org": 0.70,
    # Tier 3: General reference
    "stackoverflow.com": 0.65,
    "github.com": 0.60,
    # Tier 4: User-generated / lower editorial standards
    "medium.com": 0.40,
    "reddit.com": 0.35,
    "quora.com": 0.30,
    "blogspot.com": 0.25,
}

# TLD quality signals
TLD_SCORES: dict[str, float] = {
    ".edu": 0.80,
    ".gov": 0.85,
    ".org": 0.60,
    ".ac.uk": 0.80,
    ".edu.au": 0.80,
}


@dataclass
class CredibilityScore:
    """Detailed credibility assessment."""

    overall: float  # 0-1
    domain_authority: float
    recency_score: float
    citation_score: float
    content_type_score: float
    signals: list[str] = field(default_factory=list)


def score_credibility(  # noqa: C901
    url: str,
    published_date: str | None = None,
    citation_count: int | None = None,
    content_type: str = "web",
    has_author: bool = False,
    word_count: int = 0,
) -> CredibilityScore:
    """Score credibility based on available signals."""
    signals: list[str] = []
    parsed = urlparse(url)
    domain = parsed.netloc.lower().removeprefix("www.")

    # 1. Domain authority
    domain_score = DOMAIN_TIERS.get(domain, 0.50)
    for tld, tld_score in TLD_SCORES.items():
        if domain.endswith(tld):
            domain_score = max(domain_score, tld_score)
            signals.append(f"Trusted TLD: {tld}")
            break

    if domain in DOMAIN_TIERS:
        signals.append(f"Known domain: {domain} (score={domain_score})")

    # 2. Recency
    recency_score = 0.5
    if published_date:
        try:
            year = int(published_date[:4])
            current_year = datetime.now(UTC).year
            age = current_year - year
            if age <= 1:
                recency_score = 1.0
                signals.append("Published within last year")
            elif age <= 3:
                recency_score = 0.8
            elif age <= 5:
                recency_score = 0.6
            elif age <= 10:
                recency_score = 0.4
            else:
                recency_score = 0.2
                signals.append(f"Published {age} years ago")
        except (ValueError, TypeError):
            pass

    # 3. Citation count (academic)
    citation_score = 0.5
    if citation_count is not None:
        if citation_count >= 100:
            citation_score = 1.0
            signals.append(f"Highly cited ({citation_count} citations)")
        elif citation_count >= 20:
            citation_score = 0.8
        elif citation_count >= 5:
            citation_score = 0.6
        else:
            citation_score = 0.4

    # 4. Content type
    content_type_scores = {
        "academic": 0.85,
        "encyclopedia": 0.70,
        "news": 0.60,
        "web": 0.50,
    }
    ct_score = content_type_scores.get(content_type, 0.50)

    # 5. Additional signals
    if has_author:
        signals.append("Has named author(s)")
    if word_count > 1000:
        signals.append("Substantial content (>1000 words)")

    # Weighted average
    weights = {
        "domain": 0.35,
        "recency": 0.20,
        "citation": 0.20,
        "content_type": 0.15,
        "author": 0.10,
    }
    overall = (
        domain_score * weights["domain"]
        + recency_score * weights["recency"]
        + citation_score * weights["citation"]
        + ct_score * weights["content_type"]
        + (0.7 if has_author else 0.3) * weights["author"]
    )

    return CredibilityScore(
        overall=round(overall, 3),
        domain_authority=domain_score,
        recency_score=recency_score,
        citation_score=citation_score,
        content_type_score=ct_score,
        signals=signals,
    )
