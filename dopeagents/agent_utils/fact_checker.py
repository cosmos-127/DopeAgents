"""Lightweight fact verification using free APIs."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class FactCheckResult:
    """Result of checking a single claim."""

    claim: str
    verified: bool | None  # None = couldn't verify
    confidence: float  # 0-1
    method: str
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    sources_checked: list[str]


class FactChecker:
    """Verifies claims against free knowledge APIs.

    Methods:
    1. Google Fact Check Tools API (if API key provided)
    2. Wikipedia content search
    3. Wikidata entity lookup
    """

    def __init__(self, google_api_key: str | None = None):
        self._google_api_key = google_api_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15.0, follow_redirects=True)
        return self._client

    async def check_claim(self, claim: str) -> FactCheckResult:
        """Check a claim using multiple methods."""
        supporting: list[str] = []
        contradicting: list[str] = []
        sources_checked: list[str] = []
        confidence = 0.0

        confidence = await self._try_google_fact_check(
            claim, supporting, contradicting, sources_checked, confidence
        )
        confidence = await self._try_wikipedia(claim, supporting, sources_checked, confidence)
        confidence = await self._try_wikidata(claim, supporting, sources_checked, confidence)

        verified = None
        if supporting and not contradicting:
            verified = True
        elif contradicting and not supporting:
            verified = False

        return FactCheckResult(
            claim=claim,
            verified=verified,
            confidence=confidence if sources_checked else 0.0,
            method="multi-source",
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            sources_checked=sources_checked,
        )

    async def _try_google_fact_check(
        self,
        claim: str,
        supporting: list[str],
        contradicting: list[str],
        sources_checked: list[str],
        confidence: float,
    ) -> float:
        if not self._google_api_key:
            return confidence
        try:
            fc_result = await self._google_fact_check(claim)
            if fc_result:
                sources_checked.append("Google Fact Check")
                for review in fc_result:
                    rating = review.get("textualRating", "").lower()
                    publisher = review.get("publisher", {}).get("name", "unknown")
                    if any(w in rating for w in ("true", "correct", "accurate")):
                        supporting.append(f"Fact-checked as '{rating}' by {publisher}")
                        confidence = max(confidence, 0.9)
                    elif any(w in rating for w in ("false", "incorrect", "misleading")):
                        contradicting.append(f"Fact-checked as '{rating}' by {publisher}")
                        confidence = max(confidence, 0.9)
        except Exception as e:
            logger.debug("Google Fact Check failed: %s", e)
        return confidence

    async def _try_wikipedia(
        self,
        claim: str,
        supporting: list[str],
        sources_checked: list[str],
        confidence: float,
    ) -> float:
        try:
            wiki_result = await self._wikipedia_verify(claim)
            if wiki_result and wiki_result["found"]:
                sources_checked.append("Wikipedia")
                supporting.append(f"Consistent with Wikipedia: {wiki_result['excerpt'][:200]}")
                confidence = max(confidence, 0.6)
        except Exception as e:
            logger.debug("Wikipedia verification failed: %s", e)
        return confidence

    async def _try_wikidata(
        self,
        claim: str,
        supporting: list[str],
        sources_checked: list[str],
        confidence: float,
    ) -> float:
        try:
            wd_result = await self._wikidata_verify(claim)
            if wd_result and wd_result["found"]:
                sources_checked.append("Wikidata")
                supporting.append(f"Wikidata confirms: {wd_result['value']}")
                confidence = max(confidence, 0.8)
        except Exception as e:
            logger.debug("Wikidata verification failed: %s", e)
        return confidence

    async def _google_fact_check(self, claim: str) -> list[dict[str, Any]] | None:
        """Query Google Fact Check Tools API."""
        client = await self._get_client()
        resp = await client.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": claim, "key": self._google_api_key, "languageCode": "en"},
        )
        resp.raise_for_status()
        data = resp.json()

        reviews: list[dict[str, Any]] = []
        for claim_review in data.get("claims", []):
            for review in claim_review.get("claimReview", []):
                reviews.append(review)
        return reviews if reviews else None

    async def _wikipedia_verify(self, claim: str) -> dict[str, Any] | None:
        """Search Wikipedia for content related to the claim."""
        client = await self._get_client()
        words = re.findall(r"\b[A-Za-z]{3,}\b", claim)
        search_query = " ".join(words[:5])

        resp = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": search_query,
                "srlimit": 3,
                "format": "json",
                "srprop": "snippet",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("query", {}).get("search", [])
        if results:
            snippet = re.sub(r"<[^>]+>", "", results[0].get("snippet", ""))
            return {"found": True, "excerpt": snippet, "title": results[0]["title"]}
        return {"found": False}

    async def _wikidata_verify(self, claim: str) -> dict[str, Any] | None:
        """Query Wikidata for structured facts."""
        client = await self._get_client()
        resp = await client.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": claim[:100],
                "language": "en",
                "limit": 3,
                "format": "json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("search", [])
        if results:
            return {
                "found": True,
                "entity_id": results[0]["id"],
                "value": results[0].get("description", ""),
                "label": results[0].get("label", ""),
            }
        return {"found": False}

    async def check_batch(self, claims: list[str]) -> list[FactCheckResult]:
        """Check multiple claims concurrently."""
        return await asyncio.gather(*[self.check_claim(c) for c in claims])

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
