"""Persistent research memory for multi-session research."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ResearchSession:
    """A single research session's data."""

    session_id: str
    query: str
    timestamp: float
    synthesis: str
    key_findings: list[str]
    sources: list[dict[str, Any]]
    credibility_scores: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    quality_score: float
    information_gaps: list[str]
    follow_up_queries: list[str] = field(default_factory=list)


_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "and",
        "but",
        "or",
        "not",
        "this",
        "that",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "about",
        "between",
        "through",
        "during",
    }
)


def _extract_key_terms(text: str) -> list[str]:
    """Extract key terms for indexing."""
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [w for w in words if w not in _STOPWORDS]


class ResearchMemory:
    """Persists research sessions to disk for follow-up queries.

    Enables:
    - "Tell me more about X" (builds on previous findings)
    - "What about Y?" (uses existing sources + new search)
    - "Update the research on Z" (re-searches with old context)
    """

    def __init__(self, storage_dir: str = ".research_memory"):
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"
        self._index: list[dict[str, Any]] = self._load_index()

    def _load_index(self) -> list[dict[str, Any]]:
        if self._index_path.exists():
            data: list[dict[str, Any]] = json.loads(self._index_path.read_text(encoding="utf-8"))
            return data
        return []

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2), encoding="utf-8")

    def save_session(self, session: ResearchSession) -> None:
        """Persist a research session."""
        path = self._dir / f"{session.session_id}.json"
        data = {
            "session_id": session.session_id,
            "query": session.query,
            "timestamp": session.timestamp,
            "synthesis": session.synthesis,
            "key_findings": session.key_findings,
            "sources": session.sources,
            "credibility_scores": session.credibility_scores,
            "claims": session.claims,
            "quality_score": session.quality_score,
            "information_gaps": session.information_gaps,
            "follow_up_queries": session.follow_up_queries,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        self._index.append(
            {
                "session_id": session.session_id,
                "query": session.query,
                "timestamp": session.timestamp,
                "quality_score": session.quality_score,
                "num_sources": len(session.sources),
                "key_terms": _extract_key_terms(session.query),
            }
        )
        self._save_index()

    def load_session(self, session_id: str) -> ResearchSession | None:
        """Load a previous session by ID."""
        # Sanitize to prevent path traversal
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", session_id)
        path = self._dir / f"{safe_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return ResearchSession(**data)

    def find_related(
        self, query: str, max_results: int = 5, min_similarity: float = 0.3
    ) -> list[dict[str, Any]]:
        """Find previous sessions related to a query using Jaccard similarity."""
        query_terms = set(_extract_key_terms(query))
        if not query_terms:
            return []

        scored: list[dict[str, Any]] = []
        for entry in self._index:
            entry_terms = set(entry.get("key_terms", []))
            if not entry_terms:
                continue
            intersection = query_terms & entry_terms
            union = query_terms | entry_terms
            similarity = len(intersection) / len(union) if union else 0.0

            if similarity >= min_similarity:
                scored.append({**entry, "similarity": round(similarity, 3)})

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:max_results]

    def get_context_for_follow_up(self, query: str) -> dict[str, Any] | None:
        """Build context from previous sessions for a follow-up query."""
        related = self.find_related(query, max_results=3)
        if not related:
            return None

        all_findings: list[str] = []
        all_sources: list[dict[str, Any]] = []
        all_gaps: list[str] = []
        seen_urls: set[str] = set()

        for entry in related:
            session = self.load_session(entry["session_id"])
            if session is None:
                continue

            all_findings.extend(session.key_findings)
            for source in session.sources:
                url = source.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_sources.append(source)
            all_gaps.extend(session.information_gaps)

        return {
            "previous_findings": all_findings,
            "known_sources": all_sources,
            "known_gaps": all_gaps,
            "related_sessions": [
                {"query": e["query"], "similarity": e["similarity"]} for e in related
            ],
        }

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent sessions."""
        sorted_index = sorted(self._index, key=lambda x: x["timestamp"], reverse=True)
        return sorted_index[:limit]
