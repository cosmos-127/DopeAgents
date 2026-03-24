"""Tool definitions and executor for hybrid research agent."""

from __future__ import annotations

import logging
import re

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ToolName(StrEnum):
    """All tools available to the LLM within bounded steps."""

    FACT_CHECK = "fact_check"
    SEARCH_MORE = "search_for_more"
    LOOKUP_CITATION = "lookup_citation"
    GET_FULL_TEXT = "get_full_text"
    COMPARE_CLAIMS = "compare_claims"


@dataclass
class ToolCall:
    """A parsed tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool."""

    tool_call_id: str
    tool_name: str
    result: dict[str, Any]
    success: bool
    error: str | None = None
    tokens_used: int = 0


@dataclass
class ToolBudget:
    """Hard limits on tool usage within a single step.

    This is the KEY mechanism that bounds LLM autonomy.
    Without this, you're back to unbounded ReAct loops.
    """

    max_calls: int = 5
    max_calls_per_tool: dict[str, int] = field(
        default_factory=lambda: {
            "fact_check": 3,
            "search_for_more": 2,
            "lookup_citation": 3,
            "get_full_text": 2,
            "compare_claims": 2,
        }
    )
    calls_made: int = 0
    calls_per_tool: dict[str, int] = field(default_factory=dict)

    def can_call(self, tool_name: str) -> bool:
        if self.calls_made >= self.max_calls:
            return False
        tool_limit = self.max_calls_per_tool.get(tool_name, 2)
        current = self.calls_per_tool.get(tool_name, 0)
        return current < tool_limit

    def record_call(self, tool_name: str) -> None:
        self.calls_made += 1
        self.calls_per_tool[tool_name] = self.calls_per_tool.get(tool_name, 0) + 1

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.calls_made)

    @property
    def exhausted(self) -> bool:
        return self.calls_made >= self.max_calls

    def summary(self) -> dict[str, Any]:
        return {
            "total_calls": self.calls_made,
            "max_calls": self.max_calls,
            "remaining": self.remaining,
            "by_tool": dict(self.calls_per_tool),
        }


# ── Tool Definitions (OpenAI function-calling format) ─────────────────

ANALYSIS_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fact_check",
            "description": (
                "Verify a specific factual claim against Wikipedia and Wikidata. "
                "Use ONLY for surprising, critical, or quantitative claims that "
                "need independent verification. Do NOT use for obvious facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "The specific factual claim to verify",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this claim needs verification",
                    },
                },
                "required": ["claim", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_for_more",
            "description": (
                "Search for additional sources on a specific subtopic where "
                "current sources are insufficient. Use ONLY when you identify "
                "a clear gap that existing sources don't cover."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subtopic": {
                        "type": "string",
                        "description": "The specific subtopic to search for",
                    },
                    "reason": {
                        "type": "string",
                        "description": "What gap this search fills",
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["academic", "web", "any"],
                        "description": "Type of sources to search",
                        "default": "any",
                    },
                },
                "required": ["subtopic", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_citation",
            "description": (
                "Look up full citation details for a paper by DOI or title "
                "via CrossRef. Use when you need proper author/date/journal "
                "info for a source you want to cite."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doi": {"type": "string", "description": "DOI of the paper"},
                    "title": {"type": "string", "description": "Title of the paper"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_full_text",
            "description": (
                "Extract full readable text from a URL that was only seen as a "
                "snippet. Use when a snippet looks highly relevant but you need "
                "more context to extract claims properly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to extract full text from",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why you need the full text",
                    },
                },
                "required": ["url", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_claims",
            "description": (
                "Compare two specific claims from different sources to determine "
                "if they agree, contradict, or are unrelated. Use when you spot "
                "potential contradictions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_a": {"type": "string", "description": "First claim"},
                    "source_a": {"type": "string", "description": "Source of first claim"},
                    "claim_b": {"type": "string", "description": "Second claim"},
                    "source_b": {"type": "string", "description": "Source of second claim"},
                },
                "required": ["claim_a", "source_a", "claim_b", "source_b"],
            },
        },
    },
]


class ToolExecutor:
    """Executes tool calls against real research infrastructure."""

    def __init__(
        self,
        search_engine: Any,
        content_extractor: Any,
        fact_checker: Any,
    ):
        self._search_engine = search_engine
        self._content_extractor = content_extractor
        self._fact_checker = fact_checker

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        handlers: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
            "fact_check": self._handle_fact_check,
            "search_for_more": self._handle_search_more,
            "lookup_citation": self._handle_lookup_citation,
            "get_full_text": self._handle_get_full_text,
            "compare_claims": self._handle_compare_claims,
        }

        handler = handlers.get(tool_call.name)
        if handler is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result={"error": f"Unknown tool: {tool_call.name}"},
                success=False,
                error=f"Unknown tool: {tool_call.name}",
            )

        try:
            result = await handler(**tool_call.arguments)
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=result,
                success=True,
            )
        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_call.name, e)
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result={"error": str(e)},
                success=False,
                error=str(e),
            )

    async def _handle_fact_check(self, claim: str, _reason: str = "") -> dict[str, Any]:
        result = await self._fact_checker.check_claim(claim)
        return {
            "claim": claim,
            "verified": result.verified,
            "confidence": result.confidence,
            "supporting_evidence": result.supporting_evidence[:3],
            "contradicting_evidence": result.contradicting_evidence[:3],
            "sources_checked": result.sources_checked,
        }

    async def _handle_search_more(
        self, subtopic: str, _reason: str = "", source_type: str = "any"
    ) -> dict[str, Any]:
        content_types: list[str] | None = None
        if source_type == "academic":
            content_types = ["academic"]
        elif source_type == "web":
            content_types = ["web", "encyclopedia"]

        results = await self._search_engine.search(
            subtopic,
            max_results_per_provider=2,
            content_types=content_types,
        )
        return {
            "subtopic": subtopic,
            "results_found": len(results),
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet[:300],
                    "source_provider": r.source_provider,
                    "content_type": r.content_type,
                }
                for r in results[:5]
            ],
        }

    async def _handle_lookup_citation(self, doi: str = "", title: str = "") -> dict[str, Any]:
        from dopeagents.agent_utils.search_providers import CrossRefProvider

        provider = CrossRefProvider()
        try:
            query = doi or title
            if not query:
                return {"error": "Need either doi or title"}
            results = await provider.search(query, max_results=1)
            if results:
                r = results[0]
                return {
                    "title": r.title,
                    "authors": r.authors,
                    "doi": r.doi,
                    "url": r.url,
                    "published_date": r.published_date,
                    "citation_count": r.citation_count,
                }
            return {"error": "Citation not found"}
        finally:
            await provider.close()

    async def _handle_get_full_text(self, url: str, _reason: str = "") -> dict[str, Any]:
        result = await self._content_extractor.extract(url)
        if result.success:
            return {
                "url": url,
                "title": result.title,
                "text": result.full_text[:3000],
                "word_count": result.word_count,
            }
        return {"url": url, "error": result.error or "Extraction failed"}

    async def _handle_compare_claims(
        self,
        claim_a: str,
        _source_a: str,
        claim_b: str,
        _source_b: str,
    ) -> dict[str, Any]:
        """Compare two claims using keyword overlap + negation detection."""

        def _tokenize(text: str) -> set[str]:
            return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))

        tokens_a = _tokenize(claim_a)
        tokens_b = _tokenize(claim_b)

        if not tokens_a or not tokens_b:
            return {"relationship": "unrelated", "confidence": 0.0}

        overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

        negation_words = {
            "not",
            "no",
            "never",
            "neither",
            "nor",
            "doesn't",
            "don't",
            "didn't",
            "won't",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "cannot",
            "can't",
            "hardly",
            "unlikely",
            "false",
            "incorrect",
            "wrong",
        }
        neg_a = bool(tokens_a & negation_words)
        neg_b = bool(tokens_b & negation_words)
        negation_mismatch = neg_a != neg_b

        if overlap < 0.15:
            relationship = "unrelated"
        elif negation_mismatch and overlap > 0.3:
            relationship = "contradicting"
        elif overlap > 0.5:
            relationship = "supporting"
        else:
            relationship = "related"

        return {
            "claim_a": claim_a,
            "claim_b": claim_b,
            "relationship": relationship,
            "similarity": round(overlap, 3),
            "negation_mismatch": negation_mismatch,
        }
