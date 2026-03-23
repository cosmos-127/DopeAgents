"""Summary style formatting and truncation for DeepSummarizer (Code step).

Applies the requested output style (paragraph / bullets / tldr) and enforces
the caller-requested maximum character length — all without an LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass

from dopeagents.agent_utils.text_utils import split_sentences


@dataclass
class FormattedSummary:
    """Result of applying style and length constraints to a synthesis."""

    text: str
    word_count: int
    truncated: bool


class SummaryFormatter:
    """Apply output style and length truncation without an LLM call.

    Supported styles:

    ``paragraph``
        Return the synthesis as-is.
    ``bullets``
        Convert *key_points* (or first-pass sentence-split) into a bullet list.
    ``tldr``
        Return only the first sentence of the synthesis.
    """

    def format(
        self,
        synthesis: str,
        style: str,
        max_length: int,
        key_points: list[str],
    ) -> FormattedSummary:
        """Format and optionally truncate *synthesis*.

        Args:
            synthesis: Full synthesis text produced by the LLM.
            style: One of ``"paragraph"``, ``"bullets"``, ``"tldr"``.
            max_length: Maximum character length of the returned text.
            key_points: Pre-extracted key points used by the ``bullets`` style.

        Returns:
            :class:`FormattedSummary` with ``text``, ``word_count``, and
            ``truncated`` flag.
        """
        if style == "bullets":
            points = key_points or split_sentences(synthesis)
            formatted = "\n".join(f"\u2022 {p}" for p in points)
        elif style == "tldr":
            sentences = split_sentences(synthesis)
            formatted = sentences[0] if sentences else synthesis
        else:
            formatted = synthesis

        truncated = len(formatted) > max_length
        if truncated:
            formatted = formatted[:max_length].rsplit(" ", 1)[0] + "\u2026"

        return FormattedSummary(
            text=formatted,
            word_count=len(formatted.split()),
            truncated=truncated,
        )
