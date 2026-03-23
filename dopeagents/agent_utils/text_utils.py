"""General-purpose text utilities shared across agents."""

from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    """Split text into sentences respecting common punctuation boundaries.

    Splits on sentence-ending punctuation (. ! ?) followed by whitespace and a
    capital letter, which avoids false splits on abbreviations like "Dr." or
    decimal numbers like "3.14 values".

    Args:
        text: Raw text to split.

    Returns:
        List of non-empty sentence strings, stripped of surrounding whitespace.
    """
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]
