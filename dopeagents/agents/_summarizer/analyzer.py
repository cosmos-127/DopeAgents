"""Text structure analysis for DeepSummarizer (Code step).

Provides a zero-LLM pre-pass over the source document to determine optimal
chunking strategy before any LLM calls are made.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TextAnalysis:
    """Result of analysing a document prior to chunking."""

    recommended_chunk_size: int
    text_type: str  # "structured" | "prose"
    complexity: str  # "high" | "medium"
    paragraph_count: int


def analyze_text(text: str) -> TextAnalysis:
    """Determine optimal chunk size and document type at zero LLM cost.

    Heuristic:
    - < 2 000 chars  → single-chunk (no splitting overhead)
    - < 10 000 chars → 800-char chunks (balanced cost/coverage)
    - ≥ 10 000 chars → 1 500-char chunks (fewer, richer chunks)

    Args:
        text: Source document to analyse.

    Returns:
        TextAnalysis with recommended_chunk_size and structural metadata.
    """
    text_len = len(text)

    if text_len < 2000:
        chunk_size = max(200, text_len)
    elif text_len < 10000:
        chunk_size = 800
    else:
        chunk_size = 1500

    return TextAnalysis(
        recommended_chunk_size=chunk_size,
        text_type="structured" if re.search(r"^#+\s|\n#+\s", text) else "prose",
        complexity="high" if text_len > 10000 else "medium",
        paragraph_count=text.count("\n\n") + 1,
    )
