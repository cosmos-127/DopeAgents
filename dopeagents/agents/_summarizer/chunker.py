"""Paragraph-aware chunker for DeepSummarizer (Code step).

Splits a source document into cost-bounded chunks suitable for independent
per-chunk LLM summarisation, with a hard cap to guard cost and latency budgets.
"""

from __future__ import annotations

import warnings

from dopeagents.agent_utils.text_utils import split_sentences

MAX_CHUNKS: int = 25


class SummarizerChunker:
    """Split a document into semantically coherent, cost-bounded chunks.

    Strategy (applied in priority order):
    1. Split on paragraph boundaries (``\\n\\n``).
    2. Fall back to sentence splitting when no multi-paragraph structure exists.
    3. Merge adjacent segments until *target_size* characters is reached.
    4. Hard cap at :data:`MAX_CHUNKS` to protect cost and latency budgets.
    """

    def chunk(self, text: str, target_size: int) -> list[str]:
        """Produce a list of text chunks from *text*.

        Args:
            text: Source document.
            target_size: Target character count per chunk.

        Returns:
            Non-empty list of chunk strings, capped at :data:`MAX_CHUNKS`.
        """
        segments = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(segments) <= 1:
            segments = split_sentences(text)

        chunks: list[str] = []
        current = ""
        for seg in segments:
            if len(current) + len(seg) > target_size and current:
                chunks.append(current.strip())
                current = seg
            else:
                current = current + "\n\n" + seg if current else seg

        if current.strip():
            chunks.append(current.strip())

        if not chunks:
            return [text]

        if len(chunks) > MAX_CHUNKS:
            warnings.warn(
                f"Chunk count {len(chunks)} exceeds MAX_CHUNKS={MAX_CHUNKS}. Truncating.",
                UserWarning,
                stacklevel=2,
            )
            return chunks[:MAX_CHUNKS]

        return chunks
