"""Chunk extracted content and rank by relevance to the query."""

from __future__ import annotations

import math
import re

from collections import Counter
from dataclasses import dataclass


@dataclass
class TextChunk:
    """A chunk of text from a source."""

    text: str
    source_url: str
    source_title: str
    chunk_index: int
    word_count: int
    relevance_score: float = 0.0


class SemanticChunker:
    """Chunks long text into semantically meaningful segments.

    Strategies (applied in order):
    1. Section-based (split on headings)
    2. Paragraph-based (split on double newlines)
    3. Sliding window with overlap for oversized segments
    """

    def __init__(
        self,
        target_chunk_words: int = 300,
        overlap_words: int = 50,
        min_chunk_words: int = 50,
    ):
        self._target = target_chunk_words
        self._overlap = overlap_words
        self._min = min_chunk_words

    def chunk(
        self,
        text: str,
        source_url: str = "",
        source_title: str = "",
    ) -> list[TextChunk]:
        """Chunk text into meaningful segments."""
        sections = self._split_by_sections(text)
        if len(sections) <= 1:
            sections = self._split_by_paragraphs(text)

        chunks: list[str] = []
        for section in sections:
            words = section.split()
            if len(words) <= self._target * 1.5:
                if len(words) >= self._min:
                    chunks.append(section)
            else:
                for i in range(0, len(words), self._target - self._overlap):
                    window = " ".join(words[i : i + self._target])
                    if len(window.split()) >= self._min:
                        chunks.append(window)

        return [
            TextChunk(
                text=chunk,
                source_url=source_url,
                source_title=source_title,
                chunk_index=i,
                word_count=len(chunk.split()),
            )
            for i, chunk in enumerate(chunks)
        ]

    def _split_by_sections(self, text: str) -> list[str]:
        """Split by markdown-style headings or ALL CAPS headings."""
        pattern = r"(?=\n#{1,3}\s|\n[A-Z][A-Z\s]{5,}\n)"
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if p.strip()]

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Split by double newlines."""
        parts = re.split(r"\n\s*\n", text)
        return [p.strip() for p in parts if p.strip()]


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
        "being",
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
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "because",
        "if",
        "when",
        "where",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "their",
        "we",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric, remove stopwords."""
    tokens = re.findall(r"\b[a-z][a-z0-9]{2,}\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


class RelevanceRanker:
    """Rank chunks by relevance to the query using TF-IDF-like scoring."""

    def rank(self, query: str, chunks: list[TextChunk]) -> list[TextChunk]:
        """Rank chunks by relevance to query using term overlap scoring."""
        query_terms = set(_tokenize(query))
        if not query_terms:
            return chunks

        doc_freq: Counter[str] = Counter()
        chunk_tokens: list[set[str]] = []
        for chunk in chunks:
            tokens = set(_tokenize(chunk.text))
            chunk_tokens.append(tokens)
            for token in tokens:
                doc_freq[token] += 1

        n_docs = len(chunks)

        for i, chunk in enumerate(chunks):
            tokens = chunk_tokens[i]
            score = 0.0
            for term in query_terms:
                if term in tokens:
                    df = doc_freq.get(term, 1)
                    idf = math.log(n_docs / df) + 1.0
                    score += idf
            chunk.relevance_score = score / len(query_terms)

        chunks.sort(key=lambda c: c.relevance_score, reverse=True)
        return chunks

    def top_k(self, query: str, chunks: list[TextChunk], k: int = 10) -> list[TextChunk]:
        """Return the top-k most relevant chunks."""
        ranked = self.rank(query, chunks)
        return ranked[:k]
