"""LLM step output schemas for DeepSummarizer.

These Pydantic models define the structured response contracts for each
LLM-driven step in the 7-step summarisation workflow.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkSummary(BaseModel):
    """Output of the summarize step for a single text chunk."""

    summary: str = Field(description="Concise summary of this chunk")


class SynthesizeOut(BaseModel):
    """Output of the synthesize step."""

    synthesis: str = Field(description="Combined synthesis from all chunk summaries")
    key_points: list[str] = Field(default_factory=list, description="3-5 key points extracted")


class EvaluateOut(BaseModel):
    """Output of the evaluate step — three independent quality dimensions."""

    faithfulness_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Faithfulness to source 0-1"
    )
    completeness_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Coverage of key points 0-1"
    )
    coherence_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Structural quality 0-1"
    )
    quality_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Overall quality score 0-1"
    )
    feedback: str = Field(default="", description="Actionable feedback for the refine step")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the summary not grounded in the source text",
    )


class RefineOut(BaseModel):
    """Output of the refine step."""

    refined: str = Field(description="Improved synthesis addressing the evaluation feedback")
