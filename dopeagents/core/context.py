"""Execution context and runtime metadata."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    """Execution context passed to agents with run ID, timestamps, and metadata.

    This carries execution runtime information but NOT budget guards or retries —
    those are concerns of the Lifecycle Layer (AgentExecutor). The context is
    created once per agent.run() call and shared across all internal steps.
    """

    run_id: UUID = Field(default_factory=uuid4, description="Unique run identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this context was created",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Application-specific metadata (tracing, user info, etc.)",
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
