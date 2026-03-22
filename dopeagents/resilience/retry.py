"""Retry policies and exponential backoff (T2.8).

Handles infrastructure-level errors (timeouts, connection failures) at the
step level. Schema validation retries are handled automatically by Instructor
within _extract() and do NOT count toward RetryPolicy attempts.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RetryPolicy(BaseModel):
    """Configurable retry with exponential backoff.

    Only errors listed in ``retryable_errors`` trigger a retry.  All other
    exceptions propagate immediately without consuming retry attempts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum total attempts (1 = no retries)",
    )
    delay_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description="Base delay between attempts in seconds",
    )
    backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        description="Exponential backoff multiplier (delay * factor^attempt)",
    )
    retryable_errors: list[type[Exception]] = Field(
        default_factory=lambda: [TimeoutError, ConnectionError],  # type: ignore[arg-type]
        description="Exception types that trigger a retry",
    )
