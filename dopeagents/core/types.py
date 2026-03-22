"""Core type definitions and models."""

from datetime import datetime
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

OutputT = TypeVar("OutputT", bound=BaseModel)


class StepMetrics(BaseModel):
    """Cost and performance metrics for a single step in a multi-step agent.

    Reported by the Lifecycle Layer after each step's _extract() call completes.
    Contains per-step cost, token counts, and latency.
    """

    step_name: str = Field(description="Name of the step (e.g., 'analyze', 'chunk')")
    input_tokens: int = Field(default=0, description="Input tokens sent to LLM")
    output_tokens: int = Field(default=0, description="Output tokens received from LLM")
    total_tokens: int = Field(default=0, description="Total tokens (input + output)")
    cost_usd: float = Field(default=0.0, description="Cost in USD for this step")
    latency_seconds: float = Field(default=0.0, description="Time to execute this step")
    model_used: str | None = Field(default=None, description="Which model this step used")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionMetrics(BaseModel):
    """Metrics collected by the Lifecycle Layer for an entire agent.run() execution.

    NOT collected by the agent itself — collected by AgentExecutor from the
    aggregate of all step metrics.
    """

    run_id: UUID | None = Field(default=None, description="Execution run ID (from AgentContext)")
    # ── Executor-populated fields (DD §5.2) ──────────────────────────────
    latency_ms: float = Field(default=0.0, description="Total wall-clock time in ms")
    cost_usd: float = Field(default=0.0, description="Total cost in USD")
    token_count_in: int = Field(default=0, description="Total input tokens")
    token_count_out: int = Field(default=0, description="Total output tokens")
    llm_calls: int = Field(default=0, description="Number of LLM calls made")
    retry_count: int = Field(default=0, description="Number of retries used")
    fallback_used: str | None = Field(
        default=None, description="Name of fallback agent used, if any"
    )
    cache_hit: bool = Field(default=False, description="Whether result was served from cache")
    # ── Aggregated totals (legacy aliases kept for compatibility) ─────────
    total_cost_usd: float = Field(default=0.0, description="Total cost across all steps")
    total_input_tokens: int = Field(default=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, description="Total output tokens")
    total_tokens: int = Field(default=0, description="Total tokens")
    total_latency_seconds: float = Field(default=0.0, description="Total execution time in seconds")
    step_metrics: list[StepMetrics] = Field(default_factory=list, description="Per-step costs")
    refinement_loops: int = Field(default=0, description="Number of refinement loops executed")
    degradation_reason: str | None = Field(
        default=None, description="Reason if output was degraded"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def populate_legacy_aliases(self) -> "ExecutionMetrics":
        """Populate legacy alias fields from primary fields for backward compatibility."""
        if self.total_cost_usd == 0.0:
            self.total_cost_usd = self.cost_usd
        if self.total_input_tokens == 0:
            self.total_input_tokens = self.token_count_in
        if self.total_output_tokens == 0:
            self.total_output_tokens = self.token_count_out
        if self.total_tokens == 0:
            self.total_tokens = self.token_count_in + self.token_count_out
        if self.total_latency_seconds == 0.0:
            self.total_latency_seconds = self.latency_ms / 1000
        return self


class AgentResult(BaseModel, Generic[OutputT]):
    """Wraps agent output with execution metadata.

    Generic over the agent's OutputT so result types match agent types.
    Includes both the output and the metrics collected during execution.
    """

    output: OutputT | None = Field(default=None, description="The agent's output")
    # Executor-populated fields (DD §5.2)
    metrics: ExecutionMetrics | None = Field(
        default=None, description="Execution metrics from AgentExecutor"
    )
    execution_metrics: ExecutionMetrics | None = Field(
        default=None, description="Alias for metrics (legacy compat)"
    )
    # Result provenance (populated by AgentExecutor)
    run_id: UUID | None = Field(default=None)
    agent_name: str | None = Field(default=None)
    agent_version: str | None = Field(default=None)
    timestamp: datetime | None = Field(default=None)
    error: str | None = Field(default=None, description="Error message if execution failed")
    success: bool = Field(default=True, description="Whether execution succeeded (no error)")

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    # ── Convenience accessors for metrics ──────────────────────────────

    def tokens(self) -> int:
        """Return total tokens used (input + output)."""
        if self.metrics:
            return self.metrics.total_tokens
        return 0

    def tokens_breakdown(self) -> dict[str, int]:
        """Return token counts as a dict: {input: N, output: N, total: N}."""
        if self.metrics:
            return {
                "input": self.metrics.token_count_in,
                "output": self.metrics.token_count_out,
                "total": self.metrics.total_tokens,
            }
        return {"input": 0, "output": 0, "total": 0}

    def cost_usd(self) -> float:
        """Return execution cost in USD."""
        if self.metrics:
            return self.metrics.cost_usd or 0.0
        return 0.0

    def latency_ms(self) -> float:
        """Return execution latency in milliseconds."""
        if self.metrics:
            return self.metrics.latency_ms
        return 0.0

    def llm_calls_count(self) -> int:
        """Return number of LLM API calls made."""
        if self.metrics:
            return self.metrics.llm_calls
        return 0

    def format_metrics(self) -> str:
        """Return human-readable metrics summary."""
        if not self.metrics:
            return "No metrics available"

        parts = []
        if self.metrics.llm_calls > 0:
            parts.append(
                f"Tokens: {self.metrics.token_count_in}/{self.metrics.token_count_out} "
                f"({self.metrics.total_tokens} total)"
            )
            parts.append(f"Cost: ${self.metrics.cost_usd:.6f}")
            parts.append(f"LLM calls: {self.metrics.llm_calls}")

        parts.append(f"Latency: {self.metrics.latency_ms:.0f}ms")

        if self.metrics.fallback_used:
            parts.append(f"Fallback: {self.metrics.fallback_used}")

        return " | ".join(parts)
