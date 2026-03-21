"""Core type definitions and models."""

from pydantic import BaseModel, Field
from typing import TypeVar, Generic, Optional, Any
from uuid import UUID
from datetime import datetime

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
    model_used: Optional[str] = Field(default=None, description="Which model this step used")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionMetrics(BaseModel):
    """Metrics collected by the Lifecycle Layer for an entire agent.run() execution.
    
    NOT collected by the agent itself — collected by AgentExecutor from the
    aggregate of all step metrics.
    """

    run_id: UUID = Field(description="Execution run ID (from AgentContext)")
    total_cost_usd: float = Field(default=0.0, description="Total cost across all steps")
    total_input_tokens: int = Field(default=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, description="Total output tokens")
    total_tokens: int = Field(default=0, description="Total tokens")
    total_latency_seconds: float = Field(default=0.0, description="Total execution time")
    step_metrics: list[StepMetrics] = Field(default_factory=list, description="Per-step costs")
    refinement_loops: int = Field(default=0, description="Number of refinement loops executed")
    degradation_reason: Optional[str] = Field(
        default=None, description="Reason if output was degraded"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class AgentResult(BaseModel, Generic[OutputT]):
    """Wraps agent output with execution metadata.
    
    Generic over the agent's OutputT so result types match agent types.
    Includes both the output and the metrics collected during execution.
    """

    output: Optional[OutputT] = Field(default=None, description="The agent's output")
    execution_metrics: Optional[ExecutionMetrics] = Field(
        default=None, description="Execution metrics (step costs, latency, etc.)"
    )
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    success: bool = Field(
        default=True, description="Whether execution succeeded (no error)"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

