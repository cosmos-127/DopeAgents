"""DopeAgents error hierarchy — typed, structured errors with metadata."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class DopeAgentsError(BaseModel):
    """Base error for all DopeAgents exceptions.
    
    All errors carry structured metadata rather than bare string messages.
    Inherits from BaseModel for structured validation but acts like an Exception.
    """

    error_type: str
    message: str
    agent_name: Optional[str] = None
    original_error: Optional[str] = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def __str__(self) -> str:
        """Return the error message for logging/display."""
        return self.message


# ── Extraction Layer Errors ──────────────────────────────────────────────


class ExtractionError(DopeAgentsError):
    """Base for all extraction layer errors."""

    error_type: str = "extraction_error"
    step_name: Optional[str] = None
    input_data: Optional[str] = None


class ExtractionValidationError(ExtractionError):
    """Structured extraction failed validation (Pydantic schema mismatch).
    
    Occurs when Instructor's auto-retry cannot recover — output
    repeatedly fails Pydantic schema validation.
    """

    error_type: str = "extraction_validation_error"
    response_model: Optional[str] = None
    validation_errors: list[dict[str, Any]] = Field(default_factory=list)


class ExtractionProviderError(ExtractionError):
    """Provider-level error (rate limit, auth, service down, etc.)."""

    error_type: str = "extraction_provider_error"
    provider: Optional[str] = None
    status_code: Optional[int] = None
    retry_after: Optional[int] = None  # seconds, if applicable


# ── Cost & Budget Errors ──────────────────────────────────────────────


class CostError(DopeAgentsError):
    """Base for cost tracking and budget errors."""

    error_type: str = "cost_error"
    current_cost: float = 0.0
    budget_limit: float = 0.0


class BudgetExceededError(CostError):
    """Budget limit exceeded — execution halted.
    
    Raised when cost exceeds `max_cost_per_call`, `max_cost_per_step`,
    `max_cost_per_agent`, or `max_cost_global` with `on_exceeded="error"`.
    """

    error_type: str = "budget_exceeded_error"
    budget_type: str = Field(
        default="per_call",
        description="Type of budget exceeded: per_call, per_step, per_agent, or global",
    )


class BudgetDegradedError(CostError):
    """Budget exhausted — returned best-so-far result with degradation.
    
    Raised when cost exceeds budget with `on_exceeded="degrade"`.
    The degraded_output contains the best result produced before budget exhaustion.
    """

    error_type: str = "budget_degraded_error"
    degraded_output: Optional[Any] = None
    budget_type: str = Field(
        default="per_call",
        description="Type of budget that triggered degradation",
    )


class TokenCountError(CostError):
    """Token counting failed — cost data may be unreliable."""

    error_type: str = "token_count_error"


# ── Orchestration & Graph Errors ──────────────────────────────────────


class OrchestrationError(DopeAgentsError):
    """Base for graph orchestration errors."""

    error_type: str = "orchestration_error"
    graph_state: Optional[dict[str, Any]] = None


class GraphConstructionError(OrchestrationError):
    """Failed to build the LangGraph graph."""

    error_type: str = "graph_construction_error"


class GraphExecutionError(OrchestrationError):
    """Graph execution failed (step method raised exception, etc.)."""

    error_type: str = "graph_execution_error"
    step_name: Optional[str] = None
    step_state: Optional[dict[str, Any]] = None


# ── Contract & Composition Errors ──────────────────────────────────────


class ContractError(DopeAgentsError):
    """Base for agent contract and composition errors."""

    error_type: str = "contract_error"
    agent_a: Optional[str] = None
    agent_b: Optional[str] = None


class IncompatibleAgentsError(ContractError):
    """Two agents cannot be composed — incompatible input/output types."""

    error_type: str = "incompatible_agents_error"
    incompatibility_reason: str = ""


class PipelineValidationError(ContractError):
    """Multi-agent pipeline validation failed."""

    error_type: str = "pipeline_validation_error"
    agents: list[str] = Field(default_factory=list)
    failing_connection: Optional[tuple[str, str]] = None


# ── Type & Introspection Errors ──────────────────────────────────────


class TypeResolutionError(DopeAgentsError):
    """Failed to resolve InputT or OutputT from Agent subclass."""

    error_type: str = "type_resolution_error"
    resolution_type: str = Field(
        default="unknown", description="What failed to resolve: input_type or output_type"
    )


class SchemaExtractionError(DopeAgentsError):
    """Failed to extract Pydantic schema from agent type."""

    error_type: str = "schema_extraction_error"
    field_name: Optional[str] = None


# ── Configuration Errors ──────────────────────────────────────────────


class ConfigError(DopeAgentsError):
    """Base for configuration errors."""

    error_type: str = "config_error"


class ConfigNotFoundError(ConfigError):
    """Configuration file or environment not found."""

    error_type: str = "config_not_found_error"
    config_source: Optional[str] = None


class ConfigValidationError(ConfigError):
    """Configuration validation failed."""

    error_type: str = "config_validation_error"
    valid_keys: list[str] = Field(default_factory=list)


# ── Framework Adapter Errors ──────────────────────────────────────────


class AdapterError(DopeAgentsError):
    """Base for framework adapter errors."""

    error_type: str = "adapter_error"
    framework: Optional[str] = None


class FrameworkNotInstalledError(AdapterError):
    """Required framework dependency is not installed."""

    error_type: str = "framework_not_installed_error"


class AdapterUnsupportedError(AdapterError):
    """Agent does not support this adapter."""

    error_type: str = "adapter_unsupported_error"
    adapter_name: Optional[str] = None


# ── MCP & Protocol Errors ────────────────────────────────────────────


class MCPError(DopeAgentsError):
    """Base for Model Context Protocol errors."""

    error_type: str = "mcp_error"


class ToolRegistrationError(MCPError):
    """Failed to register agent as MCP tool."""

    error_type: str = "tool_registration_error"
    tool_name: Optional[str] = None


# ── Sandbox & Execution Errors ──────────────────────────────────────


class SandboxError(DopeAgentsError):
    """Base for sandbox execution errors."""

    error_type: str = "sandbox_error"


class SandboxTimeoutError(SandboxError):
    """Sandbox execution exceeded time limit."""

    error_type: str = "sandbox_timeout_error"
    timeout_seconds: float = 0.0


class SandboxResourceError(SandboxError):
    """Sandbox execution exceeded resource limits."""

    error_type: str = "sandbox_resource_error"
    resource_type: str = Field(
        default="memory", description="Type of resource exceeded: memory, cpu, etc."
    )
    limit: float = 0.0
    usage: float = 0.0


# ── Resilience & Retry Errors ──────────────────────────────────────


class ResilienceError(DopeAgentsError):
    """Base for resilience layer errors."""

    error_type: str = "resilience_error"


class MaxRetriesExceededError(ResilienceError):
    """Transient error persisted after max retries."""

    error_type: str = "max_retries_exceeded_error"
    retry_count: int = 0
    last_error: Optional[str] = None


class DegradationError(DopeAgentsError):
    """Entire agent degradation chain failed."""

    error_type: str = "degradation_error"
    fallback_agents: list[str] = Field(default_factory=list)


# ── Cache Errors ────────────────────────────────────────────────────


class CacheError(DopeAgentsError):
    """Base for caching layer errors."""

    error_type: str = "cache_error"


class CacheKeyError(CacheError):
    """Cache key generation or lookup failed."""

    error_type: str = "cache_key_error"


class CacheStoreError(CacheError):
    """Cache write/read operation failed."""

    error_type: str = "cache_store_error"
    operation: str = Field(default="unknown", description="Operation: read, write, or delete")


# ── Registry & Discovery Errors ──────────────────────────────────────


class RegistryError(DopeAgentsError):
    """Base for agent registry errors."""

    error_type: str = "registry_error"


class AgentNotFoundError(RegistryError):
    """Agent not found in registry."""

    error_type: str = "agent_not_found_error"
    agent_name: str = ""
    available_agents: list[str] = Field(default_factory=list)


class RegistryConflictError(RegistryError):
    """Agent name conflict in registry."""

    error_type: str = "registry_conflict_error"
    conflicting_name: str = ""


# ── Benchmark & Evaluation Errors ──────────────────────────────────


class BenchmarkError(DopeAgentsError):
    """Base for benchmarking errors."""

    error_type: str = "benchmark_error"


class MetricComputationError(BenchmarkError):
    """Failed to compute evaluation metric."""

    error_type: str = "metric_computation_error"
    metric_name: Optional[str] = None


# ── Security & PII Errors ──────────────────────────────────────────


class SecurityError(DopeAgentsError):
    """Base for security-related errors."""

    error_type: str = "security_error"


class PIIDetectionError(SecurityError):
    """PII detection or redaction failed."""

    error_type: str = "pii_detection_error"


class PIIRedactionError(SecurityError):
    """PII redaction failed — data may be unsafe."""

    error_type: str = "pii_redaction_error"
    field_name: Optional[str] = None
