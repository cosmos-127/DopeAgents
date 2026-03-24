"""DopeAgents error hierarchy — typed, structured Python exceptions with metadata.

All errors are real Python exceptions (inherit from Exception) so they can be
raised and caught with standard try/except semantics.  Structured fields are
stored as instance attributes; ``model_dump_json()`` provides JSON serialization
that mirrors the Pydantic BaseModel interface for compatibility.
"""

from __future__ import annotations

import json

from typing import Any


class DopeAgentsError(Exception):
    """Base error for all DopeAgents exceptions.

    Structured fields are stored as instance attributes; call
    ``model_dump_json()`` for JSON serialization (mirrors the Pydantic
    BaseModel interface so existing inspection code is compatible).
    """

    # Class-level default for the error type discriminator
    error_type: str = "dopeagents_error"

    def __init__(
        self,
        message: str = "",
        error_type: str | None = None,
        agent_name: str | None = None,
        original_error: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type if error_type is not None else type(self).error_type
        self.agent_name = agent_name
        self.original_error = original_error
        # Absorb any extra keyword arguments as attributes (supports subclasses
        # forwarding their own fields transparently via **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self) -> str:
        return self.message

    def model_dump_json(self, **_: Any) -> str:
        """Return a JSON string of this error's structured fields.

        Mirrors the Pydantic BaseModel.model_dump_json() interface so code
        that serialises errors doesn't need to know which base class is used.
        """
        data: dict[str, Any] = {}
        # Collect annotated field names from the whole MRO so all subclass
        # fields are included in order (reversed = base → most-derived, so
        # more-derived annotations overwrite duplicates from base)
        for cls in reversed(type(self).__mro__):
            for k in getattr(cls, "__annotations__", {}):
                if not k.startswith("_"):
                    data[k] = getattr(self, k, None)
        # Always include the core fields
        data["error_type"] = self.error_type
        data["message"] = self.message
        return json.dumps(data, default=str)

    def model_dump(self) -> dict[str, Any]:
        """Return a dict of this error's structured fields."""
        result = json.loads(self.model_dump_json())
        return result  # type: ignore[no-any-return]


# ── Extraction Layer Errors ──────────────────────────────────────────────


class ExtractionError(DopeAgentsError):
    """Base for all extraction layer errors."""

    error_type: str = "extraction_error"
    step_name: str | None = None
    input_data: str | None = None


class ExtractionValidationError(ExtractionError):
    """Structured extraction failed validation (Pydantic schema mismatch).

    Occurs when Instructor's auto-retry cannot recover — output
    repeatedly fails Pydantic schema validation.
    """

    error_type: str = "extraction_validation_error"
    response_model: str | None = None
    validation_errors: list[dict[str, Any]] | None = None

    def __init__(
        self,
        message: str = "",
        response_model: str | None = None,
        validation_errors: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)
        self.response_model = response_model
        self.validation_errors = validation_errors or []


class ExtractionProviderError(ExtractionError):
    """Provider-level error (rate limit, auth, service down, etc.)."""

    error_type: str = "extraction_provider_error"
    provider: str | None = None
    status_code: int | None = None
    retry_after: int | None = None  # seconds, if applicable


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
    budget_type: str = "per_call"


class BudgetDegradedError(CostError):
    """Budget exhausted — returned best-so-far result with degradation.

    Raised when cost exceeds budget with `on_exceeded="degrade"`.
    The degraded_output contains the best result produced before budget exhaustion.
    """

    error_type: str = "budget_degraded_error"
    degraded_output: Any | None = None
    budget_type: str = "per_call"


class TokenCountError(CostError):
    """Token counting failed — cost data may be unreliable."""

    error_type: str = "token_count_error"


# ── Orchestration & Graph Errors ──────────────────────────────────────


class OrchestrationError(DopeAgentsError):
    """Base for graph orchestration errors."""

    error_type: str = "orchestration_error"
    graph_state: dict[str, Any] | None = None


class GraphConstructionError(OrchestrationError):
    """Failed to build the LangGraph graph."""

    error_type: str = "graph_construction_error"


class GraphExecutionError(OrchestrationError):
    """Graph execution failed (step method raised exception, etc.)."""

    error_type: str = "graph_execution_error"
    step_name: str | None = None
    step_state: dict[str, Any] | None = None


# ── Contract & Composition Errors ──────────────────────────────────────


class ContractError(DopeAgentsError):
    """Base for agent contract and composition errors."""

    error_type: str = "contract_error"
    agent_a: str | None = None
    agent_b: str | None = None


class IncompatibleAgentsError(ContractError):
    """Two agents cannot be composed — incompatible input/output types."""

    error_type: str = "incompatible_agents_error"
    incompatibility_reason: str = ""


class PipelineValidationError(ContractError):
    """Multi-agent pipeline validation failed."""

    error_type: str = "pipeline_validation_error"
    agents: list[str] | None = None
    failing_connection: tuple[str, str] | None = None

    def __init__(
        self,
        message: str = "",
        agents: list[str] | None = None,
        failing_connection: tuple[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)
        self.agents = agents or []
        self.failing_connection = failing_connection


# ── Type & Introspection Errors ──────────────────────────────────────


class TypeResolutionError(DopeAgentsError):
    """Failed to resolve InputT or OutputT from Agent subclass."""

    error_type: str = "type_resolution_error"
    resolution_type: str = "unknown"


class SchemaExtractionError(DopeAgentsError):
    """Failed to extract Pydantic schema from agent type."""

    error_type: str = "schema_extraction_error"
    field_name: str | None = None


# ── Configuration Errors ──────────────────────────────────────────────


class ConfigError(DopeAgentsError):
    """Base for configuration errors."""

    error_type: str = "config_error"


class ConfigNotFoundError(ConfigError):
    """Configuration file or environment not found."""

    error_type: str = "config_not_found_error"
    config_source: str | None = None


class ConfigValidationError(ConfigError):
    """Configuration validation failed."""

    error_type: str = "config_validation_error"
    valid_keys: list[str] | None = None

    def __init__(
        self,
        message: str = "",
        valid_keys: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)
        self.valid_keys = valid_keys or []


# ── Framework Adapter Errors ──────────────────────────────────────────


class AdapterError(DopeAgentsError):
    """Base for framework adapter errors."""

    error_type: str = "adapter_error"
    framework: str | None = None


class FrameworkNotInstalledError(AdapterError):
    """Required framework dependency is not installed."""

    error_type: str = "framework_not_installed_error"


class AdapterUnsupportedError(AdapterError):
    """Agent does not support this adapter."""

    error_type: str = "adapter_unsupported_error"
    adapter_name: str | None = None


# ── MCP & Protocol Errors ────────────────────────────────────────────


class MCPError(DopeAgentsError):
    """Base for Model Context Protocol errors."""

    error_type: str = "mcp_error"


class ToolRegistrationError(MCPError):
    """Failed to register agent as MCP tool."""

    error_type: str = "tool_registration_error"
    tool_name: str | None = None


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
    resource_type: str = "memory"
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
    last_error: str | None = None


class DegradationError(DopeAgentsError):
    """Entire agent degradation chain failed."""

    error_type: str = "degradation_error"
    fallback_agents: list[str] | None = None

    def __init__(
        self,
        message: str = "",
        fallback_agents: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)
        self.fallback_agents = fallback_agents or []


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
    operation: str = "unknown"


# ── Execution Errors (used by AgentExecutor) ─────────────────────────


class AgentExecutionError(DopeAgentsError):
    """Agent execution failed — wraps the underlying exception."""

    error_type: str = "agent_execution_error"
    agent_version: str | None = None


class AllFallbacksFailedError(DopeAgentsError):
    """Every agent in a FallbackChain / DegradationChain raised an exception."""

    error_type: str = "all_fallbacks_failed_error"
    chain_agents: list[str] | None = None
    errors: list[str] | None = None

    def __init__(
        self,
        message: str = "",
        chain_agents: list[str] | None = None,
        errors: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)
        self.chain_agents = chain_agents or []
        self.errors = errors or []


class InputValidationError(DopeAgentsError):
    """Agent input failed Pydantic schema validation."""

    error_type: str = "input_validation_error"
    validation_errors: list[dict[str, Any]] | None = None

    def __init__(
        self,
        message: str = "",
        validation_errors: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)
        self.validation_errors = validation_errors or []


class OutputValidationError(DopeAgentsError):
    """Agent output failed Pydantic schema validation."""

    error_type: str = "output_validation_error"
    validation_errors: list[dict[str, Any]] | None = None

    def __init__(
        self,
        message: str = "",
        validation_errors: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)
        self.validation_errors = validation_errors or []


# ── Registry & Discovery Errors ──────────────────────────────────────


class RegistryError(DopeAgentsError):
    """Base for agent registry errors."""

    error_type: str = "registry_error"


class AgentNotFoundError(RegistryError):
    """Agent not found in registry."""

    error_type: str = "agent_not_found_error"
    agent_name: str = ""
    available_agents: list[str] | None = None

    def __init__(
        self,
        message: str = "",
        agent_name: str = "",
        available_agents: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)
        self.agent_name = agent_name
        self.available_agents = available_agents or []


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
    metric_name: str | None = None


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
    field_name: str | None = None
