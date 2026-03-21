"""DopeAgents — Production-grade AI agents with typed interfaces."""

__version__ = "0.1.0"

# Core type system and agent interface
from dopeagents.core.agent import Agent, DebugInfo, AgentDescription
from dopeagents.core.context import AgentContext
from dopeagents.core.metadata import AgentMetadata
from dopeagents.core.types import AgentResult, ExecutionMetrics, StepMetrics

# Error hierarchy
from dopeagents.errors import (
    DopeAgentsError,
    ExtractionError,
    ExtractionValidationError,
    ExtractionProviderError,
    CostError,
    BudgetExceededError,
    BudgetDegradedError,
    ContractError,
)

# Configuration
from dopeagents.config import DopeAgentsConfig, get_config, set_config, reset_config

# Observability
from dopeagents.observability.logging import get_logger

# Agents (Phase 1)
from dopeagents.agents import (
    DeepSummarizer,
    DeepSummarizerInput,
    DeepSummarizerOutput,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "Agent",
    "AgentContext",
    "AgentMetadata",
    "AgentResult",
    "ExecutionMetrics",
    "StepMetrics",
    "DebugInfo",
    "AgentDescription",
    # Errors
    "DopeAgentsError",
    "ExtractionError",
    "ExtractionValidationError",
    "ExtractionProviderError",
    "CostError",
    "BudgetExceededError",
    "BudgetDegradedError",
    "ContractError",
    # Configuration
    "DopeAgentsConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Observability
    "get_logger",
    # Agents (Phase 1)
    "DeepSummarizer",
    "DeepSummarizerInput",
    "DeepSummarizerOutput",
]

