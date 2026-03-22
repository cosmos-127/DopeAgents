"""DopeAgents — Production-grade AI agents with typed interfaces."""

__version__ = "0.1.0"

# Core type system and agent interface
# Agents (Phase 1)
from dopeagents.agents import DeepSummarizer, DeepSummarizerInput, DeepSummarizerOutput

# Configuration
from dopeagents.config import DopeAgentsConfig, get_config, reset_config, set_config
from dopeagents.core.agent import Agent, AgentDescription, DebugInfo
from dopeagents.core.context import AgentContext
from dopeagents.core.metadata import AgentMetadata
from dopeagents.core.types import AgentResult, ExecutionMetrics, StepMetrics

# Error hierarchy
from dopeagents.errors import (
    BudgetDegradedError,
    BudgetExceededError,
    ContractError,
    CostError,
    DopeAgentsError,
    ExtractionError,
    ExtractionProviderError,
    ExtractionValidationError,
)

# Observability
from dopeagents.observability.logging import get_logger

__all__ = [
    "Agent",
    "AgentContext",
    "AgentDescription",
    "AgentMetadata",
    "AgentResult",
    "BudgetDegradedError",
    "BudgetExceededError",
    "ContractError",
    "CostError",
    "DebugInfo",
    "DeepSummarizer",
    "DeepSummarizerInput",
    "DeepSummarizerOutput",
    "DopeAgentsConfig",
    "DopeAgentsError",
    "ExecutionMetrics",
    "ExtractionError",
    "ExtractionProviderError",
    "ExtractionValidationError",
    "StepMetrics",
    "__version__",
    "get_config",
    "get_logger",
    "reset_config",
    "set_config",
]
