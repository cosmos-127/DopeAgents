"""DopeAgents — Production-grade AI agents with typed interfaces."""

import builtins
import logging
import os
import warnings

from typing import Any

from dopeagents.agents import DeepSummarizer, DeepSummarizerInput, DeepSummarizerOutput
from dopeagents.config import DopeAgentsConfig, get_config, reset_config, set_config
from dopeagents.core.agent import Agent, AgentDescription, DebugInfo
from dopeagents.core.context import AgentContext
from dopeagents.core.metadata import AgentMetadata
from dopeagents.core.types import AgentResult, ExecutionMetrics, StepMetrics
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
from dopeagents.observability.logging import get_logger

__version__ = "0.1.0"

# Suppress litellm's verbose debug output (runs once on import)
os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("LITELLM_SUPPRESS_DEBUG_INFO", "True")

# Set log level for litellm and related libraries to ERROR to suppress verbose output
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("litellm_server").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Suppress litellm's deprecation warnings
warnings.filterwarnings("ignore", module="litellm")

# Suppress LiteLLM's "Provider List" diagnostic output by filtering print statements
_original_print = builtins.print


def _filtered_print(*args: Any, **kwargs: Any) -> None:
    """Filter print statements to suppress LiteLLM's diagnostic messages."""
    message = " ".join(str(arg) for arg in args)
    # Skip printing "Provider List" messages that LiteLLM outputs
    if "Provider List" not in message and "docs.litellm.ai" not in message:
        _original_print(*args, **kwargs)


builtins.print = _filtered_print

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
