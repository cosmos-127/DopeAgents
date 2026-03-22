"""Core agent base classes and type infrastructure."""

from dopeagents.core.agent import Agent, AgentDescription, DebugInfo
from dopeagents.core.context import AgentContext
from dopeagents.core.metadata import AgentMetadata
from dopeagents.core.types import AgentResult, ExecutionMetrics, StepMetrics

__all__ = [
    "Agent",
    "AgentContext",
    "AgentDescription",
    "AgentMetadata",
    "AgentResult",
    "DebugInfo",
    "ExecutionMetrics",
    "StepMetrics",
]
