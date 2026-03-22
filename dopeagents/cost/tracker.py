"""Cost tracking and aggregation."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dopeagents.core.agent import Agent
    from dopeagents.core.context import AgentContext
    from dopeagents.core.types import ExecutionMetrics


class CostTracker:
    """Tracks cumulative cost across agent executions.

    Token and request metadata are captured via Instructor hooks.
    Cost is read from LiteLLM's response metadata when available,
    then recorded via lifecycle metrics.

    Thread-safe: all mutations are guarded by a Lock.
    """

    def __init__(self) -> None:
        self._costs: dict[str, float] = defaultdict(float)
        self._global_cost: float = 0.0
        self._call_counts: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def record(
        self,
        agent: Agent,  # type: ignore[type-arg]
        context: AgentContext,  # noqa: ARG002
        metrics: ExecutionMetrics,
    ) -> None:
        """Record the cost and call count for an agent execution."""
        cost = metrics.cost_usd or metrics.total_cost_usd
        with self._lock:
            self._costs[agent.name] += cost
            self._global_cost += cost
            self._call_counts[agent.name] += 1

    def get_agent_cost(self, agent_name: str) -> float:
        """Return cumulative cost for the named agent."""
        return self._costs[agent_name]

    def get_total_cost(self) -> float:
        """Return global cumulative cost across all agents."""
        return self._global_cost

    def get_summary(self) -> dict[str, Any]:
        """Return a dict with total cost and per-agent breakdown."""
        return {
            "total_cost_usd": self._global_cost,
            "by_agent": {
                name: {
                    "cost_usd": self._costs[name],
                    "calls": self._call_counts[name],
                    "avg_cost": (
                        self._costs[name] / self._call_counts[name]
                        if self._call_counts[name] > 0
                        else 0.0
                    ),
                }
                for name in self._costs
            },
        }

    def reset(self) -> None:
        """Reset all tracked costs (useful for testing)."""
        with self._lock:
            self._costs.clear()
            self._global_cost = 0.0
            self._call_counts.clear()

    @classmethod
    def noop(cls) -> CostTracker:
        """Return a no-op tracker that records nothing."""
        return cls()
