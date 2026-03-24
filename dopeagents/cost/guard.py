"""Budget configuration and enforcement (T2.2)."""

from __future__ import annotations

import warnings

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from dopeagents.errors import BudgetDegradedError, BudgetExceededError

if TYPE_CHECKING:
    from dopeagents.core.agent import Agent
    from dopeagents.core.context import AgentContext
    from dopeagents.cost.tracker import CostTracker


class BudgetConfig(BaseModel):
    """Declarative budget limits for an agent execution.

    Passed to AgentExecutor and evaluated by BudgetGuard at pre-execution
    time and (optionally) after each step.
    """

    max_cost_per_call: float | None = Field(
        default=None, description="Total cost cap per agent.run() call"
    )
    max_cost_per_step: float | None = Field(
        default=None, description="Cost cap for each step in a multi-step agent"
    )
    max_cost_per_agent: float | None = Field(
        default=None, description="Cumulative cost cap across all calls to one agent"
    )
    max_cost_global: float | None = Field(
        default=None, description="Cumulative global cap across all agents"
    )
    max_refinement_loops: int | None = Field(
        default=None, description="Max self-evaluation refinement iterations"
    )
    on_exceeded: Literal["error", "warn", "degrade"] = Field(
        default="error",
        description=(
            "'error' raises BudgetExceededError, "
            "'warn' emits a warning (execution continues), "
            "'degrade' raises BudgetDegradedError with best-so-far output"
        ),
    )


class BudgetGuard:
    """Static helper that enforces BudgetConfig limits.

    Called by AgentExecutor before execution and (for step budgets) after
    each individual step completes.
    """

    @staticmethod
    def check_pre_execution(
        agent: Agent,  # type: ignore[type-arg]
        context: AgentContext,  # noqa: ARG004
        cost_tracker: CostTracker,
        budget: BudgetConfig | None = None,
    ) -> None:
        """Check cumulative per-agent and global limits before execution.

        Raises BudgetExceededError / BudgetDegradedError if a limit is hit.
        Does nothing if budget is None.
        """
        if budget is None:
            return

        agent_name = agent.name

        if budget.max_cost_per_agent is not None:
            current = cost_tracker.get_agent_cost(agent_name)
            if current >= budget.max_cost_per_agent:
                BudgetGuard._handle_exceeded(
                    message=(
                        f"Agent '{agent_name}' cumulative cost ${current:.4f} "
                        f">= max_cost_per_agent ${budget.max_cost_per_agent:.4f}"
                    ),
                    action=budget.on_exceeded,
                    agent_name=agent_name,
                    current_cost=current,
                    budget_limit=budget.max_cost_per_agent,
                )

        if budget.max_cost_global is not None:
            total = cost_tracker.get_total_cost()
            if total >= budget.max_cost_global:
                BudgetGuard._handle_exceeded(
                    message=(
                        f"Global cost ${total:.4f} >= max_cost_global ${budget.max_cost_global:.4f}"
                    ),
                    action=budget.on_exceeded,
                    agent_name=agent_name,
                    current_cost=total,
                    budget_limit=budget.max_cost_global,
                )

    @staticmethod
    def check_step_budget(
        step_name: str,
        step_cost: float,
        budget: BudgetConfig | None,
        agent_name: str | None = None,
    ) -> None:
        """Called after each step to enforce per-step cost limits."""
        if budget is None or budget.max_cost_per_step is None:
            return
        if step_cost >= budget.max_cost_per_step:
            BudgetGuard._handle_exceeded(
                message=(
                    f"Step '{step_name}' cost ${step_cost:.4f} "
                    f">= max_cost_per_step ${budget.max_cost_per_step:.4f}"
                ),
                action=budget.on_exceeded,
                agent_name=agent_name,
                current_cost=step_cost,
                budget_limit=budget.max_cost_per_step,
            )

    @staticmethod
    def _handle_exceeded(
        message: str,
        action: str,
        agent_name: str | None = None,
        current_cost: float = 0.0,
        budget_limit: float = 0.0,
        degraded_output: Any = None,
    ) -> None:
        """Dispatch the configured on_exceeded action."""
        if action == "error":
            raise BudgetExceededError(
                message=message,
                agent_name=agent_name,
                current_cost=current_cost,
                budget_limit=budget_limit,
            )
        elif action == "warn":
            warnings.warn(message, ResourceWarning, stacklevel=3)
        elif action == "degrade":
            raise BudgetDegradedError(
                message=message,
                agent_name=agent_name,
                current_cost=current_cost,
                budget_limit=budget_limit,
                degraded_output=degraded_output,
            )
        else:
            # Unknown action — treat as "error" for safety
            raise BudgetExceededError(
                message=message,
                agent_name=agent_name,
                current_cost=current_cost,
                budget_limit=budget_limit,
            )
