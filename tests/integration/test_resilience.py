"""Resilience integration test (T2.15).

Validates that AgentExecutor + RetryPolicy + FallbackChain + BudgetGuard
all compose correctly as a production-grade lifecycle system.

No real LLM calls.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from pydantic import BaseModel

from dopeagents.core.agent import Agent
from dopeagents.core.context import AgentContext
from dopeagents.core.types import AgentResult, ExecutionMetrics
from dopeagents.cost.guard import BudgetConfig
from dopeagents.cost.tracker import CostTracker
from dopeagents.errors import AllFallbacksFailedError, BudgetDegradedError, BudgetExceededError
from dopeagents.lifecycle.executor import AgentExecutor
from dopeagents.resilience.fallback import FallbackChain
from dopeagents.resilience.retry import RetryPolicy

# ── Fixtures ─────────────────────────────────────────────────────────────────


class TextIn(BaseModel):
    text: str


class SummaryOut(BaseModel):
    summary: str


class EchoSummaryAgent(Agent[TextIn, SummaryOut]):
    """Deterministic, no-LLM agent used as a fallback."""

    name: ClassVar[str] = "EchoSummaryAgent"
    requires_llm: ClassVar[bool] = False

    def run(
        self, input_data: TextIn, context: AgentContext | None = None
    ) -> AgentResult[SummaryOut]:
        return AgentResult(
            output=SummaryOut(summary=f"echo: {input_data.text[:30]}"),
            success=True,
        )


class FlakeyLLMAgent(Agent[TextIn, SummaryOut]):
    """Agent that fails a configurable number of times before succeeding."""

    name: ClassVar[str] = "FlakeyLLMAgent"
    requires_llm: ClassVar[bool] = False  # No real LLM in tests

    def __init__(self, fail_times: int = 2) -> None:
        super().__init__()
        self._fail_times = fail_times
        self._call_count = 0

    def run(
        self, input_data: TextIn, context: AgentContext | None = None
    ) -> AgentResult[SummaryOut]:
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise TimeoutError(f"Simulated timeout on attempt {self._call_count}")
        return AgentResult(
            output=SummaryOut(summary=f"summarised: {input_data.text[:30]}"),
            success=True,
        )


class AlwaysFailAgent(Agent[TextIn, SummaryOut]):
    """Agent that always raises."""

    name: ClassVar[str] = "AlwaysFailAgent"
    requires_llm: ClassVar[bool] = False

    def run(
        self, input_data: TextIn, context: AgentContext | None = None
    ) -> AgentResult[SummaryOut]:
        raise RuntimeError("I always fail")


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestRetryExhaustionWithFallback:
    """RetryPolicy exhausted → FallbackChain invoked → fallback succeeds."""

    def test_retry_then_fallback_success(self) -> None:
        primary = FlakeyLLMAgent(fail_times=10)  # always times out within retries
        fallback = EchoSummaryAgent()
        chain = FallbackChain([fallback])
        policy = RetryPolicy(max_attempts=2, delay_seconds=0, retryable_errors=[TimeoutError])
        executor = AgentExecutor()
        result = executor.run(
            primary,
            TextIn(text="hello world"),
            retry_policy=policy,
            fallback_chain=chain,
        )
        assert result.success is True
        assert result.output is not None
        assert result.agent_name == "FlakeyLLMAgent"  # executor uses primary's name

    def test_all_fallbacks_fail_raises(self) -> None:
        primary = AlwaysFailAgent()
        fallback = AlwaysFailAgent()
        chain = FallbackChain([fallback])
        executor = AgentExecutor()
        with pytest.raises(AllFallbacksFailedError):
            executor.run(primary, TextIn(text="x"), fallback_chain=chain)

    def test_retry_succeeds_before_exhaustion(self) -> None:
        primary = FlakeyLLMAgent(fail_times=1)  # fails once, then succeeds
        policy = RetryPolicy(max_attempts=3, delay_seconds=0, retryable_errors=[TimeoutError])
        executor = AgentExecutor()
        result = executor.run(primary, TextIn(text="hi"), retry_policy=policy)
        assert result.success is True
        assert result.metrics is not None
        assert result.metrics.retry_count == 1  # one retry was used


class TestBudgetGuardIntegration:
    """Budget exceeded scenarios with error, warn, and degrade modes."""

    def test_budget_exceeded_error_mode_before_run(self) -> None:
        tracker = CostTracker()
        agent = EchoSummaryAgent()
        ctx = AgentContext()
        # Exhaust per-agent budget
        tracker.record(agent, ctx, ExecutionMetrics(cost_usd=1.0))
        budget = BudgetConfig(max_cost_per_agent=0.5, on_exceeded="error")
        executor = AgentExecutor(cost_tracker=tracker, budget=budget)
        with pytest.raises(BudgetExceededError):
            executor.run(agent, TextIn(text="x"), context=ctx)

    def test_budget_degrade_mode_raises_degraded(self) -> None:
        tracker = CostTracker()
        agent = EchoSummaryAgent()
        ctx = AgentContext()
        tracker.record(agent, ctx, ExecutionMetrics(cost_usd=1.0))
        budget = BudgetConfig(max_cost_per_agent=0.5, on_exceeded="degrade")
        executor = AgentExecutor(cost_tracker=tracker, budget=budget)
        with pytest.raises(BudgetDegradedError) as exc_info:
            executor.run(agent, TextIn(text="x"), context=ctx)
        err = exc_info.value
        assert err.current_cost >= 1.0
        assert err.budget_limit == 0.5

    def test_no_budget_agent_runs_fine(self) -> None:
        tracker = CostTracker()
        tracker.record(  # very high cost but no budget configured
            EchoSummaryAgent(), AgentContext(), ExecutionMetrics(cost_usd=100.0)
        )
        executor = AgentExecutor(cost_tracker=tracker, budget=None)
        result = executor.run(EchoSummaryAgent(), TextIn(text="no budget"))
        assert result.success is True


class TestCostRecordingAfterRun:
    """CostTracker is updated after each successful run."""

    def test_cost_recorded_after_run(self) -> None:
        tracker = CostTracker()
        executor = AgentExecutor(cost_tracker=tracker)
        executor.run(EchoSummaryAgent(), TextIn(text="track me"))
        # Cost will be 0.0 for a no-LLM agent (no hooks), but record was called
        summary = tracker.get_summary()
        assert "EchoSummaryAgent" in summary["by_agent"]
        assert summary["by_agent"]["EchoSummaryAgent"]["calls"] == 1

    def test_multiple_runs_accumulate(self) -> None:
        tracker = CostTracker()
        executor = AgentExecutor(cost_tracker=tracker)
        for _ in range(5):
            executor.run(EchoSummaryAgent(), TextIn(text="x"))
        assert tracker.get_summary()["by_agent"]["EchoSummaryAgent"]["calls"] == 5


class TestMetricsCollection:
    """AgentResult.metrics carries timing and provenance info."""

    def test_latency_ms_always_populated(self) -> None:
        executor = AgentExecutor()
        result = executor.run(EchoSummaryAgent(), TextIn(text="timed"))
        assert result.metrics is not None
        assert result.metrics.latency_ms >= 0

    def test_agent_name_and_version_in_result(self) -> None:
        executor = AgentExecutor()
        result = executor.run(EchoSummaryAgent(), TextIn(text="meta"))
        assert result.agent_name == "EchoSummaryAgent"
        assert result.agent_version is not None

    def test_fallback_used_recorded_in_metrics(self) -> None:
        primary = AlwaysFailAgent()
        fallback = EchoSummaryAgent()
        chain = FallbackChain([fallback])
        executor = AgentExecutor()
        result = executor.run(primary, TextIn(text="fallback meta"), fallback_chain=chain)
        assert result.metrics is not None
        assert result.metrics.fallback_used == "EchoSummaryAgent"
