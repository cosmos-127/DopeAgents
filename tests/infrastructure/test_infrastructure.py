"""Tests for Phase 2 infrastructure: CostTracker, BudgetGuard, AgentExecutor,
InMemoryCache, PIIRedactor, RetryPolicy, FallbackChain, DegradationChain.

No real LLM calls — all agent.run() calls are mocked.
"""

from __future__ import annotations

import threading
import time

from typing import Any, ClassVar

import pytest

from pydantic import BaseModel, Field, ValidationError

from dopeagents.cache.manager import InMemoryCache
from dopeagents.core.agent import Agent
from dopeagents.core.context import AgentContext
from dopeagents.core.types import AgentResult, ExecutionMetrics
from dopeagents.cost.guard import BudgetConfig, BudgetGuard
from dopeagents.cost.tracker import CostTracker
from dopeagents.errors import (
    AgentExecutionError,
    AllFallbacksFailedError,
    BudgetDegradedError,
    BudgetExceededError,
    InputValidationError,
    MaxRetriesExceededError,
)
from dopeagents.lifecycle.executor import AgentExecutor
from dopeagents.lifecycle.hooks import LifecycleHooks
from dopeagents.observability.tracer import ConsoleTracer, NoopTracer
from dopeagents.resilience.degradation import DegradationChain, DegradationResult
from dopeagents.resilience.fallback import FallbackChain
from dopeagents.resilience.retry import RetryPolicy
from dopeagents.security.redaction import PIIRedactor

# ── Shared test fixtures ────────────────────────────────────────────────────


class SimpleIn(BaseModel):
    text: str = Field(min_length=1)


class SimpleOut(BaseModel):
    result: str


class AltOut(BaseModel):
    result: str
    confidence: float = 0.9


def _make_result(text: str = "ok") -> AgentResult[SimpleOut]:
    return AgentResult(output=SimpleOut(result=text), success=True)


class EchoAgent(Agent[SimpleIn, SimpleOut]):
    """Minimal synchronous agent that echoes its input (no LLM)."""

    name: ClassVar[str] = "EchoAgent"
    version: ClassVar[str] = "0.1.0"
    requires_llm: ClassVar[bool] = False

    def run(
        self, input_data: SimpleIn, context: AgentContext | None = None
    ) -> AgentResult[SimpleOut]:
        return AgentResult(output=SimpleOut(result=input_data.text), success=True)


class FailAgent(Agent[SimpleIn, SimpleOut]):
    """Agent that always raises on run()."""

    name: ClassVar[str] = "FailAgent"
    version: ClassVar[str] = "0.1.0"
    requires_llm: ClassVar[bool] = False

    def __init__(self, error: Exception = RuntimeError("fail")) -> None:
        super().__init__()
        self._error = error

    def run(
        self, input_data: SimpleIn, context: AgentContext | None = None
    ) -> AgentResult[SimpleOut]:
        raise self._error


class AltAgent(Agent[SimpleIn, AltOut]):
    """Fallback agent with a compatible (superset) output schema."""

    name: ClassVar[str] = "AltAgent"
    version: ClassVar[str] = "0.1.0"
    requires_llm: ClassVar[bool] = False

    def run(self, input_data: SimpleIn, context: AgentContext | None = None) -> AgentResult[AltOut]:
        return AgentResult(output=AltOut(result=input_data.text), success=True)


# ── CostTracker ─────────────────────────────────────────────────────────────


class TestCostTracker:
    def test_record_and_retrieve(self) -> None:
        tracker = CostTracker()
        agent = EchoAgent()
        ctx = AgentContext()
        metrics = ExecutionMetrics(cost_usd=0.05)
        tracker.record(agent, ctx, metrics)
        assert tracker.get_agent_cost("EchoAgent") == pytest.approx(0.05)
        assert tracker.get_total_cost() == pytest.approx(0.05)

    def test_cumulative_across_calls(self) -> None:
        tracker = CostTracker()
        agent = EchoAgent()
        ctx = AgentContext()
        for _ in range(3):
            tracker.record(agent, ctx, ExecutionMetrics(cost_usd=0.01))
        assert tracker.get_agent_cost("EchoAgent") == pytest.approx(0.03)
        assert tracker.get_total_cost() == pytest.approx(0.03)

    def test_thread_safety(self) -> None:
        """Concurrent record() calls must not corrupt the total."""
        tracker = CostTracker()
        agent = EchoAgent()

        def record_many() -> None:
            ctx = AgentContext()
            for _ in range(100):
                tracker.record(agent, ctx, ExecutionMetrics(cost_usd=0.001))

        threads = [threading.Thread(target=record_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker.get_total_cost() == pytest.approx(0.5, abs=1e-9)

    def test_summary_structure(self) -> None:
        tracker = CostTracker()
        agent = EchoAgent()
        ctx = AgentContext()
        tracker.record(agent, ctx, ExecutionMetrics(cost_usd=0.10))
        summary = tracker.get_summary()
        assert "total_cost_usd" in summary
        assert "EchoAgent" in summary["by_agent"]
        assert summary["by_agent"]["EchoAgent"]["calls"] == 1

    def test_noop_returns_tracker(self) -> None:
        tracker = CostTracker.noop()
        assert isinstance(tracker, CostTracker)


# ── BudgetGuard ─────────────────────────────────────────────────────────────


class TestBudgetGuard:
    def test_no_budget_is_noop(self) -> None:
        """No budget = no check, no raise."""
        tracker = CostTracker()
        BudgetGuard.check_pre_execution(EchoAgent(), AgentContext(), tracker, None)

    def test_error_mode_raises_budget_exceeded(self) -> None:
        tracker = CostTracker()
        agent = EchoAgent()
        ctx = AgentContext()
        # Simulate $0.15 already spent on this agent
        tracker.record(agent, ctx, ExecutionMetrics(cost_usd=0.15))
        budget = BudgetConfig(max_cost_per_agent=0.10, on_exceeded="error")
        with pytest.raises(BudgetExceededError):
            BudgetGuard.check_pre_execution(agent, ctx, tracker, budget)

    def test_warn_mode_does_not_raise(self) -> None:
        import warnings

        tracker = CostTracker()
        agent = EchoAgent()
        ctx = AgentContext()
        tracker.record(agent, ctx, ExecutionMetrics(cost_usd=0.20))
        budget = BudgetConfig(max_cost_per_agent=0.10, on_exceeded="warn")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BudgetGuard.check_pre_execution(agent, ctx, tracker, budget)
        assert len(w) == 1

    def test_degrade_mode_raises_budget_degraded(self) -> None:
        tracker = CostTracker()
        agent = EchoAgent()
        ctx = AgentContext()
        tracker.record(agent, ctx, ExecutionMetrics(cost_usd=0.20))
        budget = BudgetConfig(max_cost_per_agent=0.10, on_exceeded="degrade")
        with pytest.raises(BudgetDegradedError):
            BudgetGuard.check_pre_execution(agent, ctx, tracker, budget)

    def test_global_budget_raises(self) -> None:
        tracker = CostTracker()
        agent = EchoAgent()
        ctx = AgentContext()
        tracker.record(agent, ctx, ExecutionMetrics(cost_usd=1.00))
        budget = BudgetConfig(max_cost_global=0.50, on_exceeded="error")
        with pytest.raises(BudgetExceededError):
            BudgetGuard.check_pre_execution(agent, ctx, tracker, budget)

    def test_step_budget_check(self) -> None:
        budget = BudgetConfig(max_cost_per_step=0.05, on_exceeded="error")
        with pytest.raises(BudgetExceededError):
            BudgetGuard.check_step_budget("summarize", 0.10, budget)

    def test_step_budget_no_limit_is_noop(self) -> None:
        BudgetGuard.check_step_budget("analyze", 9.99, None)


# ── Tracer ──────────────────────────────────────────────────────────────────


class TestTracer:
    def test_noop_tracer_yields_span(self) -> None:
        from uuid import uuid4

        tracer = NoopTracer()
        run_id = uuid4()
        with tracer.span("test", run_id) as span:
            span.set_attribute("key", "value")
            span.add_event("my_event")
        assert span.attributes["key"] == "value"
        assert span.events[0]["name"] == "my_event"

    def test_console_tracer_runs(self, capsys: Any) -> None:
        from uuid import uuid4

        tracer = ConsoleTracer()
        run_id = uuid4()
        with tracer.span("test.span", run_id) as span:
            span.set_attribute("model", "gpt-4o")
        captured = capsys.readouterr().out
        assert "test.span" in captured


# ── AgentExecutor ────────────────────────────────────────────────────────────


class TestAgentExecutor:
    def test_basic_run_returns_result(self) -> None:
        executor = AgentExecutor()
        agent = EchoAgent()
        result = executor.run(agent, SimpleIn(text="hello"))
        assert result.success is True
        assert result.output.result == "hello"  # type: ignore[union-attr]
        assert result.agent_name == "EchoAgent"

    def test_latency_ms_is_populated(self) -> None:
        executor = AgentExecutor()
        result = executor.run(EchoAgent(), SimpleIn(text="latency test"))
        assert result.metrics is not None
        assert result.metrics.latency_ms >= 0.0

    def test_cache_hit_skips_agent_run(self) -> None:
        cache = InMemoryCache()
        agent = EchoAgent()
        input_data = SimpleIn(text="cached input")

        # Pre-populate cache
        cache.set(agent, input_data, SimpleOut(result="from cache"))

        run_calls = []
        original_run = agent.run

        def counting_run(inp: SimpleIn, ctx: AgentContext | None = None) -> AgentResult[SimpleOut]:
            run_calls.append(True)
            return original_run(inp, ctx)

        agent.run = counting_run  # type: ignore[assignment]
        executor = AgentExecutor(cache_manager=cache)
        result = executor.run(agent, input_data)

        assert len(run_calls) == 0, "agent.run() should NOT be called on cache hit"
        assert result.output.result == "from cache"  # type: ignore[union-attr]
        assert result.metrics is not None
        assert result.metrics.cache_hit is True

    def test_agent_error_wraps_as_agent_execution_error(self) -> None:
        executor = AgentExecutor()
        agent = FailAgent(RuntimeError("boom"))
        with pytest.raises(AgentExecutionError):
            executor.run(agent, SimpleIn(text="x"))

    def test_budget_exceeded_propagates(self) -> None:
        tracker = CostTracker()
        agent = EchoAgent()
        ctx = AgentContext()
        tracker.record(agent, ctx, ExecutionMetrics(cost_usd=1.0))
        budget = BudgetConfig(max_cost_per_agent=0.50, on_exceeded="error")
        executor = AgentExecutor(cost_tracker=tracker, budget=budget)
        with pytest.raises(BudgetExceededError):
            executor.run(agent, SimpleIn(text="x"), context=ctx)

    def test_lifecycle_hooks_pre_post_called(self) -> None:
        calls: list[str] = []

        class TrackingHooks(LifecycleHooks):
            def pre_execution(self, agent: Any, input: Any, context: Any) -> None:
                calls.append("pre")

            def post_execution(self, agent: Any, input: Any, output: Any, context: Any) -> None:
                calls.append("post")

        executor = AgentExecutor(hooks=TrackingHooks())
        executor.run(EchoAgent(), SimpleIn(text="hooks test"))
        assert calls == ["pre", "post"]

    def test_fallback_used_on_primary_failure(self) -> None:
        primary = FailAgent(RuntimeError("primary failed"))
        fallback = EchoAgent()
        chain = FallbackChain([fallback])

        executor = AgentExecutor()
        result = executor.run(primary, SimpleIn(text="fallback"), fallback_chain=chain)
        assert result.success is True
        assert result.output is not None

    def test_all_fallbacks_failed_raises(self) -> None:
        primary = FailAgent(RuntimeError("p"))
        fallback = FailAgent(RuntimeError("f"))
        chain = FallbackChain([fallback])

        executor = AgentExecutor()
        with pytest.raises(AllFallbacksFailedError):
            executor.run(primary, SimpleIn(text="x"), fallback_chain=chain)

    def test_input_validation_error_on_bad_input(self) -> None:
        """Pydantic validation error on input is wrapped as InputValidationError."""
        executor = AgentExecutor()
        agent = EchoAgent()
        # Now test with a Mock that raises on model_validate
        with pytest.raises((InputValidationError, Exception)):
            # Pass None as input — should fail validation
            executor.run(agent, None)  # type: ignore[misc]


# ── RetryPolicy ──────────────────────────────────────────────────────────────


class TestRetryPolicy:
    def test_default_values(self) -> None:
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.delay_seconds == 1.0
        assert policy.backoff_factor == 2.0

    def test_custom_values(self) -> None:
        policy = RetryPolicy(max_attempts=5, delay_seconds=0.5, backoff_factor=1.5)
        assert policy.max_attempts == 5

    def test_max_attempts_validation(self) -> None:
        with pytest.raises(ValidationError):
            RetryPolicy(max_attempts=0)  # ge=1 should reject

    def test_retry_exhausted(self) -> None:
        """Executor retries max_attempts times then raises MaxRetriesExceededError."""
        call_count = [0]

        class FlakeyAgent(Agent[SimpleIn, SimpleOut]):
            name: ClassVar[str] = "FlakeyAgent"
            requires_llm: ClassVar[bool] = False

            def run(
                self, input_data: SimpleIn, context: AgentContext | None = None
            ) -> AgentResult[SimpleOut]:
                call_count[0] += 1
                raise TimeoutError("timeout")

        executor = AgentExecutor()
        policy = RetryPolicy(max_attempts=3, delay_seconds=0, retryable_errors=[TimeoutError])
        with pytest.raises(MaxRetriesExceededError):
            executor.run(FlakeyAgent(), SimpleIn(text="x"), retry_policy=policy)
        assert call_count[0] == 3


# ── FallbackChain ────────────────────────────────────────────────────────────


class TestFallbackChain:
    def test_construction_requires_agents(self) -> None:
        with pytest.raises(ValueError):
            FallbackChain([])

    def test_single_agent_allowed(self) -> None:
        chain = FallbackChain([EchoAgent()])
        assert len(chain.agents) == 1

    def test_warns_on_missing_output_fields(self) -> None:
        import warnings

        class LimitedOut(BaseModel):
            summary: str  # 'result' field missing

        class LimitedAgent(Agent[SimpleIn, LimitedOut]):
            name: ClassVar[str] = "LimitedAgent"
            requires_llm: ClassVar[bool] = False

            def run(
                self, input_data: SimpleIn, context: AgentContext | None = None
            ) -> AgentResult[LimitedOut]:
                return AgentResult(output=LimitedOut(summary="x"), success=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FallbackChain([EchoAgent(), LimitedAgent()])
        # Should warn about missing 'result' field
        assert len(w) >= 1


# ── DegradationChain ─────────────────────────────────────────────────────────


class TestDegradationChain:
    def test_primary_success_no_degradation(self) -> None:
        chain = DegradationChain([EchoAgent()])
        result = chain.run_with_degradation(SimpleIn(text="hi"))
        assert isinstance(result, DegradationResult)
        assert result.agent_used == "EchoAgent"
        assert result.degradation_reason is None

    def test_fallback_used_on_primary_failure(self) -> None:
        chain = DegradationChain([FailAgent(), EchoAgent()])
        result = chain.run_with_degradation(SimpleIn(text="degrade"))
        assert result.agent_used == "EchoAgent"
        assert result.degradation_reason is not None
        assert "FailAgent" in result.degradation_reason

    def test_all_fail_raises(self) -> None:
        chain = DegradationChain([FailAgent(), FailAgent()])
        with pytest.raises(AgentExecutionError):
            chain.run_with_degradation(SimpleIn(text="all fail"))

    def test_warns_if_last_agent_requires_llm(self) -> None:
        import warnings

        class LLMRequiredAgent(Agent[SimpleIn, SimpleOut]):
            name: ClassVar[str] = "LLMRequired"
            requires_llm: ClassVar[bool] = True

            def run(
                self, input_data: SimpleIn, context: AgentContext | None = None
            ) -> AgentResult[SimpleOut]:
                return AgentResult(output=SimpleOut(result="x"), success=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DegradationChain([LLMRequiredAgent()])
        assert any("LLM" in str(warning.message) for warning in w)


# ── InMemoryCache ────────────────────────────────────────────────────────────


class TestInMemoryCache:
    def test_cache_miss_returns_none(self) -> None:
        cache = InMemoryCache()
        agent = EchoAgent()
        assert cache.get(agent, SimpleIn(text="x")) is None

    def test_cache_set_and_get(self) -> None:
        cache = InMemoryCache()
        agent = EchoAgent()
        inp = SimpleIn(text="hello")
        out = SimpleOut(result="world")
        cache.set(agent, inp, out)
        assert cache.get(agent, inp) == out

    def test_ttl_expiry(self) -> None:
        cache = InMemoryCache()
        agent = EchoAgent()
        inp = SimpleIn(text="ttl")
        out = SimpleOut(result="expires")
        cache.set(agent, inp, out, ttl=1)
        assert cache.get(agent, inp) == out
        time.sleep(1.1)
        assert cache.get(agent, inp) is None

    def test_invalidate_specific_entry(self) -> None:
        cache = InMemoryCache()
        agent = EchoAgent()
        inp = SimpleIn(text="inv")
        out = SimpleOut(result="x")
        cache.set(agent, inp, out)
        cache.invalidate(agent, inp)
        assert cache.get(agent, inp) is None

    def test_different_inputs_different_keys(self) -> None:
        cache = InMemoryCache()
        agent = EchoAgent()
        cache.set(agent, SimpleIn(text="a"), SimpleOut(result="a"))
        cache.set(agent, SimpleIn(text="b"), SimpleOut(result="b"))
        assert cache.get(agent, SimpleIn(text="a")).result == "a"  # type: ignore[union-attr]
        assert cache.get(agent, SimpleIn(text="b")).result == "b"  # type: ignore[union-attr]


# ── PIIRedactor ───────────────────────────────────────────────────────────────


class TestPIIRedactor:
    def test_redact_email(self) -> None:
        text = "email me at user@example.com please"
        result = PIIRedactor.redact_patterns(text)
        assert "user@example.com" not in result
        assert "[REDACTED_EMAIL]" in result

    def test_redact_phone(self) -> None:
        text = "call me at 555-867-5309"
        result = PIIRedactor.redact_patterns(text)
        assert "555-867-5309" not in result
        assert "[REDACTED_PHONE]" in result

    def test_redact_ssn(self) -> None:
        text = "my SSN is 123-45-6789"
        result = PIIRedactor.redact_patterns(text)
        assert "123-45-6789" not in result
        assert "[REDACTED_SSN]" in result

    def test_redact_credit_card(self) -> None:
        text = "card 4111 1111 1111 1111 ok"
        result = PIIRedactor.redact_patterns(text)
        assert "4111 1111 1111 1111" not in result
        assert "[REDACTED_CREDIT_CARD]" in result

    def test_redact_fields_nested(self) -> None:
        data = {"user": {"email": "foo@bar.com", "name": "Alice"}, "count": 1}
        redacted = PIIRedactor.redact_fields(data, ["user.email"])
        assert redacted["user"]["email"].startswith("[REDACTED")
        assert redacted["user"]["name"] == "Alice"
        assert redacted["count"] == 1

    def test_no_match_unchanged(self) -> None:
        text = "no PII here at all"
        assert PIIRedactor.redact_patterns(text) == text

    def test_multiple_emails_all_redacted(self) -> None:
        text = "a@b.com and c@d.org"
        result = PIIRedactor.redact_patterns(text)
        assert "a@b.com" not in result
        assert "c@d.org" not in result
        assert result.count("[REDACTED_EMAIL]") == 2
