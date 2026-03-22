"""Tests for token tracking and metrics collection."""

from typing import Any
from uuid import uuid4

import pytest

from dopeagents.core.types import AgentResult, ExecutionMetrics
from dopeagents.observability.tracer import Span


def test_agent_result_token_accessors() -> None:
    """Test that AgentResult provides convenient token access methods."""
    metrics = ExecutionMetrics(
        run_id=uuid4(),
        latency_ms=123.45,
        cost_usd=0.00567,
        token_count_in=150,
        token_count_out=75,
        llm_calls=2,
        total_tokens=225,
    )

    result: AgentResult[Any] = AgentResult(
        output=None,
        metrics=metrics,
        agent_name="TestAgent",
        success=True,
    )

    # Test token accessors
    assert result.tokens() == 225
    assert result.tokens_breakdown() == {
        "input": 150,
        "output": 75,
        "total": 225,
    }
    assert result.cost_usd() == pytest.approx(0.00567)
    assert result.latency_ms() == pytest.approx(123.45)
    assert result.llm_calls_count() == 2

    # Test formatted metrics
    formatted = result.format_metrics()
    assert "150/75" in formatted
    assert "225 total" in formatted
    assert "$0.00567" in formatted
    assert "2" in formatted


def test_agent_result_token_accessors_no_metrics() -> None:
    """Test that token accessors return sensible defaults when metrics are missing."""
    result: AgentResult[Any] = AgentResult(output=None, success=True)

    assert result.tokens() == 0
    assert result.tokens_breakdown() == {"input": 0, "output": 0, "total": 0}
    assert result.cost_usd() == 0.0
    assert result.llm_calls_count() == 0
    assert result.format_metrics() == "No metrics available"


def test_span_token_accumulation() -> None:
    """Test that tokens accumulate across multiple LLM calls in a span."""
    span = Span(name="test_span", run_id=uuid4())

    # Simulate first LLM call
    span.set_attribute("llm.tokens_in", 100)
    span.set_attribute("llm.tokens_out", 50)
    span.set_attribute("llm.cost_usd", 0.001)
    span.set_attribute("llm.call_count", 1)

    # Simulate second LLM call (accumulate, don't overwrite)
    span.set_attribute("llm.tokens_in", span.attributes.get("llm.tokens_in", 0) + 75)
    span.set_attribute("llm.tokens_out", span.attributes.get("llm.tokens_out", 0) + 40)
    span.set_attribute("llm.cost_usd", span.attributes.get("llm.cost_usd", 0.0) + 0.0008)
    span.set_attribute("llm.call_count", span.attributes.get("llm.call_count", 0) + 1)

    # Verify accumulation
    assert span.attributes["llm.tokens_in"] == 175
    assert span.attributes["llm.tokens_out"] == 90
    assert span.attributes["llm.cost_usd"] == pytest.approx(0.0018)
    assert span.attributes["llm.call_count"] == 2


def test_execution_metrics_aggregation() -> None:
    """Test that ExecutionMetrics properly aggregates span data."""
    metrics = ExecutionMetrics(
        run_id=uuid4(),
        latency_ms=250.5,
        cost_usd=0.00150,
        token_count_in=200,
        token_count_out=120,
        llm_calls=3,
    )

    # Verify legacy aliases are set
    assert metrics.total_cost_usd == 0.00150
    assert metrics.total_input_tokens == 200
    assert metrics.total_output_tokens == 120
    assert metrics.total_tokens == 320
    assert metrics.total_latency_seconds == pytest.approx(0.2505)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
