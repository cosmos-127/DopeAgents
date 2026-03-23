"""AgentExecutor — full lifecycle management for agent execution (T2.7)."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from pydantic import BaseModel

from dopeagents.core.context import AgentContext
from dopeagents.core.types import AgentResult, ExecutionMetrics
from dopeagents.errors import (
    AgentExecutionError,
    AllFallbacksFailedError,
    BudgetExceededError,
    MaxRetriesExceededError,
)
from dopeagents.lifecycle.hooks import LifecycleHooks
from dopeagents.observability.logging import get_logger
from dopeagents.observability.tracer import Span, Tracer

if TYPE_CHECKING:
    from dopeagents.cache.manager import CacheManager
    from dopeagents.core.agent import Agent
    from dopeagents.cost.guard import BudgetConfig
    from dopeagents.cost.tracker import CostTracker
    from dopeagents.resilience.fallback import FallbackChain
    from dopeagents.resilience.retry import RetryPolicy

logger = get_logger(__name__)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class AgentExecutor:
    """Executes agents with full lifecycle management.

    Wraps agent.run() with:
    - Input/output Pydantic validation
    - Budget pre-checking
    - Cache lookup / storage
    - Instructor observability hook attachment
    - Retry on transient errors
    - Fallback chain execution
    - Metrics collection (latency, cost, tokens)
    - PII redaction in logs
    """

    def __init__(
        self,
        tracer: Tracer | None = None,
        cost_tracker: CostTracker | None = None,
        cache_manager: CacheManager | None = None,
        budget: BudgetConfig | None = None,
        hooks: LifecycleHooks | None = None,
    ) -> None:
        from dopeagents.cost.tracker import CostTracker as _CostTracker

        self.tracer = tracer or Tracer.noop()
        self.cost_tracker = cost_tracker or _CostTracker.noop()
        self.cache_manager = cache_manager
        self.budget = budget
        self.hooks = hooks or LifecycleHooks()

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        agent: Agent[InputT, OutputT],
        input: InputT,
        context: AgentContext | None = None,
        retry_policy: RetryPolicy | None = None,
        fallback_chain: FallbackChain | None = None,
    ) -> AgentResult[OutputT]:
        """Execute an agent through the full lifecycle and return AgentResult."""
        from dopeagents.cost.guard import BudgetGuard

        context = context or AgentContext()
        # Inject infrastructure references into context for agent-level cooperation
        # (e.g., per-step tracing, budget-aware refinement loops)
        context.metadata.setdefault("tracer", self.tracer)
        context.metadata.setdefault("cost_tracker", self.cost_tracker)
        if self.budget:
            context.metadata.setdefault("budget", self.budget)
        run_id = context.run_id
        start_time = time.monotonic()

        with self.tracer.span(
            name=f"agent.{agent.name}",
            run_id=run_id,
            trace_id=getattr(context, "trace_id", None),
        ) as span:
            # ── [1] Pre-execution ──────────────────────────────────────────
            # Validate input
            validated_input = self._validate_input(agent, input)

            # Budget check (per-agent cumulative and global)
            BudgetGuard.check_pre_execution(
                agent=agent,
                context=context,
                cost_tracker=self.cost_tracker,
                budget=self.budget,
            )

            # Cache lookup
            if self.cache_manager:
                cached = self.cache_manager.get(agent, validated_input)
                if cached is not None:
                    elapsed_ms = (time.monotonic() - start_time) * 1000
                    span.set_attribute("cache_hit", True)
                    metrics = ExecutionMetrics(
                        run_id=run_id,
                        latency_ms=elapsed_ms,
                        cache_hit=True,
                    )
                    return AgentResult[OutputT](
                        run_id=run_id,
                        agent_name=agent.name,
                        agent_version=agent.version,
                        timestamp=datetime.now(UTC),
                        output=cached,  # type: ignore[arg-type]
                        metrics=metrics,
                        execution_metrics=metrics,
                        success=True,
                    )

            # Attach Instructor hooks for observability / cost capture
            self._attach_instructor_hooks(agent, span)

            # Lifecycle pre-execution hook
            self.hooks.pre_execution(agent, validated_input, context)

            # ── [2] Execution ──────────────────────────────────────────────
            output: OutputT | None = None
            retry_count = 0
            fallback_used: str | None = None

            try:
                output, retry_count = self._execute_with_retry(
                    agent, validated_input, context, retry_policy
                )
            except Exception as exec_error:
                self.hooks.on_error(agent, validated_input, exec_error, context)
                if fallback_chain:
                    try:
                        output, fallback_used = self._execute_fallback(
                            agent, fallback_chain, validated_input, context
                        )
                    except AllFallbacksFailedError:
                        raise
                else:
                    if isinstance(
                        exec_error,
                        AgentExecutionError | BudgetExceededError | MaxRetriesExceededError,
                    ):
                        raise
                    raise AgentExecutionError(
                        message=str(exec_error),
                        agent_name=agent.name,
                        agent_version=agent.version,
                        original_error=type(exec_error).__name__,
                    ) from exec_error

            # ── [3] Post-execution ─────────────────────────────────────────
            # Validate output
            validated_output = self._validate_output(agent, output)

            # Build metrics from span attributes populated by Instructor hooks
            metrics = self._build_metrics(
                run_id=run_id,
                start_time=start_time,
                retry_count=retry_count,
                fallback_used=fallback_used,
                span=span,
            )

            # Log execution metrics (tokens, cost, latency)
            self._log_execution_metrics(agent, metrics)

            # Record cost
            self.cost_tracker.record(agent, context, metrics)

            # Store in cache
            if self.cache_manager:
                self.cache_manager.set(agent, validated_input, validated_output)

            # Emit span attributes
            span.set_attribute("agent.name", agent.name)
            span.set_attribute("agent.version", agent.version)
            span.set_attribute("metrics.latency_ms", metrics.latency_ms)
            span.set_attribute("metrics.cost_usd", metrics.cost_usd)

            # Lifecycle post-execution hook
            self.hooks.post_execution(agent, validated_input, validated_output, context)

            return AgentResult[OutputT](
                run_id=run_id,
                agent_name=agent.name,
                agent_version=agent.version,
                timestamp=datetime.now(UTC),
                output=validated_output,
                metrics=metrics,
                execution_metrics=metrics,
                success=True,
            )

    # ── Private helpers ───────────────────────────────────────────────────

    def _validate_input(self, agent: Agent, input: Any) -> Any:  # type: ignore[type-arg]
        """Validate (and coerce) input against the agent's InputT schema."""
        try:
            input_type = agent.input_type()
            if isinstance(input, input_type):
                return input
            raw = input.model_dump() if hasattr(input, "model_dump") else input
            return input_type.model_validate(raw)
        except Exception as exc:
            from dopeagents.errors import InputValidationError

            raise InputValidationError(
                message=f"Input validation failed for agent '{agent.name}': {exc}",
                agent_name=agent.name,
                validation_errors=[{"error": str(exc)}],
            ) from exc

    def _validate_output(self, agent: Agent, output: Any) -> Any:  # type: ignore[type-arg]
        """Validate output against the agent's OutputT schema."""
        try:
            output_type = agent.output_type()
            if isinstance(output, output_type):
                return output
            raw = output.model_dump() if hasattr(output, "model_dump") else output
            return output_type.model_validate(raw)
        except Exception as exc:
            from dopeagents.errors import OutputValidationError

            raise OutputValidationError(
                message=f"Output validation failed for agent '{agent.name}': {exc}",
                agent_name=agent.name,
                validation_errors=[{"error": str(exc)}],
            ) from exc

    def _execute_with_retry(
        self,
        agent: Agent,  # type: ignore[type-arg]
        input: Any,
        context: AgentContext,
        retry_policy: RetryPolicy | None,
    ) -> tuple[Any, int]:
        """Execute agent.run() with optional retry on transient errors."""
        if retry_policy is None:
            result = agent.run(input, context)
            # AgentResult wraps the output; unwrap if present
            if hasattr(result, "output"):
                return result.output, 0
            return result, 0

        last_error: Exception | None = None
        retryable = tuple(retry_policy.retryable_errors)

        for attempt in range(retry_policy.max_attempts):
            try:
                result = agent.run(input, context)
                if hasattr(result, "output"):
                    return result.output, attempt
                return result, attempt
            except retryable as exc:
                last_error = exc
                self.hooks.on_retry(agent, input, attempt + 1, exc, context)
                if attempt < retry_policy.max_attempts - 1:
                    delay = retry_policy.delay_seconds * (retry_policy.backoff_factor**attempt)
                    time.sleep(delay)

        from dopeagents.errors import MaxRetriesExceededError

        raise MaxRetriesExceededError(
            message=f"Agent '{agent.name}' failed after {retry_policy.max_attempts} attempts",
            agent_name=agent.name,
            retry_count=retry_policy.max_attempts,
            last_error=str(last_error),
        )

    def _execute_fallback(
        self,
        primary_agent: Agent,  # type: ignore[type-arg]
        fallback_chain: FallbackChain,
        input: Any,
        context: AgentContext,
    ) -> tuple[Any, str]:
        """Try each agent in the fallback chain in order."""
        errors: list[str] = []
        for fallback_agent in fallback_chain.agents:
            try:
                self.hooks.on_fallback(primary_agent, fallback_agent, context)
                adapted = self._adapt_input(input, fallback_agent)
                result = fallback_agent.run(adapted, context)
                output = result.output if hasattr(result, "output") else result
                return output, fallback_agent.name
            except Exception as exc:
                errors.append(f"{fallback_agent.name}: {type(exc).__name__}: {exc}")

        raise AllFallbacksFailedError(
            message=(
                f"All fallback agents for '{primary_agent.name}' failed: " + "; ".join(errors)
            ),
            agent_name=primary_agent.name,
            chain_agents=[a.name for a in fallback_chain.agents],
            errors=errors,
        )

    @staticmethod
    def _adapt_input(input: Any, target_agent: Agent) -> Any:  # type: ignore[type-arg]
        """Attempt to adapt input for a fallback agent with a different schema."""
        target_type = target_agent.input_type()
        raw = input.model_dump() if hasattr(input, "model_dump") else input
        return target_type.model_validate(raw)

    def _attach_instructor_hooks(
        self,
        agent: Agent,  # type: ignore[type-arg]
        span: Span,
    ) -> None:
        """Attach Instructor lifecycle hooks to the agent's client for observability.

        Captures token counts and cost from LiteLLM responses so they appear
        in span attributes without any code changes inside agent.run().
        """
        if not agent.requires_llm:
            return
        try:
            from dopeagents.observability.instructor_hooks import InstructorObservabilityHooks

            client = agent._get_client()
            obs = InstructorObservabilityHooks(span)
            obs.attach(client)
        except Exception:
            # Gracefully degrade if hooks can't be attached
            pass

    def _build_metrics(
        self,
        run_id: UUID,
        start_time: float,
        retry_count: int,
        fallback_used: str | None,
        span: Span,
    ) -> ExecutionMetrics:
        """Assemble ExecutionMetrics from timing and span attributes.

        Extracts and aggregates token counts, cost, and call counts from span
        attributes populated by InstructorObservabilityHooks during LLM calls.
        """
        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Accumulate tokens across all LLM calls (multi-step agents)
        tokens_in = int(span.attributes.get("llm.tokens_in", 0))
        tokens_out = int(span.attributes.get("llm.tokens_out", 0))
        cost_usd = float(span.attributes.get("llm.cost_usd", 0.0))
        llm_calls = int(span.attributes.get("llm.call_count", 0))

        return ExecutionMetrics(
            run_id=run_id,
            latency_ms=elapsed_ms,
            cost_usd=cost_usd,
            token_count_in=tokens_in,
            token_count_out=tokens_out,
            llm_calls=llm_calls,
            retry_count=retry_count,
            fallback_used=fallback_used,
            # Legacy aliases
            total_cost_usd=cost_usd,
            total_input_tokens=tokens_in,
            total_output_tokens=tokens_out,
            total_tokens=tokens_in + tokens_out,
            total_latency_seconds=elapsed_ms / 1000,
        )

    def _log_execution_metrics(
        self,
        agent: Agent,  # type: ignore[type-arg]
        metrics: ExecutionMetrics,
    ) -> None:
        """Log execution metrics (tokens, cost, latency) for visibility.

        Makes token usage transparent to users by logging after each execution.
        """
        if metrics.llm_calls > 0:
            logger.info(
                f"[{agent.name}] Execution complete | "
                f"Tokens: {metrics.token_count_in} in, {metrics.token_count_out} out "
                f"({metrics.total_tokens} total) | "
                f"Cost: ${metrics.cost_usd:.6f} | "
                f"Latency: {metrics.latency_ms:.0f}ms | "
                f"LLM calls: {metrics.llm_calls}"
            )
        else:
            logger.info(
                f"[{agent.name}] Execution complete | "
                f"Latency: {metrics.latency_ms:.0f}ms | "
                f"(no LLM calls)"
            )
