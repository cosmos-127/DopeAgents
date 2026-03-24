"""Graceful degradation chain — most-capable to cheapest/most-reliable (T2.10)."""

from __future__ import annotations

import warnings

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from dopeagents.resilience.fallback import FallbackChain

if TYPE_CHECKING:
    from dopeagents.core.agent import Agent
    from dopeagents.core.context import AgentContext


class DegradationResult(BaseModel):
    """Output wrapper that records WHICH agent produced the result and why."""

    output: Any = None
    agent_used: str = ""
    degradation_reason: str | None = None


class DegradationChain(FallbackChain):
    """Ordered agent chain from most-capable to cheapest/most-reliable.

    Tries agents in order and returns a ``DegradationResult`` that records
    which agent produced the output and the errors from agents that were
    skipped. The last agent in the chain SHOULD be deterministic (no LLM)
    to guarantee a final result; a warning is emitted if it requires LLM.
    """

    def __init__(self, agents: list[Agent]) -> None:  # type: ignore[type-arg]
        super().__init__(agents)
        last_agent_class = type(agents[-1])
        if last_agent_class.requires_llm:
            warnings.warn(
                f"Last agent in DegradationChain ('{agents[-1].name}') requires an LLM. "
                f"Consider adding a deterministic fallback as the final option to guarantee "
                f"a result even when all LLM providers are unavailable.",
                UserWarning,
                stacklevel=2,
            )

    def run_with_degradation(
        self,
        input: Any,
        context: AgentContext | None = None,
    ) -> DegradationResult:
        """Try each agent in order; return first success with provenance metadata.

        The executor wraps this method — callers always get a
        ``DegradationResult``, never a bare exception (unless all agents fail).

        Args:
            input: Agent input (must be compatible with the primary agent's InputT).
            context: Optional execution context.

        Returns:
            DegradationResult with output, agent_used, and degradation_reason.

        Raises:
            AgentExecutionError: When every agent in the chain has failed.
        """
        from dopeagents.errors import AgentExecutionError

        errors: list[str] = []
        for agent in self.agents:
            try:
                result = agent.run(input, context)
                output = result.output if hasattr(result, "output") else result
                return DegradationResult(
                    output=output,
                    agent_used=agent.name,
                    degradation_reason="; ".join(errors) if errors else None,
                )
            except Exception as exc:
                errors.append(f"{agent.name}: {type(exc).__name__}: {exc}")

        raise AgentExecutionError(
            message=("All agents in DegradationChain failed: " + "; ".join(errors)),
            original_error="DegradationChain exhausted",
        )
