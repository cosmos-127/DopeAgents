"""Lifecycle hooks — extension points for custom logic at each lifecycle stage (T2.6)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dopeagents.core.agent import Agent
    from dopeagents.core.context import AgentContext


class LifecycleHooks:
    """Extension points for custom logic at each lifecycle stage.

    All methods have no-op default implementations.  Subclass and override
    only the hooks you care about.

    Example::

        class MyHooks(LifecycleHooks):
            def post_execution(self, agent, input, output, context):
                print(f"{agent.name} produced {output}")
    """

    def pre_execution(
        self,
        agent: Agent,  # type: ignore[type-arg]
        input: Any,
        context: AgentContext,
    ) -> None:
        """Called immediately before agent.run() is invoked."""
        pass

    def post_execution(
        self,
        agent: Agent,  # type: ignore[type-arg]
        input: Any,
        output: Any,
        context: AgentContext,
    ) -> None:
        """Called after a successful agent.run() with the validated output."""
        pass

    def on_error(
        self,
        agent: Agent,  # type: ignore[type-arg]
        input: Any,
        error: Exception,
        context: AgentContext,
    ) -> None:
        """Called when agent.run() raises an unhandled exception."""
        pass

    def on_retry(
        self,
        agent: Agent,  # type: ignore[type-arg]
        input: Any,
        attempt: int,
        error: Exception,
        context: AgentContext,
    ) -> None:
        """Called before each retry attempt after a transient failure."""
        pass

    def on_fallback(
        self,
        original_agent: Agent,  # type: ignore[type-arg]
        fallback_agent: Agent,  # type: ignore[type-arg]
        context: AgentContext,
    ) -> None:
        """Called when the primary agent fails and a fallback is attempted."""
        pass

    # ── Instructor-level hooks ────────────────────────────────────────

    def on_extraction_request(
        self,
        agent: Agent,  # type: ignore[type-arg]
        messages: list[Any],
        response_model: Any,
        context: AgentContext | None,
    ) -> None:
        """Called when the agent is about to send a prompt to the LLM."""
        pass

    def on_extraction_response(
        self,
        agent: Agent,  # type: ignore[type-arg]
        response: Any,
        usage: Any,
        context: AgentContext | None,
    ) -> None:
        """Called after the LLM returns a structured response."""
        pass

    def on_extraction_validation_error(
        self,
        agent: Agent,  # type: ignore[type-arg]
        error: Exception,
        attempt: int | None,
        context: AgentContext | None,
    ) -> None:
        """Called when Instructor fails to parse/validate the LLM output."""
        pass
