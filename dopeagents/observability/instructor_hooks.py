"""Instructor observability hooks (T2.5).

Hooks into Instructor's event system to capture LLM call details
(token counts, cost, model name) into a DopeAgents Span without
any code changes inside the agent's run() method.
"""

from __future__ import annotations

from typing import Any

from dopeagents.observability.tracer import Span


class InstructorObservabilityHooks:
    """Wires four callbacks into an Instructor client's event system.

    Instructor emits events at each lifecycle stage:
    - ``completion:kwargs``   — before the LLM call (model, message count)
    - ``completion:response`` — after the LLM call (tokens, LiteLLM cost)
    - ``completion:error``    — on LLM call failure
    - ``parse:error``         — on output validation failure (before auto-retry)
    """

    def __init__(self, span: Span) -> None:
        self.span = span

    def on_completion_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Capture the request being sent to the LLM."""
        self.span.set_attribute("llm.model", kwargs.get("model", "unknown"))
        self.span.set_attribute("llm.messages_count", len(kwargs.get("messages", [])))
        self.span.add_event(
            "llm_request_sent",
            {"model": kwargs.get("model", "unknown")},
        )

    def on_completion_response(self, response: Any) -> None:
        """Capture token counts and LiteLLM-computed cost from the response.

        Accumulates tokens across multiple LLM calls (for multi-step agents).
        """
        usage = getattr(response, "usage", None)
        if usage:
            # Accumulate tokens across all LLM calls in the span
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0

            current_in = self.span.attributes.get("llm.tokens_in", 0)
            current_out = self.span.attributes.get("llm.tokens_out", 0)

            self.span.set_attribute("llm.tokens_in", current_in + prompt_tokens)
            self.span.set_attribute("llm.tokens_out", current_out + completion_tokens)

            # Track call count for multi-step pipelines
            call_count = self.span.attributes.get("llm.call_count", 0)
            self.span.set_attribute("llm.call_count", call_count + 1)

        # Accumulate cost across calls
        hidden = getattr(response, "_hidden_params", {}) or {}
        if isinstance(hidden, dict):
            step_cost = float(hidden.get("response_cost", 0.0) or 0.0)
            current_cost = self.span.attributes.get("llm.cost_usd", 0.0)
            self.span.set_attribute("llm.cost_usd", current_cost + step_cost)

    def on_completion_error(self, error: Exception) -> None:
        """Capture LLM call failures."""
        self.span.add_event(
            "llm_call_error",
            {"error": str(error), "type": type(error).__name__},
        )

    def on_parse_error(self, error: Exception) -> None:
        """Capture Instructor validation failures before auto-retry."""
        self.span.add_event(
            "instructor_validation_error",
            {
                "error": str(error),
                "note": "Instructor will auto-retry with validation error in prompt",
            },
        )

    def attach(self, client: Any) -> None:
        """Attach all four hooks to an Instructor client via client.on()."""
        client.on("completion:kwargs", self.on_completion_kwargs)
        client.on("completion:response", self.on_completion_response)
        client.on("completion:error", self.on_completion_error)
        client.on("parse:error", self.on_parse_error)
