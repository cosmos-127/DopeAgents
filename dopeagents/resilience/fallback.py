"""Ordered agent fallback chain with output compatibility validation (T2.9)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dopeagents.core.agent import Agent


class FallbackChain:
    """Ordered list of agents tried in sequence on primary failure.

    On construction, validates that fallback agents have compatible output
    schemas (warns — does NOT block) so users catch mismatches early.

    The executor iterates ``fallback_chain.agents`` and tries each until one
    succeeds. If all fail, ``AllFallbacksFailedError`` is raised.
    """

    def __init__(self, agents: list[Agent]) -> None:  # type: ignore[type-arg]
        if not agents:
            raise ValueError("FallbackChain requires at least one agent")
        self.agents = agents
        self._validate_output_compatibility()

    def _validate_output_compatibility(self) -> None:
        """Warn when a fallback agent is missing output fields from the primary."""
        if len(self.agents) < 2:
            return
        primary_output = type(self.agents[0]).output_type()
        primary_fields = set(primary_output.model_fields.keys())

        for agent in self.agents[1:]:
            fallback_output = type(agent).output_type()
            fallback_fields = set(fallback_output.model_fields.keys())
            missing = primary_fields - fallback_fields
            if missing:
                warnings.warn(
                    f"Fallback agent '{agent.name}' is missing output fields "
                    f"that the primary agent has: {missing}. "
                    f"Callers relying on those fields may receive None.",
                    UserWarning,
                    stacklevel=3,
                )
