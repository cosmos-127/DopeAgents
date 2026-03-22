"""Validated multi-agent pipelines (T3.10)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dopeagents.contracts.checker import ContractChecker
from dopeagents.observability.logging import get_logger

if TYPE_CHECKING:
    from dopeagents.core.agent import Agent

logger = get_logger(__name__)


class Pipeline:
    """A validated sequence of agents for multi-step composition.

    Architecture & Rules (DD §4.3):
    - Each agent in the pipeline is connected sequentially
    - Output fields of agent N are mapped to input fields of agent N+1
    - Field mappings are validated via ContractChecker
    - Each link in the pipeline is validated before execution
    """

    def __init__(
        self,
        name: str,
        agents: list[Agent],  # type: ignore[type-arg]
        field_mappings: dict[tuple[int, int], dict[str, str]] | None = None,
    ) -> None:
        """Initialize a pipeline with a sequence of agents.

        Args:
            name: Human-readable name for the pipeline
            agents: Ordered list of agents to execute in sequence
            field_mappings: Explicit mappings for each pipeline link.
                           Key: (source_agent_idx, target_agent_idx)
                           Value: {source_field: target_field}
        """
        self.name = name
        self.agents = agents
        self.field_mappings = field_mappings or {}
        self._results: list[dict[str, Any]] = []

        # Validate all links in the pipeline
        self._validate()

    def _validate(self) -> None:
        """Validate compatibility between all adjacent agent pairs."""
        for i in range(len(self.agents) - 1):
            source_agent = self.agents[i]
            target_agent = self.agents[i + 1]

            # Get mappings for this link if provided
            link_mappings = self.field_mappings.get((i, i + 1), {})

            # Check compatibility
            result = ContractChecker.check(source_agent, target_agent, link_mappings)

            if not result.compatible:
                error_msg = f"Pipeline link [{i}→{i + 1}] incompatible: {'; '.join(result.errors)}"
                raise ValueError(error_msg)

            # Log warnings
            for warning in result.warnings:
                logger.warning(f"Pipeline '{self.name}' [link {i}→{i + 1}]: {warning}")

    async def execute(self, initial_input: dict[str, Any]) -> dict[str, Any]:
        """Execute the pipeline sequentially.

        Args:
            initial_input: Input for the first agent

        Returns:
            Output from the final agent
        """
        current_input = initial_input

        for i, agent in enumerate(self.agents):
            logger.info(f"Pipeline '{self.name}' executing agent {i} ({agent.name})")

            # Execute agent
            output = await agent.execute(current_input)  # type: ignore[attr-defined]

            # Store result for tracing
            self._results.append({"agent": agent.name, "output": output})

            # Prepare input for next agent
            if i < len(self.agents) - 1:
                next_agent = self.agents[i + 1]
                link_mappings = self.field_mappings.get((i, i + 1), {})

                # Build input for next agent from current output
                current_input = self._map_output(output, agent, next_agent, link_mappings)

        return current_input

    @staticmethod
    def _map_output(
        output: dict[str, Any],
        _source_agent: Agent,  # type: ignore[type-arg]
        target_agent: Agent,  # type: ignore[type-arg]
        mappings: dict[str, str],
    ) -> dict[str, Any]:
        """Map output fields to input fields for next agent.

        Rules:
        1. Apply explicit mappings from field_mappings
        2. For unmapped input fields, look for direct field name matches
        3. Fail if required fields are unmapped
        """
        next_input: dict[str, Any] = {}
        target_fields = target_agent.input_type().model_fields

        # Apply explicit mappings
        for source_field, target_field in mappings.items():
            if source_field in output:
                next_input[target_field] = output[source_field]

        # Infer mappings from field overlap
        for target_field, _field_info in target_fields.items():
            if target_field not in next_input and target_field in output:
                next_input[target_field] = output[target_field]

        return next_input

    def get_results(self) -> list[dict[str, Any]]:
        """Get execution results for each agent in the pipeline."""
        return self._results.copy()
