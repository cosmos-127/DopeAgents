"""Contract checker for agent composition (T3.11)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dopeagents.contracts.types import CompatibilityResult, FieldMapping, TypeCompatibility
from dopeagents.observability.logging import get_logger

if TYPE_CHECKING:
    from dopeagents.core.agent import Agent

logger = get_logger(__name__)


class ContractChecker:
    """Validates that two agents can be composed in sequence.

    Checks field overlap, required field coverage, and type compatibility
    between an agent's output and another agent's input.

    Rules (DD §4.3):
    - Rule 1: Field overlap — at least one output field matches an input field
    - Rule 2: Required coverage — all required input fields are satisfiable
    - Rule 3: Type compatibility — types match exactly or via allowed coercion
    """

    @staticmethod
    def check(  # noqa: C901
        source_agent: Agent,  # type: ignore[type-arg]
        target_agent: Agent,  # type: ignore[type-arg]
        field_mappings: dict[str, str] | None = None,
    ) -> CompatibilityResult:
        """Check if source_agent output is compatible with target_agent input.

        Args:
            source_agent: The agent producing output
            target_agent: The agent consuming input
            field_mappings: Optional explicit field name mappings

        Returns:
            CompatibilityResult with compatibility status and details
        """
        field_mappings = field_mappings or {}

        # Get output and input schemas
        source_output = source_agent.output_type()
        target_input = target_agent.input_type()

        source_fields = ContractChecker._extract_fields(source_output)
        target_fields = ContractChecker._extract_fields(target_input)

        # Build mappings
        inferred_mappings: list[FieldMapping] = []
        covered_target_fields: dict[str, bool] = (
            {f: f in field_mappings.values() for f in target_fields}
            if field_mappings
            else dict.fromkeys(target_fields, False)
        )

        # Apply explicit mappings first
        for source_field, target_field in field_mappings.items():
            if source_field in source_fields and target_field in target_fields:
                compatibility = ContractChecker._check_type_compatibility(
                    source_fields[source_field],
                    target_fields[target_field],
                )
                mapping = FieldMapping(
                    source_field=source_field,
                    target_field=target_field,
                    type_coercion=compatibility.coercion,
                )
                inferred_mappings.append(mapping)
                covered_target_fields[target_field] = True

        # Infer remaining mappings from field overlap
        for source_field in source_fields:
            if source_field in target_fields:
                compatibility = ContractChecker._check_type_compatibility(
                    source_fields[source_field],
                    target_fields[source_field],
                )
                if compatibility.compatible:
                    mapping = FieldMapping(
                        source_field=source_field,
                        target_field=source_field,
                        type_coercion=compatibility.coercion,
                    )
                    inferred_mappings.append(mapping)
                    covered_target_fields[source_field] = True

        # Check Rule 2: Required field coverage
        errors: list[str] = []
        for target_field, field_info in target_fields.items():
            if field_info["required"] and not covered_target_fields.get(target_field, False):
                errors.append(f"Required field '{target_field}' not covered by source agent output")

        # Check Rule 1 failure
        if not inferred_mappings and errors:
            errors.insert(0, "No field overlap between agents")

        # Build warnings
        warnings: list[str] = []
        source_fields_used = {m.source_field for m in inferred_mappings}
        for source_field in source_fields:
            if source_field not in source_fields_used:
                warnings.append(f"Source field '{source_field}' is not used by target agent")

        compatible = len(errors) == 0

        return CompatibilityResult(
            compatible=compatible,
            source_agent=source_agent.name,
            target_agent=target_agent.name,
            mappings=inferred_mappings,
            errors=errors,
            warnings=warnings,
            field_coverage=covered_target_fields,
        )

    @staticmethod
    def _extract_fields(model: type) -> dict[str, dict[str, Any]]:
        """Extract field names and type info from a Pydantic model."""
        if not hasattr(model, "model_fields"):
            return {}

        fields: dict[str, dict[str, Any]] = {}
        model_fields = model.model_fields

        for field_name, field_info in model_fields.items():
            fields[field_name] = {
                "type": field_info.annotation,
                "required": field_info.is_required(),
                "field_info": field_info,
            }

        return fields

    @staticmethod
    def _check_type_compatibility(
        source_type_info: dict[str, Any], target_type_info: dict[str, Any]
    ) -> TypeCompatibility:
        """Check if source type is compatible with target type.

        Supports:
        - Exact match: str → str
        - Coercion: int → float, str → int (if parseable)
        Returns error for incompatible: str → int, dict → list
        """
        source_type = source_type_info.get("type")
        target_type = target_type_info.get("type")

        # Exact match
        if source_type == target_type:
            return TypeCompatibility(
                source_type=str(source_type),
                target_type=str(target_type),
                compatible=True,
            )

        # Allowed coercions
        coercions = {
            (int, float): "int_to_float",
            (bool, int): "bool_to_int",
        }

        if (source_type, target_type) in coercions:
            return TypeCompatibility(
                source_type=str(source_type),
                target_type=str(target_type),
                compatible=True,
                coercion=coercions[(source_type, target_type)],  # type: ignore[index]
            )

        # Incompatible
        return TypeCompatibility(
            source_type=str(source_type),
            target_type=str(target_type),
            compatible=False,
            reason=f"No coercion path from {source_type} to {target_type}",
        )
