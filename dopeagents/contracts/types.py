"""Contract types and models (T3.10)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from dopeagents.errors import DopeAgentsError


class FieldMapping(BaseModel):
    """Describes how a field maps between two agents."""

    source_field: str = Field(description="Output field from source agent")
    target_field: str = Field(description="Input field to target agent")
    type_coercion: str | None = Field(
        default=None, description="Type coercion applied (e.g., 'int_to_float')"
    )


class TypeCompatibility(BaseModel):
    """Describes type compatibility between two fields."""

    source_type: str
    target_type: str
    compatible: bool = Field(description="Whether types are compatible")
    coercion: str | None = Field(
        default=None, description="Coercion method if compatible but not exact"
    )
    reason: str | None = Field(default=None, description="Explanation if incompatible")


class CompatibilityResult(BaseModel):
    """Result of checking agent compatibility."""

    compatible: bool = Field(description="Whether agents can be composed in sequence")
    source_agent: str = Field(description="First agent name")
    target_agent: str = Field(description="Second agent name")
    mappings: list[FieldMapping] = Field(
        default_factory=list, description="Inferred field mappings"
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Errors preventing composition (e.g., missing required fields)",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings about composition (e.g., unused fields)",
    )
    field_coverage: dict[str, bool] = Field(
        default_factory=dict,
        description="Maps target field names to coverage status",
    )


class PipelineValidationError(DopeAgentsError):
    """Raised when agent pipeline validation fails.

    Includes details about which step failed and why.
    """

    pipeline_step: int = Field(description="Zero-indexed step in the pipeline")
    source_agent: str = Field(description="Source agent name")
    target_agent: str = Field(description="Target agent name")
    errors: list[str] = Field(default_factory=list, description="Validation errors")

    def __str__(self) -> str:
        """Format error message."""
        msg = (
            f"Pipeline validation failed at step {self.pipeline_step}: "
            f"{self.source_agent} → {self.target_agent}. "
            f"Errors: {'; '.join(self.errors)}"
        )
        return msg
