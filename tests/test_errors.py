"""Tests for error handling."""

import pytest
from pydantic import BaseModel
from dopeagents.errors import (
    DopeAgentsError,
    ExtractionError,
    ExtractionValidationError,
    ExtractionProviderError,
    CostError,
    BudgetExceededError,
    BudgetDegradedError,
    TypeResolutionError,
)


class TestErrorStructure:
    """Verify error hierarchy and structured fields."""

    def test_dopeagents_error_base(self) -> None:
        """Base error carries structured metadata."""
        error = DopeAgentsError(
            error_type="test_error",
            message="Test failure",
            agent_name="TestAgent",
        )
        assert error.error_type == "test_error"
        assert error.message == "Test failure"
        assert error.agent_name == "TestAgent"
        assert str(error) == "Test failure"

    def test_extraction_validation_error(self) -> None:
        """Extraction validation error carries validation details."""
        error = ExtractionValidationError(
            error_type="extraction_validation_error",
            message="Schema validation failed",
            step_name="analyze",
            response_model="AnalyzeOutput",
            validation_errors=[
                {"loc": ("summary",), "msg": "field required"}
            ],
        )
        assert error.step_name == "analyze"
        assert len(error.validation_errors) > 0

    def test_extraction_provider_error(self) -> None:
        """Provider error carries status code and retry info."""
        error = ExtractionProviderError(
            error_type="extraction_provider_error",
            message="Rate limit exceeded",
            provider="openai",
            status_code=429,
            retry_after=60,
        )
        assert error.status_code == 429
        assert error.retry_after == 60
        assert error.provider == "openai"

    def test_budget_exceeded_error(self) -> None:
        """Budget exceeded error tracks cost and limit."""
        error = BudgetExceededError(
            error_type="budget_exceeded_error",
            message="Cost limit exceeded",
            current_cost=1.50,
            budget_limit=1.00,
            budget_type="per_call",
        )
        assert error.current_cost == 1.50
        assert error.budget_limit == 1.00
        assert error.budget_type == "per_call"

    def test_budget_degraded_error(self) -> None:
        """Degradation error carries best-so-far output."""
        error = BudgetDegradedError(
            error_type="budget_degraded_error",
            message="Budget exhausted, returning best result",
            current_cost=0.95,
            budget_limit=0.90,
            degraded_output={"summary": "partial"},
        )
        assert error.degraded_output == {"summary": "partial"}
        assert error.current_cost > error.budget_limit

    def test_type_resolution_error(self) -> None:
        """Type resolution error identifies what failed."""
        error = TypeResolutionError(
            error_type="type_resolution_error",
            message="Could not resolve InputType",
            resolution_type="input",
            agent_name="TestAgent",
        )
        assert error.resolution_type == "input"
        assert error.agent_name == "TestAgent"

    def test_error_inheritance(self) -> None:
        """Error subclasses follow inheritance hierarchy."""
        error = BudgetExceededError(
            error_type="budget_exceeded_error",
            message="Test",
            current_cost=1.0,
            budget_limit=0.5,
        )
        assert isinstance(error, DopeAgentsError)
        assert isinstance(error, CostError)
        assert isinstance(error, BaseModel)

    def test_all_errors_are_pydantic_models(self) -> None:
        """All errors are Pydantic models for serialization."""
        error = ExtractionValidationError(
            error_type="extraction_validation_error",
            message="Test",
            step_name="test",
        )
        # Should be serializable to JSON
        json_str = error.model_dump_json()
        assert "extraction_validation_error" in json_str

