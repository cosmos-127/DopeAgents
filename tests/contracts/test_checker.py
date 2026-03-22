"""Tests for ContractChecker — field mapping and type compatibility validation."""

from pydantic import BaseModel

from dopeagents.contracts.checker import ContractChecker
from dopeagents.contracts.types import CompatibilityResult, FieldMapping

# ── Test Models ────────────────────────────────────────────────────────────


class SimpleInput(BaseModel):
    """Input with basic fields."""

    text: str
    count: int


class SimpleOutput(BaseModel):
    """Output matching some input fields."""

    text: str
    count: int
    summary: str


class DisjointOutput(BaseModel):
    """Output with no overlapping fields."""

    result: str
    score: float


class RequiredInputMissing(BaseModel):
    """Input with required field not in output."""

    text: str
    count: int
    required_field: str  # No default = required


class OptionalInput(BaseModel):
    """Input with all optional fields."""

    text: str | None = None
    count: int | None = None


class IntOutput(BaseModel):
    """Output with int field."""

    number: int


class FloatInput(BaseModel):
    """Input expecting float."""

    value: float


# ── Tests ──────────────────────────────────────────────────────────────────


def test_checker_compatible_exact_match() -> None:
    """Test field overlap: output has fields matching input names."""

    # Create mock agents with proper method binding
    class SourceAgent:
        name = "Source"

        def input_type(self) -> type[BaseModel]:
            return SimpleOutput

        def output_type(self) -> type[BaseModel]:
            return SimpleOutput

    class TargetAgent:
        name = "Target"

        def input_type(self) -> type[BaseModel]:
            return SimpleInput

        def output_type(self) -> type[BaseModel]:
            return SimpleInput

    source = SourceAgent()
    target = TargetAgent()

    result = ContractChecker.check(source, target)  # type: ignore[arg-type]

    assert result.compatible is True
    assert len(result.errors) == 0
    assert len(result.mappings) >= 2  # "text" and "count" should map
    assert result.source_agent == "Source"
    assert result.target_agent == "Target"


def test_checker_incompatible_field_overlap() -> None:
    """Test Rule 1 failure: no field overlap between agents."""

    class SourceAgent:
        name = "Source"

        def input_type(self) -> type[BaseModel]:
            return DisjointOutput

        def output_type(self) -> type[BaseModel]:
            return DisjointOutput

    class TargetAgent:
        name = "Target"

        def input_type(self) -> type[BaseModel]:
            return SimpleInput

        def output_type(self) -> type[BaseModel]:
            return SimpleInput

    source = SourceAgent()
    target = TargetAgent()

    result = ContractChecker.check(source, target)  # type: ignore[arg-type]

    assert result.compatible is False
    assert any("field overlap" in err.lower() for err in result.errors)


def test_checker_required_field_not_covered() -> None:
    """Test Rule 2 failure: required input field not covered by output."""

    class SourceAgent:
        name = "Source"

        def input_type(self) -> type[BaseModel]:
            return SimpleOutput

        def output_type(self) -> type[BaseModel]:
            return SimpleOutput

    class TargetAgent:
        name = "Target"

        def input_type(self) -> type[BaseModel]:
            return RequiredInputMissing

        def output_type(self) -> type[BaseModel]:
            return RequiredInputMissing

    source = SourceAgent()
    target = TargetAgent()

    result = ContractChecker.check(source, target)  # type: ignore[arg-type]

    assert result.compatible is False
    assert any("required_field" in err.lower() for err in result.errors)


def test_checker_type_compatibility_int_to_float() -> None:
    """Test Rule 3: int → float coercion is allowed."""

    result = ContractChecker._check_type_compatibility(
        {"type": int, "required": True, "field_info": None},
        {"type": float, "required": True, "field_info": None},
    )

    assert result.compatible is True
    assert result.coercion == "int_to_float"


def test_checker_type_incompatibility() -> None:
    """Test Rule 3: incompatible types raise error."""

    result = ContractChecker._check_type_compatibility(
        {"type": str, "required": True, "field_info": None},
        {"type": int, "required": True, "field_info": None},
    )

    assert result.compatible is False
    assert result.reason is not None


def test_checker_explicit_field_mappings() -> None:
    """Test explicit field mappings override inferred mappings."""

    class SourceAgent:
        name = "Source"

        def input_type(self) -> type[BaseModel]:
            return SimpleOutput

        def output_type(self) -> type[BaseModel]:
            return SimpleOutput

    class TargetAgent:
        name = "Target"

        def input_type(self) -> type[BaseModel]:
            return SimpleInput

        def output_type(self) -> type[BaseModel]:
            return SimpleInput

    source = SourceAgent()
    target = TargetAgent()

    # Map "summary" → "text" explicitly
    result = ContractChecker.check(source, target, field_mappings={"summary": "text"})  # type: ignore[arg-type]

    assert result.compatible is True
    # "summary" → "text" mapping should be present
    assert any(m.source_field == "summary" and m.target_field == "text" for m in result.mappings)


def test_checker_optional_field_warning() -> None:
    """Test warnings for unused output fields."""

    class SourceAgent:
        name = "Source"

        def input_type(self) -> type[BaseModel]:
            return SimpleOutput

        def output_type(self) -> type[BaseModel]:
            return SimpleOutput

    class TargetAgent:
        name = "Target"

        def input_type(self) -> type[BaseModel]:
            return OptionalInput

        def output_type(self) -> type[BaseModel]:
            return OptionalInput

    source = SourceAgent()
    target = TargetAgent()

    result = ContractChecker.check(source, target)  # type: ignore[arg-type]

    assert result.compatible is True
    assert len(result.warnings) > 0  # "summary" field unused
    assert any("summary" in w for w in result.warnings)


def test_checker_extract_fields() -> None:
    """Test field extraction from Pydantic model."""

    fields = ContractChecker._extract_fields(SimpleInput)

    assert "text" in fields
    assert "count" in fields
    assert fields["text"]["type"] is str
    assert fields["count"]["type"] is int
    assert fields["text"]["required"] is True
    assert fields["count"]["required"] is True


def test_checker_extract_fields_with_optional() -> None:
    """Test field extraction handles optional fields."""

    fields = ContractChecker._extract_fields(OptionalInput)

    assert "text" in fields
    assert "count" in fields
    # Note: Pydantic marks fields with defaults as not required
    assert fields["text"]["required"] is False
    assert fields["count"]["required"] is False


def test_compatibility_result_model() -> None:
    """Test CompatibilityResult Pydantic model."""

    result = CompatibilityResult(
        compatible=True,
        source_agent="Agent1",
        target_agent="Agent2",
        mappings=[
            FieldMapping(source_field="a", target_field="a"),
            FieldMapping(source_field="b", target_field="b", type_coercion="int_to_float"),
        ],
        errors=[],
        warnings=["unused field c"],
    )

    assert result.compatible is True
    assert len(result.mappings) == 2
    assert result.mappings[1].type_coercion == "int_to_float"
    assert len(result.warnings) == 1


def test_type_compatibility_exact_match() -> None:
    """Test type compatibility with exact match."""

    result = ContractChecker._check_type_compatibility(
        {"type": str, "required": True, "field_info": None},
        {"type": str, "required": True, "field_info": None},
    )

    assert result.compatible is True
    assert result.coercion is None


def test_type_compatibility_bool_to_int() -> None:
    """Test type compatibility bool → int coercion."""

    result = ContractChecker._check_type_compatibility(
        {"type": bool, "required": True, "field_info": None},
        {"type": int, "required": True, "field_info": None},
    )

    assert result.compatible is True
    assert result.coercion == "bool_to_int"
