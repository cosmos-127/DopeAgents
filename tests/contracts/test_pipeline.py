"""Tests for Pipeline — validated multi-agent composition."""

from typing import Any

import pytest

from pydantic import BaseModel

from dopeagents.contracts.pipeline import Pipeline

# ── Test Agents ────────────────────────────────────────────────────────────


class Output1(BaseModel):
    """Output of Agent 1."""

    text: str
    score: float


class Input2(BaseModel):
    """Input of Agent 2 - compatible with Output1."""

    text: str
    score: float


class Output2(BaseModel):
    """Output of Agent 2."""

    result: str
    confidence: float


class Input3(BaseModel):
    """Input of Agent 3 - incompatible with Output2 (different field names)."""

    data: str
    level: int


class IncompatibleInput(BaseModel):
    """Input that doesn't match any output."""

    unrelated_field: str


def create_mock_agent(name: str, input_type: type, output_type: type) -> Any:
    """Create a mock agent with specified input/output types."""

    class MockAgent:
        def __init__(self) -> None:
            self.name = name
            self._input_type = input_type
            self._output_type = output_type

        def input_type(self) -> type[BaseModel]:
            return self._input_type

        def output_type(self) -> type[BaseModel]:
            return self._output_type

        async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
            """Mock execution."""
            # Return some output based on the output type
            if self._output_type == Output1:
                return {"text": "summary", "score": 0.85}
            elif self._output_type == Output2:
                return {"result": "final", "confidence": 0.92}
            else:
                return {}

    return MockAgent()


# ── Tests ──────────────────────────────────────────────────────────────────


def test_pipeline_compatible_agents() -> None:
    """Test Pipeline with compatible agents validates at construction."""

    agent1 = create_mock_agent("Agent1", IncompatibleInput, Output1)
    agent2 = create_mock_agent("Agent2", Input2, Output2)

    # Should construct without error since Output1 matches Input2
    pipeline = Pipeline(name="TestPipeline", agents=[agent1, agent2])

    assert pipeline.name == "TestPipeline"
    assert len(pipeline.agents) == 2


def test_pipeline_incompatible_agents_raises() -> None:
    """Test Pipeline with incompatible agents raises at construction."""

    agent1 = create_mock_agent("Agent1", IncompatibleInput, Output2)
    agent2 = create_mock_agent("Agent2", Input3, Output2)

    # Should raise PipelineValidationError because Output2 doesn't match Input3
    with pytest.raises(ValueError) as exc_info:
        Pipeline(name="IncompatiblePipeline", agents=[agent1, agent2])

    assert "incompatible" in str(exc_info.value).lower()


def test_pipeline_single_agent() -> None:
    """Test Pipeline with single agent (trivial case)."""

    agent = create_mock_agent("SingleAgent", IncompatibleInput, Output1)

    # Single agent should always succeed (no links to validate)
    pipeline = Pipeline(name="SingleAgentPipeline", agents=[agent])

    assert len(pipeline.agents) == 1


def test_pipeline_three_agents_compatible() -> None:
    """Test Pipeline with three agents all compatible in sequence."""

    # All agents use the same input/output
    agent1 = create_mock_agent("Agent1", IncompatibleInput, Output1)
    agent2 = create_mock_agent("Agent2", Input2, Output1)  # Input2 matches Output1
    agent3 = create_mock_agent("Agent3", Input2, Output1)  # Input2 matches Output1

    # Chain: A1 (Output1) → A2 (Input2, Output1) → A3 (Input2, Output1)
    pipeline = Pipeline(name="ThreeAgentPipeline", agents=[agent1, agent2, agent3])

    assert len(pipeline.agents) == 3


def test_pipeline_with_explicit_field_mappings() -> None:
    """Test Pipeline with explicit field mappings."""

    agent1 = create_mock_agent("Agent1", IncompatibleInput, Output1)
    agent2 = create_mock_agent("Agent2", IncompatibleInput, Output2)

    # Provide explicit mapping: Agent1.text → Agent2's compatible field
    # Since agent2 takes IncompatibleInput (unrelated_field),
    # explicit mapping is needed
    field_mappings = {
        (0, 1): {"text": "unrelated_field"}  # Map Agent1's text to Agent2's unrelated_field
    }

    pipeline = Pipeline(
        name="MappedPipeline", agents=[agent1, agent2], field_mappings=field_mappings
    )

    assert len(pipeline.agents) == 2


def test_pipeline_execution_order() -> None:
    """Test that pipeline executes agents in specified order."""

    execution_log = []

    class LoggingAgent:
        def __init__(self, name: str):
            self.name = name
            self.execution_count = 0

        def input_type(self) -> type[BaseModel]:
            return Output1  # Use Output1 as input for all (creates compatibility)

        def output_type(self) -> type[BaseModel]:
            return Output1

        async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
            execution_log.append(self.name)
            self.execution_count += 1
            return {"text": f"{self.name}_output", "score": 0.5}

    agent1 = LoggingAgent("Agent1")
    agent2 = LoggingAgent("Agent2")
    agent3 = LoggingAgent("Agent3")

    pipeline = Pipeline(name="LoggingPipeline", agents=[agent1, agent2, agent3])  # type: ignore[list-item]

    # Verify agents stored in correct order
    assert pipeline.agents[0].name == "Agent1"
    assert pipeline.agents[1].name == "Agent2"
    assert pipeline.agents[2].name == "Agent3"


def test_pipeline_validation_error_message() -> None:
    """Test PipelineValidationError includes details."""

    agent1 = create_mock_agent("FirstAgent", IncompatibleInput, Output2)
    agent2 = create_mock_agent("SecondAgent", Input3, Output2)

    with pytest.raises(ValueError) as exc_info:
        Pipeline(name="DetailedErrorPipeline", agents=[agent1, agent2])

    error_msg = str(exc_info.value)
    # Error should mention the incompatibility and agents/steps
    assert "link" in error_msg.lower() or "incompatible" in error_msg.lower()


def test_pipeline_results_storage() -> None:
    """Test Pipeline stores execution results."""

    class TrackingAgent:
        def __init__(self, name: str):
            self.name = name

        def input_type(self) -> type[BaseModel]:
            return Output1  # Ensure compatibility

        def output_type(self) -> type[BaseModel]:
            return Output1

        async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
            return {"text": "tracked", "score": 0.7}

    agent1 = TrackingAgent("TrackAgent1")
    agent2 = TrackingAgent("TrackAgent2")

    pipeline = Pipeline(name="TrackingPipeline", agents=[agent1, agent2])  # type: ignore[list-item]

    # Results should be empty until execution
    assert len(pipeline.get_results()) == 0


def test_pipeline_map_output() -> None:
    """Test Pipeline._map_output correctly constructs next agent input."""

    agent1 = create_mock_agent("Agent1", IncompatibleInput, Output1)
    agent2 = create_mock_agent("Agent2", Input2, Output2)

    # Map Output1 to Input2
    output = {"text": "source_text", "score": 0.9, "extra": "ignored"}

    mapped = Pipeline._map_output(
        output=output,
        _source_agent=agent1,
        target_agent=agent2,
        mappings={},  # No explicit mappings, infer from field overlap
    )

    # Should have "text" and "score" from overlap; extra should not be included
    assert "text" in mapped
    assert "score" in mapped
    assert mapped["text"] == "source_text"
    assert mapped["score"] == 0.9


def test_pipeline_map_output_with_explicit_mapping() -> None:
    """Test Pipeline._map_output with explicit field mappings."""

    agent1 = create_mock_agent("Agent1", IncompatibleInput, Output1)
    agent2 = create_mock_agent("Agent2", IncompatibleInput, Output2)

    output = {"text": "data", "score": 42}

    # Map "text" → "unrelated_field"
    mapped = Pipeline._map_output(
        output=output,
        _source_agent=agent1,
        target_agent=agent2,
        mappings={"text": "unrelated_field"},
    )

    assert "unrelated_field" in mapped
    assert mapped["unrelated_field"] == "data"
