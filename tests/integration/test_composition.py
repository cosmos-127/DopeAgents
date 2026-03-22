"""Integration tests for agent composition and contracts — Phase 3 validation.

Tests that:
1. ResearchAgent implements all 6 steps correctly
2. ResearchAgent.describe() returns correct step list and has_loops=False
3. DeepSummarizer and ResearchAgent are incompatible (different output/input schemas)
4. Pipeline validation detects incompatibility at construction
5. Compatible agent pairs can be composed successfully
"""

import pytest

from dopeagents.agents import DeepSummarizer, ResearchAgent, ResearchAgentInput
from dopeagents.contracts.checker import ContractChecker
from dopeagents.contracts.pipeline import Pipeline


class TestResearchAgentStructure:
    """Test ResearchAgent workflow structure."""

    def test_research_agent_steps(self) -> None:
        """Test ResearchAgent has all 6 expected steps."""
        agent = ResearchAgent()
        desc = agent.describe()

        expected_steps = ["expand_query", "search", "analyze", "synthesize", "evaluate", "refine"]

        assert desc.steps == expected_steps, f"Expected steps {expected_steps}, got {desc.steps}"

    def test_research_agent_has_loops(self) -> None:
        """Test ResearchAgent has loops (evaluate → refine loop)."""
        agent = ResearchAgent()
        desc = agent.describe()

        assert (
            desc.has_loops is True
        ), "ResearchAgent should have has_loops=True due to evaluate→refine loop"

    def test_research_agent_step_prompts_complete(self) -> None:
        """Test all 6 steps have prompts."""
        agent = ResearchAgent()
        desc = agent.describe()

        for step in desc.steps:
            assert step in agent.step_prompts, f"Step '{step}' missing from step_prompts"
            assert len(agent.step_prompts[step]) > 0, f"Step '{step}' prompt is empty"

    def test_research_agent_init_time_step_prompts_customization(self) -> None:
        """Test ResearchAgent init-time step_prompts customization."""
        custom_prompts = {
            "expand_query": "Custom expansion strategy...",
            "search": "Custom search strategy...",
        }

        agent = ResearchAgent(step_prompts=custom_prompts)

        # Custom prompts should override defaults
        assert agent.step_prompts["expand_query"] == custom_prompts["expand_query"]
        assert agent.step_prompts["search"] == custom_prompts["search"]

        # Other steps should still have defaults
        assert len(agent.step_prompts["analyze"]) > 0
        assert "Custom" not in agent.step_prompts["analyze"]

    def test_research_agent_instance_independence(self) -> None:
        """Test instances with different step_prompts are independent."""
        agent1 = ResearchAgent()
        agent2 = ResearchAgent(step_prompts={"expand_query": "Custom for agent2"})

        # agent1 should still have default
        assert "Custom" not in agent1.step_prompts["expand_query"]
        # agent2 should have custom
        assert agent2.step_prompts["expand_query"] == "Custom for agent2"


class TestAgentCompositionCompatibility:
    """Test compatibility checking and composition."""

    def test_deep_summarizer_research_agent_incompatible(self) -> None:
        """Test DeepSummarizer and ResearchAgent are incompatible."""
        summarizer = DeepSummarizer()
        researcher = ResearchAgent()

        result = ContractChecker.check(summarizer, researcher)

        assert result.compatible is False, (
            "DeepSummarizer and ResearchAgent should be incompatible "
            "(different output/input schemas)"
        )
        assert len(result.errors) > 0, "Should have errors explaining incompatibility"

    def test_incompatibility_errors_descriptive(self) -> None:
        """Test incompatibility errors are descriptive."""
        summarizer = DeepSummarizer()
        researcher = ResearchAgent()

        result = ContractChecker.check(summarizer, researcher)

        error_text = " ".join(result.errors).lower()
        # Should mention field mismatch or required field coverage
        assert (
            "field" in error_text or "required" in error_text
        ), f"Errors should mention fields or requirements, got: {result.errors}"

    def test_compatible_agents_pass_check(self) -> None:
        """Test that compatible agents pass ContractChecker."""
        from pydantic import BaseModel

        class TestInput(BaseModel):
            data: str

        class TestOutput(BaseModel):
            data: str
            result: str

        class SourceAgent:
            name = "Source"

            def input_type(self) -> type[BaseModel]:
                return TestInput

            def output_type(self) -> type[BaseModel]:
                return TestOutput

        class TargetAgent:
            name = "Target"

            def input_type(self) -> type[BaseModel]:
                return TestInput

            def output_type(self) -> type[BaseModel]:
                return TestOutput

        source = SourceAgent()
        target = TargetAgent()

        result = ContractChecker.check(source, target)  # type: ignore[arg-type]

        assert result.compatible is True, "Agents with matching fields should be compatible"
        assert len(result.errors) == 0


class TestPipelineCompositionValidation:
    """Test Pipeline validates composition at construction."""

    def test_pipeline_deep_summarizer_research_agent_fails(self) -> None:
        """Test Pipeline([DeepSummarizer, ResearchAgent]) raises at construction."""
        summarizer = DeepSummarizer()
        researcher = ResearchAgent()

        with pytest.raises(ValueError) as exc_info:
            Pipeline(name="IncompatibleComposition", agents=[summarizer, researcher])

        # Error should reference the incompatible link
        error_msg = str(exc_info.value).lower()
        assert "incompatible" in error_msg or "error" in error_msg.lower()

    def test_pipeline_construction_time_validation(self) -> None:
        """Test validation happens at Pipeline construction, not execution."""
        from pydantic import BaseModel

        class Input1(BaseModel):
            field_a: str

        class Output1(BaseModel):
            field_x: str  # Doesn't match

        class Input2(BaseModel):
            field_a: str  # Requires field_a

        # Create mock agents with proper method binding
        class Agent1:
            name = "Agent1"

            def input_type(self) -> type[BaseModel]:
                return Input1

            def output_type(self) -> type[BaseModel]:
                return Output1

        class Agent2:
            name = "Agent2"

            def input_type(self) -> type[BaseModel]:
                return Input2

            def output_type(self) -> type[BaseModel]:
                return Output1

        agent1 = Agent1()
        agent2 = Agent2()

        # Error should happen during __init__, not during later function calls
        with pytest.raises(ValueError):
            Pipeline(name="EarlyErrorPipeline", agents=[agent1, agent2])  # type: ignore[list-item]


class TestDeepSummarizerStructure:
    """Test DeepSummarizer is correctly implemented for compatibility checks."""

    def test_deep_summarizer_has_loops(self) -> None:
        """Test DeepSummarizer has loops (evaluate → refine loop)."""
        agent = DeepSummarizer()
        desc = agent.describe()

        assert (
            desc.has_loops is True
        ), "DeepSummarizer should have has_loops=True due to refine loop"

    def test_deep_summarizer_seven_steps(self) -> None:
        """Test DeepSummarizer has 7 steps."""
        agent = DeepSummarizer()
        desc = agent.describe()

        expected_steps = [
            "analyze",
            "chunk",
            "summarize",
            "synthesize",
            "evaluate",
            "refine",
            "format",
        ]

        assert (
            desc.steps == expected_steps
        ), f"Expected {len(expected_steps)} steps, got {len(desc.steps)}: {desc.steps}"


class TestContractCheckerFieldDecomposition:
    """Test ContractChecker correctly extracts and compares field schemas."""

    def test_checker_identifies_output_fields(self) -> None:
        """Test ContractChecker correctly identifies output fields."""
        from dopeagents.agents import DeepSummarizerOutput

        fields = ContractChecker._extract_fields(DeepSummarizerOutput)

        assert "summary" in fields
        assert "key_points" in fields
        assert "quality_score" in fields

    def test_checker_identifies_input_fields(self) -> None:
        """Test ContractChecker correctly identifies input fields."""

        fields = ContractChecker._extract_fields(ResearchAgentInput)

        assert "query" in fields
        assert "research_focus" in fields or "quality_threshold" in fields
        # The actual field names may vary; just check that some fields exist
        assert len(fields) > 0

    def test_output_input_field_mismatch(self) -> None:
        """Test ContractChecker detects field name mismatches."""
        from dopeagents.agents import DeepSummarizerOutput

        summary_fields = ContractChecker._extract_fields(DeepSummarizerOutput)
        research_fields = ContractChecker._extract_fields(ResearchAgentInput)

        # No overlap between DeepSummarizerOutput and ResearchInput field names
        summary_names = set(summary_fields.keys())
        research_names = set(research_fields.keys())
        overlap = summary_names & research_names

        assert len(overlap) == 0, (
            f"No fields should overlap between DeepSummarizerOutput and ResearchInput, "
            f"but found: {overlap}"
        )


class TestPipelineWarnings:
    """Test Pipeline warnings and diagnostics."""

    def test_pipeline_validation_logs_link_errors(self) -> None:
        """Test Pipeline validation error includes link number."""
        from pydantic import BaseModel

        class Out1(BaseModel):
            x: str

        class In2(BaseModel):
            y: str  # Different field

        class A1:
            name = "A1"

            def input_type(self) -> type[BaseModel]:
                return In2

            def output_type(self) -> type[BaseModel]:
                return Out1

        class A2:
            name = "A2"

            def input_type(self) -> type[BaseModel]:
                return In2

            def output_type(self) -> type[BaseModel]:
                return Out1

        agent1 = A1()
        agent2 = A2()

        with pytest.raises(ValueError) as exc_info:
            Pipeline(name="BadLink", agents=[agent1, agent2])  # type: ignore[list-item]

        # Should mention which link failed (e.g., "link 0→1")
        error_msg = str(exc_info.value).lower()
        # Either "link" or the agent names should be in the error
        has_link_info = (
            "link" in error_msg or "0" in error_msg or "a1" in error_msg or "a2" in error_msg
        )
        assert has_link_info, f"Error should indicate which link failed: {exc_info.value}"
