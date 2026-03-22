"""Agent metadata and capability tracking."""

from typing import Any

from pydantic import BaseModel, Field


class AgentMetadata(BaseModel):
    """Structured metadata about an agent, returned by Agent.metadata().

    Contains identity, versioning, capability descriptors, and schema refs
    for serialization, MCP exposure, and composition validation.
    """

    name: str = Field(description="Agent name (e.g., 'DeepSummarizer')")
    version: str = Field(default="0.0.1", description="Agent version (SemVer)")
    description: str = Field(default="", description="Human-readable description")
    capabilities: list[str] = Field(
        default_factory=list,
        description="Tags describing what agent does (e.g., ['summarization', 'multi-step'])",
    )
    tags: list[str] = Field(default_factory=list, description="Searchable tags for discovery")
    requires_llm: bool = Field(default=True, description="Whether this agent requires LLM calls")
    default_model: str | None = Field(
        default=None,
        description="Default LLM model (e.g., 'gpt-4o', None if non-LLM)",
    )
    system_prompt: str = Field(default="", description="System prompt used by this agent")
    step_prompts: dict[str, str] = Field(
        default_factory=dict,
        description="Map of step names to their prompts (multi-step agents)",
    )
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="Pydantic schema for InputT (JSON Schema format)",
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="Pydantic schema for OutputT (JSON Schema format)",
    )
