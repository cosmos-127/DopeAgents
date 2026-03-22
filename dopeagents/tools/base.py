"""Base Tool abstraction and interfaces (T3.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    pass

ToolInputT = TypeVar("ToolInputT", bound=BaseModel)
ToolOutputT = TypeVar("ToolOutputT", bound=BaseModel)


class ToolInput(BaseModel):
    """Marker base class for tool inputs."""

    pass


class ToolOutput(BaseModel):
    """Marker base class for tool outputs."""

    pass


class Tool(ABC, Generic[ToolInputT, ToolOutputT]):
    """Abstract base class for all tools.

    Tools are reusable, named functions that agents can call.
    Each tool has an input schema, output schema, and callable interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name (e.g., 'web_search', 'calculator')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass

    @abstractmethod
    def input_type(self) -> type[ToolInputT]:
        """Return the Pydantic input model for this tool."""
        pass

    @abstractmethod
    def output_type(self) -> type[ToolOutputT]:
        """Return the Pydantic output model for this tool."""
        pass

    @abstractmethod
    def call(self, input: ToolInputT) -> ToolOutputT:
        """Execute the tool with the given input.

        Args:
            input: Validated input matching self.input_type()

        Returns:
            Output matching self.output_type()

        Raises:
            ToolExecutionError: If execution fails
        """
        pass

    @abstractmethod
    def to_llm_function_schema(self) -> dict[str, Any]:
        """Generate JSON Schema for LLM function calling.

        Returns OpenAI-compatible function schema for use in LLM API calls.
        Schema includes name, description, and parameters (input schema as JSON Schema).
        """
        pass
