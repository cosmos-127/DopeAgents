"""FunctionTool — wraps a Python callable as a Tool (T3.2)."""

from __future__ import annotations

import inspect

from collections.abc import Callable as CallableType
from typing import Any, get_args, get_origin

from pydantic import BaseModel, create_model

from dopeagents.tools.base import Tool


class FunctionTool(Tool[BaseModel, BaseModel]):
    """Wraps a Python callable as a Tool.

    Introspects the function signature to:
    - Generate input schema as dynamic Pydantic model (input_type)
    - Generate JSON Schema from type annotations (function_schema)
    - Map function calls through the Tool interface

    Example:
        def search(query: str, limit: int = 10) -> list[str]:
            '''Search for documents.'''
            return ["doc1", "doc2"]

        tool = FunctionTool(search)
        output = tool.call(SearchInput(query="AI", limit=5))
    """

    def __init__(
        self,
        func: CallableType[..., Any],
        description: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize FunctionTool.

        Args:
            func: The Python callable to wrap
            description: Tool description (defaults to func docstring)
            name: Tool name (defaults to func name)
        """
        self.func = func
        self._name = name or func.__name__
        self._description = description or (func.__doc__ or "")
        self._input_model: type[BaseModel] | None = None
        self._output_model: type[BaseModel] | None = None

    @property
    def name(self) -> str:
        """Tool name."""
        return self._name

    @property
    def description(self) -> str:
        """Tool description."""
        return self._description

    def input_type(self) -> type[BaseModel]:
        """Generate and cache the Pydantic input model from function signature.

        Inspects the function's parameters and type hints to create a dynamic
        Pydantic model matching the function's input contract.

        Returns:
            Dynamic Pydantic model class
        """
        if self._input_model is not None:
            return self._input_model

        sig = inspect.signature(self.func)
        fields_dict: dict[str, Any] = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            annotation = param.annotation
            if annotation == inspect.Parameter.empty:
                annotation = str  # Default to str if no annotation

            if param.default == inspect.Parameter.empty:
                # Required parameter
                fields_dict[param_name] = (annotation, ...)
            else:
                # Optional parameter with default
                fields_dict[param_name] = (annotation, param.default)

        # Create dynamic Pydantic model
        self._input_model = create_model(
            f"{self._name}Input",
            __base__=BaseModel,
            **fields_dict,
        )
        return self._input_model

    def output_type(self) -> type[BaseModel]:
        """Return the output model.

        For FunctionTool, returns a generic BaseModel since the function
        may return any type. For typed tools, override this.

        Returns:
            OutputT model type
        """
        if self._output_model is not None:
            return self._output_model

        # Default: return the function's return type wrapped in a model
        sig = inspect.signature(self.func)
        if sig.return_annotation != inspect.Signature.empty:
            result_type = sig.return_annotation
        else:
            result_type = Any

        self._output_model = create_model(
            f"{self._name}Output",
            result=(result_type, ...),
            __base__=BaseModel,
        )
        return self._output_model

    def call(self, input: BaseModel) -> BaseModel:
        """Execute the function with the given input.

        Args:
            input: Validated input matching self.input_type()

        Returns:
            Output wrapped in self.output_type()
        """
        # Extract arguments from the Pydantic model
        args = input.model_dump() if hasattr(input, "model_dump") else dict(input)

        # Call the function
        result = self.func(**args)

        # Wrap result if needed
        output_model: type[BaseModel] = self.output_type()
        if isinstance(result, dict):
            return output_model(**result)
        else:
            return output_model(result=result)

    def function_schema(self) -> dict[str, Any]:
        """Generate JSON Schema from function type hints.

        Returns a dict with type information extracted from function signature.
        """
        sig = inspect.signature(self.func)
        schema: dict[str, Any] = {
            "name": self._name,
            "description": self._description,
            "parameters": {},
        }

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_schema = self._annotation_to_schema(param.annotation)
            if param.default == inspect.Parameter.empty:
                schema["parameters"][param_name] = {
                    **param_schema,
                    "required": True,
                }
            else:
                schema["parameters"][param_name] = {
                    **param_schema,
                    "default": param.default,
                    "required": False,
                }

        return schema

    def to_llm_function_schema(self) -> dict[str, Any]:
        """Generate OpenAI-compatible JSON Schema for function calling.

        Returns:
            Dict with name, description, and parameters (JSON Schema format)
        """
        input_model: type[BaseModel] = self.input_type()
        input_schema = input_model.model_json_schema()

        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": self._description,
                "parameters": {
                    "type": "object",
                    "properties": input_schema.get("properties", {}),
                    "required": input_schema.get("required", []),
                },
            },
        }

    @staticmethod
    def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
        """Convert a Python type annotation to a JSON Schema dict.

        Handles basic types: str, int, float, bool, list, dict
        """
        if annotation == inspect.Parameter.empty:
            return {"type": "string"}

        if annotation is str:
            return {"type": "string"}
        if annotation is int:
            return {"type": "integer"}
        if annotation is float:
            return {"type": "number"}
        if annotation is bool:
            return {"type": "boolean"}

        origin = get_origin(annotation)
        if origin is list:
            args = get_args(annotation)
            item_type = args[0] if args else Any
            return {
                "type": "array",
                "items": FunctionTool._annotation_to_schema(item_type),
            }

        if origin is dict:
            return {"type": "object"}

        # Default to string for unknown types
        return {"type": "string"}
