"""RESTTool — HTTP request wrapper (T3.3)."""

from __future__ import annotations

from typing import Any

import httpx

from pydantic import BaseModel, Field

from dopeagents.tools.base import Tool, ToolOutput


class RESTToolInput(BaseModel):
    """Input for RESTTool — HTTP request parameters."""

    method: str = Field(default="GET", pattern="^(GET|POST|PUT|DELETE|PATCH)$")
    url: str = Field(description="Full URL to request")
    headers: dict[str, str] | None = Field(default=None, description="HTTP headers")
    body: dict[str, Any] | None = Field(default=None, description="Request body")


class RESTToolOutput(ToolOutput):
    """Output from RESTTool — HTTP response."""

    status_code: int
    headers: dict[str, str]
    body: dict[str, Any] | str


class RESTTool(Tool[RESTToolInput, RESTToolOutput]):
    """HTTP-based tool for making REST API calls.

    Wraps httpx for synchronous HTTP requests.
    Useful for agents that need to call external APIs.

    Example:
        tool = RESTTool(timeout=30)
        result = tool.call(RESTToolInput(
            method="GET",
            url="https://api.example.com/search?q=python"
        ))
    """

    def __init__(
        self,
        name: str = "rest_api",
        description: str = "Call external HTTP APIs",
        timeout: float = 30.0,
    ) -> None:
        """Initialize RESTTool.

        Args:
            name: Tool name
            description: Tool description
            timeout: HTTP request timeout in seconds
        """
        self._name = name
        self._description = description
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    @property
    def name(self) -> str:
        """Tool name."""
        return self._name

    @property
    def description(self) -> str:
        """Tool description."""
        return self._description

    def input_type(self) -> type[RESTToolInput]:
        """Return input schema."""
        return RESTToolInput

    def output_type(self) -> type[RESTToolOutput]:
        """Return output schema."""
        return RESTToolOutput

    def call(self, input: RESTToolInput) -> RESTToolOutput:
        """Execute the HTTP request.

        Args:
            input: RESTToolInput with method, url, headers, body

        Returns:
            RESTToolOutput with status_code, headers, body

        Raises:
            httpx.RequestError: If the request fails
        """
        method = input.method.upper()
        url = input.url
        headers = input.headers or {}
        body = input.body

        response = self._client.request(
            method=method,
            url=url,
            headers=headers,
            json=body,
        )

        # Try to parse body as JSON, fall back to string
        try:
            response_body: dict[str, Any] | str = response.json()
        except Exception:
            response_body = response.text

        return RESTToolOutput(
            status_code=response.status_code,
            headers=dict(response.headers),
            body=response_body,
        )

    def to_llm_function_schema(self) -> dict[str, Any]:
        """Generate OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": self._description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                            "description": "HTTP method",
                        },
                        "url": {
                            "type": "string",
                            "description": "Full URL to request",
                        },
                        "headers": {
                            "type": "object",
                            "description": "Optional HTTP headers",
                        },
                        "body": {
                            "type": "object",
                            "description": "Optional request body (for POST/PUT/PATCH)",
                        },
                    },
                    "required": ["url"],
                },
            },
        }

    def __del__(self) -> None:
        """Clean up HTTP client."""
        if self._client:
            self._client.close()
