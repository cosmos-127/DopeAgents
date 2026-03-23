"""Bounded hybrid step: runs an LLM step with optional tool calling.

This is the core mechanism that makes the hybrid approach work.
It wraps a single LLM step and gives it bounded access to tools.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from dopeagents.agents._researcher.tools import ToolBudget, ToolCall, ToolExecutor, ToolResult

logger = logging.getLogger(__name__)


def _run_async_safe(coro: Any) -> Any:
    """Bridge async to sync safely."""
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


@dataclass
class HybridStepResult:
    """Result of a hybrid step execution."""

    structured_output: Any
    tool_calls: list[ToolResult] = field(default_factory=list)
    budget_summary: dict[str, Any] = field(default_factory=dict)
    llm_rounds: int = 1
    tools_enabled: bool = False


class HybridStepRunner:
    """Runs a single LLM step with bounded tool access.

    The fundamental contract:
    1. The PIPELINE decides WHEN this step runs (coded orchestration)
    2. The LLM decides WHETHER to use tools WITHIN this step (bounded autonomy)
    3. The BUDGET enforces hard limits (no runaway loops)
    4. The FINAL output is always structured (Pydantic model)
    """

    def __init__(
        self,
        extract_fn: Any,
        chat_fn: Any | None = None,
        tool_executor: ToolExecutor | None = None,
        budget: ToolBudget | None = None,
        tools_enabled: bool = True,
    ):
        self._extract = extract_fn
        self._chat = chat_fn
        self._tool_executor = tool_executor
        self._budget = budget or ToolBudget()
        self._tools_enabled = (
            tools_enabled and (chat_fn is not None) and (tool_executor is not None)
        )

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        output_model: type[BaseModel],
        model: Any = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> HybridStepResult:
        """Execute the step with optional bounded tool calling.

        If tools are enabled: prompt with tools -> tool loop -> structured extraction.
        If tools are disabled: direct structured extraction.
        """
        if not self._tools_enabled or not tools:
            output = self._extract(
                response_model=output_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                force_json_mode=True,
                allow_fallback=True,
            )
            return HybridStepResult(
                structured_output=output,
                tools_enabled=False,
                llm_rounds=1,
            )

        # ── Hybrid mode: LLM with bounded tools ──────────────────
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_with_budget(system_prompt)},
            {"role": "user", "content": user_prompt},
        ]

        all_tool_results: list[ToolResult] = []
        budget = ToolBudget(
            max_calls=self._budget.max_calls,
            max_calls_per_tool=dict(self._budget.max_calls_per_tool),
        )
        llm_rounds = 0

        # Type narrowing: these are guaranteed non-None by self._tools_enabled check
        chat_fn: Callable[..., Any] = self._chat  # type: ignore[assignment]
        tool_executor: ToolExecutor = self._tool_executor  # type: ignore[assignment]

        while not budget.exhausted:
            llm_rounds += 1

            tool_choice: str = "auto"
            if budget.remaining <= 0:
                tool_choice = "none"

            available_tools: list[dict[str, Any]] | None = [
                t for t in tools if budget.can_call(t["function"]["name"])
            ]

            if not available_tools:
                tool_choice = "none"
                available_tools = None

            response = chat_fn(
                messages=messages,
                tools=available_tools if tool_choice != "none" else None,
                tool_choice=tool_choice if available_tools else None,
                model=model,
            )

            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": self._get_response_content(response),
                    }
                )
                break

            messages.append(self._format_assistant_tool_message(response))

            for tc in tool_calls:
                if not budget.can_call(tc.name):
                    tool_result = ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        result={"error": f"Budget exceeded for tool '{tc.name}'"},
                        success=False,
                        error="Budget exceeded",
                    )
                else:
                    budget.record_call(tc.name)
                    tool_result = _run_async_safe(tool_executor.execute(tc))

                all_tool_results.append(tool_result)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result.result),
                    }
                )
        if not all_tool_results:
            logger.warning(
                "LLM did not invoke any tools despite %d being available — "
                "falling back to extraction-only analysis",
                len(tools),
            )
        # ── Final extraction: structured output ──────────────────
        messages.append(
            {
                "role": "user",
                "content": (
                    "Now provide your final analysis incorporating all tool results. "
                    "Be comprehensive and cite specific findings from the tools you used."
                ),
            }
        )

        output = self._extract(
            response_model=output_model,
            messages=messages,
            model=model,
        )

        return HybridStepResult(
            structured_output=output,
            tool_calls=all_tool_results,
            budget_summary=budget.summary(),
            llm_rounds=llm_rounds,
            tools_enabled=True,
        )

    def _build_system_with_budget(self, base_prompt: str) -> str:
        return (
            base_prompt + f"\n\n## Tool Usage Guidelines\n"
            f"You have a budget of {self._budget.max_calls} tool calls total.\n"
            f"Per-tool limits: {json.dumps(self._budget.max_calls_per_tool)}\n"
            f"Use tools STRATEGICALLY — only when the value justifies the cost.\n"
            f"Focus tools on:\n"
            f"- Claims that are surprising or critical to the research\n"
            f"- Gaps that significantly weaken the analysis\n"
            f"- Sources that seem highly relevant but need verification\n"
            f"Do NOT use tools for:\n"
            f"- Obvious or well-known facts\n"
            f"- Low-relevance tangential information\n"
            f"- Information already well-covered by existing sources"
        )

    def _parse_tool_calls(self, response: Any) -> list[ToolCall]:
        calls: list[ToolCall] = []

        # OpenAI / instructor format
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                try:
                    args = (
                        json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments
                    )
                    calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=args,
                        )
                    )
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning("Failed to parse tool call: %s", e)

        # Anthropic format
        elif hasattr(response, "content"):
            for block in getattr(response, "content", []):
                if getattr(block, "type", None) == "tool_use":
                    calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input if isinstance(block.input, dict) else {},
                        )
                    )

        return calls

    def _get_response_content(self, response: Any) -> str:
        if hasattr(response, "content") and isinstance(response.content, str):
            return response.content
        if hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content or ""
        return str(response)

    def _format_assistant_tool_message(self, response: Any) -> dict[str, Any]:
        # OpenAI format
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            return {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (msg.tool_calls or [])
                ],
            }
        return {"role": "assistant", "content": self._get_response_content(response)}
