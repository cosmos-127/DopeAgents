"""Detect whether a model supports tool calling reliably."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelCapability:
    """What a model can reliably do."""

    supports_tool_calling: bool
    supports_structured_output: bool
    max_context_tokens: int
    recommended_max_tools: int
    tier: str  # "strong", "medium", "weak"


_STRONG_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "claude-3-5-sonnet",
    "claude-3-opus",
    "claude-3-sonnet",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "llama-3.1-70b",
    "llama-3.1-405b",
    "llama-3.3-70b",
    "mistral-large",
    "mixtral-8x22b",
    "deepseek-v3",
    "deepseek-chat",
    "qwen2.5-72b",
}

_MEDIUM_MODELS = {
    "gpt-3.5-turbo",
    "claude-3-haiku",
    "gemini-1.5-flash",
    "gemini-2.0-flash-lite",
    "llama-3.1-8b",
    "llama-3.2-3b",
    "mistral-small",
    "mistral-7b",
    "qwen2.5-7b",
    "qwen2.5-14b",
    "phi-3-medium",
}

_WEAK_MODELS = {
    "llama-3.2-1b",
    "phi-3-mini",
    "phi-2",
    "gemma-2-2b",
    "gemma-2b",
    "tinyllama",
    "qwen2.5-3b",
    "qwen2.5-1.5b",
}


def detect_capability(model_name: str | None) -> ModelCapability:
    """Detect model capability from its name.

    Uses substring matching so "gpt-4o-2024-08-06" matches "gpt-4o".
    Defaults to medium tier (no tools) for unknown models.
    """
    if model_name is None:
        return ModelCapability(
            supports_tool_calling=False,
            supports_structured_output=True,
            max_context_tokens=4096,
            recommended_max_tools=0,
            tier="weak",
        )

    name = model_name.lower().strip()

    for pattern in _STRONG_MODELS:
        if pattern in name:
            return ModelCapability(
                supports_tool_calling=True,
                supports_structured_output=True,
                max_context_tokens=128_000,
                recommended_max_tools=6,
                tier="strong",
            )

    for pattern in _MEDIUM_MODELS:
        if pattern in name:
            return ModelCapability(
                supports_tool_calling=True,
                supports_structured_output=True,
                max_context_tokens=32_000,
                recommended_max_tools=3,
                tier="medium",
            )

    for pattern in _WEAK_MODELS:
        if pattern in name:
            return ModelCapability(
                supports_tool_calling=False,
                supports_structured_output=True,
                max_context_tokens=8_000,
                recommended_max_tools=0,
                tier="weak",
            )

    # Unknown model — safe default: no tools
    return ModelCapability(
        supports_tool_calling=False,
        supports_structured_output=True,
        max_context_tokens=16_000,
        recommended_max_tools=0,
        tier="medium",
    )
