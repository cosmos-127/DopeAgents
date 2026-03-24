"""PII detection and redaction for observability logs (T2.13)."""

from __future__ import annotations

import re

from typing import Any, ClassVar


class PIIRedactor:
    """Regex-based PII detection and field-level redaction.

    Used by AgentExecutor (when ``config.enable_pii_redaction`` is True and
    ``config.log_prompts`` is True) to scrub sensitive data from log output
    before it reaches any observability back-end.

    The patterns are intentionally broad to prefer false-positive redaction
    over false-negative leakage.
    """

    PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "email": re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        "phone": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    }

    @staticmethod
    def redact_fields(data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        """Return a copy of *data* with the listed field paths redacted.

        Field paths use dot notation for nested dicts, e.g. ``"user.email"``.
        """
        redacted = data.copy()
        for field_path in fields:
            parts = field_path.split(".")
            PIIRedactor._redact_nested(redacted, parts)
        return redacted

    @staticmethod
    def _redact_nested(data: dict[str, Any], path_parts: list[str]) -> None:
        if not path_parts or not isinstance(data, dict):
            return
        if len(path_parts) == 1:
            key = path_parts[0]
            if key in data:
                value = data[key]
                data[key] = (
                    f"[REDACTED: {len(value)} chars]" if isinstance(value, str) else "[REDACTED]"
                )
        else:
            key = path_parts[0]
            if key in data and isinstance(data[key], dict):
                PIIRedactor._redact_nested(data[key], path_parts[1:])

    @staticmethod
    def redact_patterns(text: str) -> str:
        """Replace any PII patterns found in *text* with labelled placeholders.

        Each match is replaced with ``[REDACTED_<TYPE>]`` where *TYPE* is one of
        ``EMAIL``, ``PHONE``, ``SSN``, or ``CREDIT_CARD``.
        """
        for pii_type, pattern in PIIRedactor.PATTERNS.items():
            text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", text)
        return text

    @staticmethod
    def add_pattern(name: str, pattern: str) -> None:
        """Register a custom regex pattern for PII detection.

        Args:
            name: Short identifier used in the ``[REDACTED_<NAME>]`` placeholder.
            pattern: Python regex string.
        """
        PIIRedactor.PATTERNS[name] = re.compile(pattern)
