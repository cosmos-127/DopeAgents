"""Tracer abstraction and implementations (T2.3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from uuid import UUID


class Span:
    """Lightweight trace span that collects attributes and events.

    Attributes are arbitrary key/value pairs (model name, token counts, etc.).
    Events are timestamped points of interest within the span duration.
    """

    def __init__(self, name: str, run_id: UUID, trace_id: UUID | None = None) -> None:
        self.name = name
        self.run_id = run_id
        self.trace_id = trace_id
        self.attributes: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute (overwrites existing key)."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Append a named event with optional attribute payload."""
        self.events.append({"name": name, "attributes": attributes or {}})


class Tracer(ABC):
    """Abstract tracer. Concrete implementations write to console, OTel, etc."""

    @abstractmethod
    @contextmanager
    def span(
        self,
        name: str,
        run_id: UUID,
        trace_id: UUID | None = None,
    ) -> Generator[Span, None, None]:
        """Open a trace span as a context manager, yielding a Span object."""
        ...

    @classmethod
    def noop(cls) -> Tracer:
        """Return a no-op tracer that discards all spans."""
        return NoopTracer()


class NoopTracer(Tracer):
    """Tracer that does nothing — the default when observability is disabled."""

    @contextmanager
    def span(
        self,
        name: str,
        run_id: UUID,
        trace_id: UUID | None = None,
    ) -> Generator[Span, None, None]:
        yield Span(name=name, run_id=run_id, trace_id=trace_id)


class ConsoleTracer(Tracer):
    """Tracer that prints span open/close to stdout — useful for development."""

    @contextmanager
    def span(
        self,
        name: str,
        run_id: UUID,
        trace_id: UUID | None = None,
    ) -> Generator[Span, None, None]:
        span = Span(name=name, run_id=run_id, trace_id=trace_id)
        print(f"[TRACE] \u25b6 {name} (run={run_id})")
        try:
            yield span
        finally:
            print(f"[TRACE] \u25c0 {name} | {span.attributes}")
