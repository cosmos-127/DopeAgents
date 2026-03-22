"""OpenTelemetry-backed tracer (T2.4)."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from uuid import UUID

from dopeagents.observability.tracer import Span, Tracer


class OTelTracer(Tracer):
    """OpenTelemetry-backed tracer.

    Bridges DopeAgents spans to OTel spans so they appear in Jaeger,
    Honeycomb, Grafana Tempo, etc.

    Requires: pip install dopeagents[otel]
    """

    def __init__(self, service_name: str = "dopeagents") -> None:
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
        except ImportError as exc:
            raise ImportError(
                "OpenTelemetry is required for OTelTracer.\n"
                "Install with: pip install dopeagents[otel]"
            ) from exc

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        otel_trace.set_tracer_provider(provider)
        self._tracer = otel_trace.get_tracer("dopeagents")

    @contextmanager
    def span(
        self,
        name: str,
        run_id: UUID,
        trace_id: UUID | None = None,
    ) -> Generator[Span, None, None]:
        dopeagents_span = Span(name=name, run_id=run_id, trace_id=trace_id)
        with self._tracer.start_as_current_span(name) as otel_span:
            otel_span.set_attribute("dopeagents.run_id", str(run_id))
            if trace_id:
                otel_span.set_attribute("dopeagents.trace_id", str(trace_id))
            try:
                yield dopeagents_span
            finally:
                import contextlib

                for key, value in dopeagents_span.attributes.items():
                    with contextlib.suppress(Exception):
                        otel_span.set_attribute(key, value)
                for event in dopeagents_span.events:
                    otel_span.add_event(event["name"], event.get("attributes", {}))
