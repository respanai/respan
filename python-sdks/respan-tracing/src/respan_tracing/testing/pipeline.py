"""Test an instrumentation through the real Respan pipeline.

Baseline wire-contract tests need the spans Respan would *export* (after the
span processor copied customer context and filtered unexportable spans), not
raw OTel spans. ``capture_exported_spans`` gives a client factory a callback
that records exactly those, and isolates the global tracer provider and the
propagated customer attributes between tests::

    def make(on_export):
        return Respan(api_key="k", base_url=UNREACHABLE,
                      instrumentations=[MyInstrumentor()],
                      span_postprocess_callback=on_export)

    with capture_exported_spans(make) as (client, spans):
        ...run the framework...
    assert all(...for span in spans)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan

from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import _PROPAGATED_ATTRIBUTES

# Nothing listens here: exporter network calls fail fast and are ignored.
UNREACHABLE_BASE_URL = "http://127.0.0.1:9/api"

Client = TypeVar("Client")


@contextmanager
def capture_exported_spans(
    make_client: Callable[[Callable[[ReadableSpan], None]], Client],
) -> Iterator[tuple[Client, list[ReadableSpan]]]:
    spans: list[ReadableSpan] = []
    # Restore by value, not token: clients are often built inside an async
    # test's task while cleanup runs in the fixture's context.
    previous_attributes = _PROPAGATED_ATTRIBUTES.get()
    _PROPAGATED_ATTRIBUTES.set({})
    previous_provider = trace._TRACER_PROVIDER
    client: Any = None
    try:
        client = make_client(spans.append)
        # OTel lets the global provider be set once per process; point it at
        # this client's provider so each test exports through its own pipeline.
        trace._TRACER_PROVIDER = RespanTracer._instance.tracer_provider
        yield client, spans
    finally:
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            shutdown()
        trace._TRACER_PROVIDER = previous_provider
        RespanTracer.reset_instance()
        _PROPAGATED_ATTRIBUTES.set(previous_attributes)
