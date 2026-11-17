"""Structural wrapper spans an instrumentation marked with
``RESPAN_INTERNAL_DROP_SPAN`` are left out, and their children re-parented by
``RESPAN_INTERNAL_EXPORT_PARENT``, as the JS exporter does (semantic style only)."""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respan_sdk.constants.span_attributes import (
    RESPAN_INTERNAL_DROP_SPAN,
    RESPAN_INTERNAL_EXPORT_PARENT,
)
from respan_tracing.exporters.respan import (
    _prepare_spans_for_export,
    _span_to_otlp_json,
)


def _spans():
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("root") as root:
        root_id = f"{root.get_span_context().span_id:016x}"
        with tracer.start_as_current_span(
            "wrapper", attributes={RESPAN_INTERNAL_DROP_SPAN: True}
        ):
            with tracer.start_as_current_span(
                "work", attributes={RESPAN_INTERNAL_EXPORT_PARENT: root_id}
            ):
                pass
    return memory.get_finished_spans(), root_id


def _sent(spans):
    return {
        span.name: _span_to_otlp_json(span) for span in _prepare_spans_for_export(spans)
    }


def test_a_marked_wrapper_is_dropped_and_its_child_moves_up():
    spans, root_id = _spans()
    sent = _sent(spans)
    assert set(sent) == {"root", "work"}
    assert sent["work"]["parentSpanId"] == root_id
    for body in sent.values():
        assert not any(a["key"].startswith("respan.internal") for a in body["attributes"])


def test_the_legacy_style_sends_the_tree_as_emitted(monkeypatch):
    monkeypatch.setenv("RESPAN_SPAN_NAME_STYLE", "legacy")
    spans, root_id = _spans()
    sent = _sent(spans)
    assert set(sent) == {"root", "wrapper", "work"}
    assert sent["work"]["parentSpanId"] != root_id
