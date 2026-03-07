import pytest
from opentelemetry.semconv_ai import SpanAttributes

from respan_tracing import RespanTelemetry, SpanLink, get_client
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.exporters.respan import _span_to_otlp_json
from respan_tracing.testing import InMemorySpanExporter


@pytest.fixture(scope="module")
def telemetry_env():
    RespanTracer.reset_instance()

    exporter = InMemorySpanExporter()
    telemetry = RespanTelemetry(
        app_name="span-link-tests",
        is_enabled=True,
        is_batching_enabled=False,
    )
    telemetry.add_processor(exporter=exporter, is_batching_enabled=False)

    yield telemetry, exporter

    exporter.clear()
    RespanTracer.reset_instance()


@pytest.fixture
def clean_exporter(telemetry_env):
    _, exporter = telemetry_env
    exporter.clear()
    yield telemetry_env
    exporter.clear()


def _resume_link() -> SpanLink:
    return SpanLink(
        trace_id="0x" + ("a" * 32),
        span_id="0x" + ("b" * 16),
        attributes={"link.type": "resume"},
    )


def _processable_attributes() -> dict[str, str]:
    return {
        "status": "resumed",
        SpanAttributes.TRACELOOP_SPAN_KIND: "workflow",
    }


def test_span_link_normalizes_prefixed_hex_ids():
    otel_link = _resume_link().to_otel_link()

    assert format(otel_link.context.trace_id, "032x") == "a" * 32
    assert format(otel_link.context.span_id, "016x") == "b" * 16
    assert otel_link.context.is_remote is True
    assert int(otel_link.context.trace_flags) == 1
    assert otel_link.attributes == {"link.type": "resume"}


def test_span_link_rejects_invalid_hex_ids():
    with pytest.raises(ValueError, match="trace_id must be 32 hex characters"):
        SpanLink(trace_id="1234", span_id="b" * 16).to_otel_link()

    with pytest.raises(ValueError, match="span_id must be a hexadecimal string"):
        SpanLink(trace_id="a" * 32, span_id="not-a-span-id!!!").to_otel_link()


def test_span_buffer_create_span_preserves_links(clean_exporter):
    telemetry, exporter = clean_exporter
    client = get_client()

    with client.get_span_buffer("resume-trace") as buffer:
        created_span_id = buffer.create_span(
            "workflow_execution",
            attributes=_processable_attributes(),
            links=[_resume_link()],
        )
        buffered_spans = buffer.get_all_spans()

    assert len(buffered_spans) == 1
    buffered_span = buffered_spans[0]
    assert format(buffered_span.get_span_context().span_id, "016x") == created_span_id
    assert len(buffered_span.links) == 1
    assert format(buffered_span.links[0].context.trace_id, "032x") == "a" * 32
    assert buffered_span.links[0].attributes == {"link.type": "resume"}

    assert client.process_spans(buffered_spans) is True
    telemetry.flush()

    exported_spans = exporter.get_finished_spans()
    assert len(exported_spans) == 1
    assert len(exported_spans[0].links) == 1
    assert format(exported_spans[0].links[0].context.span_id, "016x") == "b" * 16


def test_otlp_json_serializes_span_links(clean_exporter):
    client = get_client()

    with client.get_span_buffer("resume-trace") as buffer:
        buffer.create_span(
            "workflow_execution",
            attributes=_processable_attributes(),
            links=[_resume_link()],
        )
        buffered_span = buffer.get_all_spans()[0]

    otlp_span = _span_to_otlp_json(buffered_span)

    assert "links" in otlp_span
    assert len(otlp_span["links"]) == 1

    link_payload = otlp_span["links"][0]
    assert link_payload["traceId"] == "a" * 32
    assert link_payload["spanId"] == "b" * 16
    assert link_payload["flags"] == 257

    serialized_attributes = {
        item["key"]: item["value"]["stringValue"]
        for item in link_payload["attributes"]
    }
    assert serialized_attributes == {"link.type": "resume"}
