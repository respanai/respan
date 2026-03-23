from datetime import datetime

import pytest
from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes

from respan_sdk.constants.otlp_constants import (
    OTLP_ATTR_KEY,
    OTLP_ATTR_VALUE,
    OTLP_FLAGS_KEY,
    OTLP_LINKS_KEY,
    OTLP_SPAN_ID_KEY,
    OTLP_STRING_VALUE,
    OTLP_TRACE_ID_KEY,
)
from respan_sdk.constants.span_attributes import RESPAN_LINK_TIMESTAMP
from respan_sdk.utils.data_processing.id_processing import format_span_id, format_trace_id
from respan_tracing import RespanTelemetry, SpanLink, span_link_to_otel, span_to_link, get_client
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
    otel_link = span_link_to_otel(_resume_link())

    assert format(otel_link.context.trace_id, "032x") == "a" * 32
    assert format(otel_link.context.span_id, "016x") == "b" * 16
    assert otel_link.context.is_remote is True
    assert int(otel_link.context.trace_flags) == 1
    assert otel_link.attributes == {"link.type": "resume"}


def test_span_link_rejects_invalid_hex_ids():
    with pytest.raises(ValueError, match="trace_id must be 32 hex characters"):
        span_link_to_otel(SpanLink(trace_id="1234", span_id="b" * 16))

    with pytest.raises(ValueError, match="span_id must be a hexadecimal string"):
        span_link_to_otel(SpanLink(trace_id="a" * 32, span_id="not-a-span-id!!!"))

    with pytest.raises(Exception):
        SpanLink(trace_id=None, span_id="b" * 16)  # type: ignore[arg-type]


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
    # Span may appear twice: auto-exported on span end + process_spans
    linked_spans = [s for s in exported_spans if s.links]
    assert len(linked_spans) >= 1
    assert len(linked_spans[0].links) == 1
    assert format(linked_spans[0].links[0].context.span_id, "016x") == "b" * 16


def test_span_buffer_accepts_raw_otel_links(clean_exporter):
    raw_link = trace.Link(span_link_to_otel(_resume_link()).context, {"link.type": "resume"})
    telemetry, _ = clean_exporter
    client = get_client()

    with client.get_span_buffer("resume-trace") as buffer:
        buffer.create_span(
            "workflow_execution",
            attributes=_processable_attributes(),
            links=[raw_link],
        )
        buffered_span = buffer.get_all_spans()[0]

    assert len(buffered_span.links) == 1
    assert format(buffered_span.links[0].context.trace_id, "032x") == "a" * 32
    telemetry.flush()


def test_span_buffer_rejects_invalid_link_objects(clean_exporter):
    telemetry, _ = clean_exporter
    client = get_client()

    with client.get_span_buffer("resume-trace") as buffer:
        with pytest.raises(
            TypeError,
            match="links must contain SpanLink or opentelemetry.trace.Link instances",
        ):
            buffer.create_span(
                "workflow_execution",
                attributes=_processable_attributes(),
                links=["invalid-link"],  # type: ignore[list-item]
            )
    telemetry.flush()


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

    assert OTLP_LINKS_KEY in otlp_span
    assert len(otlp_span[OTLP_LINKS_KEY]) == 1

    link_payload = otlp_span[OTLP_LINKS_KEY][0]
    assert link_payload[OTLP_TRACE_ID_KEY] == "a" * 32
    assert link_payload[OTLP_SPAN_ID_KEY] == "b" * 16
    assert link_payload[OTLP_FLAGS_KEY] == 257

    serialized_attributes = {
        item[OTLP_ATTR_KEY]: item[OTLP_ATTR_VALUE][OTLP_STRING_VALUE]
        for item in link_payload["attributes"]
    }
    assert serialized_attributes == {"link.type": "resume"}


def test_span_link_timestamp_merged_into_attributes():
    """SpanLink.timestamp should be auto-merged into OTel link attributes."""
    link = SpanLink(
        trace_id="a" * 32,
        span_id="b" * 16,
        attributes={"link.type": "resume"},
        timestamp="2026-03-08T12:00:00Z",
    )
    otel_link = span_link_to_otel(link)
    assert otel_link.attributes[RESPAN_LINK_TIMESTAMP] == "2026-03-08T12:00:00Z"
    assert otel_link.attributes["link.type"] == "resume"


def test_span_link_no_timestamp_no_extra_attribute():
    """SpanLink without timestamp should not add respan.link.timestamp."""
    link = SpanLink(
        trace_id="a" * 32,
        span_id="b" * 16,
        attributes={"link.type": "resume"},
    )
    otel_link = span_link_to_otel(link)
    assert RESPAN_LINK_TIMESTAMP not in otel_link.attributes
    assert otel_link.attributes == {"link.type": "resume"}


def test_span_link_timestamp_does_not_mutate_original_attributes():
    """Merging timestamp must not mutate the original SpanLink.attributes dict."""
    original_attrs = {"link.type": "resume"}
    link = SpanLink(
        trace_id="a" * 32,
        span_id="b" * 16,
        attributes=original_attrs,
        timestamp="2026-03-08T12:00:00Z",
    )
    span_link_to_otel(link)
    assert RESPAN_LINK_TIMESTAMP not in original_attrs


# ---------- span_to_link() tests ----------


def test_span_to_link_captures_ids_from_live_span(clean_exporter):
    """span_to_link() should extract trace_id and span_id from a live span."""
    telemetry, _ = clean_exporter
    tracer = trace.get_tracer("test-span-to-link")

    with tracer.start_as_current_span("source-span") as span:
        link = span_to_link(span)

    ctx = span.get_span_context()
    assert link.trace_id == format_trace_id(ctx.trace_id)
    assert link.span_id == format_span_id(ctx.span_id)
    telemetry.flush()


def test_span_to_link_auto_captures_timestamp(clean_exporter):
    """span_to_link() should auto-capture start_time as ISO 8601 timestamp."""
    telemetry, _ = clean_exporter
    tracer = trace.get_tracer("test-span-to-link")

    with tracer.start_as_current_span("source-span") as span:
        link = span_to_link(span)

    assert link.timestamp is not None
    # Should be valid ISO 8601
    dt = datetime.fromisoformat(link.timestamp)
    assert dt.year >= 2020
    telemetry.flush()


def test_span_to_link_includes_custom_attributes(clean_exporter):
    """span_to_link() should pass through custom attributes."""
    telemetry, _ = clean_exporter
    tracer = trace.get_tracer("test-span-to-link")

    with tracer.start_as_current_span("source-span") as span:
        link = span_to_link(span, attributes={"link.type": "resume", "link.source": "pause"})

    assert link.attributes == {"link.type": "resume", "link.source": "pause"}
    telemetry.flush()


def test_span_to_link_rejects_invalid_span_context():
    """span_to_link() should raise ValueError for invalid spans."""
    invalid_span = trace.INVALID_SPAN
    with pytest.raises(ValueError, match="invalid SpanContext"):
        span_to_link(invalid_span)


def test_span_to_link_roundtrips_through_otel(clean_exporter):
    """SpanLink from span_to_link() should convert cleanly to OTel Link."""
    telemetry, _ = clean_exporter
    tracer = trace.get_tracer("test-span-to-link")

    with tracer.start_as_current_span("source-span") as span:
        link = span_to_link(span, attributes={"link.type": "resume"})

    otel_link = span_link_to_otel(link)
    ctx = span.get_span_context()
    assert format_trace_id(otel_link.context.trace_id) == format_trace_id(ctx.trace_id)
    assert format_span_id(otel_link.context.span_id) == format_span_id(ctx.span_id)
    # Timestamp should be merged into attributes
    assert RESPAN_LINK_TIMESTAMP in otel_link.attributes
    assert otel_link.attributes["link.type"] == "resume"
    telemetry.flush()
