"""Unit tests for span filtering, deduplication, and instrumentor (no live API)."""
import pytest

# Skip entire module if instrumentor cannot be imported (e.g. missing pkg_resources).
pytest.importorskip("respan_exporter_crewai.instrumentor")
from opentelemetry.sdk.trace.export import SpanExportResult

from respan_exporter_crewai.instrumentor import (
    _SpanDedupeCache,
    _export_crewai_spans,
    RespanCrewAIInstrumentor,
)
from respan_exporter_crewai.utils import is_crewai_span, otel_span_to_dict


# --- _SpanDedupeCache ---


def test_dedupe_cache_add_new_returns_true():
    cache = _SpanDedupeCache(max_size=100)
    assert cache.add(trace_id="trace1", span_id="span1") is True


def test_dedupe_cache_add_duplicate_returns_false():
    cache = _SpanDedupeCache(max_size=100)
    assert cache.add(trace_id="trace1", span_id="span1") is True
    assert cache.add(trace_id="trace1", span_id="span1") is False


def test_dedupe_cache_different_spans_both_added():
    cache = _SpanDedupeCache(max_size=100)
    assert cache.add(trace_id="trace1", span_id="span1") is True
    assert cache.add(trace_id="trace1", span_id="span2") is True
    assert cache.add(trace_id="trace2", span_id="span1") is True


def test_dedupe_cache_none_trace_id_returns_true():
    """Missing trace_id is not cached; allows through."""
    cache = _SpanDedupeCache(max_size=100)
    assert cache.add(trace_id=None, span_id="span1") is True
    assert cache.add(trace_id=None, span_id="span1") is True


def test_dedupe_cache_evicts_when_over_max_size():
    cache = _SpanDedupeCache(max_size=2)
    cache.add(trace_id="t1", span_id="s1")
    cache.add(trace_id="t2", span_id="s2")
    cache.add(trace_id="t3", span_id="s3")  # evicts t1:s1
    # t1:s1 should be evicted, so adding again returns True
    assert cache.add(trace_id="t1", span_id="s1") is True


# --- _export_crewai_spans ---


def _make_crewai_span(trace_id: str = "a" * 32, span_id: str = "b" * 16):
    """Minimal span-like object that is_crewai_span accepts (scope name)."""
    scope = type("Scope", (), {"name": "openinference-instrumentation-crewai"})()
    ctx = type("Ctx", (), {
        "trace_id": int(trace_id, 16) if len(trace_id) == 32 else 0xAB,
        "span_id": int(span_id, 16) if len(span_id) == 16 else 0xCD,
    })()
    return type("Span", (), {
        "instrumentation_scope": scope,
        "attributes": {},
        "context": ctx,
        "name": "test",
        "parent": None,
        "start_time": None,
        "end_time": None,
    })()


def test_export_crewai_spans_none_exporter_returns_success():
    span = _make_crewai_span()
    result = _export_crewai_spans(spans=[span], exporter=None, dedupe=_SpanDedupeCache())
    assert result == SpanExportResult.SUCCESS


def test_export_crewai_spans_filters_non_crewai_spans():
    """Non-CrewAI spans are not passed to exporter.build_payload."""
    crewai_span = _make_crewai_span()
    non_crewai = type("Span", (), {"instrumentation_scope": None, "attributes": {"other": "x"}})()

    build_payload_calls = []
    send_calls = []

    class MockExporter:
        api_key = "test-key"
        def build_payload(self, trace_or_spans):
            build_payload_calls.append(list(trace_or_spans))
            return [{"trace_id": "x"}]
        def send(self, payloads):
            send_calls.append(payloads)

    dedupe = _SpanDedupeCache()
    _export_crewai_spans(spans=[non_crewai, crewai_span], exporter=MockExporter(), dedupe=dedupe)
    assert len(build_payload_calls) == 1
    assert len(build_payload_calls[0]) == 1
    assert build_payload_calls[0][0].get("trace_id")
    assert len(send_calls) == 1


def test_export_crewai_spans_deduplicates_same_trace_span():
    """Same trace_id:span_id only exported once."""
    span = _make_crewai_span(trace_id="c" * 32, span_id="d" * 16)
    build_payload_calls = []

    class MockExporter:
        api_key = "key"
        def build_payload(self, trace_or_spans):
            build_payload_calls.append(list(trace_or_spans))
            return [{"trace_id": "c" * 32}]
        def send(self, payloads):
            pass

    dedupe = _SpanDedupeCache()
    _export_crewai_spans(spans=[span, span], exporter=MockExporter(), dedupe=dedupe)
    assert len(build_payload_calls) == 1
    assert len(build_payload_calls[0]) == 1


def test_export_crewai_spans_no_api_key_does_not_send():
    """When exporter has no api_key, send is not called."""
    span = _make_crewai_span()
    sent = []

    class MockExporter:
        api_key = None
        def build_payload(self, trace_or_spans):
            return [{"trace_id": "x"}]
        def send(self, payloads):
            sent.append(payloads)

    _export_crewai_spans(spans=[span], exporter=MockExporter(), dedupe=_SpanDedupeCache())
    assert len(sent) == 0


# --- RespanCrewAIInstrumentor ---


def test_instrumentor_instrumentation_dependencies():
    inst = RespanCrewAIInstrumentor()
    deps = inst.instrumentation_dependencies()
    assert "crewai" in str(deps)
    assert "openinference" in str(deps).lower()
