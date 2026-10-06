import json
from http import HTTPStatus
from types import SimpleNamespace

from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode
from respan_instrumentation_langfuse import LangfuseInstrumentor
from respan_instrumentation_langfuse import instrumentor as instrumentation
from respan_sdk.constants.otlp_constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.span_attributes import (
    RESPAN_CUSTOMER_PARAMS_ID,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SESSION_ID,
    RESPAN_TRACE_GROUP_ID,
)


def _source_span(
    *,
    name="generation",
    attributes=None,
    parent_id=0x0202020202020202,
    status=None,
):
    parent = SimpleNamespace(span_id=parent_id) if parent_id is not None else None
    return SimpleNamespace(
        name=name,
        attributes=attributes or {},
        context=SimpleNamespace(
            trace_id=0x01010101010101010101010101010101,
            span_id=0x0303030303030303,
        ),
        parent=parent,
        start_time=100,
        end_time=200,
        status=status or Status(StatusCode.OK),
        kind=SpanKind.INTERNAL,
    )


def test_current_generation_attributes_translate_to_canonical_chat_span():
    source = _source_span(
        attributes={
            "langfuse.observation.type": "generation",
            "langfuse.observation.input": json.dumps(
                [{"role": "user", "content": "Hello"}]
            ),
            "langfuse.observation.output": json.dumps(
                [{"role": "assistant", "content": "Hi"}]
            ),
            "langfuse.observation.model.name": "gpt-4o-mini",
            "langfuse.observation.usage_details": json.dumps(
                {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                }
            ),
        }
    )

    attrs, status_code, error_message = instrumentation._translate_span(source)

    assert status_code == HTTPStatus.OK
    assert error_message is None
    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Hello"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Hi"
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 7
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 3
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 10
    assert not any(key.startswith("langfuse.") for key in attrs)
    assert "model" not in attrs
    assert "prompt_tokens" not in attrs


def test_legacy_model_and_usage_attributes_remain_supported():
    source = _source_span(
        attributes={
            "langfuse.observation.type": "generation",
            "langfuse.observation.input": "legacy prompt",
            "langfuse.observation.output": "legacy output",
            "langfuse.observation.model": "legacy-model",
            "langfuse.usage.input": 2,
            "langfuse.usage.output": 4,
            "langfuse.usage.total": 6,
        }
    )

    attrs, _, _ = instrumentation._translate_span(source)

    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "legacy-model"
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 2
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 4
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 6


def test_generic_hierarchy_and_trace_attributes_are_preserved():
    root = _source_span(
        name="research",
        parent_id=None,
        attributes={
            "langfuse.observation.type": "span",
            "langfuse.trace.name": "langfuse_research.workflow",
            "user.id": "user-123",
            "session.id": "session-456",
            "langfuse.trace.metadata.example_run_id": "group-16-marker",
        },
    )
    child = _source_span(
        name="search",
        attributes={"langfuse.observation.type": "span"},
    )

    root_attrs, _, _ = instrumentation._translate_span(root)
    child_attrs, _, _ = instrumentation._translate_span(child)

    assert root_attrs[RESPAN_LOG_TYPE] == "workflow"
    assert root_attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
    assert root_attrs[RESPAN_TRACE_GROUP_ID] == "langfuse_research.workflow"
    assert root_attrs[RESPAN_CUSTOMER_PARAMS_ID] == "user-123"
    assert root_attrs[RESPAN_SESSION_ID] == "session-456"
    assert root_attrs[f"{RESPAN_METADATA}.example_run_id"] == "group-16-marker"
    assert child_attrs[RESPAN_LOG_TYPE] == "task"
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in child_attrs


def test_error_level_and_otel_error_status_are_preserved():
    source = _source_span(
        attributes={
            "langfuse.observation.type": "generation",
            "langfuse.observation.level": "ERROR",
            "langfuse.observation.status_message": "provider failed",
        },
        status=Status(StatusCode.ERROR, "provider failed"),
    )

    attrs, status_code, error_message = instrumentation._translate_span(source)

    assert status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert error_message == "provider failed"
    assert attrs[ERROR_MESSAGE_ATTR] == "provider failed"


def test_injected_span_uses_source_ids_and_shared_pipeline(monkeypatch):
    captured = {}
    source = _source_span()

    def fake_build_readable_span(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return SimpleNamespace(name=name, **kwargs)

    monkeypatch.setattr(
        instrumentation, "build_readable_span", fake_build_readable_span
    )
    monkeypatch.setattr(
        instrumentation,
        "inject_span",
        lambda span: captured.setdefault("injected", span) is span,
    )

    assert instrumentation._inject_langfuse_span(source) is True
    assert captured["trace_id"] == "01010101010101010101010101010101"
    assert captured["span_id"] == "0303030303030303"
    assert captured["parent_id"] == "0202020202020202"
    assert captured["merge_propagated"] is False
    assert captured["injected"].name == "generation"


def test_export_result_reflects_injection_success(monkeypatch):
    spans = [
        _source_span(name="one", attributes={"langfuse.observation.type": "span"}),
        _source_span(name="two", attributes={"langfuse.observation.type": "span"}),
    ]
    instrumentor = LangfuseInstrumentor()
    monkeypatch.setattr(
        instrumentation, "_inject_langfuse_span", lambda span: span.name == "one"
    )

    assert instrumentor._export_spans(spans) is SpanExportResult.FAILURE
    assert instrumentor.exported_span_count == 0
    assert instrumentor._export_spans(spans[:1]) is SpanExportResult.SUCCESS
    assert instrumentor.exported_span_count == 1


def test_canonical_spans_are_not_reinjected(monkeypatch):
    instrumentor = LangfuseInstrumentor()
    monkeypatch.setattr(
        instrumentation,
        "_inject_langfuse_span",
        lambda span: (_ for _ in ()).throw(AssertionError("unexpected reinjection")),
    )

    result = instrumentor._export_spans(
        [_source_span(attributes={RESPAN_LOG_TYPE: "workflow"})]
    )

    assert result is SpanExportResult.SUCCESS


def test_instrument_uninstrument_restores_exporter_without_stacking():
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    instrumentor = LangfuseInstrumentor()
    original = OTLPSpanExporter.export

    instrumentor.instrument()
    first_wrapper = OTLPSpanExporter.export
    assert first_wrapper is not original
    instrumentor.uninstrument()
    assert OTLPSpanExporter.export is original

    instrumentor.instrument()
    assert OTLPSpanExporter.export is not original
    instrumentor.uninstrument()
    assert OTLPSpanExporter.export is original


def test_exporter_detection_is_scoped_to_langfuse_endpoints():
    assert LangfuseInstrumentor._is_langfuse_exporter(
        SimpleNamespace(
            _endpoint="https://cloud.langfuse.com/api/public/otel/v1/traces"
        )
    )
    assert not LangfuseInstrumentor._is_langfuse_exporter(
        SimpleNamespace(_endpoint="https://collector.example.com/v1/traces")
    )


def test_uninstrument_preserves_a_later_exporter_wrapper(monkeypatch):
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    calls = []

    def original(self, spans):
        calls.append("original")
        return SpanExportResult.SUCCESS

    monkeypatch.setattr(OTLPSpanExporter, "export", original)
    instrumentor = LangfuseInstrumentor()
    instrumentor.instrument()
    owned = OTLPSpanExporter.export

    def later_wrapper(self, spans):
        calls.append("later")
        return owned(self, spans)

    monkeypatch.setattr(OTLPSpanExporter, "export", later_wrapper)
    instrumentor.uninstrument()
    assert OTLPSpanExporter.export is later_wrapper
    exporter = OTLPSpanExporter(
        endpoint="https://cloud.langfuse.com/api/public/otel/v1/traces"
    )
    try:
        exporter.export(
            [_source_span(attributes={"langfuse.observation.type": "span"})]
        )
        assert calls == ["later", "original"]
    finally:
        exporter.shutdown()
