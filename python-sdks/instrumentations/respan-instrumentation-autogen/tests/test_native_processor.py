import json
from types import SimpleNamespace

from opentelemetry.attributes import BoundedAttributes
from opentelemetry.trace import SpanContext, TraceFlags
from respan_instrumentation_autogen._native_processor import (
    AUTOGEN_CORE_SCOPE_NAME,
    AUTOGEN_OPENINFERENCE_SCOPE_NAME,
    AutoGenNativeSpanProcessor,
)
from respan_instrumentation_openinference._translator import OpenInferenceTranslator


def _context(span_id: int) -> SpanContext:
    return SpanContext(
        trace_id=1,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )


def _make_span(
    attrs: dict,
    *,
    scope_name: str,
    span_id: int = 2,
    parent: SpanContext | None = None,
):
    context = _context(span_id)
    return SimpleNamespace(
        name="test-span",
        _attributes=dict(attrs),
        _parent=parent,
        parent=parent,
        instrumentation_scope=SimpleNamespace(name=scope_name),
        context=context,
        get_span_context=lambda: context,
    )


def test_native_runtime_span_is_removed_from_export() -> None:
    span = _make_span(
        {
            "gen_ai.operation.name": "create_agent",
            "gen_ai.system": "autogen",
            "traceloop.entity.path": "workflow.agent",
            "llm.request.type": "chat",
            "respan.entity.log_type": "chat",
        },
        scope_name=AUTOGEN_CORE_SCOPE_NAME,
    )

    AutoGenNativeSpanProcessor().on_end(span)

    assert span._attributes == {}


def test_named_autogen_runtime_scope_is_removed_from_export() -> None:
    span = _make_span(
        {"messaging.operation": "publish"},
        scope_name="autogen SingleThreadedAgentRuntime",
    )

    AutoGenNativeSpanProcessor().on_end(span)

    assert span._attributes == {}


def test_native_runtime_span_handles_frozen_attributes() -> None:
    span = _make_span({}, scope_name=AUTOGEN_CORE_SCOPE_NAME)
    span._attributes = BoundedAttributes(
        attributes={"gen_ai.operation.name": "publish"},
        immutable=True,
    )

    AutoGenNativeSpanProcessor().on_end(span)

    assert span._attributes == {}


def test_meaningful_child_is_reparented_around_native_runtime_span() -> None:
    processor = AutoGenNativeSpanProcessor()
    workflow = _context(10)
    native = _make_span(
        {"gen_ai.operation.name": "invoke_agent"},
        scope_name=AUTOGEN_CORE_SCOPE_NAME,
        span_id=20,
        parent=workflow,
    )
    processor.on_start(native)

    child = _make_span(
        {"openinference.span.kind": "AGENT"},
        scope_name=AUTOGEN_OPENINFERENCE_SCOPE_NAME,
        span_id=30,
        parent=native.context,
    )
    processor.on_start(child)

    assert child._parent is workflow

    processor.on_end(native)
    assert processor._native_export_parents == {}


def test_nested_native_runtime_spans_share_nearest_export_parent() -> None:
    processor = AutoGenNativeSpanProcessor()
    workflow = _context(10)
    outer = _make_span(
        {},
        scope_name=AUTOGEN_CORE_SCOPE_NAME,
        span_id=20,
        parent=workflow,
    )
    processor.on_start(outer)
    inner = _make_span(
        {},
        scope_name=AUTOGEN_CORE_SCOPE_NAME,
        span_id=21,
        parent=outer.context,
    )
    processor.on_start(inner)

    child = _make_span(
        {},
        scope_name=AUTOGEN_OPENINFERENCE_SCOPE_NAME,
        span_id=30,
        parent=inner.context,
    )
    processor.on_start(child)

    assert child._parent is workflow


def test_function_result_history_is_preserved_canonically() -> None:
    span = _make_span(
        {
            "openinference.span.kind": "LLM",
            "llm.input_messages.2.message.role": "function",
            "llm.input_messages.2.function.0": json.dumps(
                {
                    "name": "estimate_latency",
                    "content": "tracing-api: about 155 ms",
                    "call_id": "call-1",
                    "is_error": False,
                }
            ),
        },
        scope_name=AUTOGEN_OPENINFERENCE_SCOPE_NAME,
    )

    AutoGenNativeSpanProcessor().on_end(span)

    assert span._attributes["llm.input_messages.2.message.role"] == "tool"
    assert "llm.input_messages.2.function.0" not in span._attributes
    assert json.loads(span._attributes["llm.input_messages.2.message.content"]) == {
        "name": "estimate_latency",
        "content": "tracing-api: about 155 ms",
        "call_id": "call-1",
        "is_error": False,
    }
    assert span._attributes["gen_ai.prompt.2.tool_call_id"] == "call-1"
    assert span._attributes["gen_ai.prompt.2.name"] == "estimate_latency"

    OpenInferenceTranslator().on_end(span)
    assert span._attributes["gen_ai.prompt.2.role"] == "tool"
    assert span._attributes["gen_ai.prompt.2.tool_call_id"] == "call-1"
    assert span._attributes["gen_ai.prompt.2.name"] == "estimate_latency"
    assert json.loads(span._attributes["gen_ai.prompt.2.content"])["content"] == (
        "tracing-api: about 155 ms"
    )


def test_processor_ignores_unrelated_scope() -> None:
    span = _make_span(
        {"gen_ai.operation.name": "invoke_agent"},
        scope_name="other-instrumentation",
    )
    original_attrs = dict(span._attributes)

    AutoGenNativeSpanProcessor().on_end(span)

    assert span._attributes == original_attrs
