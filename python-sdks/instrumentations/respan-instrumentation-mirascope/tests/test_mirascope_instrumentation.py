from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode

import respan_instrumentation_mirascope._instrumentation as instrumentation
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE


class Response:
    provider_id = "openai"
    model_id = "openai/gpt-4.1-mini"
    content = "Hello"
    text = "Hello"
    tool_calls = []
    usage = SimpleNamespace(
        input_tokens=5,
        output_tokens=2,
        cache_read_tokens=1,
        cache_write_tokens=0,
    )


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        instrumentation.trace,
        "get_tracer",
        lambda *args, **kwargs: provider.get_tracer("test.mirascope"),
    )
    monkeypatch.setattr(instrumentation, "_CAPTURE_CONTENT", True)
    return exporter


def test_sync_call_has_canonical_chat_attributes(spans) -> None:
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")

    def call(self, content, **kwargs):
        return Response()

    wrapped = instrumentation._call_wrapper(call)
    response = wrapped(model, "Say hello", tools=[lambda city: city])
    assert response.text == "Hello"

    span = spans.get_finished_spans()[0]
    attrs = span.attributes
    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[RESPAN_LOG_METHOD] == "tracing_integration"
    assert attrs[SpanAttributes.LLM_SYSTEM] == "openai"
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4.1-mini"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Say hello"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Hello"
    assert json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0]["name"]
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 5
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 2
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 7
    assert "traceloop.span.kind" not in attrs


@pytest.mark.asyncio
async def test_async_call_error_is_recorded_and_reraised(spans) -> None:
    model = SimpleNamespace(model_id="anthropic/claude-sonnet-4-5")

    async def call(self, content, **kwargs):
        raise ValueError("deterministic provider failure")

    wrapped = instrumentation._async_call_wrapper(call)
    with pytest.raises(ValueError, match="deterministic provider failure"):
        await wrapped(model, "fail")

    span = spans.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)
    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "error": "ValueError",
        "message": "deterministic provider failure",
        "status": "error",
    }


def test_sync_stream_span_ends_after_consumption(spans) -> None:
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")
    response = Response()
    response._chunk_iterator = iter(["a", "b"])

    def stream(self, content, **kwargs):
        return response

    wrapped = instrumentation._stream_wrapper(stream)
    returned = wrapped(model, "stream")
    assert spans.get_finished_spans() == ()
    assert list(returned._chunk_iterator) == ["a", "b"]
    assert len(spans.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_async_stream_error_marks_span(spans) -> None:
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")
    response = Response()

    async def chunks():
        yield "first"
        raise RuntimeError("stream failed")

    response._chunk_iterator = chunks()

    async def stream(self, content, **kwargs):
        return response

    wrapped = instrumentation._async_stream_wrapper(stream)
    returned = await wrapped(model, "stream")
    collected = []
    with pytest.raises(RuntimeError, match="stream failed"):
        async for chunk in returned._chunk_iterator:
            collected.append(chunk)
    assert collected == ["first"]
    span = spans.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    output = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert output["status"] == "error"
    assert output["error"] == "RuntimeError"


def test_tool_execution_uses_tool_contract_without_tool_calls(spans) -> None:
    tool_call = SimpleNamespace(name="weather", args={"city": "Paris"})

    def execute(self, call):
        return {"temperature": 18}

    wrapped = instrumentation._tool_wrapper(execute)
    assert wrapped(object(), tool_call) == {"temperature": 18}
    attrs = spans.get_finished_spans()[0].attributes
    assert attrs[RESPAN_LOG_TYPE] == "tool"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "weather"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {"city": "Paris"}
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "temperature": 18
    }
    assert not any(key.endswith("tool_calls") for key in attrs)


def test_capture_content_false_omits_messages_and_outputs(spans, monkeypatch) -> None:
    monkeypatch.setattr(instrumentation, "_CAPTURE_CONTENT", False)
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")

    def call(self, content, **kwargs):
        return Response()

    instrumentation._call_wrapper(call)(model, "secret")
    attrs = spans.get_finished_spans()[0].attributes
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    assert not any(key.startswith(SpanAttributes.LLM_PROMPTS) for key in attrs)
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 7
