"""End-to-end processor tests using real exported ``ReadableSpan`` objects."""

from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_instrumentation_openinference._serialization import MAX_ATTRIBUTE_CHARS
from respan_instrumentation_openinference._translator import OpenInferenceTranslator


@pytest.fixture
def real_export_pipeline():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(OpenInferenceTranslator())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider.get_tracer("openinference-contract-tests"), exporter
    provider.shutdown()


def test_real_chat_readable_span_has_canonical_contract_once(real_export_pipeline):
    tracer, exporter = real_export_pipeline
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "parameters": {"type": "object"},
            },
        }
    ]
    with tracer.start_as_current_span("openai.chat") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("llm.model_name", "gpt-4.1-mini")
        span.set_attribute("llm.system", "openai")
        span.set_attribute("llm.provider", "openai")
        span.set_attribute("llm.invocation_parameters", '{"stream":true}')
        span.set_attribute("llm.token_count.prompt", 11)
        span.set_attribute("llm.token_count.completion", 5)
        span.set_attribute("llm.token_count.total", 16)
        span.set_attribute("llm.tools", json.dumps(tools))
        span.set_attribute("llm.input_messages.0.message.role", "user")
        span.set_attribute("llm.input_messages.0.message.content", "Weather in Tokyo?")
        span.set_attribute("llm.output_messages.0.message.role", "assistant")
        span.set_attribute(
            "llm.output_messages.0.message.tool_calls.0.tool_call.id",
            "call-1",
        )
        span.set_attribute(
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name",
            "lookup_weather",
        )
        span.set_attribute(
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments",
            '{"city":"Tokyo"}',
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    readable = spans[0]
    assert isinstance(readable, ReadableSpan)
    assert readable.status.status_code is StatusCode.UNSET
    attrs = readable.attributes
    assert attrs["respan.entity.log_type"] == "chat"
    assert attrs["gen_ai.request.model"] == "gpt-4.1-mini"
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs[TLSpanAttributes.LLM_IS_STREAMING] is True
    assert json.loads(attrs["llm.request.functions"]) == tools
    assert json.loads(attrs["gen_ai.completion.0.tool_calls"])[0]["id"] == "call-1"
    assert "traceloop.span.kind" not in attrs
    assert "respan.span.tools" not in attrs
    assert "tools" not in attrs


def test_real_failed_span_keeps_otel_error_and_invents_no_usage(real_export_pipeline):
    tracer, exporter = real_export_pipeline
    error = RuntimeError("openinference deterministic provider failure")
    with tracer.start_as_current_span("openai.chat.failure") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("llm.model_name", "gpt-4.1-mini")
        span.set_attribute("llm.system", "openai")
        span.set_attribute("input.value", '{"messages":[{"role":"user"}]}')
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, str(error)))

    readable = exporter.get_finished_spans()[0]
    assert readable.status.status_code is StatusCode.ERROR
    assert readable.status.description == str(error)
    assert [event.name for event in readable.events] == ["exception"]
    assert readable.events[0].attributes["exception.type"].endswith("RuntimeError")
    for usage_key in (
        "gen_ai.usage.prompt_tokens",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.completion_tokens",
        "gen_ai.usage.output_tokens",
        "llm.usage.total_tokens",
        "prompt_tokens",
        "total_request_tokens",
    ):
        assert usage_key not in readable.attributes


def test_real_tool_and_embedding_spans_keep_hierarchy_and_content(real_export_pipeline):
    tracer, exporter = real_export_pipeline
    with tracer.start_as_current_span("workflow") as root:
        root.set_attribute("openinference.span.kind", "CHAIN")
        root_context = root.get_span_context()
        with tracer.start_as_current_span("lookup_weather") as tool:
            tool.set_attribute("openinference.span.kind", "TOOL")
            tool.set_attribute("tool.name", "lookup_weather")
            tool.set_attribute("input.value", '{"city":"Tokyo"}')
            tool.set_attribute("output.value", '{"temperature_c":22}')
        with tracer.start_as_current_span("embed.query") as embedding:
            embedding.set_attribute("openinference.span.kind", "EMBEDDING")
            embedding.set_attribute("embedding.model_name", "text-embedding-3-small")
            embedding.set_attribute(
                "embedding.embeddings.0.embedding.text",
                "Tokyo weather",
            )
            embedding.set_attribute(
                "embedding.embeddings.0.embedding.vector",
                (0.1, 0.2, 0.3),
            )

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {"workflow", "lookup_weather", "embed.query"}
    assert spans["workflow"].attributes["traceloop.entity.path"] == ""
    for name in ("lookup_weather", "embed.query"):
        assert spans[name].parent.span_id == root_context.span_id
        assert spans[name].attributes["traceloop.entity.path"] == name

    tool_input = json.loads(
        spans["lookup_weather"].attributes["traceloop.entity.input"]
    )
    assert tool_input == {
        "name": "lookup_weather",
        "arguments": {"city": "Tokyo"},
    }
    assert json.loads(spans["embed.query"].attributes["traceloop.entity.output"]) == [
        0.1,
        0.2,
        0.3,
    ]


def test_real_span_redacts_and_bounds_content_before_export(real_export_pipeline):
    tracer, exporter = real_export_pipeline
    payload = {
        "authorization": "Bearer credential-should-not-export",
        "nested": {"api_key": "sk-secret-value-12345"},
        "body": "x" * (MAX_ATTRIBUTE_CHARS * 2),
    }
    with tracer.start_as_current_span("private.chat") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("input.value", json.dumps(payload))

    entity_input = exporter.get_finished_spans()[0].attributes["traceloop.entity.input"]
    assert len(entity_input) <= MAX_ATTRIBUTE_CHARS
    assert "credential-should-not-export" not in entity_input
    assert "secret-value-12345" not in entity_input
    json.loads(entity_input)
