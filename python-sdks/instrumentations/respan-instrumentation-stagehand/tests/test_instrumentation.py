"""Unit tests for the paths a scripted client model cannot reach in a real session."""

from __future__ import annotations

import base64
import json

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from stagehand import Page, Stagehand

from respan_instrumentation_stagehand import StagehandInstrumentor, with_model_name
from respan_instrumentation_stagehand._constants import (
    ATTR_USAGE_SOURCE,
    USAGE_SOURCE_OPERATION,
)
from respan_instrumentation_stagehand._instrumentation import _SCOPE, _Session
from respan_instrumentation_stagehand._serialization import chat_messages, completion


def test_built_in_model_usage_becomes_one_content_free_chat_span(monkeypatch):
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    instrumentor = StagehandInstrumentor()
    instrumentor.activate()
    try:
        runtime = instrumentor._runtime
        session = _Session("s-1", span=None, model_name="openai/gpt-4.1-mini")
        usage = {"input_tokens": 900, "output_tokens": 40, "reasoning_tokens": 7}
        with _SCOPE.session(session):
            runtime.usage_only_chat(session, usage, {})
    finally:
        instrumentor.deactivate()

    [span] = memory.get_finished_spans()
    assert span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "openai/gpt-4.1-mini"
    assert span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 900
    assert span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 40
    assert span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 940
    assert span.attributes[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 7
    assert span.attributes[ATTR_USAGE_SOURCE] == USAGE_SOURCE_OPERATION
    assert span.attributes["respan.threads.thread_identifier"] == "s-1"
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes


def test_messages_tools_and_images_are_openai_shaped():
    png = base64.b64encode(b"\x89PNG not really").decode()
    params = {
        "system_prompt": "be careful",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "data": png, "mime_type": "image/png"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "click", "input": {"x": 1}}],
            },
            {
                "role": "user",
                "content": {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "done"}],
                },
            },
        ],
    }
    messages = chat_messages(params, keep_images=True)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert messages[2]["tool_calls"][0]["function"] == {"name": "click", "arguments": '{"x": 1}'}
    assert messages[3] == {"role": "tool", "tool_call_id": "t1", "content": "done"}
    without = chat_messages(params, keep_images=False)
    # Text only once the image is a placeholder: one plain string.
    assert without[1]["content"].startswith("look\n[image omitted: image/png")
    structured = completion(
        {"output_format": "json_schema", "structured_content": {"a": 1}, "content": []}
    )
    assert json.loads(structured["content"]) == {"a": 1}


def test_with_model_name_names_bound_methods_too():
    class Model:
        async def generate(self, params):
            return params

    named = with_model_name(Model().generate, "gpt-4.1-mini")
    assert named.__respan_model_name__ == "gpt-4.1-mini"


def test_deactivate_restores_stagehand_exactly():
    before = {name: Stagehand.__dict__[name] for name in ("create", "act", "close")}
    goto = Page.__dict__["goto"]
    instrumentor = StagehandInstrumentor()
    instrumentor.activate()
    assert Page.__dict__["goto"] is not goto
    instrumentor.deactivate()
    assert {name: Stagehand.__dict__[name] for name in before} == before
    assert Page.__dict__["goto"] is goto
