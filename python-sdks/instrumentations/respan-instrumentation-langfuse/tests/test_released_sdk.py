"""Exercise the released Langfuse SDK through its actual OTLP export path."""

import json
import uuid

import pytest
from langfuse import Langfuse, propagate_attributes
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_langfuse import LangfuseInstrumentor
from respan_instrumentation_langfuse import instrumentor as instrumentation
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_SESSION_ID


@pytest.fixture
def captured_client(monkeypatch):
    spans = []
    monkeypatch.setattr(
        instrumentation, "inject_span", lambda span: spans.append(span) or True
    )
    instrumentor = LangfuseInstrumentor()
    instrumentor.instrument()
    client = Langfuse(
        public_key=f"pk-lf-{uuid.uuid4().hex}",
        secret_key="sk-lf-local-test",
        tracer_provider=TracerProvider(),
    )
    try:
        yield client, spans
    finally:
        client.shutdown()
        instrumentor.uninstrument()


@pytest.mark.parametrize("null_content", [False, True])
def test_single_message_objects_preserve_tools_and_request_definitions(
    captured_client, null_content
):
    client, spans = captured_client
    call = {
        "id": "call-current",
        "type": "function",
        "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
    }
    historical = {**call, "id": "call-history"}
    tool = {
        "type": "function",
        "function": {"name": "weather", "parameters": {"type": "object"}},
    }
    with client.start_as_current_observation(
        name="tool-generation",
        as_type="generation",
        model="fixture-model",
        input={"role": "assistant", "content": "", "tool_calls": [historical]},
        output={
            "role": "assistant",
            "tool_calls": [call],
            **({"content": None} if null_content else {}),
        },
        model_parameters={"tools": [tool]},
    ):
        pass
    client.flush()
    attrs = spans[0].attributes
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == ""
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "assistant"
    assert (
        json.loads(attrs[f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls"])[0]["id"]
        == "call-history"
    )
    assert (
        json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])[0]["id"]
        == "call-current"
    )
    assert json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == [tool]


def test_absent_generation_content_is_not_fabricated(captured_client):
    client, spans = captured_client
    with client.start_as_current_observation(
        name="no-content",
        as_type="generation",
        model="fixture-model",
        usage_details={"input": 2, "output": 1},
    ):
        pass
    client.flush()
    attrs = spans[0].attributes
    assert not any(
        key.startswith(
            (SpanAttributes.LLM_PROMPTS + ".", SpanAttributes.LLM_COMPLETIONS + ".")
        )
        for key in attrs
    )
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs


def test_released_sdk_observation_types_and_propagation(monkeypatch):
    spans = []
    monkeypatch.setattr(
        instrumentation, "inject_span", lambda span: spans.append(span) or True
    )
    instrumentor = LangfuseInstrumentor()
    instrumentor.instrument()
    client = Langfuse(
        public_key=f"pk-lf-{uuid.uuid4().hex}",
        secret_key="sk-lf-local-test",
        tracer_provider=TracerProvider(),
    )
    try:
        with (
            propagate_attributes(
                session_id="released-sdk-session", metadata={"run_id": "released-sdk"}
            ),
            client.start_as_current_observation(
                name="workflow", input={"question": "tracing"}
            ),
        ):
            with client.start_as_current_observation(
                name="embedding",
                as_type="embedding",
                model="text-embedding-3-small",
                input=["tracing"],
                output=[[0.25, 0.5]],
                usage_details={"input": 3, "total": 3},
            ):
                pass
            with client.start_as_current_observation(
                name="guardrail",
                as_type="guardrail",
                input={"text": "safe"},
                output={"allowed": True},
            ):
                pass
        client.flush()
        by_name = {span.name: span for span in spans}
        assert set(by_name) == {"workflow", "embedding", "guardrail"}
        embedding = by_name["embedding"]
        attrs = embedding.attributes
        assert attrs[RESPAN_LOG_TYPE] == "embedding"
        assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "embedding"
        assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "text-embedding-3-small"
        assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 3
        assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == [
            [0.25, 0.5]
        ]
        assert by_name["guardrail"].attributes[RESPAN_LOG_TYPE] == "guardrail"
        assert all(
            span.attributes[RESPAN_SESSION_ID] == "released-sdk-session"
            for span in spans
        )
        assert embedding.parent.span_id == by_name["workflow"].context.span_id
        assert embedding.context.trace_id == by_name["workflow"].context.trace_id
        assert instrumentor.exported_span_count == 3
    finally:
        client.shutdown()
        instrumentor.uninstrument()
