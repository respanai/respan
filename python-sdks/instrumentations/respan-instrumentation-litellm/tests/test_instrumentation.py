import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_litellm import _callback as callback_module
from respan_instrumentation_litellm import _instrumentation as instrumentation
from respan_instrumentation_litellm._callback import RespanLiteLLMCallback
from respan_instrumentation_litellm._constants import (
    LITELLM_CHAT_SPAN_NAME,
    OFF_CONTRACT_ALIASES,
    RESPAN_SKIP_CALLBACK_KEY,
)
from respan_instrumentation_litellm._translator import build_litellm_span_data
from respan_sdk.constants.span_attributes import (
    RESPAN_CUSTOMER_PARAMS_ID,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_TRACE_GROUP_ID,
)


def _message(content, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _response(
    content="Bonjour.", usage=None, tool_calls=None, model="openai/gpt-4o-mini"
):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=_message(content, tool_calls=tool_calls))],
        usage=usage
        or SimpleNamespace(prompt_tokens=8, completion_tokens=3, total_tokens=11),
    )


def test_build_span_data_sets_canonical_chat_attributes():
    tool_schema = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    tool_call = SimpleNamespace(
        id="call_123",
        type="function",
        function=SimpleNamespace(name="get_weather", arguments={"city": "Paris"}),
    )

    span_name, attrs = build_litellm_span_data(
        kwargs={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [tool_schema],
            "litellm_params": {
                "metadata": {
                    "respan_params": {
                        "span_name": "custom.litellm",
                        "workflow_name": "litellm_weather.workflow",
                        "customer_identifier": "customer-123",
                        "metadata": {"example": "weather"},
                    }
                }
            },
        },
        response_obj=_response(content="", tool_calls=[tool_call]),
    )

    assert span_name == "custom.litellm"
    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[RESPAN_LOG_METHOD] == "tracing_integration"
    assert attrs[SpanAttributes.LLM_SYSTEM] == "openai"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.CHAT.value
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "openai/gpt-4o-mini"
    assert attrs[RESPAN_TRACE_GROUP_ID] == "litellm_weather.workflow"
    assert attrs[RESPAN_CUSTOMER_PARAMS_ID] == "customer-123"
    assert attrs[f"{RESPAN_METADATA}.example"] == "weather"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Weather in Paris?"
    assert json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == [tool_schema]
    assert json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]) == [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
            },
        }
    ]
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 8
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 3
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 8
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 3
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 11

    for alias in OFF_CONTRACT_ALIASES:
        assert alias not in attrs


def test_build_span_data_uses_standard_logging_payload_fallbacks():
    _, attrs = build_litellm_span_data(
        kwargs={
            "standard_logging_object": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hi"}],
                "response": {"choices": [{"message": {"content": "Hello"}}]},
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
                "response_cost": 0.0001,
                "cache_hit": False,
            }
        },
        response_obj={},
    )

    assert attrs[SpanAttributes.LLM_SYSTEM] == "openai"
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Hi"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == (
        '{"choices":[{"message":{"content":"Hello"}}]}'
    )
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 3
    assert attrs[f"{RESPAN_METADATA}.response_cost"] == "0.0001"
    assert attrs[f"{RESPAN_METADATA}.cache_hit"] == "false"


def test_failure_span_data_sets_error_output_without_aliases():
    _, attrs = build_litellm_span_data(
        kwargs={
            "model": "anthropic/claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
        },
        response_obj=None,
        error=RuntimeError("provider failed"),
    )

    assert attrs[SpanAttributes.LLM_SYSTEM] == "anthropic"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == "provider failed"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "provider failed"
    for alias in OFF_CONTRACT_ALIASES:
        assert alias not in attrs


def test_callback_injects_readable_span(monkeypatch):
    captured = {}

    def fake_build_readable_span(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return {"name": name, **kwargs}

    monkeypatch.setattr(
        callback_module, "build_readable_span", fake_build_readable_span
    )
    monkeypatch.setattr(
        callback_module, "inject_span", lambda span: captured.setdefault("span", span)
    )
    monkeypatch.setattr(
        callback_module, "_current_otel_parent", lambda: ("0" * 32, "1" * 16)
    )

    callback = RespanLiteLLMCallback()
    callback.log_success_event(
        kwargs={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
        },
        response_obj=_response(),
        start_time=datetime(2026, 5, 26, tzinfo=timezone.utc),
        end_time=datetime(2026, 5, 26, 0, 0, 1, tzinfo=timezone.utc),
    )

    assert captured["name"] == LITELLM_CHAT_SPAN_NAME
    assert captured["trace_id"] == "0" * 32
    assert captured["parent_id"] == "1" * 16
    assert captured["start_time_ns"] == 1779753600000000000
    assert captured["end_time_ns"] == 1779753601000000000
    assert captured["status_code"] == 200
    assert (
        captured["attributes"][SpanAttributes.LLM_REQUEST_MODEL] == "openai/gpt-4o-mini"
    )


def test_async_callback_uses_same_emission_path(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        RespanLiteLLMCallback,
        "log_success_event",
        lambda self, **kwargs: emitted.append(kwargs),
    )

    asyncio.run(
        RespanLiteLLMCallback().async_log_success_event(
            kwargs={},
            response_obj={},
            start_time=None,
            end_time=None,
        )
    )

    assert emitted == [
        {
            "kwargs": {},
            "response_obj": {},
            "start_time": None,
            "end_time": None,
        }
    ]


def test_instrumentor_registers_and_removes_litellm_callback(monkeypatch):
    fake_litellm = SimpleNamespace(callbacks=["existing"])

    monkeypatch.setattr(
        instrumentation.importlib,
        "import_module",
        lambda module_name: fake_litellm if module_name == "litellm" else None,
    )

    instrumentor = instrumentation.LiteLLMInstrumentor()
    instrumentor.activate()

    assert fake_litellm.callbacks[0] == "existing"
    assert len(fake_litellm.callbacks) == 2
    assert isinstance(fake_litellm.callbacks[1], RespanLiteLLMCallback)

    instrumentor.deactivate()

    assert fake_litellm.callbacks == ["existing"]


def test_streaming_wrapper_emits_span_after_stream_consumption(monkeypatch):
    emitted = []
    original_kwargs = {}

    def fake_completion(*args, **kwargs):
        original_kwargs.update(kwargs)
        return iter(
            [
                SimpleNamespace(
                    model="openai/gpt-4o-mini",
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="hello "))],
                ),
                SimpleNamespace(
                    model="openai/gpt-4o-mini",
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="stream"))],
                    usage=SimpleNamespace(
                        prompt_tokens=4,
                        completion_tokens=2,
                        total_tokens=6,
                    ),
                ),
            ]
        )

    fake_litellm = SimpleNamespace(callbacks=[], completion=fake_completion)
    monkeypatch.setattr(
        instrumentation.importlib,
        "import_module",
        lambda module_name: fake_litellm if module_name == "litellm" else None,
    )
    monkeypatch.setattr(
        RespanLiteLLMCallback,
        "_emit_event",
        lambda self, **kwargs: emitted.append(kwargs),
    )

    instrumentor = instrumentation.LiteLLMInstrumentor()
    instrumentor.activate()

    stream = fake_litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )
    assert [chunk for chunk in stream]

    assert original_kwargs["metadata"][RESPAN_SKIP_CALLBACK_KEY] is True
    assert len(emitted) == 1
    response_obj = emitted[0]["response_obj"]
    assert response_obj.choices[0].message.content == "hello stream"
    assert response_obj.usage.total_tokens == 6
    assert emitted[0]["kwargs"]["stream"] is True

    _, attrs = build_litellm_span_data(
        kwargs=emitted[0]["kwargs"],
        response_obj=response_obj,
    )
    assert attrs[SpanAttributes.LLM_IS_STREAMING] is True

    instrumentor.deactivate()
    assert fake_litellm.completion is fake_completion
