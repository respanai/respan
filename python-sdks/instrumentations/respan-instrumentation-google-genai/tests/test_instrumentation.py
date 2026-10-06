from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType
from typing import Any

import pytest
from opentelemetry import context as context_api
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_google_genai import (
    GoogleGenAIInstrumentor,
    _instrumentation,
)
from respan_instrumentation_google_genai._constants import (
    ASYNC_MODELS_CLASS_NAME,
    CANDIDATES_TOKEN_COUNT_KEY,
    GENERATE_CONTENT_METHOD_NAME,
    GENERATE_CONTENT_STREAM_METHOD_NAME,
    GOOGLE_GENAI_MODELS_MODULE,
    MODELS_CLASS_NAME,
    PROMPT_TOKEN_COUNT_KEY,
    TOTAL_TOKEN_COUNT_KEY,
)
from respan_instrumentation_google_genai._otel_emitter import (
    build_generate_content_attrs,
)
from respan_instrumentation_google_genai._translator import (
    extract_usage,
    request_kwargs_from_call,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT
from respan_sdk.constants.span_attributes import (
    LLM_REQUEST_MODEL,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_TYPE,
)


class Obj:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def make_response(
    text: str = "Hello",
    *,
    usage: Obj | None = None,
    parts: list[Any] | None = None,
    history: list[Any] | None = None,
) -> Obj:
    content = Obj(role="model", parts=parts if parts is not None else [Obj(text=text)])
    candidate = Obj(content=content)
    return Obj(
        text=text,
        candidates=[candidate],
        usage_metadata=usage,
        automatic_function_calling_history=history or [],
    )


def make_usage(prompt_tokens: int = 3, completion_tokens: int = 4) -> Obj:
    return Obj(
        prompt_token_count=prompt_tokens,
        candidates_token_count=completion_tokens,
        total_token_count=prompt_tokens + completion_tokens,
    )


@pytest.fixture(autouse=True)
def reset_instrumentation_globals() -> None:
    _instrumentation._original_sync_generate_content = None
    _instrumentation._original_sync_generate_content_stream = None
    _instrumentation._original_async_generate_content = None
    _instrumentation._original_async_generate_content_stream = None


@pytest.fixture()
def captured_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    spans: list[Any] = []
    monkeypatch.setattr(
        "respan_instrumentation_google_genai._otel_emitter.inject_span",
        lambda span: spans.append(span),
    )
    return spans


@pytest.fixture()
def fake_google_genai(monkeypatch: pytest.MonkeyPatch) -> tuple[type[Any], type[Any]]:
    class Models:
        def generate_content(
            self, *, model: str, contents: Any, config: Any = None
        ) -> Obj:
            return make_response(text=f"{model}: {contents}", usage=make_usage())

        def generate_content_stream(
            self,
            *,
            model: str,
            contents: Any,
            config: Any = None,
        ):
            yield make_response(text="Hello ")
            yield make_response(text="world", usage=make_usage(5, 6))

    class AsyncModels:
        async def generate_content(
            self,
            *,
            model: str,
            contents: Any,
            config: Any = None,
        ) -> Obj:
            return make_response(text=f"async {contents}", usage=make_usage(7, 8))

        async def generate_content_stream(
            self,
            *,
            model: str,
            contents: Any,
            config: Any = None,
        ):
            async def chunks():
                yield make_response(text="async ")
                yield make_response(text="stream", usage=make_usage(9, 10))

            return chunks()

    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    models_module = ModuleType(GOOGLE_GENAI_MODELS_MODULE)
    setattr(models_module, MODELS_CLASS_NAME, Models)
    setattr(models_module, ASYNC_MODELS_CLASS_NAME, AsyncModels)
    genai_module.models = models_module
    google_module.genai = genai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, GOOGLE_GENAI_MODELS_MODULE, models_module)
    return Models, AsyncModels


def weather_tool(city: str) -> str:
    """Return fake weather for a city."""
    return f"Sunny in {city}"


def test_activate_patches_sync_generate_content_and_emits_chat_span(
    fake_google_genai: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    Models, _ = fake_google_genai
    instrumentor = GoogleGenAIInstrumentor()

    instrumentor.activate()
    response = Models().generate_content(
        model="gemini-2.5-flash",
        contents="Say hello",
        config={"system_instruction": "Be brief", "tools": [weather_tool]},
    )

    assert response.text == "gemini-2.5-flash: Say hello"
    assert len(captured_spans) == 1
    attrs = captured_spans[0]._attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[LLM_REQUEST_MODEL] == "gemini-2.5-flash"
    assert attrs["gen_ai.prompt.0.role"] == "system"
    assert attrs["gen_ai.prompt.0.content"] == "Be brief"
    assert attrs["gen_ai.prompt.1.content"] == "Say hello"
    assert attrs["gen_ai.completion.0.content"] == "gemini-2.5-flash: Say hello"
    assert attrs[LLM_USAGE_PROMPT_TOKENS] == 3
    assert attrs[LLM_USAGE_COMPLETION_TOKENS] == 4
    assert (
        json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0]["function"]["name"]
        == "weather_tool"
    )
    assert "tools" not in attrs
    assert "respan.span.tools" not in attrs
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs

    instrumentor.deactivate()


def test_thinking_tokens_are_included_in_output_usage() -> None:
    usage = Obj(
        prompt_token_count=100,
        candidates_token_count=50,
        thoughts_token_count=800,
        total_token_count=950,
    )

    attrs = build_generate_content_attrs(
        request_kwargs={"model": "gemini-2.5-flash", "contents": "Explain"},
        response_or_chunks=make_response(text="Reasoned answer", usage=usage),
    )

    assert attrs[LLM_USAGE_PROMPT_TOKENS] == 100
    assert attrs[LLM_USAGE_COMPLETION_TOKENS] == 850
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 950
    assert attrs["gen_ai.usage.input_tokens"] == 100
    assert attrs["gen_ai.usage.output_tokens"] == 850


def test_stream_emits_one_span_after_iterator_is_consumed(
    fake_google_genai: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    Models, _ = fake_google_genai
    instrumentor = GoogleGenAIInstrumentor()
    instrumentor.activate()

    chunks = list(
        Models().generate_content_stream(
            model="gemini-2.5-flash",
            contents="Stream this",
        )
    )

    assert [chunk.text for chunk in chunks] == ["Hello ", "world"]
    assert len(captured_spans) == 1
    attrs = captured_spans[0]._attributes
    assert attrs["gen_ai.completion.0.content"] == "Hello world"
    assert attrs[LLM_USAGE_PROMPT_TOKENS] == 5
    assert attrs[LLM_USAGE_COMPLETION_TOKENS] == 6
    assert attrs[SpanAttributes.LLM_IS_STREAMING] is True

    instrumentor.deactivate()


def test_async_methods_emit_spans(
    fake_google_genai: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    async def run() -> None:
        _, AsyncModels = fake_google_genai
        instrumentor = GoogleGenAIInstrumentor()
        instrumentor.activate()

        response = await AsyncModels().generate_content(
            model="gemini-2.5-flash",
            contents="Async hello",
        )
        async_stream = await AsyncModels().generate_content_stream(
            model="gemini-2.5-flash",
            contents="Async stream",
        )
        chunks = [chunk async for chunk in async_stream]

        assert response.text == "async Async hello"
        assert [chunk.text for chunk in chunks] == ["async ", "stream"]
        assert len(captured_spans) == 2
        assert captured_spans[0]._attributes[LLM_USAGE_PROMPT_TOKENS] == 7
        assert (
            captured_spans[1]._attributes["gen_ai.completion.0.content"]
            == "async stream"
        )
        assert captured_spans[0]._attributes[SpanAttributes.LLM_IS_STREAMING] is False
        assert captured_spans[1]._attributes[SpanAttributes.LLM_IS_STREAMING] is True

        instrumentor.deactivate()

    asyncio.run(run())


def test_active_workflow_name_is_attached_to_injected_chat_span() -> None:
    token = context_api.attach(
        context_api.set_value(
            SpanAttributes.TRACELOOP_ENTITY_NAME,
            "google_genai_generate_content",
        )
    )
    try:
        attrs = build_generate_content_attrs(
            request_kwargs={
                "model": "gemini-2.5-flash",
                "contents": "Hello",
                "config": None,
            },
            response_or_chunks=make_response(text="Hi"),
        )
    finally:
        context_api.detach(token)

    assert (
        attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "google_genai_generate_content"
    )


def test_automatic_function_calling_history_stays_in_prompt_fields() -> None:
    function_call = Obj(id="call_1", name="get_weather", args={"city": "Tokyo"})
    history = [
        Obj(role="user", parts=[Obj(text="Weather in Tokyo?")]),
        Obj(
            role="model",
            parts=[Obj(function_call=function_call)],
        ),
    ]
    response = make_response(text="It is sunny.", history=history)

    attrs = build_generate_content_attrs(
        request_kwargs={
            "model": "gemini-2.5-flash",
            "contents": "Weather in Tokyo?",
            "config": None,
        },
        response_or_chunks=response,
    )

    assert "gen_ai.completion.0.tool_calls" not in attrs
    tool_calls = json.loads(attrs["gen_ai.prompt.1.tool_calls"])
    assert tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Tokyo"}',
            },
        }
    ]
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert "tool_calls" not in attrs
    assert "respan.span.tool_calls" not in attrs


def test_error_path_emits_failed_span(
    monkeypatch: pytest.MonkeyPatch,
    fake_google_genai: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    Models, _ = fake_google_genai

    def raise_error(self: Any, *, model: str, contents: Any, config: Any = None) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(Models, GENERATE_CONTENT_METHOD_NAME, raise_error)
    instrumentor = GoogleGenAIInstrumentor()
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="boom"):
        Models().generate_content(model="gemini-2.5-flash", contents="fail")

    assert len(captured_spans) == 1
    span = captured_spans[0]
    assert span.status.status_code.name == "ERROR"
    assert span._attributes["error.message"] == "boom"
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "error": "boom",
        "status_code": 500,
    }
    assert span._attributes[LLM_REQUEST_MODEL] == "gemini-2.5-flash"

    instrumentor.deactivate()


def test_error_path_preserves_provider_status_code(
    monkeypatch: pytest.MonkeyPatch,
    fake_google_genai: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    Models, _ = fake_google_genai

    class APIError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("model unavailable")
            self.response = Obj(status_code=503)

    def raise_error(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise APIError()

    monkeypatch.setattr(Models, GENERATE_CONTENT_METHOD_NAME, raise_error)
    instrumentor = GoogleGenAIInstrumentor()
    instrumentor.activate()

    with pytest.raises(APIError, match="model unavailable"):
        Models().generate_content("gemini-3-flash-preview", "fail")

    attrs = captured_spans[0]._attributes
    assert attrs["status_code"] == 503
    assert attrs[LLM_REQUEST_MODEL] == "gemini-3-flash-preview"
    assert "fail" in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    assert (
        json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])["status_code"] == 503
    )

    instrumentor.deactivate()


def test_request_kwargs_support_positional_sdk_arguments() -> None:
    assert request_kwargs_from_call(
        args=("gemini-3-flash-preview", "Hello", {"temperature": 0}),
        kwargs={},
    ) == {
        "model": "gemini-3-flash-preview",
        "contents": "Hello",
        "config": {"temperature": 0},
    }


def test_deactivate_restores_original_methods(
    fake_google_genai: tuple[type[Any], type[Any]],
) -> None:
    Models, AsyncModels = fake_google_genai
    original_sync = getattr(Models, GENERATE_CONTENT_METHOD_NAME)
    original_sync_stream = getattr(Models, GENERATE_CONTENT_STREAM_METHOD_NAME)
    original_async = getattr(AsyncModels, GENERATE_CONTENT_METHOD_NAME)
    original_async_stream = getattr(AsyncModels, GENERATE_CONTENT_STREAM_METHOD_NAME)

    instrumentor = GoogleGenAIInstrumentor()
    instrumentor.activate()
    assert getattr(Models, GENERATE_CONTENT_METHOD_NAME) is not original_sync
    assert getattr(AsyncModels, GENERATE_CONTENT_METHOD_NAME) is not original_async

    instrumentor.deactivate()
    assert getattr(Models, GENERATE_CONTENT_METHOD_NAME) is original_sync
    assert getattr(Models, GENERATE_CONTENT_STREAM_METHOD_NAME) is original_sync_stream
    assert getattr(AsyncModels, GENERATE_CONTENT_METHOD_NAME) is original_async
    assert (
        getattr(AsyncModels, GENERATE_CONTENT_STREAM_METHOD_NAME)
        is original_async_stream
    )


def test_thinking_tokens_reconcile_against_the_reported_total() -> None:
    """Prompt plus completion must equal the total the API returned.

    The merged change pins the individual values. This pins the invariant they have
    to satisfy, which is what anything costing off the span actually depends on.
    """
    usage = Obj(
        prompt_token_count=100,
        candidates_token_count=50,
        thoughts_token_count=800,
        total_token_count=950,
    )

    result = extract_usage(make_response(usage=usage))

    assert result[PROMPT_TOKEN_COUNT_KEY] == 100
    assert result[CANDIDATES_TOKEN_COUNT_KEY] == 850
    assert (
        result[PROMPT_TOKEN_COUNT_KEY] + result[CANDIDATES_TOKEN_COUNT_KEY]
        == result[TOTAL_TOKEN_COUNT_KEY]
    )


def test_usage_is_unchanged_when_the_model_does_not_think() -> None:
    """Control: no thoughts field at all, which is every non-thinking model.

    This is the shape of every pre-existing fixture, which is why the defect went
    unnoticed. It pins that the fold stays inert on the common path.
    """
    result = extract_usage(make_response(usage=make_usage(100, 50)))

    assert result[CANDIDATES_TOKEN_COUNT_KEY] == 50
    assert result[TOTAL_TOKEN_COUNT_KEY] == 150


def test_zero_thinking_tokens_leave_the_output_count_alone() -> None:
    """A thinking budget of zero still emits the field, and must be a no-op."""
    usage = Obj(
        prompt_token_count=100,
        candidates_token_count=50,
        thoughts_token_count=0,
        total_token_count=150,
    )

    result = extract_usage(make_response(usage=usage))

    assert result[CANDIDATES_TOKEN_COUNT_KEY] == 50
    assert result[TOTAL_TOKEN_COUNT_KEY] == 150
