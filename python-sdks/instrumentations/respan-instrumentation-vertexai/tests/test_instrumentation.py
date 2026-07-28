from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType
from typing import Any

import pytest
from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_vertexai import VertexAIInstrumentor
from respan_instrumentation_vertexai import _instrumentation
from respan_instrumentation_vertexai._constants import (
    CANDIDATES_TOKEN_COUNT_KEY,
    PROMPT_TOKEN_COUNT_KEY,
    TOTAL_TOKEN_COUNT_KEY,
    CHAT_SESSION_CLASS_NAME,
    GENERATE_CONTENT_ASYNC_METHOD_NAME,
    GENERATE_CONTENT_METHOD_NAME,
    GENERATIVE_MODEL_CLASS_NAME,
    SEND_MESSAGE_ASYNC_METHOD_NAME,
    SEND_MESSAGE_METHOD_NAME,
    VERTEXAI_GENERATIVE_MODELS_MODULE,
)
from respan_instrumentation_vertexai._otel_emitter import build_generate_content_attrs
from respan_instrumentation_vertexai._translator import extract_usage
from respan_instrumentation_vertexai._translator import request_payload_from_call
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
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
) -> Obj:
    content = Obj(role="model", parts=parts if parts is not None else [Obj(text=text)])
    candidate = Obj(content=content)
    return Obj(text=text, candidates=[candidate], usage_metadata=usage)


def make_usage(prompt_tokens: int = 3, completion_tokens: int = 4) -> Obj:
    return Obj(
        prompt_token_count=prompt_tokens,
        candidates_token_count=completion_tokens,
        total_token_count=prompt_tokens + completion_tokens,
    )


@pytest.fixture(autouse=True)
def reset_instrumentation_globals() -> None:
    _instrumentation._original_generate_content = None
    _instrumentation._original_generate_content_async = None
    _instrumentation._original_send_message = None
    _instrumentation._original_send_message_async = None


@pytest.fixture()
def captured_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    spans: list[Any] = []
    monkeypatch.setattr(
        "respan_instrumentation_vertexai._otel_emitter.inject_span",
        lambda span: spans.append(span),
    )
    return spans


@pytest.fixture()
def fake_vertexai(monkeypatch: pytest.MonkeyPatch) -> tuple[type[Any], type[Any]]:
    class GenerativeModel:
        def __init__(
            self,
            model_name: str = "gemini-2.0-flash",
            *,
            system_instruction: str | None = None,
            tools: list[Any] | None = None,
        ) -> None:
            self._model_name = model_name
            self._system_instruction = system_instruction
            self._tools = tools

        def generate_content(self, contents: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream"):
                return iter(
                    [
                        make_response(text="Hello "),
                        make_response(text="world", usage=make_usage(5, 6)),
                    ]
                )
            return make_response(
                text=f"{self._model_name}: {contents}", usage=make_usage()
            )

        async def generate_content_async(self, contents: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream"):

                async def chunks():
                    yield make_response(text="async ")
                    yield make_response(text="stream", usage=make_usage(7, 8))

                return chunks()
            return make_response(text=f"async {contents}", usage=make_usage(9, 10))

    class ChatSession:
        def __init__(self, model: GenerativeModel) -> None:
            self.model = model

        def send_message(self, content: Any, **kwargs: Any) -> Any:
            return make_response(text=f"chat: {content}", usage=make_usage(11, 12))

        async def send_message_async(self, content: Any, **kwargs: Any) -> Any:
            return make_response(
                text=f"async chat: {content}", usage=make_usage(13, 14)
            )

    vertexai_module = ModuleType("vertexai")
    generative_models_module = ModuleType(VERTEXAI_GENERATIVE_MODELS_MODULE)
    setattr(generative_models_module, GENERATIVE_MODEL_CLASS_NAME, GenerativeModel)
    setattr(generative_models_module, CHAT_SESSION_CLASS_NAME, ChatSession)
    setattr(vertexai_module, "generative_models", generative_models_module)

    monkeypatch.setitem(sys.modules, "vertexai", vertexai_module)
    monkeypatch.setitem(
        sys.modules,
        VERTEXAI_GENERATIVE_MODELS_MODULE,
        generative_models_module,
    )
    return GenerativeModel, ChatSession


def test_activate_patches_generate_content_and_emits_chat_span(
    fake_vertexai: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    GenerativeModel, _ = fake_vertexai
    tool = Obj(
        function_declarations=[
            Obj(
                name="get_weather",
                description="Get weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            )
        ]
    )
    model = GenerativeModel(
        "gemini-2.0-flash",
        system_instruction="Be brief",
        tools=[tool],
    )
    instrumentor = VertexAIInstrumentor()

    instrumentor.activate()
    response = model.generate_content("Say hello")

    assert response.text == "gemini-2.0-flash: Say hello"
    assert len(captured_spans) == 1
    attrs = captured_spans[0]._attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-2.0-flash"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Be brief"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == "Say hello"
    assert (
        attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == "gemini-2.0-flash: Say hello"
    )
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 3
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 4
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 3
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 4
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 7
    assert (
        json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0]["function"]["name"]
        == "get_weather"
    )

    instrumentor.deactivate()


def test_stream_emits_one_span_after_iterator_is_consumed(
    fake_vertexai: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    GenerativeModel, _ = fake_vertexai
    model = GenerativeModel()
    instrumentor = VertexAIInstrumentor()
    instrumentor.activate()

    chunks = list(model.generate_content("Stream this", stream=True))

    assert [chunk.text for chunk in chunks] == ["Hello ", "world"]
    assert len(captured_spans) == 1
    attrs = captured_spans[0]._attributes
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Hello world"
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 5
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 6

    instrumentor.deactivate()


def test_async_methods_emit_spans(
    fake_vertexai: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    async def run() -> None:
        GenerativeModel, ChatSession = fake_vertexai
        model = GenerativeModel()
        chat = ChatSession(model)
        instrumentor = VertexAIInstrumentor()
        instrumentor.activate()

        response = await model.generate_content_async("Async hello")
        async_stream = await model.generate_content_async("Async stream", stream=True)
        chunks = [chunk async for chunk in async_stream]
        chat_response = await chat.send_message_async("Async chat")

        assert response.text == "async Async hello"
        assert [chunk.text for chunk in chunks] == ["async ", "stream"]
        assert chat_response.text == "async chat: Async chat"
        assert len(captured_spans) == 3
        assert (
            captured_spans[0]._attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 9
        )
        assert (
            captured_spans[1]._attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
            == "async stream"
        )
        assert (
            captured_spans[2]._attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
            == 14
        )

        instrumentor.deactivate()

    asyncio.run(run())


def test_chat_session_send_message_uses_nested_model_name(
    fake_vertexai: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    GenerativeModel, ChatSession = fake_vertexai
    model = GenerativeModel("gemini-2.0-flash")
    chat = ChatSession(model)
    instrumentor = VertexAIInstrumentor()
    instrumentor.activate()

    response = chat.send_message("Continue")

    assert response.text == "chat: Continue"
    assert len(captured_spans) == 1
    attrs = captured_spans[0]._attributes
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-2.0-flash"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Continue"

    instrumentor.deactivate()


def test_active_workflow_name_is_attached_to_injected_chat_span() -> None:
    token = context_api.attach(
        context_api.set_value(
            SpanAttributes.TRACELOOP_ENTITY_NAME,
            "vertexai_generate_content_example",
        )
    )
    try:
        attrs = build_generate_content_attrs(
            request_payload={
                "model": "gemini-2.0-flash",
                "contents": "Hello",
                "system_instruction": None,
                "tools": None,
                "generation_config": None,
            },
            response_or_chunks=make_response(text="Hi"),
        )
    finally:
        context_api.detach(token)

    assert (
        attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME]
        == "vertexai_generate_content_example"
    )


def test_function_calls_use_canonical_completion_field_only() -> None:
    function_call = Obj(id="call_1", name="get_weather", args={"city": "Tokyo"})
    response = make_response(
        text="",
        parts=[Obj(function_call=function_call)],
    )

    attrs = build_generate_content_attrs(
        request_payload={
            "model": "gemini-2.0-flash",
            "contents": "Weather in Tokyo?",
            "system_instruction": None,
            "tools": None,
            "generation_config": None,
        },
        response_or_chunks=response,
    )

    tool_calls = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
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
    assert RESPAN_SPAN_TOOLS not in attrs
    assert RESPAN_SPAN_TOOL_CALLS not in attrs
    assert "tools" not in attrs
    assert "tool_calls" not in attrs


def test_error_path_emits_failed_span(
    monkeypatch: pytest.MonkeyPatch,
    fake_vertexai: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    GenerativeModel, _ = fake_vertexai

    def raise_error(self: Any, contents: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(GenerativeModel, GENERATE_CONTENT_METHOD_NAME, raise_error)
    instrumentor = VertexAIInstrumentor()
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="boom"):
        GenerativeModel("gemini-2.0-flash").generate_content("fail")

    assert len(captured_spans) == 1
    span = captured_spans[0]
    assert span.status.status_code.name == "ERROR"
    assert span._attributes["error.message"] == "boom"
    assert span._attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-2.0-flash"

    instrumentor.deactivate()


def test_deactivate_restores_original_methods(
    fake_vertexai: tuple[type[Any], type[Any]],
) -> None:
    GenerativeModel, ChatSession = fake_vertexai
    original_generate = getattr(GenerativeModel, GENERATE_CONTENT_METHOD_NAME)
    original_generate_async = getattr(
        GenerativeModel, GENERATE_CONTENT_ASYNC_METHOD_NAME
    )
    original_send = getattr(ChatSession, SEND_MESSAGE_METHOD_NAME)
    original_send_async = getattr(ChatSession, SEND_MESSAGE_ASYNC_METHOD_NAME)

    instrumentor = VertexAIInstrumentor()
    instrumentor.activate()
    assert (
        getattr(GenerativeModel, GENERATE_CONTENT_METHOD_NAME) is not original_generate
    )
    assert getattr(ChatSession, SEND_MESSAGE_METHOD_NAME) is not original_send

    instrumentor.deactivate()
    assert getattr(GenerativeModel, GENERATE_CONTENT_METHOD_NAME) is original_generate
    assert (
        getattr(GenerativeModel, GENERATE_CONTENT_ASYNC_METHOD_NAME)
        is original_generate_async
    )
    assert getattr(ChatSession, SEND_MESSAGE_METHOD_NAME) is original_send
    assert getattr(ChatSession, SEND_MESSAGE_ASYNC_METHOD_NAME) is original_send_async


def test_request_payload_reads_model_defaults(
    fake_vertexai: tuple[type[Any], type[Any]],
) -> None:
    GenerativeModel, _ = fake_vertexai
    tool = Obj(function_declarations=[Obj(name="lookup")])
    model = GenerativeModel(
        "gemini-2.0-flash",
        system_instruction="Use short answers",
        tools=[tool],
    )

    payload = request_payload_from_call(
        instance=model,
        args=("Hello",),
        kwargs={"generation_config": {"temperature": 0.1}},
    )

    assert payload["model"] == "gemini-2.0-flash"
    assert payload["contents"] == "Hello"
    assert payload["system_instruction"] == "Use short answers"
    assert payload["tools"] == [tool]


def test_thinking_tokens_fold_into_the_output_count() -> None:
    """Gemini reports thinking tokens separately but bills them at the output rate.

    Left out of the completion count the span contradicts itself: prompt + completion
    stops reconciling against the total the API returned, and the thinking tokens land
    on no attribute at all, so anything costing off the span under-reports output.
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
    assert result[TOTAL_TOKEN_COUNT_KEY] == 950
    assert (
        result[PROMPT_TOKEN_COUNT_KEY] + result[CANDIDATES_TOKEN_COUNT_KEY]
        == result[TOTAL_TOKEN_COUNT_KEY]
    )


def test_usage_is_unchanged_when_the_model_does_not_think() -> None:
    """Control: no thoughts field at all, which is every non-thinking model.

    This is why the defect went unnoticed - the existing fixtures all look like this.
    """
    result = extract_usage(make_response(usage=make_usage(100, 50)))

    assert result[CANDIDATES_TOKEN_COUNT_KEY] == 50
    assert result[TOTAL_TOKEN_COUNT_KEY] == 150


def test_zero_thinking_tokens_leave_the_output_count_alone() -> None:
    """Thinking budget set to zero still emits the field, and must be a no-op."""
    usage = Obj(
        prompt_token_count=100,
        candidates_token_count=50,
        thoughts_token_count=0,
        total_token_count=150,
    )
    result = extract_usage(make_response(usage=usage))

    assert result[CANDIDATES_TOKEN_COUNT_KEY] == 50
    assert result[TOTAL_TOKEN_COUNT_KEY] == 150
