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
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_ollama import OllamaInstrumentor
from respan_instrumentation_ollama import _instrumentation
from respan_instrumentation_ollama._constants import (
    ASYNC_CLIENT_CLASS_NAME,
    CHAT_METHOD_NAME,
    CLIENT_CLASS_NAME,
    EMBED_METHOD_NAME,
    GENERATE_METHOD_NAME,
    OLLAMA_CLIENT_MODULE,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


class Obj:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.__dict__


def get_weather(city: str) -> str:
    """Return fake weather for a city."""
    return f"Sunny in {city}"


def _chat_response(
    *,
    model: str = "llama3.2",
    content: str = "The weather is sunny.",
    tool_calls: list[Any] | None = None,
    prompt_tokens: int = 3,
    completion_tokens: int = 4,
) -> Obj:
    return Obj(
        model=model,
        message=Obj(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        ),
        prompt_eval_count=prompt_tokens,
        eval_count=completion_tokens,
    )


def _generate_response(
    *,
    model: str = "llama3.2",
    response: str = "Generated text.",
    prompt_tokens: int = 5,
    completion_tokens: int = 6,
) -> Obj:
    return Obj(
        model=model,
        response=response,
        prompt_eval_count=prompt_tokens,
        eval_count=completion_tokens,
    )


@pytest.fixture(autouse=True)
def reset_instrumentation_globals() -> None:
    _instrumentation._original_sync_chat = None
    _instrumentation._original_async_chat = None
    _instrumentation._original_sync_generate = None
    _instrumentation._original_async_generate = None
    _instrumentation._original_sync_embed = None
    _instrumentation._original_async_embed = None
    _instrumentation._original_sync_embeddings = None
    _instrumentation._original_async_embeddings = None


@pytest.fixture()
def captured_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    spans: list[Any] = []
    monkeypatch.setattr(
        "respan_instrumentation_ollama._otel_emitter.inject_span",
        lambda span: spans.append(span),
    )
    return spans


@pytest.fixture()
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> tuple[type[Any], type[Any]]:
    class Client:
        def chat(
            self,
            model: str = "",
            messages: list[dict[str, Any]] | None = None,
            *,
            tools: list[Any] | None = None,
            stream: bool = False,
        ) -> Any:
            tool_calls = [
                Obj(function=Obj(name="get_weather", arguments={"city": "Tokyo"}))
            ]
            if stream:
                return iter(
                    [
                        _chat_response(
                            model=model,
                            content="Sunny ",
                            prompt_tokens=1,
                            completion_tokens=1,
                        ),
                        _chat_response(
                            model=model, content="in Tokyo.", tool_calls=tool_calls
                        ),
                    ]
                )
            return _chat_response(model=model, tool_calls=tool_calls)

        def generate(
            self,
            model: str = "",
            prompt: str | None = None,
            *,
            system: str | None = None,
            stream: bool = False,
        ) -> Any:
            if stream:
                return iter(
                    [
                        _generate_response(
                            model=model, response="Hello ", prompt_tokens=2
                        ),
                        _generate_response(
                            model=model,
                            response="world.",
                            prompt_tokens=2,
                            completion_tokens=3,
                        ),
                    ]
                )
            return _generate_response(model=model)

        def embed(self, model: str = "", input: Any = "") -> Obj:
            return Obj(model=model, embeddings=[[0.1, 0.2]], prompt_eval_count=7)

    class AsyncClient:
        async def chat(
            self,
            model: str = "",
            messages: list[dict[str, Any]] | None = None,
            *,
            tools: list[Any] | None = None,
            stream: bool = False,
        ) -> Any:
            if stream:

                async def chunks():
                    yield _chat_response(model=model, content="Async ")
                    yield _chat_response(model=model, content="stream.")

                return chunks()
            return _chat_response(model=model, content="Async chat.")

        async def generate(
            self,
            model: str = "",
            prompt: str | None = None,
            *,
            system: str | None = None,
            stream: bool = False,
        ) -> Any:
            return _generate_response(model=model, response="Async generated.")

        async def embed(self, model: str = "", input: Any = "") -> Obj:
            return Obj(model=model, embeddings=[[0.3, 0.4]], prompt_eval_count=8)

    ollama_module = ModuleType("ollama")
    client_module = ModuleType(OLLAMA_CLIENT_MODULE)
    setattr(client_module, CLIENT_CLASS_NAME, Client)
    setattr(client_module, ASYNC_CLIENT_CLASS_NAME, AsyncClient)
    setattr(ollama_module, "_client", client_module)

    monkeypatch.setitem(sys.modules, "ollama", ollama_module)
    monkeypatch.setitem(sys.modules, OLLAMA_CLIENT_MODULE, client_module)
    return Client, AsyncClient


def _assert_no_off_contract_aliases(attrs: dict[str, Any]) -> None:
    banned_attrs = {
        "respan.span.tools",
        "respan.span.tool_calls",
        "respan.span.handoffs",
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "span_tools",
        "has_tool_calls",
        "parallel_tool_calls",
    }
    for attr in banned_attrs:
        assert attr not in attrs


def test_activate_patches_sync_chat_and_emits_contract_span(
    fake_ollama: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    Client, _ = fake_ollama
    instrumentor = OllamaInstrumentor()

    instrumentor.activate()
    response = Client().chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "Weather in Tokyo?"}],
        tools=[get_weather],
    )

    assert response.message.content == "The weather is sunny."
    assert len(captured_spans) == 1
    attrs = captured_spans[0]._attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[GenAIAttributes.GEN_AI_SYSTEM] == "ollama"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.CHAT.value
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "llama3.2"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Weather in Tokyo?"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    assert (
        attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "The weather is sunny."
    )
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 3
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 4

    tools = json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])
    assert tools[0]["function"]["name"] == "get_weather"
    tool_calls = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert tool_calls[0]["function"]["arguments"] == '{"city": "Tokyo"}'
    _assert_no_off_contract_aliases(attrs)

    instrumentor.deactivate()


def test_generate_stream_emits_one_span_after_iterator_is_consumed(
    fake_ollama: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    Client, _ = fake_ollama
    instrumentor = OllamaInstrumentor()
    instrumentor.activate()

    chunks = list(
        Client().generate(
            model="llama3.2",
            prompt="Say hello",
            system="Be brief",
            stream=True,
        )
    )

    assert [chunk.response for chunk in chunks] == ["Hello ", "world."]
    assert len(captured_spans) == 1
    attrs = captured_spans[0]._attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.CHAT.value
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == "Say hello"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Hello world."
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 2
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 3
    _assert_no_off_contract_aliases(attrs)

    instrumentor.deactivate()


def test_async_methods_emit_spans(
    fake_ollama: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    async def run() -> None:
        _, AsyncClient = fake_ollama
        instrumentor = OllamaInstrumentor()
        instrumentor.activate()

        chat_response = await AsyncClient().chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hi"}],
        )
        generate_response = await AsyncClient().generate(
            model="llama3.2",
            prompt="Hi",
        )
        stream = await AsyncClient().chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "Stream"}],
            stream=True,
        )
        chunks = [chunk async for chunk in stream]

        assert chat_response.message.content == "Async chat."
        assert generate_response.response == "Async generated."
        assert [chunk.message.content for chunk in chunks] == ["Async ", "stream."]
        assert len(captured_spans) == 3
        assert captured_spans[0]._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
        assert captured_spans[1]._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
        assert (
            captured_spans[2]._attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
            == "Async stream."
        )

        instrumentor.deactivate()

    asyncio.run(run())


def test_embedding_span_does_not_capture_vectors(
    fake_ollama: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    Client, _ = fake_ollama
    instrumentor = OllamaInstrumentor()
    instrumentor.activate()

    response = Client().embed(model="nomic-embed-text", input="hello")

    assert response.embeddings == [[0.1, 0.2]]
    assert len(captured_spans) == 1
    attrs = captured_spans[0]._attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_EMBEDDING
    assert (
        attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.EMBEDDING.value
    )
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "nomic-embed-text"
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 7
    assert "embedding" not in attrs
    assert "embeddings" not in attrs
    _assert_no_off_contract_aliases(attrs)

    instrumentor.deactivate()


def test_error_path_emits_failed_span(
    monkeypatch: pytest.MonkeyPatch,
    fake_ollama: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    Client, _ = fake_ollama

    def raise_error(
        self: Any,
        model: str = "",
        messages: list[dict[str, Any]] | None = None,
        *,
        tools: list[Any] | None = None,
        stream: bool = False,
    ) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(Client, CHAT_METHOD_NAME, raise_error)
    instrumentor = OllamaInstrumentor()
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="boom"):
        Client().chat(model="llama3.2", messages=[{"role": "user", "content": "fail"}])

    assert len(captured_spans) == 1
    span = captured_spans[0]
    assert span.status.status_code.name == "ERROR"
    assert span._attributes["error.message"] == "boom"
    assert span._attributes[SpanAttributes.LLM_REQUEST_MODEL] == "llama3.2"
    _assert_no_off_contract_aliases(span._attributes)

    instrumentor.deactivate()


def test_deactivate_restores_original_methods(
    fake_ollama: tuple[type[Any], type[Any]],
) -> None:
    Client, AsyncClient = fake_ollama
    original_sync_chat = getattr(Client, CHAT_METHOD_NAME)
    original_sync_generate = getattr(Client, GENERATE_METHOD_NAME)
    original_sync_embed = getattr(Client, EMBED_METHOD_NAME)
    original_async_chat = getattr(AsyncClient, CHAT_METHOD_NAME)

    instrumentor = OllamaInstrumentor()
    instrumentor.activate()
    assert getattr(Client, CHAT_METHOD_NAME) is not original_sync_chat
    assert getattr(Client, GENERATE_METHOD_NAME) is not original_sync_generate
    assert getattr(Client, EMBED_METHOD_NAME) is not original_sync_embed
    assert getattr(AsyncClient, CHAT_METHOD_NAME) is not original_async_chat

    instrumentor.deactivate()
    assert getattr(Client, CHAT_METHOD_NAME) is original_sync_chat
    assert getattr(Client, GENERATE_METHOD_NAME) is original_sync_generate
    assert getattr(Client, EMBED_METHOD_NAME) is original_sync_embed
    assert getattr(AsyncClient, CHAT_METHOD_NAME) is original_async_chat


def test_active_workflow_name_is_attached_to_synthetic_span() -> None:
    token = context_api.attach(
        context_api.set_value(
            SpanAttributes.TRACELOOP_ENTITY_NAME,
            "ollama_chat_example",
        )
    )
    try:
        from respan_instrumentation_ollama._otel_emitter import build_chat_attrs

        attrs = build_chat_attrs(
            request_kwargs={
                "model": "llama3.2",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            response_or_chunks=_chat_response(content="Hi"),
        )
    finally:
        context_api.detach(token)

    assert attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "ollama_chat_example"
