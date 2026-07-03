from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from opentelemetry import context as context_api
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_together import TogetherInstrumentor
from respan_instrumentation_together import _instrumentation
from respan_instrumentation_together._constants import (
    ASYNC_COMPLETIONS_RESOURCE_CLASS_NAME,
    ASYNC_EMBEDDINGS_RESOURCE_CLASS_NAME,
    ASYNC_IMAGES_RESOURCE_CLASS_NAME,
    ASYNC_RERANK_RESOURCE_CLASS_NAME,
    BANNED_ALIAS_ATTRS,
    COMPLETIONS_RESOURCE_CLASS_NAME,
    CREATE_METHOD_NAME,
    EMBEDDINGS_RESOURCE_CLASS_NAME,
    GENERATE_METHOD_NAME,
    IMAGES_RESOURCE_CLASS_NAME,
    RERANK_RESOURCE_CLASS_NAME,
    TOGETHER_CHAT_COMPLETIONS_MODULE,
    TOGETHER_EMBEDDINGS_MODULE,
    TOGETHER_IMAGES_MODULE,
    TOGETHER_RERANK_MODULE,
    TOGETHER_TEXT_COMPLETIONS_MODULE,
)
from respan_instrumentation_together._otel_emitter import (
    build_chat_attrs,
    build_completion_attrs,
    build_embedding_attrs,
    build_image_attrs,
    build_rerank_attrs,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_EMBEDDING, LOG_TYPE_TEXT
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


def obj(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def usage(prompt_tokens: int = 3, completion_tokens: int = 4) -> SimpleNamespace:
    return obj(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def chat_response(*, content: str = "hello", tool_calls: list[Any] | None = None) -> Any:
    return obj(
        model="openai/gpt-oss-20b",
        choices=[
            obj(
                finish_reason="stop",
                message=obj(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                ),
            )
        ],
        usage=usage(),
    )


def text_response() -> Any:
    return obj(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        choices=[obj(finish_reason="stop", text=" completed text")],
        usage=usage(5, 6),
    )


def embedding_response() -> Any:
    return obj(
        model="BAAI/bge-base-en-v1.5",
        data=[
            obj(index=0, object="embedding", embedding=[0.1, 0.2, 0.3]),
            obj(index=1, object="embedding", embedding=[0.4, 0.5, 0.6]),
        ],
        object="list",
    )


def rerank_response() -> Any:
    return obj(
        model="Salesforce/Llama-Rank-v1",
        object="rerank",
        results=[
            obj(index=1, relevance_score=0.95, document=obj(text="Washington, D.C.")),
        ],
        usage=usage(7, 0),
    )


def image_response() -> Any:
    return obj(
        model="black-forest-labs/FLUX.1-schnell-Free",
        object="list",
        data=[obj(index=0, type="url", url="https://example.test/image.png")],
    )


class AsyncIterator:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> "AsyncIterator":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


@pytest.fixture(autouse=True)
def reset_instrumentation_globals() -> None:
    _instrumentation._original_sync_chat_create = None
    _instrumentation._original_async_chat_create = None
    _instrumentation._original_sync_completion_create = None
    _instrumentation._original_async_completion_create = None
    _instrumentation._original_sync_embedding_create = None
    _instrumentation._original_async_embedding_create = None
    _instrumentation._original_sync_image_generate = None
    _instrumentation._original_async_image_generate = None
    _instrumentation._original_sync_rerank_create = None
    _instrumentation._original_async_rerank_create = None


@pytest.fixture()
def captured_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    spans: list[Any] = []
    monkeypatch.setattr(
        "respan_instrumentation_together._otel_emitter.inject_span",
        lambda span: spans.append(span),
    )
    return spans


@pytest.fixture()
def fake_together_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, type[Any]]:
    class ChatCompletionsResource:
        def create(self, *, model: str, messages: Any, tools: Any = None, stream: bool = False) -> Any:
            if stream:
                return iter(
                    [
                        obj(
                            choices=[
                                obj(delta=obj(role="assistant", content="hello "), finish_reason=None)
                            ],
                            usage=None,
                        ),
                        obj(
                            choices=[
                                obj(delta=obj(role="assistant", content="stream"), finish_reason="stop")
                            ],
                            usage=usage(8, 9),
                        ),
                    ]
                )
            return chat_response(content=f"{model}: {messages[0]['content']}")

    class AsyncChatCompletionsResource:
        async def create(self, *, model: str, messages: Any, stream: bool = False) -> Any:
            if stream:
                return AsyncIterator(
                    [
                        obj(choices=[obj(delta=obj(role="assistant", content="async "))]),
                        obj(choices=[obj(delta=obj(role="assistant", content="stream"))], usage=usage(10, 11)),
                    ]
                )
            return chat_response(content=f"async {messages[0]['content']}")

    class CompletionsResource:
        def create(self, *, model: str, prompt: str, stream: bool = False) -> Any:
            return text_response()

    class AsyncCompletionsResource:
        async def create(self, *, model: str, prompt: str, stream: bool = False) -> Any:
            return text_response()

    class EmbeddingsResource:
        def create(self, *, input: Any, model: str) -> Any:
            return embedding_response()

    class AsyncEmbeddingsResource:
        async def create(self, *, input: Any, model: str) -> Any:
            return embedding_response()

    class ImagesResource:
        def generate(self, *, model: str, prompt: str, **kwargs: Any) -> Any:
            return image_response()

    class AsyncImagesResource:
        async def generate(self, *, model: str, prompt: str, **kwargs: Any) -> Any:
            return image_response()

    class RerankResource:
        def create(self, *, documents: Any, model: str, query: str, **kwargs: Any) -> Any:
            return rerank_response()

    class AsyncRerankResource:
        async def create(self, *, documents: Any, model: str, query: str, **kwargs: Any) -> Any:
            return rerank_response()

    modules = {
        TOGETHER_CHAT_COMPLETIONS_MODULE: {
            COMPLETIONS_RESOURCE_CLASS_NAME: ChatCompletionsResource,
            ASYNC_COMPLETIONS_RESOURCE_CLASS_NAME: AsyncChatCompletionsResource,
        },
        TOGETHER_TEXT_COMPLETIONS_MODULE: {
            COMPLETIONS_RESOURCE_CLASS_NAME: CompletionsResource,
            ASYNC_COMPLETIONS_RESOURCE_CLASS_NAME: AsyncCompletionsResource,
        },
        TOGETHER_EMBEDDINGS_MODULE: {
            EMBEDDINGS_RESOURCE_CLASS_NAME: EmbeddingsResource,
            ASYNC_EMBEDDINGS_RESOURCE_CLASS_NAME: AsyncEmbeddingsResource,
        },
        TOGETHER_IMAGES_MODULE: {
            IMAGES_RESOURCE_CLASS_NAME: ImagesResource,
            ASYNC_IMAGES_RESOURCE_CLASS_NAME: AsyncImagesResource,
        },
        TOGETHER_RERANK_MODULE: {
            RERANK_RESOURCE_CLASS_NAME: RerankResource,
            ASYNC_RERANK_RESOURCE_CLASS_NAME: AsyncRerankResource,
        },
    }
    for module_name, attrs in modules.items():
        module = ModuleType(module_name)
        for attr_name, attr_value in attrs.items():
            setattr(module, attr_name, attr_value)
        monkeypatch.setitem(sys.modules, module_name, module)

    return {
        "chat": ChatCompletionsResource,
        "async_chat": AsyncChatCompletionsResource,
        "completion": CompletionsResource,
        "async_completion": AsyncCompletionsResource,
        "embedding": EmbeddingsResource,
        "async_embedding": AsyncEmbeddingsResource,
        "image": ImagesResource,
        "async_image": AsyncImagesResource,
        "rerank": RerankResource,
        "async_rerank": AsyncRerankResource,
    }


def assert_no_banned_aliases(attrs: dict[str, Any]) -> None:
    assert BANNED_ALIAS_ATTRS.isdisjoint(attrs)


def test_build_chat_attrs_uses_canonical_fields_without_aliases() -> None:
    tool_call = obj(
        id="call_1",
        type="function",
        function=obj(name="get_weather", arguments='{"city":"Tokyo"}'),
    )
    attrs = build_chat_attrs(
        request_kwargs={
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather.",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
        response_or_chunks=chat_response(
            content="I will check.",
            tool_calls=[tool_call],
        ),
    )

    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.CHAT.value
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "openai/gpt-oss-20b"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Weather in Tokyo?"
    assert json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0]["function"]["name"] == "get_weather"
    assert json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])[0]["function"]["name"] == "get_weather"
    assert_no_banned_aliases(attrs)


def test_build_non_chat_attrs_cover_completion_embedding_rerank_and_image() -> None:
    completion_attrs = build_completion_attrs(
        request_kwargs={"model": "code", "prompt": "Complete this"},
        response_or_chunks=text_response(),
    )
    assert completion_attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TEXT
    assert completion_attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.COMPLETION.value
    assert completion_attrs[SpanAttributes.LLM_REQUEST_MODEL] == "code"
    assert completion_attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == " completed text"
    assert_no_banned_aliases(completion_attrs)

    embedding_attrs = build_embedding_attrs(
        request_kwargs={"model": "embed", "input": ["alpha", "beta"]},
        response=embedding_response(),
    )
    assert embedding_attrs[RESPAN_LOG_TYPE] == LOG_TYPE_EMBEDDING
    assert embedding_attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.EMBEDDING.value
    assert embedding_attrs[SpanAttributes.LLM_REQUEST_MODEL] == "embed"
    assert json.loads(embedding_attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "embedding_count": 2,
        "embedding_dimensions": 3,
    }
    assert "0.1" not in embedding_attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    assert_no_banned_aliases(embedding_attrs)

    rerank_attrs = build_rerank_attrs(
        request_kwargs={
            "model": "rank",
            "query": "capital",
            "documents": ["New York", "Washington, D.C."],
        },
        response=rerank_response(),
    )
    assert rerank_attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.RERANK.value
    assert rerank_attrs[SpanAttributes.LLM_REQUEST_MODEL] == "rank"
    assert json.loads(rerank_attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[0]["index"] == 1
    assert_no_banned_aliases(rerank_attrs)

    image_attrs = build_image_attrs(
        request_kwargs={"model": "flux", "prompt": "a small robot"},
        response=image_response(),
    )
    assert image_attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.UNKNOWN.value
    assert image_attrs[SpanAttributes.LLM_REQUEST_MODEL] == "flux"
    assert json.loads(image_attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])["image_count"] == 1
    assert_no_banned_aliases(image_attrs)


def test_activate_patches_sync_chat_and_emits_span(
    fake_together_modules: dict[str, type[Any]],
    captured_spans: list[Any],
) -> None:
    ChatCompletionsResource = fake_together_modules["chat"]
    instrumentor = TogetherInstrumentor()
    instrumentor.activate()

    response = ChatCompletionsResource().create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": "Say hello"}],
    )

    assert response.choices[0].message.content == "openai/gpt-oss-20b: Say hello"
    assert len(captured_spans) == 1
    attrs = captured_spans[0]._attributes
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "openai/gpt-oss-20b"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "openai/gpt-oss-20b: Say hello"
    assert_no_banned_aliases(attrs)

    instrumentor.deactivate()


def test_stream_emits_after_consumption(
    fake_together_modules: dict[str, type[Any]],
    captured_spans: list[Any],
) -> None:
    ChatCompletionsResource = fake_together_modules["chat"]
    instrumentor = TogetherInstrumentor()
    instrumentor.activate()

    chunks = list(
        ChatCompletionsResource().create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "Stream"}],
            stream=True,
        )
    )

    assert len(chunks) == 2
    assert len(captured_spans) == 1
    attrs = captured_spans[0]._attributes
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "hello stream"
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 8
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 9
    assert_no_banned_aliases(attrs)

    instrumentor.deactivate()


def test_async_chat_emits_span(
    fake_together_modules: dict[str, type[Any]],
    captured_spans: list[Any],
) -> None:
    async def run() -> None:
        AsyncChatCompletionsResource = fake_together_modules["async_chat"]
        instrumentor = TogetherInstrumentor()
        instrumentor.activate()

        response = await AsyncChatCompletionsResource().create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "Async"}],
        )

        assert response.choices[0].message.content == "async Async"
        assert len(captured_spans) == 1
        assert captured_spans[0]._attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "async Async"

        instrumentor.deactivate()

    asyncio.run(run())


def test_active_workflow_name_is_attached_to_injected_span() -> None:
    token = context_api.attach(
        context_api.set_value(
            SpanAttributes.TRACELOOP_ENTITY_NAME,
            "together_chat_completion",
        )
    )
    try:
        attrs = build_chat_attrs(
            request_kwargs={
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            response_or_chunks=chat_response(),
        )
    finally:
        context_api.detach(token)

    assert attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "together_chat_completion"


def test_deactivate_restores_original_methods(fake_together_modules: dict[str, type[Any]]) -> None:
    ChatCompletionsResource = fake_together_modules["chat"]
    AsyncChatCompletionsResource = fake_together_modules["async_chat"]
    original_sync = getattr(ChatCompletionsResource, CREATE_METHOD_NAME)
    original_async = getattr(AsyncChatCompletionsResource, CREATE_METHOD_NAME)

    instrumentor = TogetherInstrumentor()
    instrumentor.activate()
    assert getattr(ChatCompletionsResource, CREATE_METHOD_NAME) is not original_sync
    assert getattr(AsyncChatCompletionsResource, CREATE_METHOD_NAME) is not original_async

    instrumentor.deactivate()
    assert getattr(ChatCompletionsResource, CREATE_METHOD_NAME) is original_sync
    assert getattr(AsyncChatCompletionsResource, CREATE_METHOD_NAME) is original_async
