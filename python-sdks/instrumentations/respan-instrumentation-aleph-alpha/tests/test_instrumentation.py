from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType
from typing import Any

import pytest
from opentelemetry import context as context_api
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_aleph_alpha import AlephAlphaInstrumentor
from respan_instrumentation_aleph_alpha import _instrumentation
from respan_instrumentation_aleph_alpha._constants import (
    ALEPH_ALPHA_CLIENT_MODULE,
    ASYNC_CLIENT_CLASS_NAME,
    CLIENT_CLASS_NAME,
    METHOD_CHAT,
    METHOD_COMPLETE,
)
from respan_instrumentation_aleph_alpha._otel_emitter import build_aleph_alpha_attrs
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_EMBEDDING, LOG_TYPE_TEXT
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


class Obj:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_json(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if value is not None
        }


class FakeRequest(Obj):
    pass


@pytest.fixture(autouse=True)
def reset_instrumentation_globals() -> None:
    _instrumentation._original_methods.clear()


@pytest.fixture()
def captured_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    spans: list[Any] = []
    monkeypatch.setattr(
        "respan_instrumentation_aleph_alpha._otel_emitter.inject_span",
        lambda span: spans.append(span),
    )
    return spans


@pytest.fixture()
def fake_aleph_alpha(monkeypatch: pytest.MonkeyPatch) -> tuple[type[Any], type[Any]]:
    class Client:
        def complete(self, request: Any, model: str) -> Any:
            return Obj(
                model_version=f"{model}-2026",
                completions=[Obj(completion="Completed text.", finish_reason="stop")],
                num_tokens_prompt_total=4,
                num_tokens_generated=3,
            )

        def chat(self, request: Any, model: str) -> Any:
            tool_call = Obj(
                id="call_1",
                type="function",
                function={"name": "lookup", "arguments": '{"q":"respan"}'},
            )
            return Obj(
                finish_reason="tool_calls",
                message=Obj(
                    role=Obj(value="assistant"),
                    content="I should call a tool.",
                    tool_calls=[tool_call],
                ),
            )

        def embed(self, request: Any, model: str) -> Any:
            return Obj(
                model_version=model,
                num_tokens_prompt_total=5,
                embeddings={("-1", "mean"): [0.1, 0.2, 0.3]},
                tokens=None,
            )

    class AsyncClient:
        async def complete(self, request: Any, model: str) -> Any:
            return Obj(
                model_version=model,
                completions=[Obj(completion="Async completion.")],
                num_tokens_prompt_total=6,
                num_tokens_generated=2,
            )

        async def chat(self, request: Any, model: str) -> Any:
            return Obj(
                finish_reason="stop",
                message=Obj(role=Obj(value="assistant"), content="Async chat."),
            )

        async def complete_with_streaming(self, request: Any, model: str):
            yield Obj(index=0, completion="stream ")
            yield Obj(index=0, completion="done")
            yield Obj(num_tokens_prompt_total=7, num_tokens_generated=3)

        async def chat_with_streaming(self, request: Any, model: str):
            yield Obj(role=Obj(value="assistant"), content="")
            yield Obj(content="streamed ")
            yield Obj(content="chat")
            yield Obj(prompt_tokens=8, completion_tokens=4, total_tokens=12)
            yield Obj(value="stop")

    root_module = ModuleType("aleph_alpha_client")
    client_module = ModuleType(ALEPH_ALPHA_CLIENT_MODULE)
    setattr(client_module, CLIENT_CLASS_NAME, Client)
    setattr(client_module, ASYNC_CLIENT_CLASS_NAME, AsyncClient)
    setattr(root_module, "aleph_alpha_client", client_module)

    monkeypatch.setitem(sys.modules, "aleph_alpha_client", root_module)
    monkeypatch.setitem(sys.modules, ALEPH_ALPHA_CLIENT_MODULE, client_module)
    return Client, AsyncClient


def chat_request() -> FakeRequest:
    return FakeRequest(
        model="pharia-1-chat",
        messages=[
            {"role": "system", "content": "Be brief."},
            {
                "role": "user",
                "content": "Use a tool.",
                "tool_calls": [
                    {
                        "id": "history_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a term.",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )


def completion_request() -> FakeRequest:
    return FakeRequest(
        prompt=[{"type": "text", "data": "Complete this"}],
        maximum_tokens=12,
        temperature=0.1,
    )


def embedding_request() -> FakeRequest:
    return FakeRequest(
        prompt=[{"type": "text", "data": "Embed this"}],
        layers=[-1],
        pooling=["mean"],
    )


def test_activate_patches_sync_completion_chat_and_embedding(
    fake_aleph_alpha: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    Client, _ = fake_aleph_alpha
    instrumentor = AlephAlphaInstrumentor()

    instrumentor.activate()
    complete_response = Client().complete(completion_request(), "pharia-1")
    chat_response = Client().chat(chat_request(), "pharia-1-chat")
    embed_response = Client().embed(embedding_request(), "luminous-base")

    assert complete_response.completions[0].completion == "Completed text."
    assert chat_response.message.content == "I should call a tool."
    assert embed_response.embeddings[("-1", "mean")] == [0.1, 0.2, 0.3]
    assert len(captured_spans) == 3

    completion_attrs = captured_spans[0]._attributes
    assert completion_attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TEXT
    assert completion_attrs[SpanAttributes.LLM_REQUEST_MODEL] == "pharia-1"
    assert completion_attrs["gen_ai.prompt.0.content"] == "Complete this"
    assert completion_attrs["gen_ai.completion.0.content"] == "Completed text."
    assert completion_attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 4
    assert completion_attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 3

    chat_attrs = captured_spans[1]._attributes
    assert chat_attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert chat_attrs["gen_ai.prompt.0.role"] == "system"
    assert chat_attrs["gen_ai.prompt.1.tool_calls"] == (
        '[{"id": "history_1", "type": "function", "function": '
        '{"name": "lookup", "arguments": "{}"}}]'
    )
    assert json.loads(chat_attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0]["function"][
        "name"
    ] == "lookup"
    assert json.loads(chat_attrs["gen_ai.completion.0.tool_calls"])[0]["function"][
        "name"
    ] == "lookup"
    assert "tools" not in chat_attrs
    assert "tool_calls" not in chat_attrs
    assert "respan.span.tools" not in chat_attrs
    assert "respan.span.tool_calls" not in chat_attrs

    embedding_attrs = captured_spans[2]._attributes
    assert embedding_attrs[RESPAN_LOG_TYPE] == LOG_TYPE_EMBEDDING
    assert embedding_attrs[SpanAttributes.LLM_REQUEST_TYPE] == "embedding"
    assert embedding_attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] == "Embed this"
    assert embedding_attrs["gen_ai.prompt.0.role"] == "user"
    assert embedding_attrs["gen_ai.prompt.0.content"] == "Embed this"
    embedding_summary = json.loads(embedding_attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert embedding_summary == {
        "model": "luminous-base",
        "embedding_count": 1,
        "dimensions": 3,
    }
    assert json.loads(embedding_attrs["gen_ai.completion.0.content"]) == embedding_summary
    assert "ai.embedding" not in embedding_attrs
    assert "0.1" not in embedding_attrs["gen_ai.completion.0.content"]

    instrumentor.deactivate()


def test_async_methods_and_streaming_emit_spans(
    fake_aleph_alpha: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    async def run() -> None:
        _, AsyncClient = fake_aleph_alpha
        instrumentor = AlephAlphaInstrumentor()
        instrumentor.activate()

        complete_response = await AsyncClient().complete(completion_request(), "pharia-1")
        chat_response = await AsyncClient().chat(chat_request(), "pharia-1-chat")
        completion_chunks = [
            item
            async for item in AsyncClient().complete_with_streaming(
                completion_request(),
                "pharia-1",
            )
        ]
        chat_chunks = [
            item
            async for item in AsyncClient().chat_with_streaming(
                chat_request(),
                "pharia-1-chat",
            )
        ]

        assert complete_response.completions[0].completion == "Async completion."
        assert chat_response.message.content == "Async chat."
        assert len(completion_chunks) == 3
        assert len(chat_chunks) == 5
        assert len(captured_spans) == 4
        assert captured_spans[2]._attributes["gen_ai.completion.0.content"] == (
            "stream done"
        )
        assert captured_spans[2]._attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 7
        assert captured_spans[3]._attributes["gen_ai.completion.0.content"] == (
            "streamed chat"
        )
        assert captured_spans[3]._attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 12

        instrumentor.deactivate()

    asyncio.run(run())


def test_active_workflow_name_is_attached_to_injected_span() -> None:
    token = context_api.attach(
        context_api.set_value(
            SpanAttributes.TRACELOOP_ENTITY_NAME,
            "aleph_alpha_completion_workflow",
        )
    )
    try:
        attrs = build_aleph_alpha_attrs(
            operation="complete",
            request=completion_request(),
            model="pharia-1",
            response_or_items=Obj(
                model_version="pharia-1",
                completions=[Obj(completion="Hello")],
                num_tokens_prompt_total=1,
                num_tokens_generated=1,
            ),
        )
    finally:
        context_api.detach(token)

    assert (
        attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME]
        == "aleph_alpha_completion_workflow"
    )


def test_error_path_emits_failed_span(
    monkeypatch: pytest.MonkeyPatch,
    fake_aleph_alpha: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    Client, _ = fake_aleph_alpha

    def raise_error(self: Any, request: Any, model: str) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(Client, METHOD_COMPLETE, raise_error)
    instrumentor = AlephAlphaInstrumentor()
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="boom"):
        Client().complete(completion_request(), "pharia-1")

    assert len(captured_spans) == 1
    span = captured_spans[0]
    assert span.status.status_code.name == "ERROR"
    assert span._attributes["error.message"] == "boom"
    assert span._attributes[SpanAttributes.LLM_REQUEST_MODEL] == "pharia-1"

    instrumentor.deactivate()


def test_deactivate_restores_original_methods(
    fake_aleph_alpha: tuple[type[Any], type[Any]],
) -> None:
    Client, AsyncClient = fake_aleph_alpha
    original_sync_complete = getattr(Client, METHOD_COMPLETE)
    original_sync_chat = getattr(Client, METHOD_CHAT)
    original_async_complete = getattr(AsyncClient, METHOD_COMPLETE)
    original_async_chat = getattr(AsyncClient, METHOD_CHAT)

    instrumentor = AlephAlphaInstrumentor()
    instrumentor.activate()
    assert getattr(Client, METHOD_COMPLETE) is not original_sync_complete
    assert getattr(AsyncClient, METHOD_COMPLETE) is not original_async_complete

    instrumentor.deactivate()
    assert getattr(Client, METHOD_COMPLETE) is original_sync_complete
    assert getattr(Client, METHOD_CHAT) is original_sync_chat
    assert getattr(AsyncClient, METHOD_COMPLETE) is original_async_complete
    assert getattr(AsyncClient, METHOD_CHAT) is original_async_chat
