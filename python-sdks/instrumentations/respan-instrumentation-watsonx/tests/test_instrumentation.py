from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType
from typing import Any

import pytest
from opentelemetry import context as context_api
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

from respan_instrumentation_watsonx import WatsonxInstrumentor
from respan_instrumentation_watsonx import _instrumentation
from respan_instrumentation_watsonx._constants import (
    AEMBEDDINGS_GENERATE_METHOD_NAME,
    AEMBED_DOCUMENTS_METHOD_NAME,
    AEMBED_QUERY_METHOD_NAME,
    AGENERATE_METHOD_NAME,
    AGENERATE_STREAM_METHOD_NAME,
    CHAT_METHOD_NAME,
    CHAT_STREAM_METHOD_NAME,
    EMBEDDINGS_CLASS_NAME,
    EMBEDDINGS_GENERATE_METHOD_NAME,
    EMBEDDINGS_MODULE,
    EMBED_DOCUMENTS_METHOD_NAME,
    EMBED_QUERY_METHOD_NAME,
    GENERATE_METHOD_NAME,
    GENERATE_TEXT_METHOD_NAME,
    GENERATE_TEXT_STREAM_METHOD_NAME,
    MODEL_INFERENCE_CLASS_NAME,
    MODEL_INFERENCE_MODULE,
)
from respan_instrumentation_watsonx._otel_emitter import (
    build_chat_attrs,
    build_embedding_attrs,
    build_text_attrs,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_EMBEDDING, LOG_TYPE_TEXT
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE

FORBIDDEN_ALIASES = {
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "tools",
    "tool_calls",
    "span_tools",
    "has_tool_calls",
    "parallel_tool_calls",
    "respan.span.tools",
    "respan.span.tool_calls",
    "respan.span.handoffs",
}


class Obj:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture(autouse=True)
def reset_instrumentation_globals() -> None:
    _instrumentation._original_methods.clear()


@pytest.fixture()
def captured_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    spans: list[Any] = []
    monkeypatch.setattr(
        "respan_instrumentation_watsonx._otel_emitter.inject_span",
        lambda span: spans.append(span),
    )
    return spans


@pytest.fixture()
def fake_watsonx(monkeypatch: pytest.MonkeyPatch) -> tuple[type[Any], type[Any]]:
    class ModelInference:
        model_id = "ibm/granite-3-8b-instruct"

        def generate(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
            return {
                "model_id": self.model_id,
                "results": [
                    {
                        "generated_text": f"generated: {prompt}",
                        "input_token_count": 3,
                        "generated_token_count": 4,
                    }
                ],
            }

        def generate_text(self, prompt: str | None = None, **kwargs: Any) -> str:
            return f"text: {prompt}"

        def generate_text_stream(self, prompt: str | None = None, **kwargs: Any):
            yield "stream "
            yield "text"

        async def agenerate(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
            return {
                "results": [
                    {
                        "generated_text": f"async: {prompt}",
                        "input_token_count": 5,
                        "generated_token_count": 6,
                    }
                ]
            }

        async def agenerate_stream(self, prompt: str | None = None, **kwargs: Any):
            async def chunks():
                yield "async "
                yield "stream"

            return chunks()

        def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The weather is sunny.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": {"city": "Tokyo"},
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16},
            }

        def chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any):
            yield {"choices": [{"delta": {"content": "hello "}}]}
            yield {"choices": [{"delta": {"content": "world"}}]}

        async def achat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
            return {
                "choices": [{"message": {"role": "assistant", "content": "async chat"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }

        async def achat_stream(self, messages: list[dict[str, Any]], **kwargs: Any):
            async def chunks():
                yield {"choices": [{"delta": {"content": "async "}}]}
                yield {"choices": [{"delta": {"content": "chat stream"}}]}

            return chunks()

    class Embeddings:
        model_id = "ibm/slate-125m-english-rtrvr"

        def generate(self, inputs: list[str], **kwargs: Any) -> dict[str, Any]:
            return {
                "results": [
                    {"embedding": [0.1, 0.2], "input": inputs[0]},
                    {"embedding": [0.3, 0.4], "input": inputs[1]},
                ],
                "input_token_count": 8,
            }

        def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return [[0.1, 0.2], [0.3, 0.4]]

        def embed_query(self, text: str, **kwargs: Any) -> list[float]:
            return [0.1, 0.2]

        async def agenerate(self, inputs: list[str], **kwargs: Any) -> dict[str, Any]:
            return {"results": [{"embedding": [0.1, 0.2], "input": inputs[0]}]}

        async def aembed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return [[0.1, 0.2], [0.3, 0.4]]

        async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
            return [0.1, 0.2]

    ibm_module = ModuleType("ibm_watsonx_ai")
    foundation_models_module = ModuleType("ibm_watsonx_ai.foundation_models")
    inference_module = ModuleType(MODEL_INFERENCE_MODULE)
    embeddings_module = ModuleType(EMBEDDINGS_MODULE)
    setattr(inference_module, MODEL_INFERENCE_CLASS_NAME, ModelInference)
    setattr(embeddings_module, EMBEDDINGS_CLASS_NAME, Embeddings)
    setattr(foundation_models_module, "inference", inference_module)
    setattr(foundation_models_module, "embeddings", embeddings_module)
    setattr(ibm_module, "foundation_models", foundation_models_module)

    monkeypatch.setitem(sys.modules, "ibm_watsonx_ai", ibm_module)
    monkeypatch.setitem(
        sys.modules,
        "ibm_watsonx_ai.foundation_models",
        foundation_models_module,
    )
    monkeypatch.setitem(sys.modules, MODEL_INFERENCE_MODULE, inference_module)
    monkeypatch.setitem(sys.modules, EMBEDDINGS_MODULE, embeddings_module)
    return ModelInference, Embeddings


def get_weather(city: str) -> str:
    """Return fake weather for a city."""
    return f"Sunny in {city}"


def test_activate_patches_text_generation_and_emits_text_span(
    fake_watsonx: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    ModelInference, _ = fake_watsonx
    instrumentor = WatsonxInstrumentor()

    instrumentor.activate()
    response = ModelInference().generate("Say hello")

    assert response["results"][0]["generated_text"] == "generated: Say hello"
    assert len(captured_spans) == 1
    attrs = captured_spans[0].attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TEXT
    assert attrs[TLSpanAttributes.LLM_REQUEST_TYPE] == "completion"
    assert attrs[TLSpanAttributes.LLM_REQUEST_MODEL] == "ibm/granite-3-8b-instruct"
    assert attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.content"] == "Say hello"
    assert attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"] == "generated: Say hello"
    assert attrs[TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 3
    assert attrs[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 4
    assert FORBIDDEN_ALIASES.isdisjoint(attrs)

    instrumentor.deactivate()


def test_stream_emits_one_text_span_after_iterator_is_consumed(
    fake_watsonx: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    ModelInference, _ = fake_watsonx
    instrumentor = WatsonxInstrumentor()
    instrumentor.activate()

    chunks = list(ModelInference().generate_text_stream(prompt="Stream this"))

    assert chunks == ["stream ", "text"]
    assert len(captured_spans) == 1
    assert (
        captured_spans[0].attributes[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"]
        == "stream text"
    )

    instrumentor.deactivate()


def test_chat_span_maps_tools_tool_calls_and_avoids_aliases(
    fake_watsonx: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    ModelInference, _ = fake_watsonx
    instrumentor = WatsonxInstrumentor()
    instrumentor.activate()

    response = ModelInference().chat(
        messages=[
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Weather in Tokyo?"},
        ],
        tools=[get_weather],
    )

    assert response["choices"][0]["message"]["content"] == "The weather is sunny."
    attrs = captured_spans[0].attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[TLSpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert json.loads(attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS])[0]["function"][
        "name"
    ] == "get_weather"
    assert json.loads(
        attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
    )[0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city": "Tokyo"}',
    }
    assert attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "The weather is sunny."
    )
    assert attrs[TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 9
    assert attrs[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 7
    assert FORBIDDEN_ALIASES.isdisjoint(attrs)

    instrumentor.deactivate()


def test_async_model_methods_emit_spans(
    fake_watsonx: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    async def run() -> None:
        ModelInference, _ = fake_watsonx
        instrumentor = WatsonxInstrumentor()
        instrumentor.activate()

        response = await ModelInference().achat(
            messages=[{"role": "user", "content": "Async?"}]
        )
        stream = await ModelInference().achat_stream(
            messages=[{"role": "user", "content": "Stream?"}]
        )
        chunks = [chunk async for chunk in stream]

        assert response["choices"][0]["message"]["content"] == "async chat"
        assert len(chunks) == 2
        assert len(captured_spans) == 2
        assert (
            captured_spans[0].attributes[
                f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"
            ]
            == "async chat"
        )
        assert (
            captured_spans[1].attributes[
                f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"
            ]
            == "async chat stream"
        )

        instrumentor.deactivate()

    asyncio.run(run())


def test_embedding_methods_emit_embedding_span_without_vectors(
    fake_watsonx: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    _, Embeddings = fake_watsonx
    instrumentor = WatsonxInstrumentor()
    instrumentor.activate()

    response = Embeddings().generate(inputs=["first", "second"])

    assert len(response["results"]) == 2
    attrs = captured_spans[0].attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_EMBEDDING
    assert attrs[TLSpanAttributes.LLM_REQUEST_TYPE] == "embedding"
    assert attrs[TLSpanAttributes.LLM_REQUEST_MODEL] == "ibm/slate-125m-english-rtrvr"
    assert attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.content"] == '["first", "second"]'
    assert json.loads(attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "vector_count": 2,
        "dimension": 2,
    }
    assert "embedding" not in attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    assert FORBIDDEN_ALIASES.isdisjoint(attrs)

    instrumentor.deactivate()


def test_async_embedding_methods_emit_spans(
    fake_watsonx: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    async def run() -> None:
        _, Embeddings = fake_watsonx
        instrumentor = WatsonxInstrumentor()
        instrumentor.activate()

        await Embeddings().aembed_query(text="question")
        await Embeddings().aembed_documents(texts=["one", "two"])

        assert len(captured_spans) == 2
        assert captured_spans[0].attributes[RESPAN_LOG_TYPE] == LOG_TYPE_EMBEDDING
        assert json.loads(captured_spans[1].attributes["traceloop.entity.output"]) == {
            "vector_count": 2,
            "dimension": 2,
        }

        instrumentor.deactivate()

    asyncio.run(run())


def test_active_workflow_name_is_attached_to_injected_span() -> None:
    token = context_api.attach(
        context_api.set_value(
            TLSpanAttributes.TRACELOOP_ENTITY_NAME,
            "watsonx_generate_text",
        )
    )
    try:
        attrs = build_text_attrs(
            instance=Obj(model_id="ibm/granite"),
            request_kwargs={"prompt": "Hello"},
            response_or_chunks="Hi",
        )
    finally:
        context_api.detach(token)

    assert attrs[TLSpanAttributes.TRACELOOP_WORKFLOW_NAME] == "watsonx_generate_text"


def test_builders_do_not_emit_banned_aliases() -> None:
    text_attrs = build_text_attrs(
        instance=Obj(model_id="ibm/granite"),
        request_kwargs={"prompt": "Hello"},
        response_or_chunks={"results": [{"generated_text": "Hi"}]},
    )
    chat_attrs = build_chat_attrs(
        instance=Obj(model_id="ibm/granite"),
        request_kwargs={"messages": [{"role": "user", "content": "Hi"}]},
        response_or_chunks={"choices": [{"message": {"content": "Hello"}}]},
    )
    embedding_attrs = build_embedding_attrs(
        instance=Obj(model_id="ibm/slate"),
        request_kwargs={"texts": ["Hi"]},
        response=[[0.1, 0.2]],
    )

    assert FORBIDDEN_ALIASES.isdisjoint(text_attrs)
    assert FORBIDDEN_ALIASES.isdisjoint(chat_attrs)
    assert FORBIDDEN_ALIASES.isdisjoint(embedding_attrs)


def test_error_path_emits_failed_span(
    monkeypatch: pytest.MonkeyPatch,
    fake_watsonx: tuple[type[Any], type[Any]],
    captured_spans: list[Any],
) -> None:
    ModelInference, _ = fake_watsonx

    def raise_error(self: Any, prompt: str | None = None, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(ModelInference, GENERATE_TEXT_METHOD_NAME, raise_error)
    instrumentor = WatsonxInstrumentor()
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="boom"):
        ModelInference().generate_text(prompt="fail")

    assert len(captured_spans) == 1
    span = captured_spans[0]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["error.message"] == "boom"
    assert (
        span.attributes[TLSpanAttributes.LLM_REQUEST_MODEL]
        == "ibm/granite-3-8b-instruct"
    )

    instrumentor.deactivate()


def test_deactivate_restores_original_methods(
    fake_watsonx: tuple[type[Any], type[Any]],
) -> None:
    ModelInference, Embeddings = fake_watsonx
    original_methods = {
        (ModelInference, GENERATE_METHOD_NAME): getattr(ModelInference, GENERATE_METHOD_NAME),
        (ModelInference, GENERATE_TEXT_METHOD_NAME): getattr(
            ModelInference,
            GENERATE_TEXT_METHOD_NAME,
        ),
        (ModelInference, GENERATE_TEXT_STREAM_METHOD_NAME): getattr(
            ModelInference,
            GENERATE_TEXT_STREAM_METHOD_NAME,
        ),
        (ModelInference, AGENERATE_METHOD_NAME): getattr(
            ModelInference,
            AGENERATE_METHOD_NAME,
        ),
        (ModelInference, AGENERATE_STREAM_METHOD_NAME): getattr(
            ModelInference,
            AGENERATE_STREAM_METHOD_NAME,
        ),
        (ModelInference, CHAT_METHOD_NAME): getattr(ModelInference, CHAT_METHOD_NAME),
        (ModelInference, CHAT_STREAM_METHOD_NAME): getattr(
            ModelInference,
            CHAT_STREAM_METHOD_NAME,
        ),
        (Embeddings, EMBEDDINGS_GENERATE_METHOD_NAME): getattr(
            Embeddings,
            EMBEDDINGS_GENERATE_METHOD_NAME,
        ),
        (Embeddings, EMBED_DOCUMENTS_METHOD_NAME): getattr(
            Embeddings,
            EMBED_DOCUMENTS_METHOD_NAME,
        ),
        (Embeddings, EMBED_QUERY_METHOD_NAME): getattr(Embeddings, EMBED_QUERY_METHOD_NAME),
        (Embeddings, AEMBEDDINGS_GENERATE_METHOD_NAME): getattr(
            Embeddings,
            AEMBEDDINGS_GENERATE_METHOD_NAME,
        ),
        (Embeddings, AEMBED_DOCUMENTS_METHOD_NAME): getattr(
            Embeddings,
            AEMBED_DOCUMENTS_METHOD_NAME,
        ),
        (Embeddings, AEMBED_QUERY_METHOD_NAME): getattr(Embeddings, AEMBED_QUERY_METHOD_NAME),
    }

    instrumentor = WatsonxInstrumentor()
    instrumentor.activate()
    assert getattr(ModelInference, GENERATE_METHOD_NAME) is not original_methods[
        (ModelInference, GENERATE_METHOD_NAME)
    ]
    assert getattr(Embeddings, EMBED_QUERY_METHOD_NAME) is not original_methods[
        (Embeddings, EMBED_QUERY_METHOD_NAME)
    ]

    instrumentor.deactivate()
    for (target_class, method_name), original in original_methods.items():
        assert getattr(target_class, method_name) is original
