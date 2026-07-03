"""Unit tests for the native OpenAI instrumentation (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from respan_instrumentation_openai import _otel_emitter as emitter
from respan_instrumentation_openai import _instrumentation as instr
from respan_instrumentation_openai._instrumentation import OpenAIInstrumentor


# --- fakes ------------------------------------------------------------------


def _chat_response():
    return SimpleNamespace(
        id="chatcmpl-123",
        model="gpt-4.1-nano-2025-04-14",
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello there", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3, total_tokens=11),
    )


def _embedding_response():
    return SimpleNamespace(
        model="text-embedding-3-small",
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
        usage=SimpleNamespace(prompt_tokens=5, total_tokens=5),
    )


def _responses_response():
    return SimpleNamespace(
        id="resp-1",
        model="gpt-4.1-mini",
        output_text="done",
        usage=SimpleNamespace(input_tokens=12, output_tokens=4, total_tokens=16),
    )


# --- attribute builders -----------------------------------------------------


def test_chat_attrs_are_typed_as_llm_with_tokens():
    attrs = emitter.build_chat_attrs(
        request_kwargs={"model": "gpt-4.1-nano", "messages": [{"role": "user", "content": "hi"}]},
        response=_chat_response(),
    )
    assert attrs["llm.request.type"] == "chat"
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["respan.entity.log_type"] == "chat"
    assert attrs["gen_ai.request.model"] == "gpt-4.1-nano"
    assert attrs["gen_ai.usage.prompt_tokens"] == 8
    assert attrs["gen_ai.usage.completion_tokens"] == 3
    assert attrs["gen_ai.usage.total_tokens"] == 11
    assert "Hello there" in attrs["traceloop.entity.output"]
    # input is round-trippable JSON
    assert json.loads(attrs["traceloop.entity.input"])[0]["content"] == "hi"


def test_embedding_attrs_typed_as_embedding():
    attrs = emitter.build_embedding_attrs(
        request_kwargs={"model": "text-embedding-3-small", "input": "reset password"},
        response=_embedding_response(),
    )
    assert attrs["llm.request.type"] == "embedding"
    assert attrs["respan.entity.log_type"] == "embedding"
    assert attrs["gen_ai.usage.prompt_tokens"] == 5
    assert json.loads(attrs["traceloop.entity.output"])["vector_count"] == 1
    assert json.loads(attrs["traceloop.entity.output"])["dimension"] == 3


def test_responses_attrs_map_input_output_tokens():
    attrs = emitter.build_response_attrs(
        request_kwargs={"model": "gpt-4.1-mini", "input": "hi"},
        response=_responses_response(),
    )
    assert attrs["llm.request.type"] == "chat"
    assert attrs["respan.entity.log_type"] == "response"
    assert attrs["gen_ai.usage.prompt_tokens"] == 12  # mapped from input_tokens
    assert attrs["gen_ai.usage.completion_tokens"] == 4  # mapped from output_tokens
    assert attrs["traceloop.entity.output"] == "done"


# --- emit path --------------------------------------------------------------


def test_emit_chat_injects_one_span(monkeypatch):
    captured = []
    monkeypatch.setattr(emitter, "inject_span", lambda span: captured.append(span))
    emitter.emit_chat_span(
        request_kwargs={"model": "gpt-4.1-nano", "messages": [{"role": "user", "content": "hi"}]},
        start_ns=1,
        response=_chat_response(),
    )
    assert len(captured) == 1
    span = captured[0]
    assert span.name == "openai.chat"
    assert span.attributes["llm.request.type"] == "chat"
    assert span.attributes["gen_ai.usage.total_tokens"] == 11


# --- streaming aggregation --------------------------------------------------


def _chunk(content=None, tool_deltas=None, usage=None, model="gpt-4.1-nano", cid="c1"):
    delta = SimpleNamespace(content=content, tool_calls=tool_deltas)
    return SimpleNamespace(id=cid, model=model, usage=usage, choices=[SimpleNamespace(delta=delta)])


def test_aggregate_chat_joins_text_and_usage():
    chunks = [
        _chunk(content="Hel"),
        _chunk(content="lo"),
        _chunk(usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6)),
    ]
    agg = instr._aggregate_chat(chunks)
    assert agg["choices"][0]["message"]["content"] == "Hello"
    assert emitter.tr.extract_usage(agg) == {"prompt": 4, "completion": 2, "total": 6}


def test_aggregate_chat_reassembles_streamed_tool_calls():
    def td(index, *, tid=None, name=None, args=None):
        fn = SimpleNamespace(name=name, arguments=args)
        return SimpleNamespace(index=index, id=tid, function=fn)

    chunks = [
        _chunk(tool_deltas=[td(0, tid="call_1", name="get_weather", args='{"ci')]),
        _chunk(tool_deltas=[td(0, args='ty": "Tokyo"}')]),
        _chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)),
    ]
    agg = instr._aggregate_chat(chunks)
    calls = agg["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"
    assert calls[0]["function"]["name"] == "get_weather"
    assert calls[0]["function"]["arguments"] == '{"city": "Tokyo"}'


# --- instrumentor lifecycle -------------------------------------------------


def test_activate_patches_and_deactivate_restores():
    pytest.importorskip("openai")
    from openai.resources.chat.completions import Completions
    from openai.resources.embeddings import Embeddings

    original_chat = Completions.create
    original_embed = Embeddings.create

    instr = OpenAIInstrumentor()
    instr.activate()
    try:
        assert Completions.create is not original_chat
        assert Embeddings.create is not original_embed
    finally:
        instr.deactivate()

    assert Completions.create is original_chat
    assert Embeddings.create is original_embed
