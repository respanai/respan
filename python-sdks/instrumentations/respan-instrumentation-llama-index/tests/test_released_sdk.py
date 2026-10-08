"""Controlled HTTP fixtures using released LlamaIndex providers and workflows."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index_instrumentation import root_dispatcher
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import StatusCode
from respan_instrumentation_llama_index import LlamaIndexInstrumentor
from respan_instrumentation_llama_index._serialization import extract_usage, safe_json
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

_fixture_spec = importlib.util.spec_from_file_location(
    "llama_provider_fixture", Path(__file__).with_name("_provider_fixture.py")
)
_fixture = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture)
build_fixture_embedding = _fixture.build_fixture_embedding
build_fixture_llm = _fixture.build_fixture_llm


@pytest.fixture
def runtime(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    owner = LlamaIndexInstrumentor()
    owner.activate()
    try:
        yield owner, provider, exporter
    finally:
        owner.deactivate()
        provider.shutdown()


def logical(exporter, kind="chat"):
    return [
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get(RESPAN_LOG_TYPE) == kind
    ]


def test_released_chat_source_usage_cache_reasoning_and_identity(runtime):
    _, _, exporter = runtime
    result = build_fixture_llm().chat(
        [ChatMessage(role="user", content="released prompt")]
    )
    assert "controlled tracing answer" in result.message.content
    assert result.raw.usage.prompt_tokens == 12
    span = logical(exporter)[0]
    assert span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 12
    assert span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 4
    assert span.attributes[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert span.attributes[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 2
    assert span.attributes[SpanAttributes.LLM_SYSTEM] == "openai"
    assert trace.get_current_span().get_span_context().is_valid is False


def test_released_provider_batch_embedding_full_vectors_and_actual_usage(runtime):
    _, _, exporter = runtime
    result = build_fixture_embedding().get_text_embedding_batch(["first", "second"])
    assert [len(vector) for vector in result] == [3072, 3072]
    spans = logical(exporter, "embedding")
    assert len(spans) == 1
    import json

    assert (
        json.loads(spans[0].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
        == result
    )
    assert spans[0].attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 7
    assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in spans[0].attributes


@pytest.mark.parametrize("operation", ["text", "query", "async_text", "async_query"])
def test_single_embedding_inner_native_span_provider_usage(runtime, operation):
    _, _, exporter = runtime
    model = build_fixture_embedding()
    if operation == "text":
        result = model.get_text_embedding("released single text")
    elif operation == "query":
        result = model.get_query_embedding("released single query")
    elif operation == "async_text":
        result = asyncio.run(model.aget_text_embedding("released single text"))
    else:
        result = asyncio.run(model.aget_query_embedding("released single query"))
    assert len(result) == 3072
    span = logical(exporter, "embedding")[0]
    assert span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 7
    assert span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 7
    assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in span.attributes


def test_embedding_multiple_released_http_responses_sum_actual_counts(runtime):
    _, _, exporter = runtime
    model = build_fixture_embedding()
    model.embed_batch_size = 1
    result = model.get_text_embedding_batch(["first", "second"])
    assert [len(vector) for vector in result] == [3072, 3072]
    spans = logical(exporter, "embedding")
    assert (
        sum(span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] for span in spans)
        == 14
    )
    assert (
        sum(span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] for span in spans)
        == 14
    )


@pytest.mark.parametrize("style", ["env", "context", "constructor"])
def test_released_start_privacy_and_return(runtime, monkeypatch, style):
    owner, _, exporter = runtime
    token = None
    if style == "env":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    elif style == "context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    else:
        owner.deactivate()
        owner = LlamaIndexInstrumentor(capture_content=False)
        owner.activate()
    try:
        result = build_fixture_llm().chat(
            [ChatMessage(role="user", content="private-sentinel")]
        )
        assert result.message.content
    finally:
        if token is not None:
            context.detach(token)
        if style == "constructor":
            owner.deactivate()
    assert logical(exporter)
    for span in exporter.get_finished_spans():
        assert "private-sentinel" not in str(span.attributes)
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes


def test_native_stream_snapshot_end_veto_and_context(runtime, monkeypatch):
    owner, _, exporter = runtime
    llm = build_fixture_llm()
    stream = llm.stream_chat([ChatMessage(role="user", content="stream-private")])
    assert inspect.isgenerator(stream)
    assert hasattr(stream, "send") and hasattr(stream, "throw")
    first = next(stream)
    assert first.message.content
    assert trace.get_current_span().get_span_context().is_valid is False
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    list(stream)
    for span in exporter.get_finished_spans():
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    assert not owner._span_handler.open_spans
    assert not owner._event_handler._open_event_spans


def test_native_async_stream_close_cancel_and_context(runtime):
    owner, _, exporter = runtime

    async def run():
        llm = build_fixture_llm()
        stream = await llm.astream_chat(
            [ChatMessage(role="user", content="async stream")]
        )
        assert inspect.isasyncgen(stream)
        assert hasattr(stream, "asend") and hasattr(stream, "athrow")
        async for _item in stream:
            assert trace.get_current_span().get_span_context().is_valid is False
        closing = await llm.astream_chat(
            [ChatMessage(role="user", content="close stream")]
        )
        await anext(closing)
        await closing.aclose()
        cancelled = await llm.astream_chat(
            [ChatMessage(role="user", content="cancel stream")]
        )
        await anext(cancelled)
        with pytest.raises(asyncio.CancelledError):
            await cancelled.athrow(asyncio.CancelledError())

    asyncio.run(run())
    assert len(logical(exporter)) == 3
    assert all(
        span.attributes.get(SpanAttributes.GEN_AI_IS_STREAMING)
        for span in logical(exporter)
    )
    assert not owner._span_handler.open_spans
    assert not owner._event_handler._open_event_spans


@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_released_suppression(runtime, key):
    owner, _, exporter = runtime
    token = context.attach(context.set_value(key, True))
    try:
        assert (
            build_fixture_llm()
            .chat([ChatMessage(role="user", content="suppressed")])
            .message.content
        )
    finally:
        context.detach(token)
    assert exporter.get_finished_spans() == ()
    assert not owner._span_handler._suppressed_ids


def test_released_sampler_off(monkeypatch):
    provider = TracerProvider(sampler=ALWAYS_OFF)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    owner = LlamaIndexInstrumentor()
    owner.activate()
    try:
        build_fixture_llm().chat([ChatMessage(role="user", content="not sampled")])
        assert exporter.get_finished_spans() == ()
    finally:
        owner.deactivate()
        provider.shutdown()


def test_released_provider_error_has_no_synthetic_status_output_usage(runtime):
    owner, _, exporter = runtime
    with pytest.raises(Exception) as caught:
        build_fixture_llm(fail=True).chat(
            [ChatMessage(role="user", content="controlled error")]
        )
    assert type(caught.value).__name__ == "APIConnectionError"
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert all(span.status.status_code is StatusCode.ERROR for span in spans)
    for span in spans:
        assert "status_code" not in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
        assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in span.attributes
    assert not owner._event_handler._open_event_spans


def test_shared_owner_settings_and_foreign_listener_preservation(runtime):
    first, _, exporter = runtime
    second = LlamaIndexInstrumentor()
    second.activate()
    with pytest.raises(ValueError, match="different settings"):
        LlamaIndexInstrumentor(capture_content=False).activate()
    assert second._event_handler is first._event_handler
    first.deactivate()
    build_fixture_llm().chat([ChatMessage(role="user", content="remaining owner")])
    assert len(logical(exporter)) == 1
    second.deactivate()
    assert first._event_handler not in root_dispatcher.event_handlers


def test_transaction_partial_registration_rolls_back(monkeypatch):
    owner = LlamaIndexInstrumentor()
    old_span, old_event = (
        list(root_dispatcher.span_handlers),
        list(root_dispatcher.event_handlers),
    )
    original = type(root_dispatcher).add_event_handler

    def fail(self, handler):
        original(self, handler)
        raise RuntimeError("partial registration")

    monkeypatch.setattr(type(root_dispatcher), "add_event_handler", fail)
    with pytest.raises(RuntimeError, match="partial registration"):
        owner.activate()
    assert root_dispatcher.span_handlers == old_span
    assert root_dispatcher.event_handlers == old_event
    assert not owner._owners


def test_function_agent_native_current_tools_and_ids(runtime):
    _, _, exporter = runtime

    def multiply_numbers(a: int, b: int) -> int:
        return a * b

    async def run():
        agent = FunctionAgent(
            tools=[FunctionTool.from_defaults(fn=multiply_numbers)],
            llm=build_fixture_llm(),
            streaming=False,
        )
        return await agent.run(user_msg="Use the multiply_numbers tool.")

    result = asyncio.run(run())
    assert result.response.content
    chats = logical(exporter)
    first = next(
        span
        for span in chats
        if f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls" in span.attributes
    )
    import json

    call = json.loads(
        first.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
    )[0]
    assert call["id"] == "fixture-tool-call-1"
    assert json.loads(call["function"]["arguments"]) == {"a": 7, "b": 6}
    assert (
        json.loads(first.attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0][
            "function"
        ]["name"]
        == "multiply_numbers"
    )
    execution = logical(exporter, "tool")[0]
    assert execution.attributes["gen_ai.tool.call.id"] == call["id"]
    assert (
        sum(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls" in span.attributes
            for span in chats
        )
        == 1
    )


def test_bounded_hostile_serialization_and_complete_vectors():
    class Hostile:
        @property
        def model_dump(self):
            raise AssertionError("hostile property")

        def __str__(self):
            raise AssertionError("hostile string")

    text = safe_json(
        {"api_key": "never-print", "hostile": Hostile(), "content": "😀" * 10000}
    )
    assert "never-print" not in text
    assert len(text.encode()) <= 16000
    import json

    vectors = [float(index) for index in range(5000)]
    assert json.loads(safe_json(vectors, complete=True)) == vectors


def test_usage_no_bool_negative_or_partial_fabrication():
    assert extract_usage(
        {"usage": {"prompt_tokens": True, "completion_tokens": -1}}
    ) == (None, None, None)
    assert extract_usage({"usage": {"input_tokens": 3}}) == (3, None, None)
    assert extract_usage({"usage": {"input_tokens": 0, "output_tokens": 0}}) == (
        0,
        0,
        0,
    )


def test_generic_sparse_numeric_vector_mapping_is_complete():
    import json

    vector = {index: float(index) for index in range(256)}
    assert json.loads(safe_json(vector)) == {
        str(index): float(index) for index in range(256)
    }


def test_released_sparse_sdk_complete_vector(runtime):
    try:
        from llama_index.core.base.embeddings.base_sparse import BaseSparseEmbedding
    except ImportError:
        pytest.skip("Sparse embedding surface unavailable in installed core")
    _, _, exporter = runtime

    class FixtureSparse(BaseSparseEmbedding):
        model_name: str = "fixture-sparse"

        def _get_query_embedding(self, query: str):
            return {index: float(index) for index in range(256)}

        async def _aget_query_embedding(self, query: str):
            return self._get_query_embedding(query)

        def _get_text_embedding(self, text: str):
            return self._get_query_embedding(text)

        async def _aget_text_embedding(self, text: str):
            return self._get_query_embedding(text)

    result = FixtureSparse().get_text_embedding("sparse controlled input")
    import json

    span = logical(exporter, "embedding")[0]
    stored = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[0]
    assert stored == {str(key): value for key, value in result.items()}
    assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in span.attributes
    native_outputs = [
        json.loads(item.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
        for item in exporter.get_finished_spans()
        if item.attributes.get(RESPAN_LOG_TYPE) != "embedding"
        and SpanAttributes.TRACELOOP_ENTITY_OUTPUT in item.attributes
    ]
    assert native_outputs
    assert all(item == stored for item in native_outputs)


def test_shared_provider_hook_foreign_owner_and_inert_retained_hook(
    runtime, monkeypatch
):
    from openai.resources.embeddings import Embeddings

    owner, _, exporter = runtime
    wrapper = Embeddings.create
    original = next(
        original for cls, _, original, _ in owner._patches if cls is Embeddings
    )

    def foreign(*args, **kwargs):
        return wrapper(*args, **kwargs)

    monkeypatch.setattr(Embeddings, "create", foreign)
    owner.deactivate()
    assert Embeddings.create is foreign
    # Retained foreign wrapper calls the original SDK with instrumentation inert.
    build_fixture_embedding().get_text_embedding("inert fixture")
    assert not logical(exporter, "embedding")
    monkeypatch.setattr(Embeddings, "create", original)


def test_start_disabled_event_does_not_inspect_completion_payload(runtime, monkeypatch):
    owner, _, exporter = runtime
    from llama_index.core.instrumentation.events.llm import LLMChatStartEvent

    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    event = LLMChatStartEvent(
        messages=[ChatMessage(role="user", content="private")],
        additional_kwargs={},
        model_dict={},
        span_id="private-event",
    )
    owner._event_handler.handle(event)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")

    class HostileResponse:
        @property
        def message(self):
            raise AssertionError("must not inspect opted-out message")

    owner._event_handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatEndEvent",
            span_id="private-event",
            response=HostileResponse(),
        )
    )
    span = logical(exporter)[0]
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes


def test_native_tool_arguments_and_long_output_complete(runtime):
    import json

    _, _, exporter = runtime
    arguments = {f"field-{index}": index for index in range(120)}
    output = "controlled-result " * 2000

    def execute(**kwargs):
        assert kwargs == arguments
        return output

    result = FunctionTool.from_defaults(fn=execute, name="large-tool").call(**arguments)
    assert result.raw_output == output
    span = logical(exporter, "tool")[0]
    assert (
        json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])["arguments"]
        == arguments
    )
    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == output


def test_observed_veto_remains_bound_for_reenabled_and_ended_parent(
    runtime, monkeypatch
):
    owner, _, exporter = runtime
    handler = owner._span_handler
    args = SimpleNamespace(args=("private-frame",), kwargs={})
    handler.span_enter(id_="parent", bound_args=args)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    handler.span_enter(id_="child-first", bound_args=args, parent_id="parent")
    handler.span_exit(id_="child-first", bound_args=args, result="private-first")
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    handler.span_enter(id_="child-second", bound_args=args, parent_id="parent")
    handler.span_exit(id_="child-second", bound_args=args, result="private-second")
    handler.span_exit(id_="parent", bound_args=args, result="private-parent")
    handler.span_enter(id_="child-late", bound_args=args, parent_id="parent")
    handler.span_exit(id_="child-late", bound_args=args, result="private-late")
    for span in exporter.get_finished_spans():
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes


def test_unknown_completion_does_not_invoke_str_or_invent_output(runtime):
    owner, _, exporter = runtime

    class Unknown:
        def __str__(self):
            raise AssertionError("unknown output string hook must not run")

    handler = owner._event_handler
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMCompletionStartEvent",
            span_id="unknown-event",
            prompt="controlled",
            model_dict={},
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMCompletionEndEvent",
            span_id="unknown-event",
            response=Unknown(),
        )
    )
    span = logical(exporter, "text")[0]
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes


def test_event_veto_before_reenable_removes_started_llm_payload(runtime, monkeypatch):
    owner, _, exporter = runtime
    span_handler, event_handler = owner._span_handler, owner._event_handler
    args = SimpleNamespace(args=("private generic input",), kwargs={})
    span_handler.span_enter(id_="veto-event", bound_args=args)
    event_handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatStartEvent",
            span_id="veto-event",
            messages=[ChatMessage(role="user", content="private input")],
            model_dict={},
        )
    )
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    event_handler.handle(
        SimpleNamespace(class_name=lambda: "ObservedPrivacyEvent", span_id="veto-event")
    )
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    response = SimpleNamespace(
        message=ChatMessage(role="assistant", content="private output"), raw={}
    )
    event_handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatEndEvent",
            span_id="veto-event",
            response=response,
        )
    )
    span_handler.span_exit(
        id_="veto-event", bound_args=args, result="private generic output"
    )
    for span in exporter.get_finished_spans():
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
