"""Released OITracer and OpenAI delegate contracts; transport is controlled."""

from __future__ import annotations

import inspect
import json

import httpx
import openai
import pytest
from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import context, trace
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_instrumentation_openinference import (
    OpenInferenceInstrumentor,
    OpenInferenceTranslator,
)
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


@pytest.fixture
def pipeline():
    provider = TracerProvider()
    translator = OpenInferenceTranslator()
    sink = InMemorySpanExporter()
    provider.add_span_processor(translator)
    provider.add_span_processor(SimpleSpanProcessor(sink))
    yield provider, translator, sink
    provider.shutdown()


def oi(provider, **kwargs):
    return OITracer(
        provider.get_tracer("released-oi-sdk"), config=TraceConfig(**kwargs)
    )


def test_native_decorators_preserve_values_errors_and_hierarchy(pipeline):
    provider, _, sink = pipeline
    tracer = oi(provider)
    value = {"answer": "controlled"}
    failure = RuntimeError("controlled native error")

    @tracer.tool(name="lookup")
    def lookup() -> dict:
        return value

    @tracer.chain
    def work():
        assert lookup() is value
        raise failure

    with pytest.raises(RuntimeError) as caught:
        work()
    assert caught.value is failure
    tool, root = sink.get_finished_spans()
    assert tool.parent.span_id == root.context.span_id
    assert json.loads(tool.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == value
    assert (
        json.loads(tool.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])["name"]
        == "lookup"
    )
    assert root.status.status_code is StatusCode.ERROR
    assert "status_code" not in root.attributes
    assert not trace.get_current_span().get_span_context().is_valid


@pytest.mark.asyncio
async def test_native_async_decorator_keeps_context_and_result(pipeline):
    provider, _, sink = pipeline
    tracer = oi(provider)
    value = {"answer": 42}

    @tracer.chain
    async def work():
        return value

    assert await work() is value
    assert len(sink.get_finished_spans()) == 1
    assert not trace.get_current_span().get_span_context().is_valid


@pytest.mark.parametrize("policy", ["environment", "context", "final_veto"])
def test_private_start_cannot_be_reenabled_and_final_veto_removes_content(
    pipeline, monkeypatch, policy
):
    provider, _, sink = pipeline
    tracer = oi(provider)
    token = None
    if policy == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    if policy == "context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        span = tracer.start_span("private", openinference_span_kind="llm")
        if token is not None:
            context.detach(token)
            token = None
        monkeypatch.setenv(
            "TRACELOOP_TRACE_CONTENT", "false" if policy == "final_veto" else "true"
        )
        span.set_attribute("input.value", "private-input-sentinel")
        span.set_attribute("output.value", "private-output-sentinel")
        span.set_attribute("llm.tools", '[{"name":"private-schema-sentinel"}]')
        span.set_attribute("llm.token_count.prompt", 11)
        span.end()
    finally:
        if token is not None:
            context.detach(token)
    attrs = dict(sink.get_finished_spans()[0].attributes)
    assert "private-" not in json.dumps(attrs)
    assert attrs["gen_ai.usage.input_tokens"] == 11


def test_finished_private_parent_bounds_late_child(pipeline):
    provider, _, sink = pipeline
    tracer = oi(provider)
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    parent = tracer.start_span("private-root", openinference_span_kind="chain")
    parent_context = trace.set_span_in_context(parent)
    context.detach(token)
    parent.end()
    with tracer.start_as_current_span(
        "child", context=parent_context, openinference_span_kind="llm"
    ) as span:
        span.set_attribute("input.value", "late-child-sentinel")
    assert "late-child-sentinel" not in json.dumps(
        dict(sink.get_finished_spans()[-1].attributes)
    )


@pytest.mark.parametrize("mode", ["sampler", "suppression"])
def test_native_oi_obeys_sampling_and_suppression(mode):
    provider = (
        TracerProvider(sampler=ALWAYS_OFF) if mode == "sampler" else TracerProvider()
    )
    translator = OpenInferenceTranslator()
    sink = InMemorySpanExporter()
    provider.add_span_processor(translator)
    provider.add_span_processor(SimpleSpanProcessor(sink))
    token = (
        context.attach(context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True))
        if mode == "suppression"
        else None
    )
    try:

        @oi(provider).chain
        def work():
            return "native result"

        assert work() == "native result"
        assert not sink.get_finished_spans()
        assert not translator._policy._active
    finally:
        if token is not None:
            context.detach(token)
        provider.shutdown()


def test_complete_vector_tool_schema_arguments_results_and_ids(pipeline):
    provider, _, sink = pipeline
    tracer = oi(provider)
    vector = [float(i) / 10000 for i in range(5000)]
    schema = {
        "name": "lookup",
        "parameters": {
            "properties": {f"p{i}": {"type": "string"} for i in range(1200)}
        },
    }
    arguments = {"values": vector, "text": "a" * 35000}
    call_id = "call-" + "x" * 700
    with tracer.start_as_current_span("root", openinference_span_kind="llm") as root:
        root.set_attribute("llm.tools", json.dumps([schema]))
        root.set_attribute(
            "llm.input_messages.0.message.tool_calls.0.tool_call.id", "historical-only"
        )
        root.set_attribute(
            "llm.output_messages.0.message.tool_calls.0.tool_call.id", call_id
        )
        root.set_attribute(
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name",
            "lookup",
        )
        root.set_attribute(
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments",
            json.dumps(arguments),
        )
        with tracer.start_as_current_span(
            "lookup", openinference_span_kind="tool"
        ) as tool:
            tool.set_attribute("tool.name", "lookup")
            tool.set_attribute("tool.id", call_id)
            tool.set_attribute("input.value", json.dumps(arguments))
            tool.set_attribute("output.value", json.dumps({"vector": vector}))
    tool, root = sink.get_finished_spans()
    attrs = root.attributes
    assert json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == [schema]
    calls = json.loads(attrs["gen_ai.completion.0.tool_calls"])
    assert calls[0]["id"] == call_id
    assert json.loads(calls[0]["function"]["arguments"]) == arguments
    assert "historical-only" not in attrs["gen_ai.completion.0.tool_calls"]
    assert tool.attributes["gen_ai.tool.call.id"] == call_id
    assert json.loads(tool.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "vector": vector
    }
    assert (
        json.loads(tool.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])["arguments"]
        == arguments
    )
    with tracer.start_as_current_span(
        "embedding", openinference_span_kind="embedding"
    ) as span:
        span.set_attribute("embedding.embeddings.0.embedding.vector", vector)
    assert (
        json.loads(
            sink.get_finished_spans()[-1].attributes[
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT
            ]
        )
        == vector
    )


def test_modern_multimodal_and_actual_usage_fields(pipeline):
    provider, _, sink = pipeline
    tracer = oi(provider)
    with tracer.start_as_current_span("modern", openinference_span_kind="llm") as span:
        span.set_attribute("llm.request.model_name", "requested-model")
        span.set_attribute("llm.response.model_name", "returned-model")
        span.set_attribute(
            "llm.input_messages.0.message.contents.0.message_content.type", "text"
        )
        span.set_attribute(
            "llm.input_messages.0.message.contents.0.message_content.text", "caption"
        )
        span.set_attribute(
            "llm.input_messages.0.message.contents.1.message_content.type", "image"
        )
        span.set_attribute(
            "llm.input_messages.0.message.contents.1.message_content.image.image.url",
            "https://fixture.invalid/image.png",
        )
        span.set_attribute("llm.token_count.prompt", 0)
        span.set_attribute("llm.token_count.completion", 0)
        span.set_attribute("llm.token_count.prompt_details.cache_read", 0)
        span.set_attribute("llm.token_count.prompt_details.cache_write", 3)
        span.set_attribute("llm.token_count.completion_details.reasoning", 2)
    attrs = sink.get_finished_spans()[0].attributes
    assert json.loads(attrs["gen_ai.prompt.0.content"]) == [
        {"type": "text", "text": "caption"},
        {
            "type": "image",
            "image": {"image": {"url": "https://fixture.invalid/image.png"}},
        },
    ]
    assert attrs["gen_ai.usage.input_tokens"] == 0
    assert attrs["gen_ai.usage.output_tokens"] == 0
    assert attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 0
    assert attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 3
    assert attrs[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 2
    from openinference.semconv.trace import SpanAttributes as OI

    if hasattr(OI, "LLM_REQUEST_MODEL_NAME"):
        assert attrs["gen_ai.request.model"] == "requested-model"
        assert attrs["gen_ai.response.model"] == "returned-model"


@pytest.mark.parametrize("invalid", [True, -1, 1.25, "11"])
def test_invalid_usage_is_not_published(pipeline, invalid):
    provider, _, sink = pipeline
    with oi(provider).start_as_current_span(
        "invalid", openinference_span_kind="llm"
    ) as span:
        span.set_attribute("llm.token_count.prompt", invalid)
        span.set_attribute("llm.token_count.completion", invalid)
        span.set_attribute("llm.token_count.total", invalid)
    assert not any("tokens" in k for k in sink.get_finished_spans()[0].attributes)


def test_current_native_genai_projection_respects_privacy(pipeline):
    if "enable_genai_semconv" not in inspect.signature(TraceConfig).parameters:
        pytest.skip("GenAI projection absent in OpenInference0.1.32")
    provider, _, sink = pipeline
    tracer = oi(provider, enable_genai_semconv=True)
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        with tracer.start_as_current_span(
            "private", openinference_span_kind="llm"
        ) as span:
            span.set_attribute("llm.input_messages.0.message.role", "user")
            span.set_attribute(
                "llm.input_messages.0.message.content", "genai-private-sentinel"
            )
    finally:
        context.detach(token)
    assert "genai-private-sentinel" not in json.dumps(
        dict(sink.get_finished_spans()[0].attributes)
    )


@pytest.mark.parametrize("kind", ["retriever", "reranker", "guardrail", "evaluator"])
def test_new_native_decorators_map_to_contract(pipeline, kind):
    provider, _, sink = pipeline
    tracer = oi(provider)
    if not hasattr(tracer, kind):
        pytest.skip(f"{kind} decorator absent in OpenInference0.1.32")
    result = {"fixture": kind}

    def work():
        return result

    wrapped = getattr(tracer, kind)(work)
    assert wrapped() is result
    attrs = sink.get_finished_spans()[0].attributes
    assert attrs["respan.entity.log_type"] == (
        "guardrail" if kind == "guardrail" else "task"
    )
    assert "traceloop.span.kind" not in attrs


@pytest.mark.parametrize("stream", [False, True])
def test_released_openai_delegate_preserves_response_or_chunks_and_actual_usage(
    pipeline, monkeypatch, stream
):
    provider, _, sink = pipeline
    monkeypatch.setattr(
        "respan_instrumentation_openinference._instrumentation.trace.get_tracer_provider",
        lambda: provider,
    )
    wrapper = OpenInferenceInstrumentor(OpenAIInstrumentor)
    wrapper.activate()
    payload = {
        "id": "fixture-response",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "fixture answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }

    def handler(request):
        if stream:
            chunk = {
                "id": "fixture-response",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "fixture answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": payload["usage"],
            }
            return httpx.Response(
                200,
                text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=payload)

    client = openai.OpenAI(
        api_key="fixture-only",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    try:
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "fixture prompt"}],
            stream=stream,
        )
        if stream:
            assert isinstance(result, openai.Stream)
            chunks = list(result)
            assert chunks[0].choices[0].delta.content == "fixture answer"
            result.close()
        else:
            assert isinstance(result, openai.types.chat.ChatCompletion)
            assert result.choices[0].message.content == "fixture answer"
        span = sink.get_finished_spans()[0]
        assert span.attributes["gen_ai.usage.input_tokens"] == 11
        assert span.attributes["gen_ai.usage.output_tokens"] == 7
        assert "fixture answer" in span.attributes["traceloop.entity.output"]
        assert not trace.get_current_span().get_span_context().is_valid
    finally:
        client.close()
        wrapper.deactivate()


def test_serializer_fault_cannot_replace_application_result(pipeline, monkeypatch):
    provider, _, sink = pipeline
    import respan_instrumentation_openinference._translator as module

    def fail(*args, **kwargs):
        raise RuntimeError("controlled serializer failure")

    monkeypatch.setattr(module, "bounded_json", fail)
    value = {"native": 42}

    @oi(provider).chain
    def work():
        return value

    assert work() is value
    assert len(sink.get_finished_spans()) == 1
    assert "input.value" not in sink.get_finished_spans()[0].attributes


def test_processor_constructor_options_and_foreign_order_are_retained(
    pipeline, monkeypatch
):
    provider, _, _ = pipeline
    original = provider._active_span_processor._span_processors
    monkeypatch.setattr(
        "respan_instrumentation_openinference._instrumentation.trace.get_tracer_provider",
        lambda: provider,
    )

    class Source(SpanProcessor):
        def __init__(self, flag=True):
            self.flag = flag

        def on_start(self, span, parent_context=None):
            pass

        def on_end(self, span):
            pass

        def shutdown(self):
            pass

        def force_flush(self, timeout_millis=30000):
            return True

    wrapper = OpenInferenceInstrumentor(Source, flag=False)
    wrapper.activate()
    assert wrapper._instrumentor.flag is False
    foreign = Source()
    provider.add_span_processor(foreign)
    wrapper.deactivate()
    assert provider._active_span_processor._span_processors == (*original, foreign)


def test_private_exception_diagnostics_do_not_export_payload(pipeline):
    provider, _, sink = pipeline
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:

        @oi(provider).chain
        def fail():
            raise RuntimeError("private-exception-sentinel")

        with pytest.raises(RuntimeError, match="private-exception-sentinel"):
            fail()
    finally:
        context.detach(token)
    span = sink.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description is None
    assert "private-exception-sentinel" not in json.dumps(
        [dict(e.attributes) for e in span.events]
    )


def test_credential_shapes_are_redacted_without_losing_complete_tool_data(pipeline):
    provider, _, sink = pipeline
    with oi(provider).start_as_current_span(
        "lookup", openinference_span_kind="tool"
    ) as span:
        span.set_attribute(
            "input.value",
            json.dumps(
                {
                    "text": 'password="a secret with spaces" https://user:pass@fixture.invalid/path',
                    "values": list(range(5000)),
                }
            ),
        )
        span.set_attribute("output.value", "Bearer abcdefgh12345678")
    attrs = dict(sink.get_finished_spans()[0].attributes)
    value = json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert value["arguments"]["values"] == list(range(5000))
    assert "a secret with spaces" not in json.dumps(attrs)
    assert "user:pass" not in json.dumps(attrs)
    assert "abcdefgh12345678" not in json.dumps(attrs)


def test_vectors_in_message_and_numeric_sparse_result_are_complete(pipeline):
    provider, _, sink = pipeline
    dense = list(range(5000))
    sparse = {str(i): i / 10000 for i in range(5000)}
    with oi(provider).start_as_current_span(
        "chat", openinference_span_kind="llm"
    ) as span:
        span.set_attribute("llm.input_messages.0.message.role", "tool")
        span.set_attribute(
            "llm.input_messages.0.message.content", json.dumps({"vector": dense})
        )
    attrs = sink.get_finished_spans()[0].attributes
    assert json.loads(attrs["gen_ai.prompt.0.content"])["vector"] == dense
    assert (
        json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT])["messages"][0][
            "content"
        ]["vector"]
        == dense
    )
    with oi(provider).start_as_current_span(
        "embedding", openinference_span_kind="embedding"
    ) as span:
        span.set_attribute("output.value", json.dumps(sparse))
    assert (
        json.loads(
            sink.get_finished_spans()[-1].attributes[
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT
            ]
        )
        == sparse
    )


@pytest.mark.parametrize("private", [False, True])
def test_span_level_diagnostics_are_redacted_and_private_content_is_removed(
    pipeline, private
):
    provider, _, sink = pipeline
    token = (
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        if private
        else None
    )
    try:
        with oi(provider).start_as_current_span(
            "diagnostic", openinference_span_kind="llm"
        ) as span:
            span.set_attribute("error.message", "token=controlled-secret")
            span.set_attribute("exception.message", "password=controlled-secret")
            span.set_attribute(
                "exception.stacktrace", "Bearer controlled-secret-123456"
            )
            span.set_attribute("error.type", "NativeFixtureError")
    finally:
        if token is not None:
            context.detach(token)
    attrs = dict(sink.get_finished_spans()[0].attributes)
    assert attrs["error.type"] == "NativeFixtureError"
    assert "controlled-secret" not in json.dumps(attrs)
    if private:
        assert all(
            key not in attrs
            for key in ("error.message", "exception.message", "exception.stacktrace")
        )


def test_late_activation_does_not_assume_unknown_start_was_public(monkeypatch):
    provider = TracerProvider()
    sink = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(sink))
    tracer = oi(provider)
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    span = tracer.start_span("started-before-activation", openinference_span_kind="llm")
    context.detach(token)
    span.set_attribute("input.value", "unknown-start-private-sentinel")
    monkeypatch.setattr(
        "respan_instrumentation_openinference._instrumentation.trace.get_tracer_provider",
        lambda: provider,
    )

    class Source(SpanProcessor):
        pass

    wrapper = OpenInferenceInstrumentor(Source)
    wrapper.activate()
    try:
        span.end()
        assert "unknown-start-private-sentinel" not in json.dumps(
            dict(sink.get_finished_spans()[0].attributes)
        )
    finally:
        wrapper.deactivate()
        provider.shutdown()


def test_native_callback_end_veto_observed_before_sdk_context_detach(monkeypatch):
    provider = TracerProvider()
    sink = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(sink))
    monkeypatch.setattr(
        "respan_instrumentation_openinference._instrumentation.trace.get_tracer_provider",
        lambda: provider,
    )
    original = OITracer.start_as_current_span

    class Source(SpanProcessor):
        pass

    first = OpenInferenceInstrumentor(Source)
    second = OpenInferenceInstrumentor(Source)
    first.activate()
    second.activate()
    value = "callback-private-sentinel"

    @oi(provider).chain
    def work():
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        return value

    try:
        assert work() is value
        assert "callback-private-sentinel" not in json.dumps(
            dict(sink.get_finished_spans()[0].attributes)
        )
        assert context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
        first.deactivate()
        assert OITracer.start_as_current_span is not original
    finally:
        second.deactivate()
        provider.shutdown()
    assert OITracer.start_as_current_span is original


def test_foreign_scope_replacement_is_not_overwritten_on_teardown(monkeypatch):
    provider = TracerProvider()
    monkeypatch.setattr(
        "respan_instrumentation_openinference._instrumentation.trace.get_tracer_provider",
        lambda: provider,
    )

    class Source(SpanProcessor):
        pass

    wrapper = OpenInferenceInstrumentor(Source)
    original = OITracer.start_as_current_span
    wrapper.activate()

    def foreign(*args, **kwargs):
        return original(*args, **kwargs)

    OITracer.start_as_current_span = foreign
    try:
        wrapper.deactivate()
        assert OITracer.start_as_current_span is foreign
    finally:
        OITracer.start_as_current_span = original
        provider.shutdown()


@pytest.mark.parametrize("kind", ["agent", "tool", "chain"])
def test_native_non_model_spans_do_not_promote_inherited_llm_fields(pipeline, kind):
    provider, _, sink = pipeline
    with oi(provider).start_as_current_span(
        "native-work", openinference_span_kind=kind
    ) as span:
        span.set_attribute("llm.model_name", "parent-model-hint")
        span.set_attribute("llm.provider", "openai")
        span.set_attribute("llm.token_count.prompt", 11)
        span.set_attribute("gen_ai.request.model", "parent-model-hint")
        span.set_attribute("gen_ai.usage.input_tokens", 11)
        span.set_attribute("input.value", '{"native":"input"}')
        span.set_attribute("output.value", '{"native":"output"}')
    attrs = dict(sink.get_finished_spans()[0].attributes)
    assert "gen_ai.request.model" not in attrs
    assert "gen_ai.system" not in attrs
    assert "gen_ai.provider.name" not in attrs
    assert not any(k.startswith(("gen_ai.usage.", "llm.usage.")) for k in attrs)
    assert "native" in attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
