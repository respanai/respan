"""Released native Braintrust lifecycle, export records and privacy contracts."""

import asyncio
import inspect
import json
from functools import wraps

import braintrust
import pytest
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_braintrust import BraintrustInstrumentor
from respan_instrumentation_braintrust import _instrumentation as module
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


class Sink:
    def __init__(self):
        self.rows = []
        self.resolved = []

    def log(self, *rows):
        self.rows.extend(rows)

    def flush(self, *args, **kwargs):
        rows, self.rows = self.rows, []
        for row in rows:
            self.resolved.append(row.get())

    def enforce_queue_size_limit(self, enforce):
        pass

    def set_masking_function(self, function):
        pass


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
    braintrust._internal_reset_global_state()
    state = braintrust._internal_get_global_state()
    sink = Sink()
    state._override_bg_logger.logger = sink
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    plugin = BraintrustInstrumentor(tracer_provider=provider)
    plugin.activate()
    logger = braintrust.init_logger(
        project="fixture",
        project_id="fixture-project",
        api_key=braintrust.logger.TEST_API_KEY,
        async_flush=False,
        set_current=True,
    )
    yield provider, exporter, plugin, logger, sink
    plugin.deactivate()
    if module._RUNTIME:
        module._RUNTIME.close()
        module._RUNTIME = None
    provider.shutdown()
    braintrust._internal_reset_global_state()


def records(exporter, kind=None):
    return [
        s
        for s in exporter.get_finished_spans()
        if kind is None or s.attributes.get(RESPAN_LOG_TYPE) == kind
    ]


def test_native_rows_topology_usage_source_and_sink_preserved(runtime):
    provider, exporter, _, logger, sink = runtime
    with provider.get_tracer("caller").start_as_current_span("caller") as caller:
        with (
            logger.start_span(name="evaluation", type="eval") as root,
            root.start_span(name="generation", type="llm") as span,
        ):
            span.log(
                input=[{"role": "user", "content": "question"}],
                output={"role": "assistant", "content": "answer"},
                metadata={"model": "fixture", "provider": "openai"},
                metrics={
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                    "cache_read_tokens": 3,
                    "reasoning_tokens": 2,
                },
            )
            assert trace.get_current_span() is not caller
        assert trace.get_current_span() is caller
        logger.flush()
    assert len(sink.resolved) == 5
    llm = records(exporter, "chat")[0]
    workflow = records(exporter, "workflow")[0]
    assert (
        llm.parent.span_id == workflow.context.span_id
        and workflow.parent.span_id == caller.context.span_id
    )
    assert llm.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 18
    assert llm.attributes[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 2
    assert llm.attributes[SpanAttributes.LLM_SYSTEM] == "openai"
    assert not module._RUNTIME.states
    assert all(
        SpanAttributes.TRACELOOP_SPAN_KIND not in s.attributes
        for s in records(exporter)
    )


@pytest.mark.parametrize("mode", ["environment", "context", "option"])
def test_initial_privacy_bounds_native_payload_capture(runtime, monkeypatch, mode):
    provider, exporter, plugin, logger, sink = runtime
    token = None
    if mode == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    elif mode == "context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    else:
        plugin.deactivate()
        plugin = BraintrustInstrumentor(tracer_provider=provider, include_content=False)
        plugin.activate()
    try:
        with logger.start_span(name="private", type="llm") as span:
            span.log(
                input="private prompt",
                output="private output",
                metadata={"secret": "private secret", "model": "fixture"},
                metrics={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        logger.flush()
    finally:
        if token is not None:
            context.detach(token)
        if mode == "option":
            plugin.deactivate()
    attrs = records(exporter, "chat")[0].attributes
    assert "private" not in str(
        {k: v for k, v in attrs.items() if k != "traceloop.entity.name"}
    )
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 0
    assert len(sink.resolved) == 3


def test_late_privacy_veto_and_no_reenable_before_lazy_flush(runtime, monkeypatch):
    _, exporter, _, logger, _ = runtime
    with logger.start_span(name="veto", type="llm") as span:
        span.log(
            input="private input",
            output="private output",
            metadata={"model": "fixture"},
        )
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    logger.flush()
    attrs = records(exporter, "chat")[0].attributes
    assert (
        SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
        and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    )


@pytest.mark.parametrize(
    "key",
    [context._SUPPRESS_INSTRUMENTATION_KEY, "suppress_language_model_instrumentation"],
)
def test_suppressed_native_sdk_still_forwards_native_rows(runtime, key):
    _, exporter, _, logger, sink = runtime
    token = context.attach(context.set_value(key, True))
    try:
        with logger.start_span(name="suppressed", type="tool") as span:
            span.log(input={"value": 1}, output={"value": 2})
        logger.flush()
    finally:
        context.detach(token)
    assert not records(exporter) and len(sink.resolved) == 3


def test_sampled_off_never_serializes_but_native_sink_works(runtime, monkeypatch):
    _, exporter, plugin, logger, sink = runtime
    plugin.deactivate()
    provider = TracerProvider(sampler=ALWAYS_OFF)
    plugin = BraintrustInstrumentor(tracer_provider=provider)
    plugin.activate()

    def forbidden(*args):
        raise AssertionError("telemetry serialized")

    monkeypatch.setattr(module, "plain", forbidden)
    with logger.start_span(name="off", type="llm") as span:
        span.log(input="actual", output="actual")
    logger.flush()
    assert not records(exporter) and len(sink.resolved) == 3
    assert not module._RUNTIME.states
    plugin.deactivate()
    provider.shutdown()


def test_shared_owners_native_sink_foreign_hook_and_inert_retained_wrappers(
    runtime, monkeypatch
):
    provider, exporter, first, logger, _ = runtime
    original = braintrust.logger.SpanImpl.end
    second = BraintrustInstrumentor(tracer_provider=provider)
    second.activate()
    with logger.start_span(name="first", type="tool"):
        pass
    first.deactivate()
    with logger.start_span(name="second", type="tool"):
        pass
    logger.flush()
    assert len(records(exporter, "tool")) == 2

    @wraps(original)
    def foreign(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(braintrust.logger.SpanImpl, "end", foreign)
    second.deactivate()
    assert braintrust.logger.SpanImpl.end is foreign
    with logger.start_span(name="inert", type="tool"):
        pass
    logger.flush()
    assert len(records(exporter, "tool")) == 2
    third = BraintrustInstrumentor(tracer_provider=provider)
    third.activate()
    with logger.start_span(name="reactivated", type="tool"):
        pass
    logger.flush()
    assert len(records(exporter, "tool")) == 3
    third.deactivate()
    assert braintrust.logger.SpanImpl.end is foreign


def test_partial_install_rollback(runtime, monkeypatch):
    _, _, plugin, _, _ = runtime
    plugin.deactivate()
    original = braintrust.logger.SpanImpl.__init__
    patch = module._Runtime.patch

    def fail(self, owner, name, factory):
        if name == "log_internal":
            raise ValueError("controlled install failure")
        return patch(self, owner, name, factory)

    monkeypatch.setattr(module._Runtime, "patch", fail)
    with pytest.raises(ValueError):
        BraintrustInstrumentor().activate()
    assert braintrust.logger.SpanImpl.__init__ is original and module._RUNTIME is None


def test_actual_error_identity_no_invented_http_or_completion(runtime):
    _, exporter, _, logger, _ = runtime
    error = ValueError("controlled failure")
    with (
        pytest.raises(ValueError) as caught,
        logger.start_span(name="failure", type="llm") as span,
    ):
        span.log(input="question", metadata={"model": "fixture"})
        raise error
    assert caught.value is error
    logger.flush()
    span = records(exporter, "chat")[0]
    assert span.status.status_code is trace.StatusCode.ERROR
    assert not any(
        k.startswith(SpanAttributes.LLM_COMPLETIONS) for k in span.attributes
    )
    assert "http.response.status_code" not in span.attributes


def test_complete_tool_calls_definitions_results_and_vectors(runtime):
    _, exporter, _, logger, _ = runtime
    call = {
        "id": "source-call",
        "type": "function",
        "function": {"name": "lookup", "arguments": {"value": 2}},
    }
    with logger.start_span(name="model", type="llm") as model:
        model.log(
            input=[
                {"role": "assistant", "tool_calls": [dict(call, id="past-call")]},
                {"role": "tool", "content": "past", "tool_call_id": "past-call"},
            ],
            output={"role": "assistant", "tool_calls": [call]},
            metadata={
                "model": "fixture",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    f"field_{i}": {"type": "string"} for i in range(100)
                                },
                            },
                        },
                    }
                ],
            },
        )
        with model.start_span(name="lookup", type="tool") as tool:
            tool.log(
                input={"value": 2},
                output={"vector": list(range(5000))},
                metadata={"tool_call_id": "source-call"},
            )
    with logger.start_span(name="embedding", type="embedding") as span:
        span.log(
            input=["first"],
            output=[list(range(5000))],
            metadata={"model": "fixture-embed"},
        )
    logger.flush()
    llm = records(exporter, "chat")[0].attributes
    tool = records(exporter, "tool")[0].attributes
    embedding = records(exporter, "embedding")[0].attributes
    current = json.loads(llm[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert current[0]["id"] == "source-call" and json.loads(
        current[0]["function"]["arguments"]
    ) == {"value": 2}
    assert llm[f"{SpanAttributes.LLM_PROMPTS}.1.tool_call_id"] == "past-call"
    assert (
        len(
            json.loads(llm[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0]["function"][
                "parameters"
            ]["properties"]
        )
        == 100
    )
    assert (
        tool["gen_ai.tool.call.id"] == "source-call"
        and len(json.loads(tool[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])["vector"])
        == 5000
    )
    assert len(json.loads(embedding[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[0]) == 5000
    assert not any("usage." in k for k in embedding)


def test_unknown_and_invalid_usage_stays_absent(runtime):
    _, exporter, _, logger, _ = runtime
    with logger.start_span(name="unknown", type="llm") as span:
        span.log(
            output="answer",
            metrics={
                "prompt_tokens": -1,
                "completion_tokens": True,
                "total_tokens": -1,
            },
        )
    logger.flush()
    assert not any("usage." in k for k in records(exporter, "chat")[0].attributes)


def test_sdk_masking_and_current_native_customizers_are_preserved(runtime):
    if not hasattr(braintrust, "set_span_customizers"):
        pytest.skip("Native export customizers introduced after declared minimum")
    _, exporter, _, logger, _ = runtime

    class Customizer(braintrust.SpanCustomizer):
        def on_span_export(self, record):
            if "output" in record:
                record["output"] = {"changed": "actual customizer"}
            return record

    braintrust.set_span_customizers([Customizer()])
    try:
        with logger.start_span(name="custom", type="tool") as span:
            span.log(output={"original": "value"})
        logger.flush()
    finally:
        braintrust.set_span_customizers(None)
    assert json.loads(
        records(exporter, "tool")[0].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    ) == {"changed": "actual customizer"}


def test_native_traced_sync_and_async_results(runtime):
    _, exporter, _, logger, _ = runtime

    @braintrust.traced(name="sync")
    def sync(value):
        return {"value": value}

    @braintrust.traced(name="async")
    async def asynchronous(value):
        return {"value": value}

    assert sync(3) == {"value": 3}
    assert asyncio.run(asynchronous(4)) == {"value": 4}
    logger.flush()
    assert len(records(exporter, "tool")) == 2


def test_native_traced_generators_detach_otel_context_each_advance(runtime):
    provider, exporter, _, logger, _ = runtime

    @braintrust.traced(name="generator")
    def generate():
        yield "first"
        yield "second"

    with provider.get_tracer("caller").start_as_current_span("caller") as caller:
        assert inspect.isgeneratorfunction(generate)
        iterator = generate()
        assert next(iterator) == "first"
        assert trace.get_current_span() is caller
        assert iterator.send(None) == "second"
        assert trace.get_current_span() is caller
        iterator.close()
        assert trace.get_current_span() is caller
    logger.flush()
    assert len(records(exporter, "tool")) == 1


@pytest.mark.asyncio
async def test_native_async_traced_generator_detaches(runtime):
    provider, exporter, _, logger, _ = runtime

    @braintrust.traced(name="async-generator")
    async def generate():
        yield "first"
        yield "second"

    with provider.get_tracer("caller").start_as_current_span("caller") as caller:
        assert inspect.isasyncgenfunction(generate)
        iterator = generate()
        assert await iterator.__anext__() == "first"
        assert trace.get_current_span() is caller
        assert await iterator.asend(None) == "second"
        assert trace.get_current_span() is caller
        await iterator.aclose()
        assert trace.get_current_span() is caller
    logger.flush()
    assert len(records(exporter, "tool")) == 1


def test_final_owner_release_inside_span_restores_caller_context(runtime):
    provider, exporter, plugin, logger, _ = runtime
    with provider.get_tracer("caller").start_as_current_span("caller") as caller:
        with logger.start_span(name="pending", type="tool") as span:
            span.log(input="private content")
            plugin.deactivate()
        assert trace.get_current_span() is caller
    logger.flush()
    assert len(records(exporter, "tool")) == 1


def test_actual_braintrust_openai_wrapper_current_calls_and_source_usage(runtime):
    from _fixtures import client_context

    _, exporter, _, logger, _ = runtime
    with client_context() as client:
        result = client.chat.completions.create(
            model="fixture-model",
            messages=[{"role": "user", "content": "fixture"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ],
        )
        assert result.choices[0].message.tool_calls[0].id == "current-source-id"
    logger.flush()
    attrs = records(exporter, "chat")[0].attributes
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 18
    assert attrs[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 2
    assert attrs[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert (
        json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])[0]["id"]
        == "current-source-id"
    )


def test_actual_braintrust_openai_embedding_preserves_complete_source_vector(runtime):
    from _fixtures import client_context

    _, exporter, _, logger, _ = runtime
    with client_context() as client:
        result = client.embeddings.create(
            model="fixture-embed", input=["fixture"], encoding_format="float"
        )
    assert len(result.data[0].embedding) == 5000
    logger.flush()
    attrs = records(exporter, "embedding")[0].attributes
    assert len(json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[0]) == 5000
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 9


def test_delayed_child_cannot_outlive_parent_privacy_veto(runtime, monkeypatch):
    _, exporter, _, logger, sink = runtime
    with logger.start_span(name="parent", type="eval") as root:
        with root.start_span(name="child", type="llm") as child:
            child.log(
                input="private child input",
                output="private child output",
                metadata={
                    "model": "fixture",
                    "provider": "openai",
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                },
            )
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    sink.rows = list(reversed(sink.rows))
    logger.flush()
    attrs = records(exporter, "chat")[0].attributes
    assert (
        SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
        and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    )
    assert (
        attrs[SpanAttributes.LLM_REQUEST_MODEL] == "fixture"
        and attrs[SpanAttributes.LLM_SYSTEM] == "openai"
    )
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 0


def test_policy_fault_after_span_start_has_cleanup_and_native_result(
    runtime, monkeypatch
):
    _, exporter, _, logger, _ = runtime

    def fail():
        raise ValueError("controlled policy fault")

    monkeypatch.setattr(module, "_allowed", fail)
    with logger.start_span(name="policy-fault", type="tool") as span:
        span.log(output="native value")
    logger.flush()
    assert len(records(exporter, "tool")) == 1 and not module._RUNTIME.states
    assert (
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT
        not in records(exporter, "tool")[0].attributes
    )


def test_native_local_sink_mask_is_not_bypassed_or_invoked_twice(runtime):
    _, exporter, _, logger, sink = runtime
    calls = []
    sink._masking_function = lambda value: calls.append(value) or "[native-masked]"
    with logger.start_span(name="masked", type="tool") as span:
        span.log(input="sensitive", output="sensitive")
    logger.flush()
    assert not calls
    attrs = records(exporter, "tool")[0].attributes
    assert (
        SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
        and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    )


def test_actual_provider_stream_native_chunks_and_caller_context(runtime):
    from _fixtures import client_context

    provider, exporter, _, logger, _ = runtime
    with provider.get_tracer("caller").start_as_current_span("caller") as caller:
        with client_context() as client:
            stream = client.chat.completions.create(
                model="fixture-model",
                messages=[{"role": "user", "content": "fixture"}],
                stream=True,
                stream_options={"include_usage": True},
            )
            before = trace.get_current_span()
            chunks = []
            for chunk in stream:
                chunks.append(chunk)
                assert trace.get_current_span() is before
            assert chunks[0].choices[0].delta.content == "controlled stream"
            assert trace.get_current_span() is caller
        logger.flush()
    attrs = records(exporter, "chat")[0].attributes
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 11
    assert attrs[SpanAttributes.GEN_AI_IS_STREAMING] is True


def test_masking_start_bound_and_shared_owner_policy(runtime):
    provider, exporter, plugin, logger, _ = runtime

    def hide(value):
        return "[hidden at start]"

    plugin.set_masking_function(hide)
    with logger.start_span(name="masked-start", type="tool") as span:
        span.log(input="private payload", output="private result")
        plugin.set_masking_function(None)
    logger.flush()
    attrs = records(exporter, "tool")[0].attributes
    assert (
        "private" not in str(attrs)
        and "hidden at start" in attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    second = BraintrustInstrumentor(tracer_provider=provider)
    second.activate()
    with pytest.raises(ValueError):
        plugin.set_masking_function(hide)
    second.deactivate()


def test_preexisting_native_generator_keeps_caller_context(runtime):
    provider, exporter, plugin, logger, _ = runtime
    plugin.deactivate()

    @braintrust.traced(name="preexisting-generator")
    def generate():
        yield "native"

    plugin.activate()
    with provider.get_tracer("caller").start_as_current_span("caller") as caller:
        iterator = generate()
        assert next(iterator) == "native"
        assert trace.get_current_span() is caller
        iterator.close()
        assert trace.get_current_span() is caller
    logger.flush()
    assert len(records(exporter, "tool")) == 1


def test_private_usage_reader_does_not_visit_non_counter_fields():
    from respan_instrumentation_braintrust._mapping import private_usage

    class Usage(dict):
        def items(self):
            raise AssertionError("private mapping traversal")

    value = Usage(
        input_tokens=0,
        total_tokens=0,
        private_payload=object(),
        input_tokens_details=Usage(cached_tokens=3, private_payload=object()),
    )
    assert private_usage(value) == {
        "input_tokens": 0,
        "total_tokens": 0,
        "input_tokens_details": {"cached_tokens": 3},
    }
