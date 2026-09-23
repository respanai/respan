from __future__ import annotations

import asyncio
import json

import ai
import ai.ops
import ai.testing
import pytest
from ai import experimental_telemetry as telemetry
from ai.providers.base import Provider
from ai.types.messages import FilePart
from ai.types.usage import Usage
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_vercel import VercelInstrumentor
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


@pytest.fixture
def setup():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = VercelInstrumentor(tracer_provider=provider)
    instrumentor.activate()
    yield instrumentor, provider, exporter
    instrumentor.deactivate()
    provider.shutdown()


async def test_generate_and_stream_capture_real_sdk_messages(setup):
    _, _, exporter = setup
    for streaming in (False, True):
        prompt = ai.user_message("hello")
        response = ai.assistant_message("world").model_copy(
            update={"usage": Usage(input_tokens=3, output_tokens=2)}
        )
        model = ai.testing.FakeModel([prompt, response])
        if streaming:
            async with ai.stream(model, [prompt]) as stream:
                async for _ in stream:
                    pass
            assert stream.message.text == "world"
        else:
            assert (await ai.experimental_generate(model, [prompt])).text == "world"
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    for index, span in enumerate(spans):
        attrs = span.attributes
        assert attrs[RESPAN_LOG_TYPE] == "chat"
        assert attrs["gen_ai.prompt.0.content"] == "hello"
        assert attrs["gen_ai.completion.0.content"] == "world"
        if index == 0:
            assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 5
        else:
            # The SDK FakeModel's stream emits text events without Usage events.
            assert SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in attrs
        assert "traceloop.span.kind" not in attrs


async def test_agent_tools_have_connected_tree_and_no_duplicate_usage(setup):
    _, provider, exporter = setup

    @ai.tool
    async def multiply(value: int) -> int:
        """Double the input."""
        return value * 2

    prompt = ai.user_message("double 6")
    model = ai.testing.FakeModel(
        [
            prompt,
            ai.assistant_message(ai.testing.tool_call(multiply, value=6)),
            ai.assistant_message("12"),
        ]
    )
    with provider.get_tracer("test").start_as_current_span("parent") as parent:
        async with ai.Agent(tools=[multiply]).run(model, [prompt]) as stream:
            async for _ in stream:
                pass
    spans = exporter.get_finished_spans()
    ids = {span.context.span_id for span in spans}
    assert all(span.parent is None or span.parent.span_id in ids for span in spans)
    assert {span.context.trace_id for span in spans} == {
        parent.get_span_context().trace_id
    }
    tool = next(s for s in spans if s.attributes.get(RESPAN_LOG_TYPE) == "tool")
    assert json.loads(tool.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])[
        "arguments"
    ] == {"value": 6}
    assert json.loads(tool.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == 12
    assert "gen_ai.tool.name" not in tool.attributes
    agent = next(s for s in spans if s.attributes.get(RESPAN_LOG_TYPE) == "agent")
    assert not any("usage" in key for key in agent.attributes)
    chats = [s for s in spans if s.attributes.get(RESPAN_LOG_TYPE) == "chat"]
    assert len(chats) == 2
    assert "gen_ai.completion.0.tool_calls" not in chats[-1].attributes


async def test_privacy_suppresses_inputs_outputs_and_tool_payloads(setup):
    original, provider, exporter = setup
    original.deactivate()
    private = VercelInstrumentor(tracer_provider=provider, capture_content=False)
    private.activate()
    try:
        prompt = ai.user_message("secret prompt")
        model = ai.testing.FakeModel([prompt, ai.assistant_message("secret result")])
        await ai.experimental_generate(model, [prompt])
        async with telemetry.span(
            telemetry.ToolExecutionSpanData(
                tool_name="secret_tool",
                tool_call_id="x",
                args={"secret": True},
                result="secret",
            )
        ):
            pass
        for span in exporter.get_finished_spans():
            assert "secret prompt" not in str(span.attributes)
            assert not any(
                k.endswith((".input", ".output", ".content", ".tool_calls"))
                for k in span.attributes
            )
    finally:
        private.deactivate()


class OperationProvider(Provider):
    provider_class_id: str = "respan-vercel-test"
    name: str = "test"
    default_base_url: str = "https://example.invalid"

    async def embed(self, model, values, *, params):
        if values == ["fail"]:
            raise ValueError("embedding failure")
        return ai.ops.Item(
            value=[[0.1, 0.2] for _ in values], usage=Usage(input_tokens=7)
        )

    async def generate_image(self, model, prompt, *, params):
        return ai.ops.Item(value=[FilePart(data=b"image", media_type="image/png")])

    async def generate_video(self, model, prompt, *, params):
        return ai.ops.Item(value=[FilePart(data=b"video", media_type="video/mp4")])

    async def generate_audio(self, model, prompt, *, params):
        return ai.ops.Item(value=[FilePart(data=b"audio", media_type="audio/wav")])

    async def transcribe(self, model, audio, *, params):
        return ai.ops.Item(value=ai.ops.Transcription(text="hello", language="en"))

    async def rerank(self, model, documents, query, *, params):
        return ai.ops.Item(value=[ai.ops.RankedDocument(index=1, score=0.9)])

    async def evaluate(self, model, state, questions, *, params):
        return ai.ops.Item(
            value=ai.ops.Evaluation(
                answers={"correct": ai.ops.BooleanAnswer(probability=0.95)}
            )
        )


@pytest.mark.parametrize(
    "operation,arguments",
    [
        ("generate_image", ["image"]),
        ("generate_video", ["video"]),
        ("generate_audio", ["audio"]),
        ("transcribe", [b"audio"]),
        ("rerank", [["low", "high"], "high"]),
        (
            "experimental_evaluate",
            ["4 is even", {"correct": ai.ops.BooleanQuestion(instructions="correct?")}],
        ),
    ],
)
async def test_non_chat_operations_preserve_results_without_chat_coercion(
    setup, operation, arguments
):
    _, _, exporter = setup
    from respan_instrumentation_vercel._translator import json_value

    result = await getattr(ai.ops, operation)(
        ai.Model(id="media-test", provider=OperationProvider()), *arguments
    )
    (span,) = exporter.get_finished_spans()
    assert span.attributes[RESPAN_LOG_TYPE] == "task"
    assert SpanAttributes.LLM_REQUEST_MODEL not in span.attributes
    assert SpanAttributes.LLM_REQUEST_TYPE not in span.attributes
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == json_value(
        result.value
    )


async def test_stream_metadata_parameters_and_usage_from_native_events(setup):
    from ai.models.core.params import (
        InferenceRequestParams,
        OutputParams,
        TemperatureSamplerParams,
    )
    from opentelemetry.semconv._incubating.attributes import gen_ai_attributes

    _, _, exporter = setup
    params = InferenceRequestParams(
        sampling={TemperatureSamplerParams: TemperatureSamplerParams(temperature=0.2)},
        output=OutputParams(max_tokens=20),
    )
    data = telemetry.AiStreamSpanData(
        model="test", provider="test", messages=[ai.user_message("hi")], params=params
    )
    async with telemetry.span(data) as span:
        span.add_event(telemetry.FIRST_TOKEN)
        span.data.message = ai.assistant_message("hello")
        span.data.usage = Usage(
            input_tokens=5, output_tokens=3, cache_read_tokens=2, reasoning_tokens=1
        )
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs[SpanAttributes.GEN_AI_IS_STREAMING] is True
    assert attrs[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.2
    assert attrs[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 20
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 8
    assert attrs[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 2
    assert attrs[gen_ai_attributes.GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] >= 0


async def test_default_params_do_not_break_telemetry(setup):
    from ai.models.core.params import InferenceRequestParams

    _, _, exporter = setup
    prompt = ai.user_message("hi")
    model = ai.testing.FakeModel([prompt, ai.assistant_message("hello")])
    await ai.experimental_generate(model, [prompt], params=InferenceRequestParams())
    assert len(exporter.get_finished_spans()) == 1


def test_packaging_declares_plugin():
    from importlib.metadata import entry_points

    (plugin,) = [
        entry
        for entry in entry_points(group="respan.instrumentations")
        if entry.name == "vercel"
    ]
    assert plugin.load() is VercelInstrumentor


async def test_embedding_keeps_vectors_and_real_usage_once(setup):
    _, _, exporter = setup
    model = ai.Model(id="embedding-test", provider=OperationProvider())
    result = await ai.ops.embed(model, ["hello", "world"])
    assert result.value == [[0.1, 0.2], [0.1, 0.2]]
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[RESPAN_LOG_TYPE] == "embedding"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
        "hello",
        "world",
    ]
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == result.value
    assert attrs["gen_ai.usage.input_tokens"] == 7


async def test_provider_exception_is_preserved_and_traced(setup):
    _, _, exporter = setup
    with pytest.raises(ValueError, match="embedding failure"):
        await ai.ops.embed(
            ai.Model(id="embedding-test", provider=OperationProvider()), ["fail"]
        )
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == trace.StatusCode.ERROR
    assert "embedding failure" in span.status.description


async def test_tool_error_result_sets_error_status(setup):
    _, _, exporter = setup
    async with telemetry.span(
        telemetry.ToolExecutionSpanData(
            tool_name="failing", tool_call_id="call1", is_error=True, result="failed"
        )
    ):
        pass
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == trace.StatusCode.ERROR


async def test_concurrent_calls_do_not_cross_parent_contexts(setup):
    _, provider, exporter = setup

    async def run(index):
        prompt = ai.user_message(str(index))
        model = ai.testing.FakeModel([prompt, ai.assistant_message(str(index))])
        with provider.get_tracer("test").start_as_current_span(
            f"parent-{index}"
        ) as parent:
            await asyncio.sleep(0)
            await ai.experimental_generate(model, [prompt])
            return parent.get_span_context().trace_id

    ids = await asyncio.gather(run(1), run(2))
    assert ids[0] != ids[1]
    for span in exporter.get_finished_spans():
        if span.attributes.get(RESPAN_LOG_TYPE) == "chat":
            value = span.attributes["gen_ai.prompt.0.content"]
            assert span.context.trace_id == ids[int(value) - 1]


async def test_live_native_parent_keeps_intervening_application_span(setup):
    _, provider, exporter = setup
    prompt = ai.user_message("hi")
    model = ai.testing.FakeModel([prompt, ai.assistant_message("hello")])
    async with telemetry.span("native parent"):
        with provider.get_tracer("app").start_as_current_span(
            "application task"
        ) as app:
            await ai.experimental_generate(model, [prompt])
    chat = next(
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get(RESPAN_LOG_TYPE) == "chat"
    )
    assert chat.parent.span_id == app.get_span_context().span_id


def test_activation_is_reference_counted_and_restores_patches():
    provider = TracerProvider()
    original = ai.ops.embed
    ids = provider.id_generator
    one = VercelInstrumentor(tracer_provider=provider)
    two = VercelInstrumentor(tracer_provider=provider)
    one.activate()
    wrapper = ai.ops.embed
    one.activate()
    two.activate()
    one.deactivate()
    assert ai.ops.embed is wrapper
    two.deactivate()
    assert ai.ops.embed is original
    assert provider.id_generator is ids
    two.deactivate()
    provider.shutdown()


async def test_hook_content_and_status(setup):
    _, _, exporter = setup
    data = telemetry.HookSpanData(
        label="approval", hook_type="tool", metadata={"action": "read"}
    )
    async with telemetry.span(data) as span:
        span.data.status = "resolved"
        span.data.resolution = {"granted": True}
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs[RESPAN_LOG_TYPE] == "task"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])["resolution"] == {
        "granted": True
    }


async def test_retained_operation_alias_is_inert_after_deactivate(setup):
    plugin, _, exporter = setup
    alias = ai.ops.embed
    plugin.deactivate()
    await alias(ai.Model(id="test", provider=OperationProvider()), ["private"])
    assert exporter.get_finished_spans() == ()


async def test_retained_alias_uses_new_provider_and_privacy_settings(setup):
    first, _, first_exporter = setup
    alias = ai.ops.embed
    first.deactivate()
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    private = VercelInstrumentor(tracer_provider=provider, capture_content=False)
    private.activate()
    try:
        await alias(ai.Model(id="test", provider=OperationProvider()), ["private"])
        assert first_exporter.get_finished_spans() == ()
        (span,) = exporter.get_finished_spans()
        assert span.attributes[RESPAN_LOG_TYPE] == "embedding"
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    finally:
        private.deactivate()
        provider.shutdown()


async def test_final_deactivate_waits_for_in_flight_operation_wrappers(setup):
    plugin, _, exporter = setup
    started = asyncio.Event()
    finish = asyncio.Event()

    class WaitingProvider(OperationProvider):
        async def embed(self, model, values, *, params):
            started.set()
            await finish.wait()
            return await super().embed(model, values, params=params)

    task = asyncio.create_task(
        ai.ops.embed(ai.Model(id="test", provider=WaitingProvider()), ["hello"])
    )
    await started.wait()
    try:
        with pytest.raises(RuntimeError, match="in-flight"):
            plugin.deactivate()
    finally:
        finish.set()
        await task
    plugin.deactivate()
    assert len(exporter.get_finished_spans()) == 1


async def test_operation_cancellation_releases_activation_guard(setup):
    plugin, _, _ = setup
    started = asyncio.Event()

    class WaitingProvider(OperationProvider):
        async def embed(self, model, values, *, params):
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(
        ai.ops.embed(ai.Model(id="test", provider=WaitingProvider()), ["hello"])
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    plugin.deactivate()


@pytest.mark.parametrize("outer_parent", [False, True])
@pytest.mark.parametrize("generic_data", [False, True])
async def test_json_dict_sink_replay_preserves_messages_usage_and_whole_tree(
    setup, outer_parent, generic_data
):
    from contextlib import nullcontext

    _, provider, exporter = setup

    @ai.tool
    async def double(value: int) -> int:
        """Double a value."""
        return value * 2

    prompt = ai.user_message("double 7")
    model = ai.testing.FakeModel(
        [
            prompt,
            ai.assistant_message(ai.testing.tool_call(double, value=7)),
            ai.assistant_message("14"),
        ]
    )
    sink = telemetry.DictSink()
    async with (
        telemetry.use_sink(sink),
        ai.Agent(tools=[double]).run(model, [prompt]) as stream,
    ):
        async for _ in stream:
            pass
    payload = json.loads(
        json.dumps([span.model_dump(mode="json") for span in sink.finished_spans])
    )
    if generic_data:
        from typing import Any

        payload = [telemetry.Span[Any].model_validate(item) for item in payload]
    observed_data = []

    class Observer:
        async def on_span_start(self, span):
            observed_data.append(span.data)

        async def on_span_event(self, span, event):
            pass

        async def on_span_end(self, span):
            observed_data.append(span.data)

    observer = Observer()
    telemetry.register(observer)
    try:
        parent_context = (
            provider.get_tracer("test").start_as_current_span("outer")
            if outer_parent
            else nullcontext()
        )
        with parent_context:
            await telemetry.push_all(payload)
    finally:
        telemetry.unregister(observer)
    spans = exporter.get_finished_spans()
    assert len(spans) == len(payload) + int(outer_parent)
    if generic_data:
        assert all(isinstance(data, dict) for data in observed_data)
    assert len({span.context.trace_id for span in spans}) == 1
    ids = {span.context.span_id for span in spans}
    assert all(span.parent is None or span.parent.span_id in ids for span in spans)
    chats = [span for span in spans if span.attributes.get(RESPAN_LOG_TYPE) == "chat"]
    assert len(chats) == 2
    assert chats[0].attributes["gen_ai.prompt.0.content"] == "double 7"
    assert chats[-1].attributes["gen_ai.completion.0.content"] == "14"
    tool = next(
        span for span in spans if span.attributes.get(RESPAN_LOG_TYPE) == "tool"
    )
    assert json.loads(tool.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == 14


@pytest.mark.parametrize("gate", ["environment", "context"])
async def test_runtime_content_opt_out_controls_native_and_wrapped_calls(
    setup, monkeypatch, gate
):
    from opentelemetry import context
    from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

    _, _, exporter = setup
    token = None
    if gate == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "off")
    else:
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        prompt = ai.user_message("SECRET input")
        model = ai.testing.FakeModel([prompt, ai.assistant_message("SECRET output")])
        await ai.experimental_generate(model, [prompt])
        await ai.ops.embed(
            ai.Model(id="test", provider=OperationProvider()), ["SECRET embedding"]
        )
    finally:
        if token is not None:
            context.detach(token)
    assert len(exporter.get_finished_spans()) == 2
    for span in exporter.get_finished_spans():
        assert "SECRET" not in str(span.attributes)
        assert not any(
            key.endswith((".input", ".output", ".content")) for key in span.attributes
        )


async def test_native_content_opt_out_at_start_cannot_be_relaxed_before_end(
    setup, monkeypatch
):
    _, _, exporter = setup
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    async with telemetry.span(
        telemetry.AiGenerateSpanData(
            model="test", provider="test", messages=[ai.user_message("SECRET input")]
        )
    ) as span:
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        span.data.message = ai.assistant_message("SECRET output")
    (exported,) = exporter.get_finished_spans()
    assert "SECRET" not in str(exported.attributes)


async def test_native_content_opt_out_at_end_suppresses_completed_content(
    setup, monkeypatch
):
    _, _, exporter = setup
    async with telemetry.span(
        telemetry.AiGenerateSpanData(
            model="test", provider="test", messages=[ai.user_message("SECRET input")]
        )
    ) as span:
        span.data.message = ai.assistant_message("SECRET output")
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    (exported,) = exporter.get_finished_spans()
    assert "SECRET" not in str(exported.attributes)


@pytest.mark.parametrize("enabled_at_start", [False, True])
async def test_wrapped_content_requires_permission_at_start_and_end(
    setup, monkeypatch, enabled_at_start
):
    _, _, exporter = setup
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", str(enabled_at_start))

    class SwitchingProvider(OperationProvider):
        async def embed(self, model, values, *, params):
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", str(not enabled_at_start))
            return await super().embed(model, values, params=params)

    await ai.ops.embed(ai.Model(id="test", provider=SwitchingProvider()), ["SECRET"])
    (span,) = exporter.get_finished_spans()
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes


@pytest.mark.parametrize("outer_parent", [False, True])
@pytest.mark.parametrize(
    "operation,arguments,log_type",
    [
        ("embed", [["hello", "world"]], "embedding"),
        ("generate_image", ["image"], "task"),
        ("generate_video", ["video"], "task"),
        ("generate_audio", ["audio"], "task"),
        ("transcribe", [b"audio"], "task"),
        ("rerank", [["low", "high"], "high"], "task"),
        (
            "experimental_evaluate",
            ["4 is even", {"correct": ai.ops.BooleanQuestion(instructions="correct?")}],
            "task",
        ),
    ],
)
async def test_operation_dict_sink_defers_export_and_replays_full_content(
    setup, operation, arguments, log_type, outer_parent
):
    from contextlib import nullcontext

    from respan_instrumentation_vercel._constants import AI_OPERATION_CONTENT
    from respan_instrumentation_vercel._translator import json_value
    from respan_sdk.constants.span_attributes import RESPAN_METADATA

    _, provider, exporter = setup
    model = ai.Model(id="operation-test", provider=OperationProvider())
    await getattr(ai.ops, operation)(model, *arguments)
    (direct,) = exporter.get_finished_spans()
    exporter.clear()
    sink = telemetry.DictSink()
    async with (
        telemetry.use_sink(sink),
        telemetry.span("serialized workflow") as parent,
    ):
        parent.trace_attrs["caller"] = "preserved"
        result = await getattr(ai.ops, operation)(model, *arguments)
        assert exporter.get_finished_spans() == ()
    payload = json.loads(
        json.dumps([s.model_dump(mode="json") for s in sink.finished_spans])
    )
    assert len(payload) == 2
    operation_snapshot = next(
        item for item in payload if item["data"]["kind"] != "custom"
    )
    marker = operation_snapshot["trace_attrs"][AI_OPERATION_CONTENT]
    assert marker["span_id"] == operation_snapshot["id"]
    assert marker["output"] == json_value(result.value)
    assert exporter.get_finished_spans() == ()
    with (
        provider.get_tracer("test").start_as_current_span("outer application")
        if outer_parent
        else nullcontext()
    ):
        await telemetry.push_all(payload)
    spans = exporter.get_finished_spans()
    assert len(spans) == 2 + int(outer_parent)
    assert len({s.context.trace_id for s in spans}) == 1
    ids = {s.context.span_id for s in spans}
    assert all(s.parent is None or s.parent.span_id in ids for s in spans)
    replayed = next(
        s
        for s in spans
        if s.attributes.get(SpanAttributes.TRACELOOP_ENTITY_NAME)
        == operation_snapshot["data"]["kind"]
    )
    assert replayed.attributes[RESPAN_LOG_TYPE] == log_type
    for key, value in direct.attributes.items():
        if key not in {RESPAN_METADATA, SpanAttributes.TRACELOOP_ENTITY_OUTPUT}:
            assert replayed.attributes.get(key) == value
    assert replayed.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == json_value(
        result.value
    )
    metadata = json.loads(replayed.attributes[RESPAN_METADATA])
    assert metadata["caller"] == "preserved"
    for key, value in json.loads(direct.attributes.get(RESPAN_METADATA, "{}")).items():
        assert metadata[key] == value
    assert AI_OPERATION_CONTENT not in str(replayed.attributes)
    if operation == "embed":
        assert (
            json.loads(replayed.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
            == result.value
        )
        assert sum(s.attributes.get("gen_ai.usage.input_tokens", 0) for s in spans) == 7


@pytest.mark.parametrize("capture_content", [False, True])
async def test_deferred_operation_error_preserves_exception_status_and_privacy(
    setup, capture_content
):
    from respan_instrumentation_vercel._constants import AI_OPERATION_CONTENT

    first, provider, exporter = setup
    first.deactivate()
    plugin = VercelInstrumentor(
        tracer_provider=provider, capture_content=capture_content
    )
    plugin.activate()
    try:
        sink = telemetry.DictSink()
        async with telemetry.use_sink(sink):
            with pytest.raises(ValueError, match="embedding failure"):
                await ai.ops.embed(
                    ai.Model(id="error-test", provider=OperationProvider()), ["fail"]
                )
        assert exporter.get_finished_spans() == ()
        payload = json.loads(
            json.dumps([s.model_dump(mode="json") for s in sink.finished_spans])
        )
        assert len(payload) == 1
        assert (AI_OPERATION_CONTENT in payload[0]["trace_attrs"]) is capture_content
        await telemetry.push_all(payload)
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == trace.StatusCode.ERROR
        assert "embedding failure" in span.status.description
        assert (
            SpanAttributes.TRACELOOP_ENTITY_INPUT in span.attributes
        ) is capture_content
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
        assert AI_OPERATION_CONTENT not in str(span.attributes)
    finally:
        plugin.deactivate()


@pytest.mark.parametrize("enabled_at_start", [False, True])
async def test_deferred_content_requires_runtime_gate_at_both_boundaries(
    setup, monkeypatch, enabled_at_start
):
    from respan_instrumentation_vercel._constants import AI_OPERATION_CONTENT

    _, _, exporter = setup
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", str(enabled_at_start))

    class SwitchingProvider(OperationProvider):
        async def embed(self, model, values, *, params):
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", str(not enabled_at_start))
            return await super().embed(model, values, params=params)

    sink = telemetry.DictSink()
    async with telemetry.use_sink(sink):
        await ai.ops.embed(
            ai.Model(id="private-test", provider=SwitchingProvider()), ["SECRET"]
        )
    payload = json.loads(
        json.dumps([s.model_dump(mode="json") for s in sink.finished_spans])
    )
    assert "SECRET" not in json.dumps(payload)
    assert AI_OPERATION_CONTENT not in payload[0]["trace_attrs"]
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    await telemetry.push_all(payload)
    (span,) = exporter.get_finished_spans()
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes


async def test_replay_privacy_gate_suppresses_durable_operation_payload(
    setup, monkeypatch
):
    _, _, exporter = setup
    sink = telemetry.DictSink()
    async with telemetry.use_sink(sink):
        await ai.ops.embed(
            ai.Model(id="private-test", provider=OperationProvider()), ["SECRET"]
        )
    payload = json.loads(
        json.dumps([s.model_dump(mode="json") for s in sink.finished_spans])
    )
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    await telemetry.push_all(payload)
    (span,) = exporter.get_finished_spans()
    assert "SECRET" not in str(span.attributes)
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes


async def test_deferred_sink_failures_do_not_change_operation_result(setup):
    _, _, exporter = setup

    class FailingSink:
        async def on_push(self, span):
            raise RuntimeError("sink unavailable")

    async with telemetry.use_sink(FailingSink()):
        result = await ai.ops.embed(
            ai.Model(id="test", provider=OperationProvider()), ["hello"]
        )
    assert result.value == [[0.1, 0.2]]
    assert exporter.get_finished_spans() == ()
