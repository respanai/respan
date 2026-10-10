"""Released native SDK compatibility, source mapping, privacy and ownership."""

import ast
import asyncio
import json

import pytest
from livekit.agents import llm, telemetry
from livekit.agents._exceptions import APIStatusError
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider, sampling
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_instrumentation_livekit import LiveKitInstrumentor
from respan_instrumentation_livekit import _instrumentation as lifecycle
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

from ._fixtures import (
    APIConnectOptions,
    Model,
    chat_context,
    provider_model,
    value_tool,
    vector_tool,
)


@pytest.fixture
def native(monkeypatch):
    assert lifecycle._RUNTIME is None
    provider = TracerProvider()
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    owner = LiveKitInstrumentor()
    owner.activate()
    try:
        yield provider, exporter, owner
    finally:
        owner.deactivate()
        provider.shutdown()


def chat_spans(exporter):
    return [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get("respan.entity.log_type") == "chat"
    ]


@pytest.mark.asyncio
async def test_released_stream_values_usage_and_current_tools_full_payloads(native):
    _provider, exporter, _owner = native
    result = (
        await Model(tools=True)
        .chat(chat_ctx=chat_context(True), tools=[vector_tool])
        .collect()
    )
    assert result.text == "PRIVATE_OUTPUT" and len(result.tool_calls) == 2
    tools = [
        await llm.execute_function_call(c, llm.ToolContext([vector_tool]))
        for c in result.tool_calls
    ]
    assert all(len(t.raw_output["dense"]) == 5000 for t in tools)
    chats = chat_spans(exporter)
    assert len(chats) == 1
    a = chats[0].attributes
    current = json.loads(a["gen_ai.completion.0.tool_calls"])
    assert [c["id"] for c in current] == ["actual-call-0", "actual-call-1"]
    for c in current:
        args = json.loads(c["function"]["arguments"])
        assert len(args["dense"]) == 5000 and len(args["sparse"]) == 256
        assert args["api_key"] == "[REDACTED]"
        assert isinstance(args, dict)
    history = [
        json.loads(v)
        for k, v in a.items()
        if k.startswith("gen_ai.prompt.") and k.endswith(".tool_calls")
    ]
    assert len(history) == 1 and history[0][0]["id"] == "history-only"
    definitions = json.loads(a["llm.request.functions"])
    props = definitions[0]["function"]["parameters"]["properties"]
    assert (
        len(props) == 123
        and props["api_key"]["type"] == "string"
        and props["api_key"]["default"] == "[REDACTED]"
    )
    assert [
        a[k]
        for k in [
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "llm.usage.total_tokens",
            "gen_ai.usage.cache_read_input_tokens",
        ]
    ] == [11, 7, 18, 3]
    toolspans = [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get("respan.entity.log_type") == "tool"
    ]
    assert len(toolspans) == 2
    for s in toolspans:
        assert s.parent.span_id == chats[0].context.span_id and s.attributes[
            "gen_ai.tool.call.id"
        ] in ["actual-call-0", "actual-call-1"]
        out = json.loads(s.attributes["traceloop.entity.output"])
        assert out["dense"] == list(range(5000)) and len(out["sparse"]) == 256
    assert all(
        not any(
            k.startswith(("lk.", "respan.span."))
            or k in {"status_code", "tools", "tool_calls"}
            for k in s.attributes
        )
        for s in exporter.get_finished_spans()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        False,
        {"prompt_tokens": False, "completion_tokens": False, "total_tokens": False},
        {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    ],
)
async def test_released_usage_constructor_provenance(native, usage):
    _, exporter, _ = native
    result = await Model(usage=usage).chat(chat_ctx=chat_context()).collect()
    assert result.text == "PRIVATE_OUTPUT"
    attrs = chat_spans(exporter)[0].attributes
    if usage and type(usage["prompt_tokens"]) is int and usage["prompt_tokens"] == 0:
        assert attrs["gen_ai.usage.input_tokens"] == 0
    else:
        assert not any("usage" in k for k in attrs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        True,
        False,
        {"prompt_tokens": False, "completion_tokens": False, "total_tokens": False},
        {"prompt_tokens": "11", "completion_tokens": "7", "total_tokens": "18"},
    ],
)
async def test_actual_openai_plugin_http_sse_source_usage(native, usage):
    _, exporter, _ = native
    model, client = provider_model(usage=usage)
    try:
        result = await model.chat(chat_ctx=chat_context()).collect()
        assert result.text == "controlled provider output"
    finally:
        await client.close()
        await model.aclose()
    attrs = chat_spans(exporter)[0].attributes
    assert (
        json.loads(attrs["traceloop.entity.output"])[0]["content"]
        == "controlled provider output"
    )
    if usage is True:
        assert [
            attrs[k]
            for k in [
                "gen_ai.usage.input_tokens",
                "gen_ai.usage.output_tokens",
                "llm.usage.total_tokens",
            ]
        ] == [11, 7, 18]
    else:
        assert not any("usage" in k for k in attrs)


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["env", "context", "constructor", "otel-env"])
async def test_start_opt_out_cannot_be_reenabled(native, monkeypatch, setting):
    _, exporter, owner = native
    pause = asyncio.Event()
    if setting == "constructor":
        owner.deactivate()
        owner = LiveKitInstrumentor(capture_content=False)
        owner.activate()
    if setting == "env":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    if setting == "otel-env":
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false"
        )
    token = (
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        if setting == "context"
        else None
    )
    stream = Model(pause=pause).chat(chat_ctx=chat_context())
    await asyncio.sleep(0)
    if token is not None:
        context.detach(token)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
    pause.set()
    assert (await stream.collect()).text == "PRIVATE_OUTPUT"
    attrs = chat_spans(exporter)[0].attributes
    assert "PRIVATE" not in json.dumps(dict(attrs))
    assert not exporter.get_finished_spans()[0].events
    owner.deactivate()


@pytest.mark.asyncio
async def test_closed_foreign_parent_veto_before_detach(native, monkeypatch):
    provider, exporter, _ = native
    pause = asyncio.Event()
    with provider.get_tracer("caller").start_as_current_span("foreign-parent"):
        stream = Model(pause=pause).chat(chat_ctx=chat_context())
        await asyncio.sleep(0)
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    pause.set()
    assert (await stream.collect()).text == "PRIVATE_OUTPUT"
    assert "PRIVATE" not in json.dumps(dict(chat_spans(exporter)[0].attributes))


@pytest.mark.asyncio
async def test_explicit_private_parent_context(native):
    provider, exporter, _ = native
    parent = provider.get_tracer("caller").start_span(
        "explicit-parent", context=context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
    )
    token = context.attach(trace.set_span_in_context(parent))
    try:
        assert (
            await Model().chat(chat_ctx=chat_context()).collect()
        ).text == "PRIVATE_OUTPUT"
    finally:
        context.detach(token)
        parent.end()
    assert "PRIVATE" not in json.dumps(dict(chat_spans(exporter)[0].attributes))


@pytest.mark.asyncio
async def test_end_veto_native_return_preserved(native, monkeypatch):
    _, exporter, _ = native

    def boundary():
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    result = await Model(boundary=boundary).chat(chat_ctx=chat_context()).collect()
    assert result.text == "PRIVATE_OUTPUT"
    assert "PRIVATE" not in json.dumps(dict(chat_spans(exporter)[0].attributes))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suppression",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
async def test_suppression(native, suppression):
    _, exporter, _ = native
    token = context.attach(context.set_value(suppression, True))
    try:
        assert (
            await Model().chat(chat_ctx=chat_context()).collect()
        ).text == "PRIVATE_OUTPUT"
    finally:
        context.detach(token)
    assert not chat_spans(exporter)


@pytest.mark.asyncio
async def test_sampler_off_skips_own_tool_serialization(monkeypatch):
    provider = TracerProvider(sampler=sampling.ALWAYS_OFF)
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    owner = LiveKitInstrumentor()
    owner.activate()

    def fail(*args, **kwargs):
        raise AssertionError("unsampled payload serialized")

    monkeypatch.setattr(lifecycle, "normalize_livekit_tools", fail)
    try:
        assert (
            await Model().chat(chat_ctx=chat_context(), tools=[value_tool]).collect()
        ).text == "PRIVATE_OUTPUT"
    finally:
        owner.deactivate()
        provider.shutdown()


@pytest.mark.asyncio
async def test_native_tool_error_has_error_status_without_output_or_http(native):
    _, exporter, _ = native
    call = llm.FunctionToolCall(
        name="missing_tool", arguments="{}", call_id="actual-error"
    )
    result = await llm.execute_function_call(call, llm.ToolContext([]))
    assert result.fnc_call_out.is_error and isinstance(result.raw_exception, ValueError)
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code.name == "ERROR"
    assert not any(
        k in span.attributes
        for k in ["traceloop.entity.output", "status_code", "http.response.status_code"]
    )


@pytest.mark.asyncio
async def test_actual_http_error_exception_preserved(native):
    _, exporter, _ = native
    model, client = provider_model(error=True)
    try:
        with pytest.raises(APIStatusError) as caught:
            await model.chat(
                chat_ctx=chat_context(), conn_options=APIConnectOptions(max_retry=0)
            ).collect()
        assert caught.value.status_code == 429
    finally:
        await client.close()
        await model.aclose()
    chat = chat_spans(exporter)[0]
    assert chat.status.status_code.name == "ERROR"
    assert chat.attributes["http.response.status_code"] == 429
    assert "traceloop.entity.output" not in chat.attributes


@pytest.mark.asyncio
async def test_own_tool_observer_fault_preserves_native_result(native, monkeypatch):
    _, exporter, _ = native

    def fail(*args, **kwargs):
        raise RuntimeError("observer")

    monkeypatch.setattr(lifecycle, "build_tool_span_attrs", fail)
    call = llm.FunctionToolCall(
        name="value_tool", arguments='{"value":"actual"}', call_id="actual-call"
    )
    result = await llm.execute_function_call(call, llm.ToolContext([value_tool]))
    assert result.raw_output == "actual"
    assert "traceloop.entity.output" not in exporter.get_finished_spans()[0].attributes


def test_shared_and_foreign_hooks_restore(native):
    _provider, _, owner = native
    second = LiveKitInstrumentor()
    second.activate()
    installed = llm.execute_function_call
    private = LiveKitInstrumentor(capture_content=False)
    with pytest.raises(ValueError):
        private.activate()
    owner.deactivate()
    assert llm.execute_function_call is installed

    async def foreign(*args, **kwargs):
        return await installed(*args, **kwargs)

    llm.execute_function_call = foreign
    try:
        second.deactivate()
        assert llm.execute_function_call is foreign
    finally:
        llm.execute_function_call = installed.__wrapped__


def test_activation_failure_rolls_back_owned_fields(monkeypatch):
    provider = TracerProvider()
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    previous = (telemetry.tracer._tracer_provider, telemetry.tracer._tracer)
    native = llm.LLMStream._main_task
    original = lifecycle.Runtime.patch

    def patch(runtime, owner, name, replacement):
        if name == "__init__":
            raise RuntimeError("registration fault")
        return original(runtime, owner, name, replacement)

    monkeypatch.setattr(lifecycle.Runtime, "patch", patch)
    with pytest.raises(RuntimeError):
        LiveKitInstrumentor().activate()
    assert (
        llm.LLMStream._main_task is native
        and (telemetry.tracer._tracer_provider, telemetry.tracer._tracer) == previous
    )
    assert lifecycle._RUNTIME is None
    provider.shutdown()


@pytest.mark.asyncio
async def test_early_close_preserves_native_iterator_and_caller_context(native):
    provider, exporter, _ = native
    finish = asyncio.Event()
    with provider.get_tracer("caller").start_as_current_span("caller") as parent:
        stream = Model(finish=finish).chat(chat_ctx=chat_context())
        assert stream.__aiter__() is stream
        chunk = await stream.__anext__()
        assert (
            isinstance(chunk, llm.ChatChunk) and chunk.delta.content == "PRIVATE_OUTPUT"
        )
        assert trace.get_current_span() is parent
        assert await stream.aclose() is None
        assert trace.get_current_span() is parent
    attrs = chat_spans(exporter)[0].attributes
    assert not any("usage" in k for k in attrs)
    assert attrs["gen_ai.completion.0.content"] == "PRIVATE_OUTPUT"


@pytest.mark.asyncio
async def test_inflight_deactivation_keeps_private_bounds_and_then_restores(
    native, monkeypatch
):
    _, exporter, owner = native
    pause = asyncio.Event()
    stream = Model(pause=pause).chat(chat_ctx=chat_context())
    await asyncio.sleep(0)
    owner.deactivate()
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    pause.set()
    assert (await stream.collect()).text == "PRIVATE_OUTPUT"
    assert "PRIVATE" not in json.dumps(dict(chat_spans(exporter)[0].attributes))
    assert lifecycle._RUNTIME is None
    owner.activate()
    owner.deactivate()


@pytest.mark.asyncio
@pytest.mark.parametrize("callback", ["start", "end", "observe"])
async def test_own_policy_fault_preserves_actual_sdk_result(
    native, monkeypatch, callback
):
    _, exporter, _ = native

    def fail(*args, **kwargs):
        raise RuntimeError("observer fault")

    monkeypatch.setattr(lifecycle._RUNTIME.processor.policy, callback, fail)
    assert (
        await Model().chat(chat_ctx=chat_context()).collect()
    ).text == "PRIVATE_OUTPUT"
    assert "PRIVATE" not in json.dumps(dict(chat_spans(exporter)[0].attributes))


@pytest.mark.asyncio
async def test_tool_error_status_precedes_own_serialization_fault(native, monkeypatch):
    _, exporter, _ = native

    def fail(*args, **kwargs):
        raise RuntimeError("observer fault")

    monkeypatch.setattr(lifecycle, "build_tool_span_attrs", fail)
    result = await llm.execute_function_call(
        llm.FunctionToolCall(name="missing", arguments="{}", call_id="actual"),
        llm.ToolContext([]),
    )
    assert isinstance(result.raw_exception, ValueError)
    assert exporter.get_finished_spans()[0].status.status_code.name == "ERROR"


@pytest.mark.asyncio
async def test_completed_call_private_bound_survives_many_live_call_objects(
    native, monkeypatch
):
    _, exporter, _ = native
    from ._fixtures import Stream

    calls = [
        llm.FunctionToolCall(
            name="value_tool",
            arguments='{"value":"PRIVATE_TOOL"}',
            call_id=f"actual-{i}",
        )
        for i in range(2050)
    ]

    class ManyStream(Stream):
        async def _run(self):
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id="actual",
                    delta=llm.ChoiceDelta(role="assistant", tool_calls=calls),
                )
            )

    model = Model()
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    response = await ManyStream(
        model,
        chat_ctx=chat_context(),
        tools=[value_tool],
        conn_options=APIConnectOptions(max_retry=0),
    ).collect()
    assert response.tool_calls[0] is calls[0]
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    result = await llm.execute_function_call(calls[0], llm.ToolContext([value_tool]))
    assert result.raw_output == "PRIVATE_TOOL"
    toolspan = exporter.get_finished_spans()[-1]
    assert "PRIVATE" not in json.dumps(dict(toolspan.attributes))


@pytest.mark.asyncio
async def test_raw_usage_provenance_survives_many_live_usage_objects(native):
    _, exporter, _ = native
    from ._fixtures import Stream

    values = [
        llm.CompletionUsage(
            prompt_tokens=False, completion_tokens=False, total_tokens=False
        )
        for _ in range(2050)
    ]

    class SavedStream(Stream):
        async def _run(self):
            self._event_ch.send_nowait(llm.ChatChunk(id="actual", usage=values[0]))

    await SavedStream(
        Model(),
        chat_ctx=chat_context(),
        tools=[],
        conn_options=APIConnectOptions(max_retry=0),
    ).collect()
    assert not any("usage" in k for k in chat_spans(exporter)[0].attributes)


@pytest.mark.asyncio
async def test_native_capture_api_false_is_respected(native):
    _, exporter, _ = native
    try:
        from livekit.agents.telemetry import gen_ai
    except ImportError:
        pytest.skip("native GenAI capture API introduced after LiveKit1.6")
    previous = gen_ai.capture_content_enabled()
    gen_ai.set_capture_content(False)
    try:
        assert (
            await Model().chat(chat_ctx=chat_context()).collect()
        ).text == "PRIVATE_OUTPUT"
    finally:
        gen_ai.set_capture_content(previous)
    assert "PRIVATE" not in json.dumps(dict(chat_spans(exporter)[0].attributes))


@pytest.mark.asyncio
async def test_native_allow_pii_false_and_foreign_same_provider_survive(native):
    provider, exporter, owner = native
    import inspect

    if "allow_pii" not in inspect.signature(telemetry.set_tracer_provider).parameters:
        pytest.skip("native allow_pii introduced after LiveKit1.6")
    previous = (telemetry.tracer._tracer_provider, telemetry.tracer._tracer)
    telemetry.set_tracer_provider(provider, allow_pii=False)
    foreign = telemetry.tracer._tracer
    try:
        assert (
            await Model().chat(chat_ctx=chat_context()).collect()
        ).text == "PRIVATE_OUTPUT"
        owner.deactivate()
        assert (
            telemetry.tracer._tracer_provider is provider
            and telemetry.tracer._tracer is foreign
        )
    finally:
        telemetry.tracer._tracer_provider, telemetry.tracer._tracer = previous
    assert "PRIVATE" not in json.dumps(dict(chat_spans(exporter)[0].attributes))


def test_native_provider_partial_failure_rolls_back(monkeypatch):
    provider = TracerProvider()
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    previous = (telemetry.tracer._tracer_provider, telemetry.tracer._tracer)
    original = trace.get_tracer

    def fail(name, *args, **kwargs):
        if name == "livekit-agents" and kwargs.get("tracer_provider") is provider:
            raise RuntimeError("native provider failure")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(trace, "get_tracer", fail)
    with pytest.raises(RuntimeError):
        LiveKitInstrumentor().activate()
    assert (
        telemetry.tracer._tracer_provider,
        telemetry.tracer._tracer,
    ) == previous and lifecycle._RUNTIME is None
    provider.shutdown()


@pytest.mark.asyncio
async def test_actual_usage_before_native_failure_is_retained(native):
    _, exporter, _ = native
    from ._fixtures import Stream

    class UsageThenError(Stream):
        async def _run(self):
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id="actual",
                    usage=llm.CompletionUsage(
                        prompt_tokens=11, completion_tokens=7, total_tokens=18
                    ),
                )
            )
            raise APIStatusError(
                "actual stream failure", status_code=429, retryable=False
            )

    with pytest.raises(APIStatusError):
        await UsageThenError(
            Model(),
            chat_ctx=chat_context(),
            tools=[],
            conn_options=APIConnectOptions(max_retry=0),
        ).collect()
    attrs = chat_spans(exporter)[0].attributes
    assert [
        attrs[k]
        for k in [
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "llm.usage.total_tokens",
        ]
    ] == [11, 7, 18]
    assert (
        attrs["http.response.status_code"] == 429
        and "traceloop.entity.output" not in attrs
    )


@pytest.mark.asyncio
async def test_actual_tool_benign_url_query_and_fragment_preserved(native):
    _, exporter, _ = native
    url = "https://fixture.invalid/search?q=vector%20query#section"
    result = await llm.execute_function_call(
        llm.FunctionToolCall(
            name="value_tool",
            arguments=json.dumps({"value": url}),
            call_id="actual-url",
        ),
        llm.ToolContext([value_tool]),
    )
    assert (
        result.raw_output == url
        and json.loads(
            exporter.get_finished_spans()[0].attributes["traceloop.entity.output"]
        )
        == url
    )
    from respan_instrumentation_livekit._serialization import safe_json

    redacted = json.loads(
        safe_json(
            {
                "url": "https://user:pass@fixture.invalid/search?q=vector&api_key=actual-secret#section"
            }
        )
    )["url"]
    assert (
        "user:pass" not in redacted
        and "actual-secret" not in redacted
        and "q=vector" in redacted
        and "#section" in redacted
    )


@pytest.mark.asyncio
async def test_room_free_native_session_tool_spans_are_promoted(native):
    _, exporter, _ = native
    from livekit.agents import Agent, AgentSession

    class OnceModel(Model):
        def chat(self, *, chat_ctx, tools=None, **kwargs):
            self.tools_enabled = not any(
                item.type == "function_call_output" for item in chat_ctx.items
            )
            return super().chat(chat_ctx=chat_ctx, tools=tools, **kwargs)

    session = AgentSession(llm=OnceModel())
    try:
        await session.start(
            agent=Agent(
                instructions="controlled room-free fixture", tools=[vector_tool]
            )
        )
        await asyncio.wait_for(session.run(user_input="controlled tool request"), 10)
    finally:
        await session.aclose()
    tools = [s for s in exporter.get_finished_spans() if s.name == "function_tool"]
    assert len(tools) == 2
    for s in tools:
        attrs = s.attributes
        assert attrs["respan.entity.log_type"] == "tool" and attrs[
            "gen_ai.tool.call.id"
        ] in ["actual-call-0", "actual-call-1"]
        assert json.loads(attrs["traceloop.entity.input"])["name"] == "vector_tool"
        output = json.loads(attrs["traceloop.entity.output"])
        # Native Session intentionally renders the tool value with str(value).
        # Keep that source string rather than inventing a structured result.
        assert isinstance(output, str)
        parsed = ast.literal_eval(output)
        assert parsed["dense"] == list(range(5000)) and len(parsed["sparse"]) == 256
        assert not any(
            k.startswith("gen_ai.tool.") and k != "gen_ai.tool.call.id" for k in attrs
        )
    assert not any(s.name == "livekit.tool" for s in exporter.get_finished_spans())
    assert all(
        not any(k.startswith(("gen_ai.", "llm.")) for k in s.attributes)
        for s in exporter.get_finished_spans()
        if s.attributes.get("respan.entity.log_type") == "task"
    )


@pytest.mark.asyncio
async def test_room_free_native_session_error_flag_becomes_error_without_output_http(
    native,
):
    _, exporter, _ = native
    from livekit.agents import Agent, AgentSession, function_tool

    from ._fixtures import SCHEMA

    @function_tool(raw_schema=SCHEMA)
    async def failing_tool(raw_arguments: dict[str, object]):
        raise ValueError("controlled native function failure")

    class OnceModel(Model):
        def chat(self, *, chat_ctx, tools=None, **kwargs):
            self.tools_enabled = not any(
                item.type == "function_call_output" for item in chat_ctx.items
            )
            return super().chat(chat_ctx=chat_ctx, tools=tools, **kwargs)

    session = AgentSession(llm=OnceModel())
    try:
        await session.start(
            agent=Agent(
                instructions="controlled room-free failure", tools=[failing_tool]
            )
        )
        await asyncio.wait_for(session.run(user_input="controlled failure request"), 10)
    finally:
        await session.aclose()
    tools = [s for s in exporter.get_finished_spans() if s.name == "function_tool"]
    assert len(tools) == 2
    for s in tools:
        assert (
            s.status.status_code.name == "ERROR"
            and s.attributes["respan.entity.log_type"] == "tool"
        )
        assert (
            "traceloop.entity.output" not in s.attributes
            and "http.response.status_code" not in s.attributes
            and "status_code" not in s.attributes
        )


@pytest.mark.asyncio
async def test_end_content_veto_retains_valid_source_usage(native, monkeypatch):
    _, exporter, _ = native
    from ._fixtures import Stream

    class UsageThenVeto(Stream):
        async def _run(self):
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id="actual",
                    usage=llm.CompletionUsage(
                        prompt_tokens=11, completion_tokens=7, total_tokens=18
                    ),
                )
            )
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    await UsageThenVeto(
        Model(),
        chat_ctx=chat_context(),
        tools=[],
        conn_options=APIConnectOptions(max_retry=0),
    ).collect()
    attrs = chat_spans(exporter)[0].attributes
    assert [
        attrs[k]
        for k in [
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "llm.usage.total_tokens",
        ]
    ] == [11, 7, 18]
    assert (
        "traceloop.entity.input" not in attrs and "traceloop.entity.output" not in attrs
    )


def test_native_stamped_redaction_bound_is_private(native):
    from livekit.agents import types
    from livekit.agents.telemetry import trace_types

    if not hasattr(types, "ATTRIBUTE_REDACTION_ENABLED"):
        pytest.skip("Native stamped project redaction flag absent in LiveKit1.6")
    ATTRIBUTE_REDACTION_ENABLED = types.ATTRIBUTE_REDACTION_ENABLED
    _, exporter, _ = native
    with telemetry.tracer.start_as_current_span(
        "function_tool",
        attributes={
            ATTRIBUTE_REDACTION_ENABLED: True,
            trace_types.ATTR_FUNCTION_TOOL_NAME: "actual",
            trace_types.ATTR_FUNCTION_TOOL_ID: "actual-call",
            trace_types.ATTR_FUNCTION_TOOL_ARGS: '{"content":"PRIVATE_INPUT"}',
            trace_types.ATTR_FUNCTION_TOOL_OUTPUT: "PRIVATE_OUTPUT",
        },
    ):
        pass
    attrs = exporter.get_finished_spans()[0].attributes
    assert "traceloop.entity.input" not in attrs
    assert "traceloop.entity.output" not in attrs


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["env", "context"])
async def test_unobserved_pre_activation_parent_stays_private(monkeypatch, boundary):
    provider = TracerProvider()
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    owner = LiveKitInstrumentor()
    initial = None
    permitted = None
    if boundary == "env":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    else:
        initial = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        with provider.get_tracer("caller").start_as_current_span(
            "preexisting.private", attributes={"caller.field": "untouched"}
        ) as parent:
            owner.activate()
            if boundary == "env":
                monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
            else:
                permitted = context.attach(
                    context.set_value(ENABLE_CONTENT_TRACING_KEY, True)
                )
            try:
                result = await Model().chat(chat_ctx=chat_context()).collect()
            finally:
                if permitted is not None:
                    context.detach(permitted)
                owner.deactivate()
            assert result.text == "PRIVATE_OUTPUT"
            assert parent.attributes == {"caller.field": "untouched"}
    finally:
        if initial is not None:
            context.detach(initial)
        provider.shutdown()
    chats = chat_spans(exporter)
    assert len(chats) == 1
    attrs = chats[0].attributes
    assert "traceloop.entity.input" not in attrs
    assert "traceloop.entity.output" not in attrs
    assert attrs["gen_ai.usage.input_tokens"] == 11
    assert attrs["gen_ai.usage.output_tokens"] == 7
    assert attrs["llm.usage.total_tokens"] == 18
