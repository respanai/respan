"""Controlled native Burr applications against the released SDK."""

import asyncio
import inspect
import json
from collections.abc import Generator

import pytest
from burr.core import ApplicationBuilder, State, action
from burr.core.action import streaming_action
from burr.visibility import TracerFactory
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SpanAttributes as AI
from opentelemetry.trace import StatusCode
from respan_instrumentation_burr import BurrInstrumentor
from respan_instrumentation_burr._adapter import BurrLifecycleAdapter
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

SECRET = "controlled-burr-private-marker"


@action(reads=["value"], writes=["result"])
def echo(state: State, argument=None) -> State:
    return state.update(result=state["value"], argument=argument)


def builder(fn=echo, value=SECRET):
    return (
        ApplicationBuilder()
        .with_actions(fn)
        .with_state(value=value)
        .with_identifiers(
            app_id="burr-controlled-app", partition_key="burr-controlled-thread"
        )
        .with_entrypoint(fn.__name__)
    )


def run_app(adapter, fn=echo, value=SECRET, inputs=None):
    app = builder(fn, value).with_hooks(adapter).build()
    return app.run(halt_after=[fn.__name__], inputs=inputs or {})


def contains_secret(spans):
    return SECRET in str(
        [(dict(s.attributes), s.events, s.status.description) for s in spans]
    )


def test_native_result_and_input_identity_and_full_state(telemetry):
    _, exporter = telemetry
    adapter = BurrLifecycleAdapter()
    payload = {
        "history": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query":"fixture"}',
                        },
                    }
                ],
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
        "dense": [0.25, 0.5],
        "sparse": {"indices": [2, 9], "values": [0.1, 0.9]},
    }
    identity = object()
    result = run_app(adapter, value=payload, inputs={"argument": identity})
    assert result[2]["result"] is payload
    assert result[2]["argument"] is identity
    spans = exporter.get_finished_spans()
    task = next(s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "task")
    assert (
        json.loads(task.attributes[AI.TRACELOOP_ENTITY_OUTPUT])["state"]["result"]
        == payload
    )
    assert (
        task.parent.span_id
        == next(
            s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "workflow"
        ).context.span_id
    )
    assert not any("status_code" in s.attributes for s in spans)
    assert not any(AI.TRACELOOP_SPAN_KIND in s.attributes for s in spans)


def test_native_exception_identity_and_root_and_custom_error(telemetry):
    _, exporter = telemetry
    error = RuntimeError(SECRET)

    @action(reads=["value"], writes=[])
    def fail(state: State, __tracer: TracerFactory):
        with __tracer("native-custom-error"):
            raise error

    owner = BurrInstrumentor()
    owner.activate()
    try:
        with pytest.raises(RuntimeError) as caught:
            builder(fail).build().run(halt_after=["fail"])
    finally:
        owner.deactivate()
    assert caught.value is error
    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    assert all(s.status.status_code is StatusCode.ERROR for s in spans)
    assert all(any(e.name == "exception" for e in s.events) for s in spans)
    assert all("status_code" not in s.attributes for s in spans)
    assert (
        AI.TRACELOOP_ENTITY_OUTPUT
        not in next(s for s in spans if s.name == "native-custom-error").attributes
    )


def test_custom_attributes_accumulate_without_changing_native_tracer(telemetry):
    _, exporter = telemetry

    @action(reads=["value"], writes=["result"])
    def custom(state: State, __tracer: TracerFactory):
        with __tracer("custom", span_dependencies=["prior"]) as span:
            span.log_attributes(first={"dense": [0.1, 0.2]})
            span.log_attributes(second={"sparse": {"indices": [1], "values": [0.3]}})
        return state.update(result="done")

    run_app(BurrLifecycleAdapter(), custom)
    span = next(s for s in exporter.get_finished_spans() if s.name == "custom")
    metadata = json.loads(span.attributes["respan.metadata.burr"])
    assert metadata["logged_attributes"] == {
        "first": {"dense": [0.1, 0.2]},
        "second": {"sparse": {"indices": [1], "values": [0.3]}},
    }
    assert metadata["span_dependencies"] == ["prior"]
    assert metadata["span_id"]


@streaming_action(reads=["value"], writes=["result"])
def streaming(state: State) -> Generator[tuple[dict, State | None], None, None]:
    yield {"delta": "first"}, None
    yield {"delta": "second"}, None
    yield {"done": True}, state.update(result="firstsecond")


def test_native_stream_container_items_and_get(telemetry):
    _, exporter = telemetry
    app = builder(streaming).with_hooks(BurrLifecycleAdapter()).build()
    native = builder(streaming).build().stream_result(halt_after=["streaming"])[1]
    native_type = type(native)
    list(native)
    native.get()
    _, stream = app.stream_result(halt_after=["streaming"])
    assert type(stream) is native_type
    assert list(stream) == [{"delta": "first"}, {"delta": "second"}]
    result, state = stream.get()
    assert result == {"done": True} and state["result"] == "firstsecond"
    task = next(
        s
        for s in exporter.get_finished_spans()
        if s.attributes[RESPAN_LOG_TYPE] == "task"
    )
    assert [e.name for e in task.events] == [
        "burr.stream.start",
        "burr.stream.item",
        "burr.stream.item",
        "burr.stream.end",
    ]
    assert (
        json.loads(task.attributes["respan.metadata.burr"])["stream"]["item_count"] == 2
    )


def test_native_iterate_generator_close_and_context(telemetry):
    app = builder().with_hooks(BurrLifecycleAdapter()).build()
    iterator = app.iterate(halt_after=["echo"])
    assert inspect.isgenerator(iterator)
    next(iterator)
    iterator.close()
    assert not trace.get_current_span().get_span_context().is_valid


def test_async_builder_and_run(telemetry):
    _, exporter = telemetry

    @action(reads=["value"], writes=["result"])
    async def native_async(state: State):
        await asyncio.sleep(0)
        return state.update(result=state["value"])

    owner = BurrInstrumentor()
    owner.activate()

    async def execute():
        app = await builder(native_async).abuild()
        return await app.arun(halt_after=["native_async"])

    try:
        assert asyncio.run(execute())[2]["result"] == SECRET
    finally:
        owner.deactivate()
    assert len(exporter.get_finished_spans()) == 2


@pytest.mark.parametrize(
    "source", ["setting", "environment", "respan_environment", "context"]
)
def test_initial_privacy_bound_cannot_widen(telemetry, monkeypatch, source):
    _, exporter = telemetry
    token = None
    if source == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    if source == "respan_environment":
        monkeypatch.setenv("RESPAN_TRACE_CONTENT", "false")
    if source == "context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))

    @action(reads=["value"], writes=["result"])
    def widen(state: State):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        monkeypatch.setenv("RESPAN_TRACE_CONTENT", "true")
        return state.update(result=state["value"])

    try:
        run_app(BurrLifecycleAdapter(capture_content=source != "setting"), widen)
    finally:
        if token is not None:
            context.detach(token)
    assert not contains_secret(exporter.get_finished_spans())


@pytest.mark.parametrize("ctx", ["ambient", "supplied"])
def test_ambient_and_supplied_parent_bounds_are_combined(telemetry, ctx):
    provider, exporter = telemetry
    adapter = BurrLifecycleAdapter()
    denied = context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
    allowed = context.set_value(ENABLE_CONTENT_TRACING_KEY, True)
    token = context.attach(denied if ctx == "ambient" else allowed)
    try:
        parent = provider.get_tracer("native-parent").start_span(
            "parent", context=allowed if ctx == "ambient" else denied
        )
    finally:
        context.detach(token)
    with trace.use_span(parent, end_on_exit=True):
        run_app(adapter)
    assert not contains_secret(exporter.get_finished_spans())


def test_unobserved_recording_parent_is_private(telemetry):
    provider, exporter = telemetry
    with provider.get_tracer("native-parent").start_as_current_span("preexisting"):
        run_app(BurrLifecycleAdapter())
    assert not contains_secret(exporter.get_finished_spans())


def test_finished_parent_privacy_is_retained(telemetry, monkeypatch):
    provider, exporter = telemetry
    adapter = BurrLifecycleAdapter()
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    parent = provider.get_tracer("native-parent").start_span("finished")
    parent.end()
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    with trace.use_span(parent):
        run_app(adapter)
    assert not contains_secret(exporter.get_finished_spans())


def test_pre_detach_veto_scrubs_input_metadata_events_and_error(telemetry):
    _, exporter = telemetry
    error = RuntimeError(SECRET)

    @action(reads=["value"], writes=[])
    def veto(state: State, __tracer: TracerFactory):
        with __tracer("veto") as span:
            span.log_attributes(secret=SECRET)
            token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
            context.detach(token)
            raise error

    owner = BurrInstrumentor()
    owner.activate()
    try:
        with pytest.raises(RuntimeError) as caught:
            builder(veto).build().run(halt_after=["veto"])
    finally:
        owner.deactivate()
    assert caught.value is error
    spans = exporter.get_finished_spans()
    assert not contains_secret(spans)
    assert all(s.status.status_code is StatusCode.ERROR for s in spans)
    assert all(s.status.description is None and not s.events for s in spans)


@pytest.mark.parametrize("suppression", ["general", "language_model"])
def test_suppression_before_content_extraction(telemetry, suppression):
    _, exporter = telemetry
    from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY

    k = (
        context._SUPPRESS_INSTRUMENTATION_KEY
        if suppression == "general"
        else SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    )
    token = context.attach(context.set_value(k, True))
    try:
        assert run_app(BurrLifecycleAdapter())[2]["result"] == SECRET
    finally:
        context.detach(token)
    assert exporter.get_finished_spans() == ()


def test_sampling_precedes_content_extraction(telemetry):
    provider = TracerProvider(sampler=ALWAYS_OFF)

    class Payload:
        def model_dump(self, **kwargs):
            raise AssertionError("payload should not be extracted")

    assert isinstance(
        run_app(
            BurrLifecycleAdapter(tracer=provider.get_tracer("dropped")), value=Payload()
        )[2]["result"],
        Payload,
    )
    provider.shutdown()


@pytest.mark.parametrize("fault", ["start", "input", "status", "output", "end"])
def test_telemetry_fault_preserves_native_results_and_context(
    telemetry, monkeypatch, fault
):
    provider, _ = telemetry
    tracer = provider.get_tracer("faulted")
    original = tracer.start_span

    def fail(*args, **kwargs):
        raise RuntimeError("controlled-telemetry-fault")

    def start(*args, **kwargs):
        if fault == "start":
            return fail()
        span = original(*args, **kwargs)
        if fault == "status":
            monkeypatch.setattr(span, "set_status", fail)
        elif fault == "end":
            monkeypatch.setattr(span, "end", fail)
        elif fault in {"input", "output"}:
            set_attribute = span.set_attribute

            def attribute(k, value):
                if k == (
                    AI.TRACELOOP_ENTITY_INPUT
                    if fault == "input"
                    else AI.TRACELOOP_ENTITY_OUTPUT
                ):
                    fail()
                return set_attribute(k, value)

            monkeypatch.setattr(span, "set_attribute", attribute)
        return span

    monkeypatch.setattr(tracer, "start_span", start)
    assert run_app(BurrLifecycleAdapter(tracer=tracer))[2]["result"] == SECRET
    assert not trace.get_current_span().get_span_context().is_valid


def test_telemetry_event_fault_preserves_stream(telemetry, monkeypatch):
    provider, _ = telemetry
    tracer = provider.get_tracer("event-fault")
    original = tracer.start_span

    def start(*args, **kwargs):
        span = original(*args, **kwargs)

        def fail(*args, **kwargs):
            raise RuntimeError("controlled-telemetry-fault")

        monkeypatch.setattr(span, "add_event", fail)
        return span

    monkeypatch.setattr(tracer, "start_span", start)
    app = builder(streaming).with_hooks(BurrLifecycleAdapter(tracer=tracer)).build()
    _, stream = app.stream_result(halt_after=["streaming"])
    assert list(stream) == [{"delta": "first"}, {"delta": "second"}]
    assert stream.get()[1]["result"] == "firstsecond"


def test_async_stream_is_native_and_terminal_events_are_complete(telemetry):
    _, exporter = telemetry

    @streaming_action(reads=["value"], writes=["result"])
    async def stream_async(state: State):
        yield {"delta": "first"}, None
        await asyncio.sleep(0)
        yield {"result": SECRET}, state.update(result=SECRET)

    async def execute():
        app = await builder(stream_async).with_hooks(BurrLifecycleAdapter()).abuild()
        _, stream = await app.astream_result(halt_after=["stream_async"])
        assert type(stream).__module__ == "burr.core.action"
        assert [item async for item in stream] == [{"delta": "first"}]
        return await stream.get()

    assert asyncio.run(execute())[1]["result"] == SECRET
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    task = next(s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "task")
    assert [event.name for event in task.events] == [
        "burr.stream.start",
        "burr.stream.item",
        "burr.stream.end",
    ]


def test_stream_error_retains_native_exception_identity(telemetry):
    _, exporter = telemetry
    error = RuntimeError(SECRET)

    @streaming_action(reads=["value"], writes=["result"])
    def failing_stream(state: State):
        yield {"delta": "first"}, None
        raise error

    app = builder(failing_stream).with_hooks(BurrLifecycleAdapter()).build()
    _, stream = app.stream_result(halt_after=["failing_stream"])
    with pytest.raises(RuntimeError) as caught:
        list(stream)
    assert caught.value is error
    assert all(
        s.status.status_code is StatusCode.ERROR for s in exporter.get_finished_spans()
    )
    assert not trace.get_current_span().get_span_context().is_valid


def test_late_provider_registers_policy_before_capture(telemetry, monkeypatch):
    from opentelemetry.trace import ProxyTracerProvider

    monkeypatch.setattr(trace, "_TRACER_PROVIDER", ProxyTracerProvider())
    adapter = BurrLifecycleAdapter()
    provider = TracerProvider()
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    try:
        run_app(adapter)
        assert len(exporter.get_finished_spans()) == 2
        assert contains_secret(exporter.get_finished_spans())
    finally:
        provider.shutdown()


def test_late_provider_unobserved_private_parent_cannot_widen(telemetry, monkeypatch):
    from opentelemetry.trace import ProxyTracerProvider

    monkeypatch.setattr(trace, "_TRACER_PROVIDER", ProxyTracerProvider())
    adapter = BurrLifecycleAdapter()
    provider = TracerProvider()
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    try:
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        with provider.get_tracer("preexisting").start_as_current_span("private"):
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
            run_app(adapter)
        assert not contains_secret(exporter.get_finished_spans())
    finally:
        provider.shutdown()


def test_parent_end_veto_is_retained_after_finished_context(telemetry):
    provider, exporter = telemetry
    adapter = BurrLifecycleAdapter()
    parent = provider.get_tracer("observed").start_span("parent")
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        parent.end()
    finally:
        context.detach(token)
    with trace.use_span(parent):
        run_app(adapter)
    assert not contains_secret(exporter.get_finished_spans())


def test_record_exception_fault_preserves_native_error_and_cleans_context(
    telemetry, monkeypatch
):
    provider, _ = telemetry
    error = RuntimeError(SECRET)

    @action(reads=["value"], writes=[])
    def native_error(state: State):
        raise error

    tracer = provider.get_tracer("exception-fault")
    original = tracer.start_span

    def start(*args, **kwargs):
        span = original(*args, **kwargs)

        def fail(*args, **kwargs):
            raise RuntimeError("controlled-telemetry-fault")

        monkeypatch.setattr(span, "record_exception", fail)
        return span

    monkeypatch.setattr(tracer, "start_span", start)
    with pytest.raises(RuntimeError) as caught:
        run_app(BurrLifecycleAdapter(tracer=tracer), native_error)
    assert caught.value is error
    assert not trace.get_current_span().get_span_context().is_valid


def test_concurrent_async_errors_remain_context_local(telemetry):
    _, exporter = telemetry
    error = RuntimeError(SECRET)

    @action(reads=["value"], writes=["result"])
    async def concurrent(state: State, fail: bool):
        await asyncio.sleep(0)
        if fail:
            raise error
        return state.update(result=42)

    async def execute():
        adapter = BurrLifecycleAdapter()
        failed = builder(concurrent).with_hooks(adapter).build()
        success = builder(concurrent).with_hooks(adapter).build()
        return await asyncio.gather(
            failed.arun(halt_after=["concurrent"], inputs={"fail": True}),
            success.arun(halt_after=["concurrent"], inputs={"fail": False}),
            return_exceptions=True,
        )

    result = asyncio.run(execute())
    assert result[0] is error and result[1][2]["result"] == 42
    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    assert [s.status.status_code for s in spans].count(StatusCode.ERROR) == 2
    assert [s.status.status_code for s in spans].count(StatusCode.OK) == 2


def test_successful_custom_span_inside_exception_handler_has_no_false_error(telemetry):
    _, exporter = telemetry

    @action(reads=["value"], writes=["result"])
    def handled(state: State, __tracer: TracerFactory):
        try:
            raise RuntimeError(SECRET)
        except RuntimeError:
            with __tracer("handled-custom"):
                pass
        return state.update(result=42)

    owner = BurrInstrumentor()
    owner.activate()
    try:
        assert builder(handled).build().run(halt_after=["handled"])[2]["result"] == 42
    finally:
        owner.deactivate()
    assert all(
        s.status.status_code is StatusCode.OK for s in exporter.get_finished_spans()
    )
    assert not any(s.events for s in exporter.get_finished_spans())


@pytest.mark.parametrize("remote", [True, False])
def test_unknown_parent_carriers_have_conservative_local_bounds(telemetry, remote):
    _, exporter = telemetry
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    parent = NonRecordingSpan(
        SpanContext(
            trace_id=123, span_id=456, is_remote=remote, trace_flags=TraceFlags(1)
        )
    )
    adapter = BurrLifecycleAdapter()
    with trace.use_span(parent):
        run_app(adapter)
    assert contains_secret(exporter.get_finished_spans()) is remote
    assert all(s.context.trace_id == 123 for s in exporter.get_finished_spans())


def test_unknown_mapping_and_serializer_protocols_are_not_invoked(telemetry):
    from collections.abc import Mapping

    calls = []

    class HostileMapping(Mapping):
        def __iter__(self):
            calls.append("iterate")
            raise AssertionError("hostile iterate")

        def __len__(self):
            calls.append("length")
            raise AssertionError("hostile length")

        def __getitem__(self, key):
            calls.append("lookup")
            raise AssertionError("hostile lookup")

        def items(self):
            calls.append("items")
            raise AssertionError("hostile items")

    class HostileDumper:
        def model_dump(self, **kwargs):
            calls.append("model_dump")
            raise AssertionError("hostile dumper")

        def get_all(self):
            calls.append("get_all")
            raise AssertionError("hostile state protocol")

        def __repr__(self):
            calls.append("repr")
            raise AssertionError("hostile repr")

    payload = {"mapping": HostileMapping(), "dumper": HostileDumper()}
    assert run_app(BurrLifecycleAdapter(), value=payload)[2]["result"] is payload
    assert calls == []
    _, exporter = telemetry
    task = next(
        s
        for s in exporter.get_finished_spans()
        if s.attributes[RESPAN_LOG_TYPE] == "task"
    )
    assert json.loads(task.attributes[AI.TRACELOOP_ENTITY_OUTPUT])["state"][
        "result"
    ] == {"mapping": {"type": "HostileMapping"}, "dumper": {"type": "HostileDumper"}}


def test_credentials_are_redacted_while_schema_fields_remain_complete(telemetry):
    _, exporter = telemetry
    schema = {
        "type": "object",
        "properties": {
            "api_key": {"type": "string", "description": "API key argument"},
            "password": {"type": "string"},
        },
        "required": ["api_key"],
    }
    payload = {
        "Authorization": "controlled-credential-value",
        "credentials": {"secret": "controlled-credential-value"},
        "api_key": "controlled-credential-value",
        "tool": {"parameters": schema},
        "dense": [0.1, 0.2],
    }
    assert run_app(BurrLifecycleAdapter(), value=payload)[2]["result"] is payload
    spans = exporter.get_finished_spans()
    assert "controlled-credential-value" not in str([dict(s.attributes) for s in spans])
    task = next(s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "task")
    output = json.loads(task.attributes[AI.TRACELOOP_ENTITY_OUTPUT])["state"]["result"]
    assert output["tool"]["parameters"] == schema
    assert output["api_key"] == "[REDACTED]"
    assert output["dense"] == [0.1, 0.2]
