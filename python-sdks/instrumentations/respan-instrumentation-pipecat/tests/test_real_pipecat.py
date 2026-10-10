import json

import pytest
from _fixtures import run
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes as TL
from opentelemetry.trace import StatusCode
from respan_instrumentation_pipecat import PipecatInstrumentor
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


@pytest.fixture
def telemetry():
    p = TracerProvider()
    e = InMemorySpanExporter()
    p.add_span_processor(SimpleSpanProcessor(e))
    i = PipecatInstrumentor(tracer_provider=p)
    i.activate()
    yield p, e, i
    i.deactivate()
    p.shutdown()


@pytest.mark.asyncio
async def test_actual_values_usage_and_parent_context(telemetry):
    p, e, _i = telemetry
    t = p.get_tracer("fixture")
    with t.start_as_current_span("root"):
        collector, _worker = await run()
    assert any(
        getattr(f, "text", None) == "Actual native text." for f in collector.frames
    )
    spans = e.get_finished_spans()
    assert len(spans) == 3
    llm = next(s for s in spans if s.name == "pipecat.llm")
    a = llm.attributes
    assert (
        a[TL.LLM_USAGE_PROMPT_TOKENS] == 11
        and a[TL.LLM_USAGE_COMPLETION_TOKENS] == 7
        and a[TL.LLM_USAGE_TOTAL_TOKENS] == 23
    )
    assert (
        a[TL.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 4
        and a[TL.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 5
    )
    assert a[TL.LLM_USAGE_REASONING_TOKENS] == 2
    assert (
        json.loads(a[TL.TRACELOOP_ENTITY_INPUT])[0]["content"]
        == "Actual native prompt."
    )
    assert json.loads(a[TL.TRACELOOP_ENTITY_OUTPUT]) == "Actual native text."
    assert TL.TRACELOOP_SPAN_KIND not in a
    byid = {s.context.span_id: s for s in spans}
    for s in spans:
        if s.parent:
            parent = byid[s.parent.span_id]
            assert parent.start_time <= s.start_time <= s.end_time <= parent.end_time
    assert trace.get_current_span().get_span_context() == trace.INVALID_SPAN_CONTEXT


@pytest.mark.asyncio
async def test_native_status_error_without_synthetic_output(telemetry):
    _p, e, _i = telemetry
    await run(fail=True)
    spans = e.get_finished_spans()
    assert len(spans) == 2
    assert all(s.status.status_code.name == "ERROR" for s in spans)
    assert all(s.attributes["http.response.status_code"] == 401 for s in spans)
    assert all(
        "status_code" not in s.attributes
        and TL.TRACELOOP_ENTITY_OUTPUT not in s.attributes
        and TL.LLM_USAGE_PROMPT_TOKENS not in s.attributes
        for s in spans
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["constructor", "env", "context", "mid-context", "mid-env"]
)
async def test_content_initial_and_end_bounds(telemetry, monkeypatch, mode):
    p, e, i = telemetry
    veto = None
    if mode == "constructor":
        i.deactivate()
        i = PipecatInstrumentor(tracer_provider=p, capture_content=False)
        i.activate()
    if mode == "env":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    token = None
    if mode == "context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    if mode == "mid-context":
        veto = lambda: context.attach(
            context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
        )
    if mode == "mid-env":
        veto = lambda: monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    try:
        await run(veto=veto)
    finally:
        if token is not None:
            context.detach(token)
        i.deactivate()
    spans = e.get_finished_spans()
    assert len(spans) == 2
    assert all(
        not any(
            k in s.attributes
            for k in [TL.TRACELOOP_ENTITY_INPUT, TL.TRACELOOP_ENTITY_OUTPUT]
        )
        for s in spans
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
async def test_native_suppression(telemetry, key):
    _p, e, _i = telemetry
    token = context.attach(context.set_value(key, True))
    try:
        await run()
    finally:
        context.detach(token)
    assert not e.get_finished_spans()


@pytest.mark.asyncio
async def test_sampler_drops_native_spans():
    p = TracerProvider(sampler=ALWAYS_OFF)
    e = InMemorySpanExporter()
    p.add_span_processor(SimpleSpanProcessor(e))
    i = PipecatInstrumentor(tracer_provider=p)
    i.activate()
    try:
        await run()
    finally:
        i.deactivate()
        p.shutdown()
    assert not e.get_finished_spans()


@pytest.mark.asyncio
async def test_complete_actual_tool_vector_and_schema(telemetry):
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    _p, e, _i = telemetry
    properties = {"field" + str(n): {"type": "string"} for n in range(120)}
    properties["api_key"] = {"type": "string", "default": "synthetic secret"}
    tools = ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="vector_tool",
                description="Actual schema",
                properties=properties,
                required=[],
            )
        ]
    )
    await run(
        tool=True,
        messages=[{"role": "user", "content": "history" + str(n)} for n in range(150)],
        tools=tools,
    )
    spans = e.get_finished_spans()
    assert len(spans) == 3
    llm = next(s for s in spans if s.name == "pipecat.llm")
    tool = next(s for s in spans if ".tool." in s.name)
    assert len(json.loads(llm.attributes[TL.TRACELOOP_ENTITY_INPUT])) == 150
    definitions = json.loads(llm.attributes[TL.LLM_REQUEST_FUNCTIONS])
    assert len(definitions[0]["function"]["parameters"]["properties"]) == 121
    assert (
        definitions[0]["function"]["parameters"]["properties"]["api_key"]["type"]
        == "string"
    )
    assert "synthetic secret" not in llm.attributes[TL.LLM_REQUEST_FUNCTIONS]
    assert tool.attributes["gen_ai.tool.call.id"] == "current-call"
    calls = json.loads(llm.attributes[TL.LLM_COMPLETIONS + ".0.tool_calls"])
    assert (
        calls[0]["id"] == "current-call"
        and len(json.loads(calls[0]["function"]["arguments"])["values"]) == 120
    )
    output = json.loads(tool.attributes[TL.TRACELOOP_ENTITY_OUTPUT])
    assert len(output["dense"]) == 5000 and len(output["sparse"]) == 256
    assert (
        len(
            json.loads(tool.attributes[TL.TRACELOOP_ENTITY_INPUT])["arguments"][
                "values"
            ]
        )
        == 120
    )


@pytest.mark.asyncio
async def test_native_static_config_input_output_masks(provider=None):
    from openinference.instrumentation import TraceConfig

    p = TracerProvider()
    e = InMemorySpanExporter()
    p.add_span_processor(SimpleSpanProcessor(e))
    i = PipecatInstrumentor(
        tracer_provider=p, config=TraceConfig(hide_inputs=True, hide_outputs=True)
    )
    i.activate()
    try:
        collector, _worker = await run(tool=True)
    finally:
        i.deactivate()
        p.shutdown()
    assert any(
        getattr(f, "text", None) == "Actual native text." for f in collector.frames
    )
    assert len(e.get_finished_spans()) == 3
    assert all(
        not any(
            k in s.attributes
            for k in [
                TL.TRACELOOP_ENTITY_INPUT,
                TL.TRACELOOP_ENTITY_OUTPUT,
                TL.LLM_REQUEST_FUNCTIONS,
            ]
        )
        for s in e.get_finished_spans()
    )


@pytest.mark.asyncio
async def test_delayed_finished_private_parent(telemetry):
    p, e, _i = telemetry
    t = p.get_tracer("fixture")
    started = __import__("asyncio").Event()
    resume = __import__("asyncio").Event()

    async def child():
        started.set()
        await resume.wait()
        return await run()

    with t.start_as_current_span("manual.root"):
        task = __import__("asyncio").create_task(child())
        await started.wait()
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    # Deliberately retain the false value until native use_span detach observed it.
    resume.set()
    await task
    owned = [s for s in e.get_finished_spans() if s.name != "manual.root"]
    assert len(owned) == 2
    assert all(
        not any(
            k in s.attributes
            for k in [TL.TRACELOOP_ENTITY_INPUT, TL.TRACELOOP_ENTITY_OUTPUT]
        )
        for s in owned
    )


@pytest.mark.asyncio
async def test_actual_released_openai_service_http_sse(telemetry):
    from _http_service import ProviderService

    _p, e, _i = telemetry
    service = ProviderService(api_key="fixture", model="fixture-native-model")
    try:
        collector, _worker = await run(service=service)
    finally:
        await service._client.close()
    assert any(
        getattr(f, "text", None) == "Actual HTTP text." for f in collector.frames
    )
    span = next(s for s in e.get_finished_spans() if s.name == "pipecat.llm")
    assert (
        span.attributes[TL.LLM_USAGE_PROMPT_TOKENS] == 11
        and span.attributes[TL.LLM_USAGE_COMPLETION_TOKENS] == 7
        and span.attributes[TL.LLM_USAGE_TOTAL_TOKENS] == 18
    )
    assert (
        span.attributes[TL.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 4
        and span.attributes[TL.LLM_USAGE_REASONING_TOKENS] == 2
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["stt", "tts"])
async def test_native_voice_service_frames(telemetry, kind):
    from _voice_service import STT, TTS

    _p, e, _i = telemetry
    collector, _worker = await run(service=STT() if kind == "stt" else TTS())
    voice = next(s for s in e.get_finished_spans() if s.name == "pipecat." + kind)
    assert voice.attributes["respan.entity.log_type"] == (
        "transcription" if kind == "stt" else "speech"
    )
    assert not any(
        k.startswith(("gen_ai.usage.", "llm.usage.")) for k in voice.attributes
    )
    assert any(
        type(f).__name__ == ("TranscriptionFrame" if kind == "stt" else "TTSTextFrame")
        for f in collector.frames
    )


@pytest.mark.asyncio
async def test_actual_argument_json_redaction_fidelity(telemetry):
    from _fixtures import Service

    _p, e, _i = telemetry
    arguments = {
        "note": "Bearer synthetic-token",
        "label": "okay",
        "api_key": 'synthetic secret with "quotes"',
    }
    collector, _worker = await run(service=Service(tool=True, tool_args=arguments))
    native = next(
        f for f in collector.frames if type(f).__name__ == "FunctionCallResultFrame"
    )
    assert native.arguments == arguments
    span = next(s for s in e.get_finished_spans() if s.name == "pipecat.llm")
    calls = json.loads(span.attributes[TL.LLM_COMPLETIONS + ".0.tool_calls"])
    parsed = json.loads(calls[0]["function"]["arguments"])
    assert (
        parsed["label"] == "okay"
        and parsed["note"] == "Bearer <redacted>"
        and parsed["api_key"] == "<redacted>"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["before", "during"])
async def test_native_pipeline_survives_deactivation(telemetry, when):
    _p, e, i = telemetry
    collector, _worker = await run(
        setup=(lambda worker: i.deactivate()) if when == "before" else None,
        veto=i.deactivate if when == "during" else None,
    )
    assert any(
        getattr(f, "text", None) == "Actual native text." for f in collector.frames
    )
    spans = e.get_finished_spans()
    assert len(spans) == (0 if when == "before" else 2)
    assert all("openinference.span.kind" not in s.attributes for s in spans)


@pytest.mark.asyncio
async def test_preexisting_parent_veto_before_first_native_span():
    p = TracerProvider()
    e = InMemorySpanExporter()
    p.add_span_processor(SimpleSpanProcessor(e))
    tracer = p.get_tracer("fixture")
    with tracer.start_as_current_span("preexisting.root"):
        i = PipecatInstrumentor(tracer_provider=p)
        i.activate()
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        context.detach(token)
        try:
            await run()
        finally:
            i.deactivate()
    p.shutdown()
    assert all(
        not any(
            k in s.attributes
            for k in [TL.TRACELOOP_ENTITY_INPUT, TL.TRACELOOP_ENTITY_OUTPUT]
        )
        for s in e.get_finished_spans()
        if s.name.startswith("pipecat.")
    )


@pytest.mark.asyncio
async def test_raw_provider_counts_before_native_dto_coercion(telemetry):
    from _http_service import ProviderService

    _p, e, _i = telemetry
    service = ProviderService(
        api_key="fixture",
        settings=ProviderService.Settings(model="fixture-native-model"),
        raw_usage={
            "prompt_tokens": True,
            "completion_tokens": False,
            "total_tokens": 12,
        },
    )
    try:
        collector, _worker = await run(service=service)
    finally:
        await service._client.close()
    assert any(
        getattr(f, "text", None) == "Actual HTTP text." for f in collector.frames
    )
    span = next(s for s in e.get_finished_spans() if s.name == "pipecat.llm")
    assert (
        TL.LLM_USAGE_PROMPT_TOKENS not in span.attributes
        and TL.LLM_USAGE_COMPLETION_TOKENS not in span.attributes
    )
    assert span.attributes[TL.LLM_USAGE_TOTAL_TOKENS] == 12


@pytest.mark.asyncio
async def test_actual_prompt_json_bearer_redaction_preserves_history(telemetry):
    _p, e, _i = telemetry
    await run(
        messages=[{"role": "user", "content": 'Bearer fixture-token with "quotes"'}]
    )
    span = next(s for s in e.get_finished_spans() if s.name == "pipecat.llm")
    values = json.loads(span.attributes[TL.TRACELOOP_ENTITY_INPUT])
    assert (
        isinstance(values, list)
        and values[0]["content"] == 'Bearer <redacted> with "quotes"'
    )


@pytest.mark.asyncio
async def test_native_cancel_frame_and_partial_text_preserved(telemetry):
    from _fixtures import Service

    _, exporter, _ = telemetry
    collector, _ = await run(service=Service(cancel=True))
    assert any(type(f).__name__ == "CancelFrame" for f in collector.frames)
    assert any(
        getattr(f, "text", None) == "Actual native text." for f in collector.frames
    )
    spans = exporter.get_finished_spans()
    assert {s.name for s in spans} == {"pipecat.conversation.turn", "pipecat.llm"}
    assert not any("http.response.status_code" in s.attributes for s in spans)
    assert not any(TL.LLM_USAGE_PROMPT_TOKENS in s.attributes for s in spans)


@pytest.mark.asyncio
async def test_actual_partial_output_survives_native_error(telemetry):
    from _fixtures import Service

    _, exporter, _ = telemetry
    collector, _ = await run(service=Service(fail=True, partial=True))
    assert any(
        getattr(f, "text", None) == "Actual partial text." for f in collector.frames
    )
    span = next(s for s in exporter.get_finished_spans() if s.name == "pipecat.llm")
    assert span.status.status_code is StatusCode.ERROR
    assert "Actual partial text." in span.attributes[TL.TRACELOOP_ENTITY_OUTPUT]


@pytest.mark.asyncio
async def test_actual_proxy_provider_rejected_before_native_hooks():
    import inspect

    from _fixtures import PipelineWorker
    from openinference.instrumentation.pipecat._observer import OpenInferenceObserver
    from respan_instrumentation_pipecat import _instrumentation as implementation

    original = inspect.getattr_static(PipelineWorker, "__init__")
    observer = inspect.getattr_static(OpenInferenceObserver, "on_push_frame")
    instrumentor = PipecatInstrumentor(tracer_provider=trace.ProxyTracerProvider())
    instrumentor.activate()
    assert not instrumentor._is_instrumented and implementation._RUNTIME is None
    assert inspect.getattr_static(PipelineWorker, "__init__") is original
    assert inspect.getattr_static(OpenInferenceObserver, "on_push_frame") is observer
    collector, _ = await run()
    assert any(
        getattr(f, "text", None) == "Actual native text." for f in collector.frames
    )
    instrumentor.deactivate()


@pytest.mark.asyncio
async def test_preexisting_private_parent_initial_bound_is_unobserved():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = PipecatInstrumentor(tracer_provider=provider)
    private = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        with provider.get_tracer("fixture").start_as_current_span(
            "preexisting.private"
        ):
            instrumentor.activate()
            permitted = context.attach(
                context.set_value(ENABLE_CONTENT_TRACING_KEY, True)
            )
            try:
                collector, _ = await run()
            finally:
                context.detach(permitted)
                instrumentor.deactivate()
    finally:
        context.detach(private)
        provider.shutdown()
    assert any(
        getattr(f, "text", None) == "Actual native text." for f in collector.frames
    )
    native = [s for s in exporter.get_finished_spans() if s.name.startswith("pipecat.")]
    assert len(native) == 2
    assert all(
        TL.TRACELOOP_ENTITY_INPUT not in s.attributes
        and TL.TRACELOOP_ENTITY_OUTPUT not in s.attributes
        for s in native
    )


@pytest.mark.asyncio
async def test_ambient_private_initial_bound_with_permitted_supplied_parent(telemetry):
    provider, exporter, _ = telemetry
    private = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        supplied = context.set_value(ENABLE_CONTENT_TRACING_KEY, True)
        parent = provider.get_tracer("fixture").start_span(
            "ambient.private", context=supplied
        )
        permit = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, True))
        try:
            with trace.use_span(parent, end_on_exit=True):
                collector, _ = await run()
        finally:
            context.detach(permit)
    finally:
        context.detach(private)
    assert any(
        getattr(f, "text", None) == "Actual native text." for f in collector.frames
    )
    native = [s for s in exporter.get_finished_spans() if s.name.startswith("pipecat.")]
    assert len(native) == 2
    assert all(
        TL.TRACELOOP_ENTITY_INPUT not in s.attributes
        and TL.TRACELOOP_ENTITY_OUTPUT not in s.attributes
        for s in native
    )
