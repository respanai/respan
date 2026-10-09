"""Real SDK clients, protobuf responses and local gRPC transport frames."""

import asyncio
import inspect
import json
import types

import pytest
from google.api_core.exceptions import ServiceUnavailable
from opentelemetry import context, trace
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from respan_instrumentation_vertexai import VertexAIInstrumentor
from respan_instrumentation_vertexai import _instrumentation as adapter
from respan_instrumentation_vertexai._serialization import json_dumps, safe_text
from respan_instrumentation_vertexai._translator import native_value
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from vertexai.generative_models import FunctionDeclaration, Part, Tool

from tests._native import NativeRuntime

INPUT = SpanAttributes.TRACELOOP_ENTITY_INPUT
OUTPUT = SpanAttributes.TRACELOOP_ENTITY_OUTPUT


@pytest.fixture
def runtime():
    r = NativeRuntime()
    p = TracerProvider()
    e = InMemorySpanExporter()
    p.add_span_processor(SimpleSpanProcessor(e))
    inst = VertexAIInstrumentor(tracer_provider=p)
    inst.activate()
    yield r, p, e, inst
    inst.deactivate()
    r.close()
    p.shutdown()


def test_native_generation_response_parent_and_sourced_usage(runtime):
    r, p, e, _ = runtime
    model = r.model()
    with p.get_tracer("parent").start_as_current_span("outer") as parent:
        response = model.generate_content("hello")
    assert response.text == "native response" and len(r.requests) == 1
    span = e.get_finished_spans()[0]
    assert span.parent.span_id == parent.context.span_id
    assert (
        json.loads(span.attributes[OUTPUT])[0]["candidates"][0]["content"]["parts"][0][
            "text"
        ]
        == "native response"
    )
    assert span.attributes["gen_ai.usage.input_tokens"] == 7
    assert span.attributes["gen_ai.usage.output_tokens"] == 3
    assert span.attributes["llm.usage.total_tokens"] == 10
    assert "status_code" not in span.attributes
    assert "traceloop.span.kind" not in span.attributes


def test_native_stream_preserves_generator_and_all_seventy_chunks(runtime):
    r, _, e, _ = runtime
    source = r.model().generate_content("stream", stream=True)
    assert iter(source) is source and type(source.source) is types.GeneratorType
    chunks = list(source)
    assert len(chunks) == 70 and len(r.requests) == 1
    span = e.get_finished_spans()[0]
    assert span.attributes["gen_ai.completion.0.content"] == "".join(
        c.text for c in chunks
    )
    assert len(json.loads(span.attributes[OUTPUT])) == 70
    assert span.attributes["gen_ai.usage.input_tokens"] == 7


def test_native_early_close_finishes_span_and_native_generator(runtime):
    r, _, e, _ = runtime
    source = r.model().generate_content("stream", stream=True)
    assert next(source).text == "0,"
    source.close()
    assert inspect.getgeneratorstate(source) == "GEN_CLOSED"
    assert len(e.get_finished_spans()) == 1
    assert e.get_finished_spans()[0].attributes["gen_ai.completion.0.content"] == "0,"


def test_native_function_calls_and_full_tool_schema_history(runtime):
    r, _, e, _ = runtime
    declaration = FunctionDeclaration(
        name="get_weather",
        description="lookup",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "api_key": {"type": "string", "description": "credential parameter"},
            },
            "required": ["city"],
        },
    )
    model = r.model()
    chat = model.start_chat()
    first = chat.send_message(
        "weather", tools=[Tool(function_declarations=[declaration])]
    )
    assert first.candidates[0].content.parts[0].function_call.name == "get_weather"
    second = chat.send_message(
        Part.from_function_response(name="get_weather", response={"weather": "sunny"})
    )
    assert second.text == "native response" and len(r.requests) == 2
    spans = e.get_finished_spans()
    first_attrs = spans[0].attributes
    calls = json.loads(first_attrs["gen_ai.completion.0.tool_calls"])
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Tokyo"}
    schema = json.loads(first_attrs["llm.request.functions"])[0]["function"][
        "parameters"
    ]
    assert schema["properties"]["api_key"]["type"] == "string"
    history = json.loads(spans[1].attributes[INPUT])["messages"]
    assert history[0]["content"] == "weather"
    assert history[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert history[2]["role"] == "tool"
    assert "gen_ai.completion.0.tool_calls" not in spans[1].attributes


def test_native_embeddings_preserve_full_5001_vectors_and_only_source_usage(runtime):
    r, _, e, _ = runtime
    response = r.embedding().get_embeddings(["first", "second"])
    assert len(response) == 2 and all(len(v.values) == 5001 for v in response)
    span = e.get_finished_spans()[0]
    assert json.loads(span.attributes[OUTPUT]) == [v.values for v in response]
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert "gen_ai.usage.output_tokens" not in span.attributes
    assert "llm.usage.total_tokens" not in span.attributes


def test_native_async_generation_stream_and_close(runtime):
    r, _, e, _ = runtime

    async def run():
        model, channel = await r.async_model()
        try:
            response = await model.generate_content_async("hello")
            assert response.text == "native response"
            source = await model.generate_content_async("stream", stream=True)
            assert (
                source.__aiter__() is source
                and type(source.source) is types.AsyncGeneratorType
            )
            chunks = [c async for c in source]
            assert len(chunks) == 70
            early = await model.generate_content_async("stream", stream=True)
            assert (await early.__anext__()).text == "0,"
            await early.aclose()
        finally:
            await channel.close()

    asyncio.run(run())
    spans = e.get_finished_spans()
    assert len(spans) == 3
    assert "69," in spans[1].attributes["gen_ai.completion.0.content"]


def test_native_provider_error_identity_and_no_invented_result(runtime):
    r, _, e, _ = runtime
    with pytest.raises(ServiceUnavailable) as caught:
        r.model().generate_content("failure")
    span = e.get_finished_spans()[0]
    assert span.status.status_code is trace.StatusCode.ERROR
    assert span.attributes[ERROR_TYPE] == type(caught.value).__name__
    assert span.attributes["http.response.status_code"] == 503
    assert OUTPUT not in span.attributes


@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_suppression_precedes_native_payload_inspection(runtime, monkeypatch, key):
    r, _, e, _ = runtime
    monkeypatch.setattr(
        adapter, "request_payload_from_call", lambda **kw: pytest.fail("inspected")
    )
    token = context.attach(context.set_value(key, True))
    try:
        assert r.model().generate_content("hello").text == "native response"
    finally:
        context.detach(token)
    assert not e.get_finished_spans()


def test_sampling_precedes_native_payload_inspection(runtime, monkeypatch):
    r, p, e, _ = runtime
    p.sampler = ALWAYS_OFF
    monkeypatch.setattr(
        adapter, "request_payload_from_call", lambda **kw: pytest.fail("inspected")
    )
    source = r.model().generate_content("stream", stream=True)
    assert type(source) is types.GeneratorType and len(list(source)) == 70
    assert not e.get_finished_spans()


@pytest.mark.parametrize("veto", ["context", "environment", "capture"])
def test_initial_privacy_skips_body_and_native_diagnostics(runtime, monkeypatch, veto):
    r, _p, e, inst = runtime
    if veto == "capture":
        inst.deactivate()
        inst._capture_content = False
        inst.activate()
    if veto == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    token = (
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        if veto == "context"
        else None
    )
    monkeypatch.setattr(
        adapter, "request_payload_from_call", lambda **kw: pytest.fail("inspected")
    )
    try:
        with pytest.raises(ServiceUnavailable):
            r.model().generate_content("failure")
    finally:
        if token is not None:
            context.detach(token)
    span = e.get_finished_spans()[0]
    assert (
        INPUT not in span.attributes
        and OUTPUT not in span.attributes
        and ERROR_MESSAGE not in span.attributes
    )
    assert span.status.description is None and not span.events


def test_stream_late_privacy_clears_previous_content_and_retention(runtime):
    r, _, e, _ = runtime
    source = r.model().generate_content("stream", stream=True)
    assert next(source).text == "0,"
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        assert next(source).text == "1,"
    finally:
        context.detach(token)
    list(source)
    span = e.get_finished_spans()[0]
    assert INPUT not in span.attributes and OUTPUT not in span.attributes
    assert not span.events and span.status.description is None


@pytest.mark.parametrize("finished", [False, True])
def test_observed_parent_initial_veto_survives_context_restore(runtime, finished):
    r, p, e, _ = runtime
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    parent = p.get_tracer("test").start_span("outer")
    context.detach(token)
    if finished:
        parent.end()
    with trace.use_span(parent):
        r.model().generate_content("hello")
    if not finished:
        parent.end()
    child = next(
        s for s in e.get_finished_spans() if s.name == "vertexai.generate_content"
    )
    assert INPUT not in child.attributes and OUTPUT not in child.attributes


def test_active_child_veto_is_irreversible_for_parent_stream_and_sibling(runtime):
    r, p, e, _ = runtime
    with p.get_tracer("test").start_as_current_span("parent"):
        source = r.model().generate_content("stream", stream=True)
        next(source)
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        try:
            r.model().generate_content("hello")
        finally:
            context.detach(token)
        list(source)
        r.model().generate_content("hello")
    for s in e.get_finished_spans()[:-1]:
        assert INPUT not in s.attributes and OUTPUT not in s.attributes


def test_late_provider_unknown_recording_and_finished_carrier_closed(runtime):
    r, _, _, inst = runtime
    inst.deactivate()
    p = TracerProvider()
    e = InMemorySpanExporter()
    p.add_span_processor(SimpleSpanProcessor(e))
    recording = p.get_tracer("test").start_span("unknown")
    finished = p.get_tracer("test").start_span("finished")
    finished.end()
    other = VertexAIInstrumentor(tracer_provider=p)
    other.activate()
    try:
        for parent in [recording, trace.NonRecordingSpan(finished.context)]:
            with trace.use_span(parent):
                r.model().generate_content("hello")
    finally:
        other.deactivate()
        recording.end()
        p.shutdown()
    assert all(
        INPUT not in s.attributes
        for s in e.get_finished_spans()
        if s.name == "vertexai.generate_content"
    )


@pytest.mark.parametrize(
    "fault", ["start", "request", "attribute", "output", "end", "detach"]
)
def test_telemetry_faults_preserve_actual_native_result_and_stream_cleanup(
    runtime, monkeypatch, fault
):
    r, p, _e, _ = runtime

    def fail(*a, **k):
        raise RuntimeError("telemetry fault")

    if fault == "start":
        monkeypatch.setattr(p.get_tracer("vertexai"), "start_span", fail)
    elif fault == "request":
        monkeypatch.setattr(adapter, "request_payload_from_call", fail)
    elif fault == "output":
        monkeypatch.setattr(adapter, "response_attributes", fail)
    elif fault == "detach":
        monkeypatch.setattr(adapter.context, "detach", fail)
    else:
        monkeypatch.setattr(
            Span, "set_attribute" if fault == "attribute" else "end", fail
        )
    model = r.model()
    response = model.generate_content("hello")
    assert response.text == "native response"
    source = model.generate_content("stream", stream=True)
    assert next(source).text == "0,"
    source.close()
    assert inspect.getgeneratorstate(source) == "GEN_CLOSED"


def test_shared_lifecycle_foreign_wrapper_and_configuration(runtime):
    _, p, _, first = runtime
    second = VertexAIInstrumentor(tracer_provider=p)
    second.activate()
    with pytest.raises(ValueError):
        VertexAIInstrumentor(capture_content=False, tracer_provider=p).activate()
    patch = adapter._PATCHES[0]
    wrapped = patch.wrapper

    @__import__("functools").wraps(wrapped)
    def foreign(*args, **kwargs):
        return wrapped(*args, **kwargs)

    setattr(patch.cls, patch.method_name, foreign)
    try:
        first.deactivate()
        assert adapter._ENABLED
        second.deactivate()
        assert inspect.getattr_static(patch.cls, patch.method_name) is foreign
    finally:
        setattr(patch.cls, patch.method_name, patch.original)


def test_partial_install_rollback_preserves_native_descriptors(runtime, monkeypatch):
    _, p, _, first = runtime
    first.deactivate()
    targets = adapter._load_targets()
    originals = [inspect.getattr_static(t[0], t[1]) for t in targets]
    monkeypatch.setattr(
        adapter,
        "_load_targets",
        lambda: [
            targets[0],
            (targets[0][0], "does_not_exist", "invalid", False, False),
        ],
    )
    with pytest.raises(AttributeError):
        VertexAIInstrumentor(tracer_provider=p).activate()
    assert all(
        inspect.getattr_static(t[0], t[1]) is orig
        for t, orig in zip(targets, originals)
    )
    assert not adapter._PATCHES


def test_complete_json_schema_history_vectors_and_unknown_hooks():
    class Unknown:
        def model_dump(self, *args, **kwargs):
            pytest.fail("customer hook")

        def __getattribute__(self, key):
            if key in {"to_dict", "model_dump", "__dict__"}:
                pytest.fail("customer hook")
            return object.__getattribute__(self, key)

    assert native_value(Unknown()) == {"type": "Unknown"}
    value = {
        "vector": list(range(5001)),
        "history": [{"role": "user", "content": "x" * 6000}] * 80,
        "arguments": '{"api_key":"two word secret","enabled":false,"count":0}',
        "schema": {"properties": {"api_key": {"type": "string", "default": "private"}}},
    }
    parsed = json.loads(json_dumps(value))
    assert len(parsed["vector"]) == 5001 and len(parsed["history"]) == 80
    args = json.loads(parsed["arguments"])
    assert args == {"api_key": "[REDACTED]", "enabled": False, "count": 0}
    assert parsed["schema"]["properties"]["api_key"] == {
        "type": "string",
        "default": "[REDACTED]",
    }
    cleaned = safe_text(
        'secret="two word secret" Bearer private Basic cHJpdmF0ZQ== https://user:pass@example.com/?token=private'
    )
    assert (
        "private" not in cleaned
        and "two word" not in cleaned
        and "user:pass" not in cleaned
    )


def test_native_stream_close_before_first_read_finishes_without_rpc(runtime):
    r, _, e, _ = runtime
    source = r.model().generate_content("stream", stream=True)
    source.close()
    assert not r.requests and source.gi_frame is None
    assert (
        len(e.get_finished_spans()) == 1
        and OUTPUT not in e.get_finished_spans()[0].attributes
    )


def test_native_async_stream_aclose_before_first_read_finishes_without_consumption(
    runtime,
):
    r, _, e, _ = runtime

    async def run():
        model, channel = await r.async_model()
        try:
            source = await model.generate_content_async("stream", stream=True)
            await source.aclose()
            assert source.ag_frame is None
        finally:
            await channel.close()

    asyncio.run(run())
    assert len(r.requests) <= 1 and len(e.get_finished_spans()) == 1
    assert OUTPUT not in e.get_finished_spans()[0].attributes


@pytest.mark.parametrize(
    "flag", ["override_enable_content_tracing", "enable_content_tracing"]
)
def test_traceloop_and_respan_context_vetoes_apply(runtime, flag):
    r, _, e, _ = runtime
    key = ENABLE_CONTENT_TRACING_KEY if flag == "enable_content_tracing" else flag
    token = context.attach(context.set_value(key, False))
    try:
        r.model().generate_content("hello")
    finally:
        context.detach(token)
    assert INPUT not in e.get_finished_spans()[0].attributes


def test_respan_environment_veto_applies(runtime, monkeypatch):
    r, _, e, _ = runtime
    monkeypatch.setenv("RESPAN_TRACE_CONTENT", "false")
    r.model().generate_content("hello")
    assert INPUT not in e.get_finished_spans()[0].attributes


@pytest.mark.parametrize("finished", [False, True])
def test_active_and_finished_ancestor_attribute_veto_is_irreversible(runtime, finished):
    r, p, e, _ = runtime
    with p.get_tracer("test").start_as_current_span("parent") as parent:
        parent.set_attribute("traceloop.enable_content_tracing", False)
        if finished:
            parent.end()
        r.model().generate_content("hello")
        parent.set_attribute("traceloop.enable_content_tracing", True)
        r.model().generate_content("hello")
    assert all(
        INPUT not in s.attributes
        for s in e.get_finished_spans()
        if s.name == "vertexai.generate_content"
    )


def test_detach_and_runtime_fault_restore_exact_native_context(runtime, monkeypatch):
    r, _p, _e, _ = runtime
    ambient = context.get_current()

    def fail(*args, **kwargs):
        raise RuntimeError("telemetry context fault")

    monkeypatch.setattr(context, "detach", fail)
    monkeypatch.setattr(context._RUNTIME_CONTEXT, "detach", fail)
    assert r.model().generate_content("hello").text == "native response"
    assert context.get_current() is ambient


def test_owned_observer_removal_preserves_foreign_processors(runtime):
    _r, p, _, inst = runtime
    foreign = SimpleSpanProcessor(InMemorySpanExporter())
    p.add_span_processor(foreign)
    owned = adapter._POLICIES[p]
    inst.deactivate()
    assert any(x is foreign for x in p._active_span_processor._span_processors)
    assert all(x is not owned for x in p._active_span_processor._span_processors)


def test_unknown_exception_descriptor_hooks_are_omitted():
    from respan_instrumentation_vertexai._serialization import (
        provider_status_code,
        safe_exception_message,
    )

    class Unknown(RuntimeError):
        @property
        def code(self):
            pytest.fail("code hook")

        def __getattribute__(self, key):
            if key in {"args", "response", "status_code"}:
                pytest.fail("error hook")
            return object.__getattribute__(self, key)

    Unknown.__module__ = "google.api_core.exceptions"
    error = Unknown("actual builtin argument")
    assert safe_exception_message(error) == "actual builtin argument"
    assert provider_status_code(error) is None


def test_native_async_embeddings_complete_vectors_and_statistics(runtime):
    r, _, e, _ = runtime

    async def run():
        model, channel = await r.async_embedding()
        try:
            result = await model.get_embeddings_async(["first"])
            assert len(result[0].values) == 5001
            return result
        finally:
            await channel.close()

    result = asyncio.run(run())
    span = e.get_finished_spans()[0]
    assert json.loads(span.attributes[OUTPUT]) == [result[0].values]
    assert span.attributes["gen_ai.usage.input_tokens"] == 5


def test_native_cached_usage_and_full_response_fields(runtime):
    from google.cloud.aiplatform_v1.types import GenerateContentResponse

    if (
        "cached_content_token_count"
        not in GenerateContentResponse.UsageMetadata.meta.fields
    ):
        pytest.skip("installed native SDK has no cache token field")
    r, _, e, _ = runtime
    response = r.model().generate_content("hello")
    span = e.get_finished_spans()[0]
    assert (
        span.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS]
        == response.usage_metadata.cached_content_token_count
        == 2
    )
    raw = json.loads(span.attributes[OUTPUT])[0]
    assert raw["candidates"][0]["finish_reason"] == "STOP"
    assert raw["usage_metadata"]["total_token_count"] == 10


def test_supplied_parent_context_suppression_and_content_are_both_vetoes(runtime):
    _, _p, _, _ = runtime
    from respan_instrumentation_vertexai._policy import content_allowed

    supplied = context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
    assert not content_allowed(True, supplied)
    token = context.attach(supplied)
    try:
        assert not content_allowed(
            True, context.set_value(ENABLE_CONTENT_TRACING_KEY, True)
        )
    finally:
        context.detach(token)
    assert not content_allowed(
        True, context.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
    )


def test_native_known_reasoning_signature_fields_preserved(runtime):
    from google.cloud.aiplatform_v1.types import GenerateContentResponse
    from google.cloud.aiplatform_v1.types import Part as NativePart

    if "thought_signature" not in NativePart._meta.fields:
        pytest.skip("Native SDK does not expose thought_signature")
    r, _, e, _ = runtime
    original = r.response

    def response(*args, **kwargs):
        return GenerateContentResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "text": "thought",
                                    "thought": True,
                                    "thought_signature": b"complete-signature",
                                }
                            ],
                        },
                        "finish_reason": "STOP",
                    }
                ]
            }
        )

    r.response = response
    native = r.model().generate_content("hello")
    assert (
        native.candidates[0].content.parts[0]._raw_part.thought_signature
        == b"complete-signature"
    )
    raw = json.loads(e.get_finished_spans()[0].attributes[OUTPUT])[0]
    part = raw["candidates"][0]["content"]["parts"][0]
    assert (
        part["thought"] is True
        and part["thought_signature"] == "Y29tcGxldGUtc2lnbmF0dXJl"
    )
    r.response = original


def test_native_startup_fault_restores_exact_ambient_context(runtime, monkeypatch):
    r, _p, _, _ = runtime
    ambient = context.get_current()
    original = adapter.request_attributes

    def fault(payload):
        context.attach(context.set_value("poisoned", True))
        raise RuntimeError("telemetry")

    monkeypatch.setattr(adapter, "request_attributes", fault)
    response = r.model().generate_content("hello")
    assert response.text == "native response" and len(r.requests) == 1
    assert context.get_current() is ambient
    monkeypatch.setattr(adapter, "request_attributes", original)


def test_owned_policy_activation_rollback_removes_only_own_processor(
    runtime, monkeypatch
):
    _, p, _, inst = runtime
    inst.deactivate()
    foreign = p._active_span_processor._span_processors
    original = adapter._policy

    def fault():
        original()
        raise RuntimeError("telemetry startup")

    monkeypatch.setattr(adapter, "_policy", fault)
    with pytest.raises(RuntimeError):
        VertexAIInstrumentor(tracer_provider=p).activate()
    assert p._active_span_processor._span_processors == foreign
    assert not adapter._POLICIES and not adapter._PATCHES


def test_propagated_unknown_hooks_and_credentials_are_safe(runtime):
    from respan_tracing.utils.span_factory import _PROPAGATED_ATTRIBUTES

    class Unknown:
        def __str__(self):
            raise AssertionError("customer formatting called")

    r, _, e, _ = runtime
    token = _PROPAGATED_ATTRIBUTES.set(
        {"metadata": {"opaque": Unknown(), "api_key": "private-value", "zero": 0}}
    )
    try:
        assert r.model().generate_content("hello").text == "native response"
    finally:
        _PROPAGATED_ATTRIBUTES.reset(token)
    attrs = e.get_finished_spans()[0].attributes
    assert attrs["respan.metadata.api_key"] == "[REDACTED]"
    assert attrs["respan.metadata.zero"] == "0"
    assert "Unknown" in attrs["respan.metadata.opaque"]


def test_two_pending_native_streams_under_parent_keep_content_after_export(runtime):
    r, p, e, _ = runtime
    with p.get_tracer("parent").start_as_current_span("outer"):
        first = r.model().generate_content("first", stream=True)
        second = r.model().generate_content("second", stream=True)
        assert next(first).text == next(second).text == "0,"
        first.close()
        assert next(second).text == "1,"
        second.close()
    spans = [
        span for span in e.get_finished_spans() if span.name.startswith("vertexai.")
    ]
    assert len(spans) == 2
    assert all(INPUT in span.attributes and OUTPUT in span.attributes for span in spans)
    assert spans[1].attributes["gen_ai.completion.0.content"] == "0,1,"


def test_native_processor_start_fault_preserves_native_ambient(runtime):
    from opentelemetry.sdk.trace import SpanProcessor

    r, p, _, _ = runtime
    ambient = context.get_current()

    class FaultProcessor(SpanProcessor):
        def on_start(self, span, parent_context=None):
            context.attach(context.set_value("processor-poison", True))
            raise RuntimeError("processor startup")

    p.add_span_processor(FaultProcessor())
    assert r.model().generate_content("hello").text == "native response"
    assert len(r.requests) == 1 and context.get_current() is ambient


def test_full_native_request_config_safety_tools_and_labels(runtime):
    from vertexai.generative_models import (
        GenerationConfig,
        HarmBlockThreshold,
        HarmCategory,
        SafetySetting,
        ToolConfig,
    )

    r, _, e, _ = runtime
    config = GenerationConfig(
        temperature=0,
        top_p=0.25,
        stop_sequences=["done"],
        candidate_count=1,
        max_output_tokens=32,
    )
    tools = ToolConfig(
        function_calling_config=ToolConfig.FunctionCallingConfig(
            mode=ToolConfig.FunctionCallingConfig.Mode.AUTO
        )
    )
    safety = SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    )
    assert (
        r.model()
        .generate_content(
            "hello",
            generation_config=config,
            tool_config=tools,
            safety_settings=[safety],
            labels={"controlled": "value"},
        )
        .text
        == "native response"
    )
    value = json.loads(e.get_finished_spans()[0].attributes[INPUT])["request"]
    assert value["generation_config"]["top_p"] == 0.25
    assert value["generation_config"]["stop_sequences"] == ["done"]
    assert value["tool_config"]["function_calling_config"]["mode"] == "AUTO"
    assert value["safety_settings"][0]["threshold"] == "BLOCK_NONE"
    assert value["labels"] == {"controlled": "value"}
    assert len(r.requests) == 1 and r.requests[0].generation_config.top_p == 0.25


def test_native_blocked_prompt_feedback_is_preserved_without_invented_completion(
    runtime,
):
    r, _, e, _ = runtime
    response = r.model().generate_content("blocked")
    assert not response.candidates and len(r.requests) == 1
    span = e.get_finished_spans()[0]
    assert (
        json.loads(span.attributes[OUTPUT])[0]["prompt_feedback"]["block_reason"]
        == "SAFETY"
    )
    assert "gen_ai.completion.0.content" not in span.attributes
    assert "gen_ai.usage.input_tokens" not in span.attributes
