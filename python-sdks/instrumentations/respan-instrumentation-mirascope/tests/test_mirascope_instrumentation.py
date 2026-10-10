from __future__ import annotations

import gc
import json

import pytest
from _fixtures import Provider, async_vector_tool, model, openai_model, vector_tool
from mirascope import llm
from opentelemetry import context, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as G
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv._incubating.attributes.http_attributes import (
    HTTP_RESPONSE_STATUS_CODE,
)
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes as A
from respan_instrumentation_mirascope import _translation as translation
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


@pytest.mark.parametrize(
    "method",
    [
        "call",
        "context_call",
        "call_async",
        "context_call_async",
        "stream",
        "context_stream",
        "stream_async",
        "context_stream_async",
    ],
)
async def test_native_model_surfaces_preserve_values_and_detach(capture, method):
    _, provider, exporter = capture
    native, source = model(temperature=0, max_tokens=0)
    kwargs = {"ctx": llm.Context(deps={})} if "context" in method else {}
    with provider.get_tracer("application").start_as_current_span("parent") as parent:
        response = getattr(native, method)("actual prompt", **kwargs)
        if "async" in method:
            response = await response
        assert response is source.last
        assert trace.get_current_span() is parent
        if "stream" in method:
            if "async" in method:
                assert (
                    "".join([part async for part in response.text_stream()])
                    == "native stream\n"
                )
            else:
                assert "".join(response.text_stream()) == "native stream\n"
            assert trace.get_current_span() is parent
            assert all(
                sc.span_id != parent.get_span_context().span_id
                for sc in source.observed
            )
        else:
            assert response.text() == "native result"
    span = next(s for s in exporter.get_finished_spans() if s.name == "llm")
    assert span.parent.span_id == parent.get_span_context().span_id
    assert span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert span.attributes[A.LLM_REQUEST_TEMPERATURE] == 0
    assert span.attributes[A.LLM_REQUEST_MAX_TOKENS] == 0
    assert span.attributes[A.GEN_AI_IS_STREAMING] is ("stream" in method)
    assert json.loads(span.attributes[A.TRACELOOP_ENTITY_OUTPUT])[0]["content"] in {
        "native result",
        "native stream",
    }


@pytest.mark.parametrize(
    "owner", ["Toolkit", "ContextToolkit", "AsyncToolkit", "AsyncContextToolkit"]
)
async def test_actual_toolkits_complete_dense_sparse_ids_schema(capture, owner):
    _, _, exporter = capture
    asynchronous = owner.startswith("Async")
    tool = async_vector_tool if asynchronous else vector_tool
    toolkit = getattr(llm, owner)([tool])
    call = llm.ToolCall(
        id="tool-id-" + "i" * 600,
        name=tool.name,
        args=json.dumps({"values": list(range(5000))}),
    )
    args = [llm.Context(deps={}), call] if "Context" in owner else [call]
    result = toolkit.execute(*args)
    if asynchronous:
        result = await result
    assert result.error is None and result.id == call.id
    span = exporter.get_finished_spans()[0]
    assert span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert span.attributes[G.GEN_AI_TOOL_CALL_ID] == call.id
    payload = json.loads(span.attributes[A.TRACELOOP_ENTITY_OUTPUT])
    assert len(payload if asynchronous else payload["embedding"]) == 5000
    if not asynchronous:
        assert len(payload["sparse_vector"]) == 5000
    assert A.LLM_SYSTEM not in span.attributes
    assert G.GEN_AI_PROVIDER_NAME not in span.attributes


@pytest.mark.parametrize("setting", ["false", "  FALSE  ", "0", "off", "no"])
def test_environment_privacy_does_not_inspect_content(capture, monkeypatch, setting):
    _, _, exporter = capture
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", setting)
    monkeypatch.setattr(
        translation, "messages", lambda value: pytest.fail("private serializer called")
    )
    native, _ = model()
    assert native.call("private prompt").text() == "native result"
    attrs = exporter.get_finished_spans()[0].attributes
    assert (
        A.TRACELOOP_ENTITY_INPUT not in attrs and A.TRACELOOP_ENTITY_OUTPUT not in attrs
    )
    assert attrs[G.GEN_AI_USAGE_INPUT_TOKENS] == 9


@pytest.mark.parametrize(
    "setting",
    [
        ENABLE_CONTENT_TRACING_KEY,
        _SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_context_privacy_and_suppression(capture, setting):
    _, _, exporter = capture
    token = context.attach(
        context.set_value(setting, setting != ENABLE_CONTENT_TRACING_KEY)
    )
    try:
        native, _ = model()
        native.call("private prompt")
    finally:
        context.detach(token)
    spans = exporter.get_finished_spans()
    if setting == ENABLE_CONTENT_TRACING_KEY:
        assert len(spans) == 1 and A.TRACELOOP_ENTITY_INPUT not in spans[0].attributes
    else:
        assert not spans


def test_delayed_stream_finished_parent_veto_is_sticky(capture):
    _, provider, exporter = capture
    native, _ = model()
    with provider.get_tracer("application").start_as_current_span("parent"):
        response = native.stream("private later")
        # Native application scope detaches before Span.end: checkpoint must see this.
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    assert "".join(response.text_stream()) == "native stream\n"
    child = next(s for s in exporter.get_finished_spans() if s.name == "llm")
    assert A.TRACELOOP_ENTITY_INPUT not in child.attributes
    assert A.TRACELOOP_ENTITY_OUTPUT not in child.attributes


def test_start_bound_cannot_be_unmasked_and_end_veto(capture):
    _, _, exporter = capture
    native, _ = model()
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    response = native.stream("private initial")
    context.detach(token)
    list(response.text_stream())
    assert A.TRACELOOP_ENTITY_INPUT not in exporter.get_finished_spans()[0].attributes
    exporter.clear()
    response = native.stream("private final")
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    list(response.text_stream())
    context.detach(token)
    assert A.TRACELOOP_ENTITY_INPUT not in exporter.get_finished_spans()[0].attributes


def test_sampling_skips_all_content_inspection(capture, monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from respan_instrumentation_mirascope import MirascopeInstrumentor

    capture[0].deactivate()
    provider = TracerProvider(sampler=ALWAYS_OFF)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = MirascopeInstrumentor(tracer_provider=provider)
    calls = []
    original = translation.messages

    def observed(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(translation, "messages", observed)
    instrumentor.activate()
    try:
        native, _ = model()
        assert native.call("actual prompt").text() == "native result"
        assert not exporter.get_finished_spans() and not calls
    finally:
        instrumentor.deactivate()
        provider.shutdown()


@pytest.mark.parametrize("error_first", [True, False])
def test_native_stream_error_retains_actual_partial_only(capture, error_first):
    _, _, exporter = capture
    error = llm.ServerError("provider error", provider="controlled", status_code=503)
    chunks = (
        [error]
        if error_first
        else [llm.TextStartChunk(), llm.TextChunk(delta="partial"), error]
    )
    native, _ = model(Provider(chunks=chunks))
    response = native.stream("prompt")
    with pytest.raises(llm.ServerError) as raised:
        list(response.text_stream())
    assert raised.value is error
    span = exporter.get_finished_spans()[0]
    assert span.attributes[HTTP_RESPONSE_STATUS_CODE] == 503
    assert span.attributes[ERROR_TYPE] == "ServerError"
    assert (A.TRACELOOP_ENTITY_OUTPUT not in span.attributes) is error_first
    if not error_first:
        assert "partial" in span.attributes[A.TRACELOOP_ENTITY_OUTPUT]


def test_unadvanced_close_abandonment_and_generator_protocol(capture):
    instrumentor, _, exporter = capture
    native, source = model()
    response = native.stream("prompt")
    response._chunk_iterator.close()
    assert len(exporter.get_finished_spans()) == 1
    assert A.TRACELOOP_ENTITY_OUTPUT not in exporter.get_finished_spans()[0].attributes
    response = native.stream("abandoned")
    del response
    source.last = None
    gc.collect()
    assert not instrumentor.runtime.calls
    assert len(exporter.get_finished_spans()) == 2
    assert not hasattr(native.stream("prompt")._chunk_iterator, "__enter__")


def test_native_stream_send_throw_and_return(capture):
    _, _, exporter = capture
    native, _ = model()
    response = native.stream("prompt")
    assert response._chunk_iterator.send(None).type == "text_start_chunk"
    error = RuntimeError("native thrown error")
    with pytest.raises(RuntimeError) as raised:
        response._chunk_iterator.throw(error)
    assert raised.value is error
    assert len(exporter.get_finished_spans()) == 1


async def test_native_async_stream_asend_athrow_aclose(capture):
    _, _, exporter = capture
    native, _ = model()
    response = await native.stream_async("prompt")
    assert (await response._chunk_iterator.asend(None)).type == "text_start_chunk"
    error = RuntimeError("native async thrown error")
    with pytest.raises(RuntimeError) as raised:
        await response._chunk_iterator.athrow(error)
    assert raised.value is error
    response = await native.stream_async("unadvanced")
    await response._chunk_iterator.aclose()
    assert len(exporter.get_finished_spans()) == 2


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        {"prompt_tokens": True, "completion_tokens": -1, "total_tokens": False},
        {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    ],
)
def test_controlled_native_http_usage_is_source_only(capture, usage):
    _, _, exporter = capture
    native, provider, requests = openai_model(usage=usage)
    try:
        response = native.call("actual prompt")
        assert response.text() == "actual provider result"
        assert requests[0]["messages"][0]["content"] == "actual prompt"
        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[HTTP_RESPONSE_STATUS_CODE] == 200
        if usage is None or isinstance(usage["prompt_tokens"], bool):
            assert G.GEN_AI_USAGE_INPUT_TOKENS not in attrs
            assert A.LLM_USAGE_TOTAL_TOKENS not in attrs
        else:
            assert attrs[G.GEN_AI_USAGE_INPUT_TOKENS] == usage["prompt_tokens"]
            if usage["prompt_tokens"]:
                assert attrs[A.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] == 4
                assert attrs[A.GEN_AI_USAGE_REASONING_TOKENS] == 1
    finally:
        provider._completions_provider.client.close()
        provider._responses_provider.client.close()


@pytest.mark.parametrize("xai", [False, True])
async def test_native_responses_and_xai2_5(capture, xai):
    _, _, exporter = capture
    native, provider, requests = openai_model(
        mode="responses",
        xai=xai,
        usage={
            "input_tokens": 6,
            "output_tokens": 2,
            "total_tokens": 8,
            "input_tokens_details": {"cached_tokens": 1, "cache_write_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
    )
    response = await native.call_async("native responses")
    assert response.text() == "actual provider result"
    assert (
        requests
        and exporter.get_finished_spans()[0].attributes[G.GEN_AI_USAGE_INPUT_TOKENS]
        == 6
    )
    assert (
        exporter.get_finished_spans()[0].attributes[
            A.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS
        ]
        == 3
    )
    owner = provider if xai else provider._responses_provider
    await owner.async_client.close()
    owner.client.close()


def test_native_http_stream_raw_invalid_counters(capture):
    _, _, exporter = capture
    native, provider, _ = openai_model(
        stream=True,
        usage={"prompt_tokens": True, "completion_tokens": 2, "total_tokens": False},
    )
    assert "".join(native.stream("prompt").text_stream()) == "actual streamed result\n"
    attrs = exporter.get_finished_spans()[0].attributes
    assert G.GEN_AI_USAGE_INPUT_TOKENS not in attrs
    assert attrs[G.GEN_AI_USAGE_OUTPUT_TOKENS] == 2
    assert A.LLM_USAGE_TOTAL_TOKENS not in attrs
    provider._completions_provider.client.close()


def test_complete_current_history_calls_schema_and150_messages(capture):
    _, _, exporter = capture
    call = llm.ToolCall(
        id="id-" + "x" * 600,
        name="vector_tool",
        args=json.dumps({"values": list(range(5000)), "api_key": "secret with spaces"}),
    )
    native, _ = model(Provider(content=call))
    history = [llm.messages.user(str(i)) for i in range(150)]
    response = native.call(history, tools=[vector_tool])
    assert response.tool_calls[0] is call
    attrs = exporter.get_finished_spans()[0].attributes
    assert len(json.loads(attrs[A.TRACELOOP_ENTITY_INPUT])) == 150
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT and attrs[A.LLM_REQUEST_MODEL]
    output = json.loads(attrs[A.TRACELOOP_ENTITY_OUTPUT])[0]["tool_calls"][0]
    assert output["id"] == call.id
    assert len(json.loads(output["function"]["arguments"])["values"]) == 5000
    assert "secret with spaces" not in attrs[A.TRACELOOP_ENTITY_OUTPUT]
    schema = json.loads(attrs[A.LLM_REQUEST_FUNCTIONS])[0]["function"]["parameters"]
    assert schema["properties"]["api_key"]["type"] == "string"
    assert schema["properties"]["api_key"]["default"] == "[REDACTED]"
    exporter.clear()
    native.call(response.messages)
    old = json.loads(
        exporter.get_finished_spans()[0].attributes[A.TRACELOOP_ENTITY_INPUT]
    )[-1]["tool_calls"][0]
    assert old == output


def test_telemetry_fault_preserves_native_values_errors_and_cleanup(
    capture, monkeypatch
):
    instrumentor, _, exporter = capture
    monkeypatch.setattr(
        translation,
        "prepare",
        lambda value: (_ for _ in ()).throw(RuntimeError("observer failure")),
    )
    native, source = model()
    assert native.call("prompt") is source.last
    error = llm.ServerError("native failure", provider="controlled", status_code=401)
    native, _ = model(Provider(error=error))
    with pytest.raises(llm.ServerError) as raised:
        native.call("prompt")
    assert raised.value is error
    assert not instrumentor.runtime.calls
    assert len(exporter.get_finished_spans()) == 2
    assert (
        exporter.get_finished_spans()[-1].attributes[HTTP_RESPONSE_STATUS_CODE] == 401
    )


def test_hostile_native_exception_and_private_diagnostics(capture):
    _, _, exporter = capture

    class NativeFailure(RuntimeError):
        def __str__(self):
            raise AssertionError("exception stringification changed application")

    error = NativeFailure(
        'api_key="fixture secret with spaces" Bearer fixture-token https://user:password@fixture.invalid'
    )
    native, _ = model(Provider(error=error))
    with pytest.raises(NativeFailure) as raised:
        native.call("prompt")
    assert raised.value is error
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs[ERROR_TYPE] == "NativeFailure"
    assert "fixture secret" not in str(dict(attrs))
    assert A.TRACELOOP_ENTITY_OUTPUT not in attrs
    exporter.clear()
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        with pytest.raises(NativeFailure):
            native.call("private prompt")
    finally:
        context.detach(token)
    private = exporter.get_finished_spans()[0]
    assert not private.events and private.status.description is None
    assert "error.message" not in private.attributes


def test_no_computed_total_from_invalid_native_usage(capture):
    _, _, exporter = capture
    native, _ = model(Provider(usage=llm.Usage(input_tokens=-1, output_tokens=3)))
    native.call("prompt")
    attrs = exporter.get_finished_spans()[0].attributes
    assert G.GEN_AI_USAGE_INPUT_TOKENS not in attrs
    assert attrs[G.GEN_AI_USAGE_OUTPUT_TOKENS] == 3
    assert A.LLM_USAGE_TOTAL_TOKENS not in attrs


def test_raw_tool_stream_current_partial_arguments(capture):
    _, _, exporter = capture
    native, _ = model(
        Provider(
            chunks=[
                llm.ToolCallStartChunk(id="current-id", name="vector_tool"),
                llm.ToolCallChunk(
                    id="current-id",
                    delta='{"api_key":"fixture credential", "values":[1,2]}',
                ),
                llm.ToolCallEndChunk(id="current-id"),
            ]
        )
    )
    response = native.stream("current tools")
    list(response.chunk_stream())
    calls = json.loads(
        exporter.get_finished_spans()[0].attributes[f"{A.LLM_COMPLETIONS}.0.tool_calls"]
    )
    assert calls[0]["id"] == "current-id"
    assert json.loads(calls[0]["function"]["arguments"])["api_key"] == "[REDACTED]"


def test_native_partial_tool_error_redacts_unfinished_credential(capture):
    _, _, exporter = capture
    error = RuntimeError("controlled stream failure")
    native, _ = model(
        Provider(
            chunks=[
                llm.ToolCallStartChunk(id="partial-native", name="vector_tool"),
                llm.ToolCallChunk(
                    id="partial-native",
                    delta='{"api_key":"synthetic secret with spaces',
                ),
                error,
            ]
        )
    )
    response = native.stream("partial native tool")
    with pytest.raises(RuntimeError) as raised:
        list(response.chunk_stream())
    assert raised.value is error
    attrs = exporter.get_finished_spans()[0].attributes
    assert "synthetic secret" not in str(dict(attrs))
    calls = json.loads(attrs[f"{A.LLM_COMPLETIONS}.0.tool_calls"])
    assert (
        calls[0]["id"] == "partial-native"
        and "[REDACTED]" in calls[0]["function"]["arguments"]
    )


def test_native_http_schema_omits_unset_model_fields_preserves_raw_null(
    capture, monkeypatch
):
    _, _, exporter = capture
    monkeypatch.setitem(vector_tool.parameters.properties["values"], "default", None)
    native, provider, requests = openai_model()
    try:
        response = native.call("actual native tool-schema HTTP", tools=[vector_tool])
        assert response.text() == "actual provider result"
        native_parameters = requests[0]["tools"][0]["function"]["parameters"]
        attrs = exporter.get_finished_spans()[0].attributes
        parameters = json.loads(attrs[A.LLM_REQUEST_FUNCTIONS])[0]["function"][
            "parameters"
        ]
        assert "$defs" not in native_parameters and "$defs" not in parameters
        assert not any(value is None for value in parameters.values())
        assert parameters["properties"]["values"]["default"] is None
        assert parameters["properties"]["api_key"]["type"] == "string"
        assert parameters["properties"]["api_key"]["default"] == "[REDACTED]"
    finally:
        provider._completions_provider.client.close()
        provider._responses_provider.client.close()
