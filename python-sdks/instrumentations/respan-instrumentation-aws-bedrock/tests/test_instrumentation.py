from __future__ import annotations

import json

import pytest
from botocore.client import BaseClient
from botocore.eventstream import EventStream
from botocore.exceptions import ClientError, IncompleteReadError
from botocore.response import StreamingBody
from native_fixtures import (
    client,
    converse_frames,
    converse_response,
    frame,
    invoke_frames,
    transport,
)
from opentelemetry import context, trace
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_instrumentation_aws_bedrock import AWSBedrockInstrumentor
from respan_instrumentation_aws_bedrock._otel_emitter import build_bedrock_attrs
from respan_instrumentation_aws_bedrock._privacy import json_text, value

PARAMS = {
    "modelId": "anthropic.controlled",
    "messages": [{"role": "user", "content": [{"text": "prompt"}]}],
}
INVOKE = {
    "modelId": "anthropic.controlled",
    "body": json.dumps({"messages": [{"role": "user", "content": "prompt"}]}),
    "contentType": "application/json",
}


@pytest.fixture(autouse=True)
def native_capabilities(request):
    # The declared InvokeModel floor precedes native Converse. Skip only SDK
    # surfaces absent from that released client, not adapter failures.
    converse_tests = {
        "test_native_converse_identity_tools_reasoning_zero_cache",
        "test_native_error_response_and_status_are_sourced",
        "test_native_event_error_is_preserved",
        "test_suppressed_native_calls_create_no_span",
        "test_native_sampler_before_capture",
        "test_late_or_initial_content_veto_clears_body_data",
        "test_unknown_local_carrier_veto",
        "test_faulted_mapping_and_end_do_not_change_native_result",
    }
    c = client()
    try:
        if not hasattr(c, "converse") and (
            request.node.originalname in converse_tests
            or (
                request.node.originalname
                == "test_native_event_frames_identity_fragment_tools"
                and request.node.callspec.params["key"] == "stream"
            )
        ):
            pytest.skip(
                "Converse is absent from the released boto3 1.34.0 service model"
            )
    finally:
        c.close()


@pytest.fixture
def setup():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AWSBedrockInstrumentor(tracer_provider=provider)
    instrumentor.activate()
    yield provider, exporter, instrumentor
    instrumentor.deactivate()
    provider.shutdown()


def test_native_converse_identity_tools_reasoning_zero_cache(setup):
    provider, exporter, _ = setup
    c = client()
    transport(c, converse_response())
    with provider.get_tracer("test").start_as_current_span("workflow"):
        response = c.converse(**PARAMS)
    span = exporter.get_finished_spans()[0]
    assert response["output"]["message"]["content"][0]["text"] == "native hello"
    attrs = span.attributes
    assert attrs["gen_ai.usage.input_tokens"] == 0
    assert attrs["gen_ai.usage.cache_read_input_tokens"] == 0
    assert attrs["gen_ai.usage.cache_creation_input_tokens"] == 3
    assert "controlled reasoning" in attrs["traceloop.entity.output"]
    assert json.loads(attrs["gen_ai.completion.0.tool_calls"])[0]["id"] == "call-1"
    assert span.parent.span_id == exporter.get_finished_spans()[1].context.span_id
    assert "status_code" not in attrs


@pytest.mark.parametrize("mode", ["read", "chunks", "lines", "readinto"])
def test_invoke_body_remains_native_unread_and_byte_exact(setup, mode):
    _, exporter, _ = setup
    c = client()
    payload = b'{"content":[{"type":"text","text":"native hello"}],"usage":{"input_tokens":0,"output_tokens":1}}'
    raw = transport(c, payload)
    response = c.invoke_model(**INVOKE)
    body = response["body"]
    assert type(body) is StreamingBody
    assert body.tell() == 0
    assert not exporter.get_finished_spans()
    if mode == "read":
        assert body.read(0) == b""
        data = body.read()
    elif mode == "chunks":
        data = b"".join(body.iter_chunks(7))
    elif mode == "lines":
        data = b"".join(body.iter_lines(7))
    else:
        if not hasattr(body, "readinto"):
            body.close()
            pytest.skip(
                "native StreamingBody.readinto is unavailable at this SDK floor"
            )
        buffer = bytearray(7)
        parts = []
        while count := body.readinto(buffer):
            parts.append(bytes(buffer[:count]))
        data = b"".join(parts)
    assert data == payload
    assert len(exporter.get_finished_spans()) == 1
    assert exporter.get_finished_spans()[0].attributes["gen_ai.usage.input_tokens"] == 0
    assert "llm.usage.total_tokens" not in exporter.get_finished_spans()[0].attributes
    body.close()
    assert raw.raw.closed


@pytest.mark.parametrize(
    "operation,frames,key",
    [
        ("converse_stream", converse_frames, "stream"),
        ("invoke_model_with_response_stream", invoke_frames, "body"),
    ],
)
def test_native_event_frames_identity_fragment_tools(setup, operation, frames, key):
    _, exporter, _ = setup
    c = client()
    transport(c, frames(), event_stream=True)
    response = getattr(c, operation)(**(PARAMS if key == "stream" else INVOKE))
    stream = response[key]
    assert type(stream) is EventStream
    assert not exporter.get_finished_spans()
    events = list(stream)
    assert events
    attrs = exporter.get_finished_spans()[0].attributes
    calls = json.loads(attrs["gen_ai.completion.0.tool_calls"])
    assert json.loads(calls[0]["function"]["arguments"])["city"] == "Shanghai"
    assert attrs["gen_ai.usage.input_tokens"] == 0
    assert attrs["gen_ai.completion.0.content"] == "hello"
    stream.close()


def test_native_error_response_and_status_are_sourced(setup):
    _, exporter, _ = setup
    c = client()
    transport(
        c,
        {"message": "password=controlled-secret", "__type": "ValidationException"},
        status=400,
    )
    with pytest.raises(ClientError) as caught:
        c.converse(**PARAMS)
    span = exporter.get_finished_spans()[0]
    assert caught.value.response["ResponseMetadata"]["HTTPStatusCode"] == 400
    assert span.status.status_code is trace.StatusCode.ERROR
    assert span.attributes["http.response.status_code"] == 400
    assert span.attributes["error.type"] == type(caught.value).__name__
    assert "controlled-secret" not in span.attributes["error.message"]
    assert "traceloop.entity.output" not in span.attributes


def test_native_event_error_is_preserved(setup):
    _, exporter, _ = setup
    c = client()
    transport(
        c,
        frame("validationException", {"message": "controlled"}, "exception"),
        event_stream=True,
    )
    stream = c.converse_stream(**PARAMS)["stream"]
    with pytest.raises(ClientError):
        list(stream)
    assert exporter.get_finished_spans()[0].status.status_code is trace.StatusCode.ERROR


@pytest.mark.parametrize(
    "key", [_SUPPRESS_INSTRUMENTATION_KEY, SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY]
)
def test_suppressed_native_calls_create_no_span(setup, key):
    _, exporter, _ = setup
    c = client()
    transport(c, converse_response())
    token = context.attach(context.set_value(key, True))
    try:
        assert c.converse(**PARAMS)
    finally:
        context.detach(token)
    assert not exporter.get_finished_spans()


def test_native_sampler_before_capture():
    provider = TracerProvider(sampler=ALWAYS_OFF)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    inst = AWSBedrockInstrumentor(tracer_provider=provider)
    inst.activate()
    try:
        c = client()
        transport(c, converse_response())
        assert c.converse(**PARAMS)
        assert not exporter.get_finished_spans()
    finally:
        inst.deactivate()
        provider.shutdown()


@pytest.mark.parametrize("veto", ["ambient", "parent", "finished", "initial"])
def test_late_or_initial_content_veto_clears_body_data(setup, veto):
    provider, exporter, _ = setup
    c = client()
    transport(c, converse_frames(), event_stream=True)
    initial = (
        context.attach(context.set_value("override_enable_content_tracing", False))
        if veto == "initial"
        else None
    )
    with provider.get_tracer("test").start_as_current_span("parent") as parent:
        stream = c.converse_stream(**PARAMS)["stream"]
        if veto in ("parent", "finished"):
            parent.set_attribute("traceloop.enable_content_tracing", False)
        if veto != "finished":
            token = (
                context.attach(
                    context.set_value("override_enable_content_tracing", False)
                )
                if veto == "ambient"
                else None
            )
            try:
                assert list(stream)
            finally:
                if token:
                    context.detach(token)
    if veto == "finished":
        assert list(stream)
    if initial:
        context.detach(initial)
    child = next(s for s in exporter.get_finished_spans() if s.name == "ConverseStream")
    assert "traceloop.entity.input" not in child.attributes
    assert "traceloop.entity.output" not in child.attributes
    assert not any(k.startswith("gen_ai.prompt.") for k in child.attributes)


def test_unknown_local_carrier_veto(setup):
    _provider, exporter, _ = setup
    unknown = trace.NonRecordingSpan(
        trace.SpanContext(
            trace_id=123, span_id=123, is_remote=False, trace_flags=trace.TraceFlags(1)
        )
    )
    token = context.attach(trace.set_span_in_context(unknown))
    try:
        c = client()
        transport(c, converse_response())
        c.converse(**PARAMS)
    finally:
        context.detach(token)
    assert "traceloop.entity.input" not in exporter.get_finished_spans()[0].attributes


def test_partial_close_has_no_invented_output(setup):
    _, exporter, _ = setup
    c = client()
    raw = transport(c, b'{"content":[]}')
    body = c.invoke_model(**INVOKE)["body"]
    assert body.read(2) == b'{"'
    body.close()
    body.close()
    assert raw.raw.closed
    assert len(exporter.get_finished_spans()) == 1
    assert "traceloop.entity.output" not in exporter.get_finished_spans()[0].attributes


def test_body_native_incomplete_read_error_occurs_on_user_read(setup):
    _, exporter, _ = setup
    c = client()
    raw = transport(c, b'{"content":[]}')
    raw.headers["content-length"] = "100"
    body = c.invoke_model(**INVOKE)["body"]
    with pytest.raises(IncompleteReadError):
        body.read()
    assert (
        exporter.get_finished_spans()[0].attributes["error.type"]
        == "IncompleteReadError"
    )


def test_shared_owner_conflict_and_foreign_wrapper_preserved(setup):
    provider, _, first = setup
    installed = BaseClient._make_api_call
    second = AWSBedrockInstrumentor(tracer_provider=provider)
    second.activate()
    first.deactivate()
    assert BaseClient._make_api_call is installed
    with pytest.raises(ValueError):
        AWSBedrockInstrumentor(
            capture_content=False, tracer_provider=provider
        ).activate()

    def foreign(*args, **kwargs):
        return installed(*args, **kwargs)

    BaseClient._make_api_call = foreign
    second.deactivate()
    assert BaseClient._make_api_call is foreign
    BaseClient._make_api_call = installed.__wrapped__


def test_faulted_mapping_and_end_do_not_change_native_result(setup, monkeypatch):
    _, exporter, _ = setup
    monkeypatch.setattr(
        "respan_instrumentation_aws_bedrock._instrumentation.build_bedrock_attrs",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("telemetry")),
    )
    c = client()
    transport(c, converse_response())
    assert c.converse(**PARAMS)["stopReason"] == "tool_use"
    assert len(exporter.get_finished_spans()) == 1


def test_full_embedding_schema_and_credential_redaction():
    vector = [float(i) for i in range(5001)]
    attrs = build_bedrock_attrs(
        operation_name="InvokeModel",
        api_params={
            "modelId": "amazon.titan-embed",
            "body": json.dumps({"inputText": "hello"}),
        },
        response_payload={"embedding": vector, "inputTextTokenCount": 0},
    )
    assert json.loads(attrs["traceloop.entity.output"]) == vector
    assert attrs["respan.entity.log_type"] == "embedding"
    schema = {
        "parameters": {
            "type": "object",
            "properties": {
                "api_key": {"type": "string", "default": "controlled-secret"}
            },
        },
        "url": "https://user:pass@example.test/?token=controlled-secret",
        "text": 'api_key="controlled-secret" Bearer controlled-secret',
    }
    encoded = json_text(schema)
    assert "controlled-secret" not in encoded
    assert "api_key" in json.loads(encoded)["parameters"]["properties"]
    assert json_text(json.loads(encoded)) == encoded


def test_unknown_hooks_are_never_called():
    class Unknown:
        def __iter__(self):
            raise AssertionError("iterated")

        def __str__(self):
            raise AssertionError("stringified")

        def model_dump(self):
            raise AssertionError("serialized")

    assert value({"unknown": Unknown()}) == {"unknown": None}


def test_native_context_manager_returns_same_raw_and_captures(setup):
    _, exporter, _ = setup
    c = client()
    raw = transport(c, {"content": [{"type": "text", "text": "context hello"}]})
    body = c.invoke_model(**INVOKE)["body"]
    with body as entered:
        assert entered is raw.raw
        assert json.loads(entered.read())["content"][0]["text"] == "context hello"
    assert raw.raw.closed
    assert len(exporter.get_finished_spans()) == 1
    assert (
        exporter.get_finished_spans()[0].attributes["gen_ai.completion.0.content"]
        == "context hello"
    )


@pytest.mark.parametrize(
    "policy", ["respan-context", "environment", "no-widen", "late-suppression"]
)
def test_invoke_floor_privacy_policy(setup, monkeypatch, policy):
    from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

    _, exporter, _ = setup
    c = client()
    transport(c, {"content": [{"type": "text", "text": "private"}]})
    token = None
    if policy == "respan-context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    elif policy in ("environment", "no-widen"):
        monkeypatch.setenv("RESPAN_TRACE_CONTENT", "false")
    try:
        body = c.invoke_model(**INVOKE)["body"]
        if policy == "no-widen":
            monkeypatch.setenv("RESPAN_TRACE_CONTENT", "true")
        if policy == "late-suppression":
            token = context.attach(
                context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True)
            )
        assert body.read()
        body.close()
    finally:
        if token is not None:
            context.detach(token)
    assert "traceloop.entity.input" not in exporter.get_finished_spans()[0].attributes
    assert "traceloop.entity.output" not in exporter.get_finished_spans()[0].attributes


@pytest.mark.parametrize(
    "key", [_SUPPRESS_INSTRUMENTATION_KEY, SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY]
)
def test_invoke_floor_suppression(setup, key):
    _, exporter, _ = setup
    c = client()
    transport(c, {"content": []})
    token = context.attach(context.set_value(key, True))
    try:
        body = c.invoke_model(**INVOKE)["body"]
        assert body.read()
        body.close()
    finally:
        context.detach(token)
    assert not exporter.get_finished_spans()


def test_foreign_end_processor_fault_preserves_native_body(setup):
    from opentelemetry.sdk.trace import SpanProcessor

    provider, exporter, _ = setup

    class Fault(SpanProcessor):
        def on_end(self, span):
            raise RuntimeError("telemetry end")

    provider.add_span_processor(Fault())
    c = client()
    raw = transport(c, {"content": []})
    body = c.invoke_model(**INVOKE)["body"]
    assert json.loads(body.read()) == {"content": []}
    body.close()
    assert raw.raw.closed
    assert len(exporter.get_finished_spans()) == 1


def test_unknown_error_diagnostic_hook_is_not_executed(setup):
    _, exporter, _ = setup

    class UnknownError(Exception):
        def __getattribute__(self, name):
            if name == "response":
                raise AssertionError("unknown response hook executed")
            return super().__getattribute__(name)

        def __str__(self):
            raise AssertionError("unknown error hook executed")

    error = UnknownError()
    c = client()

    def fail(request):
        raise error

    c._endpoint.http_session.send = fail
    with pytest.raises(UnknownError) as caught:
        c.invoke_model(**INVOKE)
    assert caught.value is error
    assert exporter.get_finished_spans()[0].attributes["error.type"] == "UnknownError"
    assert "error.message" not in exporter.get_finished_spans()[0].attributes


def test_full_schema_arguments_and_history_survive_native_transport(setup):
    _, exporter, _ = setup
    c = client()
    if not hasattr(c, "converse"):
        pytest.skip("Converse absent at native floor")
    params = dict(PARAMS)
    schema = {
        "type": "object",
        "properties": {
            "api_key": {"type": "string", "default": "controlled-secret"},
            "city": {"type": "string"},
        },
        "additionalProperties": False,
    }
    params["toolConfig"] = {
        "tools": [{"toolSpec": {"name": "weather", "inputSchema": {"json": schema}}}]
    }
    params["messages"] = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "historical",
                        "name": "weather",
                        "input": {"city": "Paris", "zero": 0},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "historical",
                        "content": [{"json": {"empty": "", "zero": 0, "false": False}}],
                    }
                }
            ],
        },
    ]
    transport(c, converse_response())
    c.converse(**params)
    attrs = exporter.get_finished_spans()[0].attributes
    assert "historical" in attrs["gen_ai.prompt.0.tool_calls"]
    assert "historical" not in attrs["gen_ai.completion.0.tool_calls"]
    assert "controlled-secret" not in attrs["llm.request.functions"]
    encoded = json.loads(attrs["llm.request.functions"])[0]["function"]["parameters"]
    assert encoded["properties"]["api_key"]["default"] == "[REDACTED]"
    assert encoded["additionalProperties"] is False
    assert (
        json.loads(attrs["traceloop.entity.input"])["messages"][1]["content"][0][
            "toolResult"
        ]["content"][0]["json"]["false"]
        is False
    )


def test_invoke_json_schema_credential_defaults_remain_valid(setup):
    _, exporter, _ = setup
    c = client()
    transport(c, {"content": []})
    body = {
        "messages": [],
        "tools": [
            {
                "name": "controlled",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "default": "controlled-secret"}
                    },
                },
            }
        ],
    }
    params = dict(INVOKE, body=json.dumps(body))
    native = c.invoke_model(**params)["body"]
    assert native.read()
    native.close()
    attrs = exporter.get_finished_spans()[0].attributes
    schema = json.loads(attrs["llm.request.functions"])[0]["function"]["parameters"]
    assert schema["properties"]["api_key"]["default"] == "[REDACTED]"
    assert "controlled-secret" not in json.dumps(dict(attrs))


def test_partial_native_stream_close_has_only_observed_events(setup):
    _, exporter, _ = setup
    c = client()
    if not hasattr(c, "converse_stream"):
        pytest.skip("ConverseStream absent at native floor")
    raw = transport(c, converse_frames(), event_stream=True)
    stream = c.converse_stream(**PARAMS)["stream"]
    iterator = iter(stream)
    assert next(iterator) == {"messageStart": {"role": "assistant"}}
    stream.close()
    stream.close()
    assert raw.raw.closed
    assert len(exporter.get_finished_spans()) == 1
    attrs = exporter.get_finished_spans()[0].attributes
    assert json.loads(attrs["traceloop.entity.output"]) == [
        {"messageStart": {"role": "assistant"}}
    ]
    assert "gen_ai.completion.0.content" not in attrs


@pytest.mark.parametrize(
    "veto", ["active-ancestor", "generic-child-start", "generic-child-end"]
)
def test_irreversible_native_ancestor_content_veto(setup, veto):
    provider, exporter, _ = setup
    c = client()
    raw = transport(c, {"content": [{"type": "text", "text": "outer private"}]})

    def send(request):
        if veto == "active-ancestor":
            ancestor = trace.get_current_span()
            ancestor.set_attribute("traceloop.enable_content_tracing", False)
            nested = client()
            transport(nested, {"content": []})
            body = nested.invoke_model(**INVOKE)["body"]
            body.read()
            body.close()
            ancestor.set_attribute("traceloop.enable_content_tracing", True)
        elif veto == "generic-child-start":
            token = context.attach(
                context.set_value("override_enable_content_tracing", False)
            )
            try:
                with provider.get_tracer("test").start_as_current_span("generic-child"):
                    pass
            finally:
                context.detach(token)
        else:
            generic = provider.get_tracer("test").start_span("generic-child")
            token = context.attach(
                context.set_value("override_enable_content_tracing", False)
            )
            try:
                generic.end()
            finally:
                context.detach(token)
        return raw

    c._endpoint.http_session.send = send
    body = c.invoke_model(**INVOKE)["body"]
    assert body.read()
    body.close()
    spans = [
        span for span in exporter.get_finished_spans() if span.name == "InvokeModel"
    ]
    assert spans
    assert all(
        "traceloop.entity.input" not in span.attributes
        and "traceloop.entity.output" not in span.attributes
        for span in spans
    )


@pytest.mark.parametrize("runtime_fault", [False, True])
def test_detach_fault_restores_exact_native_context(setup, monkeypatch, runtime_fault):
    _, exporter, _ = setup
    ambient = context.get_current()

    def fail(token):
        raise RuntimeError("controlled detach fault")

    monkeypatch.setattr(context, "detach", fail)
    if runtime_fault:
        monkeypatch.setattr(context._RUNTIME_CONTEXT, "detach", fail)
    c = client()
    transport(c, {"content": []})
    body = c.invoke_model(**INVOKE)["body"]
    assert context.get_current() is ambient
    assert body.read()
    body.close()
    assert len(exporter.get_finished_spans()) == 1


def test_exact_ai_floor_cache_counts_from_invoke(setup):
    _, exporter, _ = setup
    c = client()
    transport(
        c,
        {
            "content": [],
            "usage": {
                "input_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 3,
            },
        },
    )
    body = c.invoke_model(**INVOKE)["body"]
    body.read()
    body.close()
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs["gen_ai.usage.cache_read_input_tokens"] == 0
    assert attrs["gen_ai.usage.cache_creation_input_tokens"] == 3


@pytest.mark.parametrize(
    "operation,params,key,payload",
    [
        ("invoke_model", INVOKE, "body", b'{"content":[]}'),
        ("converse_stream", PARAMS, "stream", None),
    ],
)
def test_response_and_body_identity_match_native_parser_callback(
    setup, operation, params, key, payload
):
    _, exporter, _ = setup
    c = client()
    if not hasattr(c, operation):
        pytest.skip("native operation absent at declared floor")
    observed = []

    def capture(parsed, **kwargs):
        observed.append((parsed, parsed[key]))

    c.meta.events.register("after-call.bedrock-runtime.*", capture)
    transport(
        c,
        converse_frames() if payload is None else payload,
        event_stream=payload is None,
    )
    result = getattr(c, operation)(**params)
    assert result is observed[0][0]
    assert result[key] is observed[0][1]
    if key == "body":
        result[key].read()
    else:
        list(result[key])
    result[key].close()
    assert len(exporter.get_finished_spans()) == 1


def test_unknown_native_error_subclass_hook_is_not_executed(setup):
    _, exporter, _ = setup

    class UnknownClientError(ClientError):
        def __str__(self):
            raise AssertionError("unknown subclass hook executed")

    error = UnknownClientError(
        {"Error": {"Code": "controlled", "Message": "private"}}, "InvokeModel"
    )
    c = client()

    def fail(request):
        raise error

    c._endpoint.http_session.send = fail
    with pytest.raises(UnknownClientError) as caught:
        c.invoke_model(**INVOKE)
    assert caught.value is error
    assert (
        exporter.get_finished_spans()[0].attributes["error.type"]
        == "UnknownClientError"
    )
    assert "error.message" not in exporter.get_finished_spans()[0].attributes


def test_partial_read_veto_irreversibly_denies_parent_and_sibling(setup):
    provider, exporter, _ = setup
    with provider.get_tracer("test").start_as_current_span("parent"):
        c = client()
        transport(c, {"content": [{"type": "text", "text": "private"}]})
        body = c.invoke_model(**INVOKE)["body"]
        token = context.attach(
            context.set_value("override_enable_content_tracing", False)
        )
        try:
            assert body.read(2)
        finally:
            context.detach(token)
        assert body.read()
        body.close()
        sibling = client()
        transport(sibling, {"content": [{"type": "text", "text": "sibling private"}]})
        native = sibling.invoke_model(**INVOKE)["body"]
        native.read()
        native.close()
    children = [
        span for span in exporter.get_finished_spans() if span.name == "InvokeModel"
    ]
    assert len(children) == 2
    assert all(
        "traceloop.entity.input" not in span.attributes
        and "traceloop.entity.output" not in span.attributes
        for span in children
    )


def test_native_retry_and_callbacks_produce_one_span(setup):
    _, exporter, _ = setup
    c = client(retries=1)
    first = transport(
        c,
        {"message": "controlled throttle", "__type": "ThrottlingException"},
        status=429,
    )
    second = transport(c, {"content": [{"type": "text", "text": "after retry"}]})
    responses = iter((first, second))
    requests = []
    callbacks = []

    def send(request):
        requests.append(request)
        return next(responses)

    c._endpoint.http_session.send = send
    c.meta.events.register(
        "after-call.bedrock-runtime.*",
        lambda parsed, **kwargs: callbacks.append(parsed),
    )
    result = c.invoke_model(**INVOKE)
    assert json.loads(result["body"].read())["content"][0]["text"] == "after retry"
    result["body"].close()
    assert len(requests) == 2 and callbacks == [result]
    assert len(exporter.get_finished_spans()) == 1
