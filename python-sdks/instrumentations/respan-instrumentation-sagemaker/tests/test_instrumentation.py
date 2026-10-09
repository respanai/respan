"""Native SDK transport/objects/data/ownership, no fake vendor modules."""

import inspect
import json

import pytest
from _fixtures import client, frame
from botocore.client import BaseClient
from botocore.eventstream import EventStream
from botocore.exceptions import ClientError, EventStreamError
from botocore.response import StreamingBody
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes as AI
from respan_instrumentation_sagemaker import SageMakerInstrumentor
from respan_instrumentation_sagemaker import _instrumentation as module
from respan_instrumentation_sagemaker._serialization import json_dumps, safe_text
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


@pytest.fixture
def pipeline(monkeypatch):
    for key in ("TRACELOOP_TRACE_CONTENT", "RESPAN_TRACE_CONTENT"):
        monkeypatch.delenv(key, raising=False)
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    owners = []

    def owner(capture=True):
        o = SageMakerInstrumentor(tracer_provider=provider, capture_content=capture)
        o.activate()
        owners.append(o)
        return o

    yield provider, memory, owner
    for o in reversed(owners):
        o.deactivate()
    provider.shutdown()


def invoke(c, body=None, **extra):
    return c.invoke_endpoint(
        EndpointName="controlled-endpoint",
        Body=json.dumps(
            body
            if body is not None
            else {"messages": [{"role": "user", "content": "native prompt"}]}
        ).encode(),
        ContentType="application/json",
        **extra,
    )


def test_native_body_identity_lazy_consumption_full_values_and_sourced_zero_usage(
    pipeline,
):
    _, m, owner = pipeline
    owner()
    payload = {
        "model": "controlled-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": {
                        "dense": list(range(5001)),
                        "sparse": {"indices": [0, 5000], "values": [0, False]},
                        "false": False,
                        "zero": 0,
                    },
                    "tool_calls": [
                        {
                            "id": "native-id",
                            "type": "function",
                            "function": {
                                "name": "native_tool",
                                "arguments": json.dumps(
                                    {
                                        "history": list(range(75)),
                                        "zero": 0,
                                        "false": False,
                                    }
                                ),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"input_tokens": 0, "output_tokens": 2, "total_tokens": 0},
    }
    c, _, _ = client(payload)
    response = invoke(
        c,
        {
            "model": "requested-model",
            "messages": [{"role": "user", "content": f"item-{i}"} for i in range(75)],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "native_tool",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "api_key": {
                                    "type": "string",
                                    "default": "controlled secret",
                                }
                            },
                        },
                    },
                }
            ],
        },
    )
    body = response["Body"]
    assert (
        type(body) is StreamingBody
        and body._amount_read == 0
        and len(m.get_finished_spans()) == 0
    )
    native_bytes = body.read()
    assert json.loads(native_bytes) == payload
    assert "read" not in body.__dict__
    span = m.get_finished_spans()[0]
    attrs = span.attributes
    assert json.loads(attrs[AI.TRACELOOP_ENTITY_OUTPUT]) == payload
    assert len(json.loads(attrs[AI.TRACELOOP_ENTITY_INPUT])["messages"]) == 75
    assert attrs[AI.LLM_REQUEST_MODEL] == "controlled-model"
    assert (
        attrs[AI.LLM_USAGE_PROMPT_TOKENS] == 0 and attrs[AI.LLM_USAGE_TOTAL_TOKENS] == 0
    )
    assert attrs[HTTP_RESPONSE_STATUS_CODE] == 200
    assert (
        json.loads(attrs[AI.LLM_REQUEST_FUNCTIONS])[0]["function"]["parameters"][
            "properties"
        ]["api_key"]["default"]
        == "[REDACTED]"
    )
    body.close()
    c.close()


def test_usage_absent_and_total_not_invented(pipeline):
    _, m, owner = pipeline
    owner()
    for payload in (
        {"generated_text": "native", "usage": {"input_tokens": 0, "output_tokens": 2}},
        {"generated_text": "native"},
    ):
        c, _, _ = client(payload)
        r = invoke(c, {"inputs": "native prompt"})
        r["Body"].read()
        r["Body"].close()
        c.close()
    assert all(
        AI.LLM_USAGE_TOTAL_TOKENS not in s.attributes for s in m.get_finished_spans()
    )
    assert AI.LLM_USAGE_PROMPT_TOKENS not in m.get_finished_spans()[1].attributes


def test_endpoint_is_not_fabricated_model_and_native_ml_is_task(pipeline):
    _, m, owner = pipeline
    owner()
    c, _, _ = client({"predictions": [False, 0]})
    r = invoke(c, {"instances": [[0, False]]})
    r["Body"].read()
    r["Body"].close()
    c.close()
    attrs = m.get_finished_spans()[0].attributes
    assert attrs[RESPAN_LOG_TYPE] == "task"
    assert AI.LLM_REQUEST_MODEL not in attrs
    assert json.loads(attrs[AI.TRACELOOP_ENTITY_OUTPUT]) == {"predictions": [False, 0]}


def test_native_embedding_vectors_complete(pipeline):
    _, m, owner = pipeline
    owner()
    payload = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": list(range(5001))}],
        "usage": {"input_tokens": 0},
    }
    c, _, _ = client(payload)
    r = invoke(c, {"model": "native-embedding", "input": ["native"]})
    r["Body"].read()
    r["Body"].close()
    c.close()
    attrs = m.get_finished_spans()[0].attributes
    assert attrs[RESPAN_LOG_TYPE] == "embedding"
    assert json.loads(attrs[AI.TRACELOOP_ENTITY_OUTPUT]) == [list(range(5001))]


def test_async_submission_retains_actual_sdk_fields_without_result_guess(pipeline):
    _, m, owner = pipeline
    owner()
    c, _, _ = client(
        {},
        status=202,
        headers={
            "x-amzn-sagemaker-inference-id": "native-inference",
            "x-amzn-sagemaker-outputlocation": "s3://controlled/output",
            "x-amzn-sagemaker-failurelocation": "s3://controlled/failure",
        },
    )
    result = c.invoke_endpoint_async(
        EndpointName="controlled-endpoint",
        InputLocation="s3://controlled/input",
        ContentType="application/json",
    )
    c.close()
    attrs = m.get_finished_spans()[0].attributes
    payload = json.loads(attrs[AI.TRACELOOP_ENTITY_OUTPUT])
    assert {k: v for k, v in payload.items() if k != "ResponseMetadata"} == {
        k: v for k, v in result.items() if k != "ResponseMetadata"
    }
    assert "state" not in payload
    assert attrs[HTTP_RESPONSE_STATUS_CODE] == 202


def test_native_error_no_output_and_error_identity(pipeline):
    _, m, owner = pipeline
    owner()
    c, _, _ = client(
        {"Message": "controlled native error"},
        status=429,
        headers={"x-amzn-errortype": "ThrottlingException"},
    )
    with pytest.raises(ClientError) as error:
        invoke(c)
    assert error.value.response["ResponseMetadata"]["HTTPStatusCode"] == 429
    span = m.get_finished_spans()[0]
    assert span.status.status_code is trace.StatusCode.ERROR
    assert AI.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    assert span.attributes[HTTP_RESPONSE_STATUS_CODE] == 429
    assert span.attributes[ERROR_MESSAGE] == "controlled native error"
    c.close()


@pytest.mark.parametrize("mode", ["constructor", "env", "context"])
def test_private_native_body_untouched_no_content_or_diagnostics(
    pipeline, monkeypatch, mode
):
    _, m, owner = pipeline
    token = None
    if mode == "env":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    if mode == "context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        owner(mode != "constructor")
        c, _, _ = client({"generated_text": "PRIVATE result"})
        r = invoke(c, {"inputs": "PRIVATE prompt"})
        body = r["Body"]
        assert type(body) is StreamingBody and "read" not in body.__dict__
        body.read()
        body.close()
        c.close()
    finally:
        if token:
            context.detach(token)
    assert len(m.get_finished_spans()) == 1
    assert "PRIVATE" not in str(m.get_finished_spans()[0].attributes)
    assert (
        AI.TRACELOOP_ENTITY_INPUT not in m.get_finished_spans()[0].attributes
        and AI.TRACELOOP_ENTITY_OUTPUT not in m.get_finished_spans()[0].attributes
    )


@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_suppression_zero_extra_body_activity(pipeline, key):
    _, m, owner = pipeline
    owner()
    t = context.attach(context.set_value(key, True))
    try:
        c, _, _ = client({"generated_text": "native"})
        r = invoke(c)
        assert "read" not in r["Body"].__dict__
        r["Body"].read()
        r["Body"].close()
        c.close()
    finally:
        context.detach(t)
    assert not m.get_finished_spans()


def test_sampling_no_extraction(monkeypatch):
    p = TracerProvider(sampler=ALWAYS_OFF)
    m = InMemorySpanExporter()
    p.add_span_processor(SimpleSpanProcessor(m))
    o = SageMakerInstrumentor(tracer_provider=p)
    o.activate()
    monkeypatch.setattr(
        module,
        "request_body",
        lambda p: (_ for _ in ()).throw(AssertionError("should not extract")),
    )
    try:
        c, _, _ = client({"generated_text": "native"})
        r = invoke(c)
        assert "read" not in r["Body"].__dict__
        r["Body"].read()
        r["Body"].close()
        c.close()
    finally:
        o.deactivate()
        p.shutdown()
    assert not m.get_finished_spans()


@pytest.mark.parametrize("parent", ["unobserved", "initial", "finished"])
def test_native_parent_bounds_cannot_widen(pipeline, parent):
    p, m, owner = pipeline
    if parent == "unobserved":
        span = p.get_tracer("native").start_span("parent")
        owner()
    else:
        owner()
        t = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        span = p.get_tracer("native").start_span("parent")
        context.detach(t)
        if parent == "finished":
            span.end()
    t = context.attach(trace.set_span_in_context(span))
    try:
        c, _, _ = client({"generated_text": "PRIVATE native"})
        r = invoke(c, {"inputs": "PRIVATE prompt"})
        r["Body"].read()
        r["Body"].close()
        c.close()
    finally:
        context.detach(t)
        span.end()
    sage = next(s for s in m.get_finished_spans() if s.name.startswith("sagemaker"))
    assert (
        AI.TRACELOOP_ENTITY_INPUT not in sage.attributes
        and AI.TRACELOOP_ENTITY_OUTPUT not in sage.attributes
    )


def test_pre_detach_and_late_body_veto_clear_retained_content(pipeline):
    _, m, owner = pipeline
    owner()
    c, _, _ = client({"generated_text": "PRIVATE output"})
    r = invoke(c, {"inputs": "PRIVATE input"})
    b = r["Body"]
    b.read(5)
    state = next(iter(module._MANAGER.states))
    assert state.chunks
    t = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    b.read()
    context.detach(t)
    assert not state.chunks and not state.params
    span = m.get_finished_spans()[0]
    assert (
        AI.TRACELOOP_ENTITY_INPUT not in span.attributes
        and AI.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    )
    b.close()
    c.close()


def test_native_partial_read_and_readinto_preserve_full_bytes(pipeline):
    _, m, owner = pipeline
    owner()
    payload = {"generated_text": "native string"}
    c, _, _ = client(payload)
    r = invoke(c, {"inputs": "native"})
    b = r["Body"]
    prefix = b.read(2)
    if not hasattr(b, "readinto"):
        b.close()
        c.close()
        pytest.skip("native SDK floor has no StreamingBody.readinto")
    chunks = [prefix]
    buffer = bytearray(7)
    while count := b.readinto(buffer):
        chunks.append(bytes(buffer[:count]))
    assert json.loads(b"".join(chunks)) == payload
    assert (
        json.loads(m.get_finished_spans()[0].attributes[AI.TRACELOOP_ENTITY_OUTPUT])
        == payload
    )
    b.close()
    c.close()


def test_native_context_manager_return_unchanged(pipeline):
    _, m, owner = pipeline
    owner()
    c, _, raws = client({"generated_text": "native"})
    r = invoke(c, {"inputs": "native"})
    with r["Body"] as raw:
        assert raw is raws[0]
        data = raw.read()
    assert json.loads(data) == {"generated_text": "native"}
    assert len(m.get_finished_spans()) == 1
    c.close()


def test_native_event_stream_identity_fragmented_json_tool_args_usage(pipeline):
    _, m, owner = pipeline
    owner()
    frames = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "native-call",
                                "type": "function",
                                "function": {
                                    "name": "native_tool",
                                    "arguments": '{"value":',
                                },
                            }
                        ],
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'false,"zero":0}'}}
                        ]
                    },
                }
            ],
            "usage": {"input_tokens": 0, "output_tokens": 2},
        },
    ]
    encoded = b"".join((json.dumps(x) + "\n").encode() for x in frames)
    data = frame(encoded[:13]) + frame(encoded[13:40]) + frame(encoded[40:])
    c, _, _ = client(events=data)
    r = c.invoke_endpoint_with_response_stream(
        EndpointName="controlled-endpoint",
        Body=b'{"messages":[{"role":"user","content":"native"}]}',
        ContentType="application/json",
    )
    body = r["Body"]
    assert type(body) is EventStream and not m.get_finished_spans()
    events = list(body)
    assert b"".join(e["PayloadPart"]["Bytes"] for e in events) == encoded
    attrs = m.get_finished_spans()[0].attributes
    tool = json.loads(attrs[AI.LLM_COMPLETIONS + ".0.tool_calls"])[0]
    assert tool["id"] == "native-call"
    assert json.loads(tool["function"]["arguments"]) == {"value": False, "zero": 0}
    assert attrs[AI.LLM_USAGE_PROMPT_TOKENS] == 0
    assert AI.LLM_USAGE_TOTAL_TOKENS not in attrs
    body.close()
    c.close()


def test_native_event_stream_error_partial_result_not_fabricated(pipeline):
    _, m, owner = pipeline
    owner()
    data = frame(b'{"token":{"text":"native part"}}\n') + frame(
        b'{"Message":"controlled stream error"}',
        kind="ModelStreamError",
        message_type="exception",
    )
    c, _, _ = client(events=data)
    r = c.invoke_endpoint_with_response_stream(
        EndpointName="controlled-endpoint",
        Body=b'{"inputs":"native"}',
        ContentType="application/json",
    )
    with pytest.raises(EventStreamError):
        list(r["Body"])
    span = m.get_finished_spans()[0]
    assert span.status.status_code is trace.StatusCode.ERROR
    assert "error" not in json.loads(span.attributes[AI.TRACELOOP_ENTITY_OUTPUT])
    assert (
        json.loads(span.attributes[AI.TRACELOOP_ENTITY_OUTPUT])["token"]["text"]
        == "native part"
    )
    r["Body"].close()
    c.close()


@pytest.mark.parametrize("fault", ["start", "extract", "attributes", "end"])
def test_observer_faults_preserve_native_response_and_cleanup(
    pipeline, monkeypatch, fault
):
    _, _, owner = pipeline
    owner()

    def fail(*a, **k):
        raise RuntimeError("controlled observer fault")

    if fault == "start":
        monkeypatch.setattr(module._Call, "__init__", fail)
    elif fault == "extract":
        monkeypatch.setattr(module, "request_body", fail)
    elif fault == "attributes":
        monkeypatch.setattr(module, "build_sagemaker_attrs", fail)
    c, _, _ = client({"generated_text": "native"})
    r = invoke(c)
    if fault == "end" and module._MANAGER.states:
        state = next(iter(module._MANAGER.states))
        monkeypatch.setattr(state.span, "end", fail)
    assert json.loads(r["Body"].read()) == {"generated_text": "native"}
    r["Body"].close()
    assert "read" not in r["Body"].__dict__
    assert not module._MANAGER.states
    c.close()


def test_shared_idempotent_conflict_and_foreign_wrapper_ownership(pipeline):
    p, _, owner = pipeline
    original = inspect.getattr_static(BaseClient, "_make_api_call")
    first = owner()
    first.activate()
    second = owner()
    first.deactivate()
    assert inspect.getattr_static(BaseClient, "_make_api_call") is not original
    with pytest.raises(RuntimeError):
        SageMakerInstrumentor(tracer_provider=p, capture_content=False).activate()
    owned = inspect.getattr_static(BaseClient, "_make_api_call")

    def foreign(*a, **k):
        return owned(*a, **k)

    BaseClient._make_api_call = foreign
    second.deactivate()
    assert BaseClient._make_api_call is foreign
    BaseClient._make_api_call = original


def test_serializer_no_unknown_hooks_valid_idempotent_encoded_schema():
    class Unknown:
        def model_dump(self):
            raise AssertionError("unknown hook")

        def __str__(self):
            raise AssertionError("unknown str")

    data = {
        "unknown": Unknown(),
        "schema": {
            "type": "object",
            "properties": {"api_key": {"type": "string", "default": "secret value"}},
            "required": ["api_key"],
        },
        "text": "x" * 20000,
    }
    parsed = json.loads(json_dumps(data))
    assert len(parsed["text"]) == 20000
    assert parsed["unknown"] == {"type": "Unknown"}
    encoded = safe_text(json_dumps(data))
    assert safe_text(encoded) == encoded
    assert (
        json.loads(encoded)["schema"]["properties"]["api_key"]["default"]
        == "[REDACTED]"
    )
    assert json.loads(safe_text('{"api_key":"secret words","ok":false}')) == {
        "api_key": "[REDACTED]",
        "ok": False,
    }


def test_preimported_detach_alias_and_active_parent_attribute_latch(pipeline):
    from opentelemetry.context import detach as alias

    p, m, owner = pipeline
    owner()
    with p.get_tracer("native").start_as_current_span("parent") as parent:
        c, _, _ = client({"generated_text": "PRIVATE output"})
        r = invoke(c, {"inputs": "PRIVATE input"})
        t = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        alias(t)
        r["Body"].read()
        r["Body"].close()
        c.close()
    assert (
        AI.TRACELOOP_ENTITY_OUTPUT
        not in next(
            s for s in m.get_finished_spans() if s.name.startswith("sagemaker")
        ).attributes
    )
    with p.get_tracer("native").start_as_current_span("second") as parent:
        c, _, _ = client({"generated_text": "PRIVATE output"})
        r = invoke(c, {"inputs": "PRIVATE input"})
        parent.set_attribute("trace_content", False)
        r["Body"].read()
        r["Body"].close()
        c.close()
    assert all(
        AI.TRACELOOP_ENTITY_OUTPUT not in s.attributes
        for s in m.get_finished_spans()
        if s.name.startswith("sagemaker")
    )


def test_no_implicit_native_depth_limit_or_stream_role_default():
    from respan_instrumentation_sagemaker._translator import StreamData

    data = {"leaf": False}
    for _ in range(80):
        data = {"next": data}
    assert json.loads(json_dumps(data)) == data
    stream = StreamData()
    stream.add(
        {
            "PayloadPart": {
                "Bytes": b'{"choices":[{"index":0,"delta":{"content":"native"}}]}\n{"choices":[{"index":0,"delta":{"content":" part"}}]}\n'
            }
        }
    )
    assert "role" not in stream.payload()["choices"][0]["message"]
