"""Actual released SDK clients, HTTP/SSE frames and OTel providers."""

import asyncio
import inspect
import json

import pytest
from opentelemetry import context, trace
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from respan_instrumentation_together import TogetherInstrumentor
from respan_instrumentation_together import _instrumentation as adapter
from respan_instrumentation_together._serialization import (
    json_dumps,
    provider_status_code,
    safe_text,
)
from respan_instrumentation_together._translator import native_value
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from together import APIError, AsyncStream, RateLimitError, Stream

from tests._native import NativeRuntime

INPUT = SpanAttributes.TRACELOOP_ENTITY_INPUT
OUTPUT = SpanAttributes.TRACELOOP_ENTITY_OUTPUT


def chat(client, content="hello", **kwargs):
    return client.chat.completions.create(
        model="native-model", messages=[{"role": "user", "content": content}], **kwargs
    )


@pytest.fixture
def runtime():
    r = NativeRuntime()
    p = TracerProvider()
    e = InMemorySpanExporter()
    p.add_span_processor(SimpleSpanProcessor(e))
    inst = TogetherInstrumentor(tracer_provider=p)
    inst.activate()
    yield r, p, e, inst
    inst.deactivate()
    p.shutdown()


def test_native_unary_parent_full_fields_only_sourced_usage(runtime):
    r, p, e, _ = runtime
    with (
        r.client() as client,
        p.get_tracer("parent").start_as_current_span("outer") as parent,
    ):
        response = chat(client)
    span = e.get_finished_spans()[0]
    assert (
        span.parent.span_id == parent.context.span_id
        and response.choices[0].message.content == "native response"
    )
    output = json.loads(span.attributes[OUTPUT])[0]
    assert (
        output["id"] == response.id
        and output["choices"][0]["message"]["reasoning"] == "native reasoning"
        and output["choices"][0]["seed"] == 0
    )
    assert (
        span.attributes["gen_ai.usage.input_tokens"] == 7
        and span.attributes["gen_ai.usage.output_tokens"] == 3
        and span.attributes["llm.usage.total_tokens"] == 10
    )
    assert (
        HTTP_RESPONSE_STATUS_CODE not in span.attributes
        and "status_code" not in span.attributes
        and "traceloop.span.kind" not in span.attributes
    )
    assert len(r.requests) == r.callback_count == 1


def test_native_stream_identity_no_eager_reads_all70_and_close(runtime):
    r, _, e, _ = runtime
    with r.client() as client:
        source = chat(client, stream=True)
        assert type(source) is Stream and r.bodies[-1].reads == 0
        chunks = list(source)
        assert (
            len(chunks) == 70 and r.bodies[-1].closes == 1 and source.response.is_closed
        )
        assert (
            json.loads(e.get_finished_spans()[0].attributes[OUTPUT])[69]["usage"][
                "total_tokens"
            ]
            == 10
        )
        assert e.get_finished_spans()[0].attributes[
            "gen_ai.completion.0.content"
        ] == "".join(x.choices[0].delta.content for x in chunks)
        assert source.close() is None
    assert len(r.requests) == 1 and r.callback_count == 1


@pytest.mark.parametrize("consumed", [0, 1])
def test_native_context_manager_and_early_unread_close(runtime, consumed):
    r, _, e, _ = runtime
    with r.client() as client:
        source = chat(client, stream=True)
        assert source.__enter__() is source
        if consumed:
            assert next(source).choices[0].delta.content == "0,"
        assert source.__exit__(None, None, None) is None
    span = e.get_finished_spans()[0]
    assert source.response.is_closed and r.bodies[-1].closes == 1
    assert (OUTPUT in span.attributes) == bool(consumed)
    if consumed:
        assert span.attributes["gen_ai.completion.0.content"] == "0,"
    assert "gen_ai.usage.input_tokens" not in span.attributes


def test_native_text_stream_and_unary_complete(runtime):
    r, _, e, _ = runtime
    with r.client() as client:
        native = client.completions.create(
            model="native-model", prompt="complete", temperature=0, max_tokens=32
        )
        assert native.choices[0].text == "native completion"
        stream = client.completions.create(
            model="native-model", prompt="complete", stream=True
        )
        assert type(stream) is Stream and len(list(stream)) == 70
    assert len(e.get_finished_spans()) == 2 and e.get_finished_spans()[1].attributes[
        "gen_ai.completion.0.content"
    ].endswith("69,")


@pytest.mark.parametrize("operation", ["embedding", "rerank", "image"])
def test_native_all_other_operations_preserve_complete_data(runtime, operation):
    r, _, e, _ = runtime
    with r.client() as client:
        if operation == "embedding":
            response = client.embeddings.create(
                model="native-embedding", input=["first", "second"]
            )
            assert len(response.data[0].embedding) == 5001
        elif operation == "rerank":
            response = client.rerank.create(
                model="native-rerank",
                query="native query",
                documents=["first", "second"],
                return_documents=True,
            )
        else:
            response = client.images.generate(
                model="native-image", prompt="controlled", response_format="b64_json"
            )
    output = json.loads(e.get_finished_spans()[0].attributes[OUTPUT])
    if operation == "embedding":
        assert (
            len(output) == 2
            and len(output[0]) == 5001
            and output[1][-1] == response.data[1].embedding[-1]
        )
    elif operation == "rerank":
        assert output[0]["results"][0]["document"] == {
            "text": "native doc",
            "flag": False,
            "zero": 0,
        }
    else:
        assert output[0]["data"][0]["b64_json"] == response.data[0].b64_json


def test_native_empty_response_keeps_full_feedback_no_projection_or_usage(runtime):
    r, _, e, _ = runtime
    with r.client() as client:
        response = chat(client, "empty")
    span = e.get_finished_spans()[0]
    assert response.choices == []
    raw = json.loads(span.attributes[OUTPUT])[0]
    assert (
        raw["choices"] == []
        and raw["native_feedback"]["blocked"] is True
        and raw["zero"] == 0
        and raw["flag"] is False
    )
    assert (
        "gen_ai.completion.0.content" not in span.attributes
        and "gen_ai.usage.input_tokens" not in span.attributes
    )


def test_native_error_exact_class_diagnostics_source_code_no_result(runtime):
    r, _, e, _ = runtime
    with r.client() as client, pytest.raises(RateLimitError) as caught:
        chat(client, "failure")
    span = e.get_finished_spans()[0]
    assert (
        span.status.status_code.name == "ERROR"
        and span.attributes[HTTP_RESPONSE_STATUS_CODE]
        == caught.value.status_code
        == 429
    )
    assert (
        OUTPUT not in span.attributes
        and span.attributes[ERROR_TYPE] == "RateLimitError"
    )
    assert (
        "controlled provider failure" in span.attributes[ERROR_MESSAGE]
        and len(r.requests) == 1
    )


def test_native_sse_error_preserves_partial_output_and_original_error(runtime):
    r, _, e, _ = runtime
    with r.client() as client:
        source = chat(client, "stream-error", stream=True)
        assert next(source).choices[0].delta.content == "0,"
        assert next(source).choices[0].delta.content == "1,"
        with pytest.raises(APIError) as caught:
            next(source)
    span = e.get_finished_spans()[0]
    assert (
        caught.value.args
        and source.response.is_closed
        and span.status.status_code.name == "ERROR"
    )
    assert (
        span.attributes["gen_ai.completion.0.content"] == "0,1,"
        and HTTP_RESPONSE_STATUS_CODE not in span.attributes
    )
    assert "gen_ai.usage.input_tokens" not in span.attributes


def test_native_retry_and_callback_count_unmodified(runtime):
    r, _, e, _ = runtime
    r.retry = True
    with r.client(retries=1) as client:
        response = chat(client)
    assert (
        response.choices[0].message.content == "native response"
        and r.attempts == r.callback_count == 2
        and len(e.get_finished_spans()) == 1
    )


def test_native_async_every_operation_stream_and_original_contextmanager(runtime):
    r, _, e, _ = runtime

    async def run():
        async with r.async_client() as client:
            assert (
                await client.chat.completions.create(
                    model="native-model",
                    messages=[{"role": "user", "content": "hello"}],
                )
            ).choices[0].message.content == "native response"
            assert (
                await client.completions.create(model="native-model", prompt="hello")
            ).choices[0].text == "native completion"
            assert (
                len(
                    (
                        await client.embeddings.create(
                            model="native-embedding", input=["hello"]
                        )
                    )
                    .data[0]
                    .embedding
                )
                == 5001
            )
            await client.rerank.create(
                model="native-rerank", query="native", documents=["first", "second"]
            )
            await client.images.generate(
                model="native-image", prompt="native", response_format="b64_json"
            )
            source = await client.chat.completions.create(
                model="native-model",
                messages=[{"role": "user", "content": "stream"}],
                stream=True,
            )
            assert (
                type(source) is AsyncStream
                and r.bodies[-1].reads == 0
                and await source.__aenter__() is source
            )
            chunks = [chunk async for chunk in source]
            assert len(chunks) == 70
            assert await source.__aexit__(None, None, None) is None
            unread = await client.chat.completions.create(
                model="native-model",
                messages=[{"role": "user", "content": "stream"}],
                stream=True,
            )
            assert await unread.close() is None and unread.response.is_closed

    asyncio.run(run())
    assert len(e.get_finished_spans()) == 7 and all(
        body.closes == 1 for body in r.bodies
    )


@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_suppression_precedes_content_inspection_native_work_still_once(
    runtime, monkeypatch, key
):
    r, _, e, _ = runtime
    inspected = []

    def spy(*args):
        inspected.append(True)
        raise AssertionError("content inspection attempted")

    monkeypatch.setattr(adapter, "request_attributes", spy)
    token = context.attach(context.set_value(key, True))
    try:
        with r.client() as client:
            assert chat(client).choices[0].message.content == "native response"
    finally:
        context.detach(token)
    assert not e.get_finished_spans() and len(r.requests) == 1

    assert not inspected


def test_sampling_precedes_capture_and_preserves_original_stream_type(
    runtime, monkeypatch
):
    r, _, e, inst = runtime
    inst.deactivate()
    p = TracerProvider(sampler=ALWAYS_OFF)
    new = TogetherInstrumentor(tracer_provider=p)
    new.activate()
    inspected = []

    def spy(*args):
        inspected.append(True)
        raise AssertionError("content inspection attempted")

    monkeypatch.setattr(adapter, "request_attributes", spy)
    try:
        with r.client() as client:
            source = chat(client, stream=True)
            assert type(source) is Stream and "close" not in source.__dict__
            assert len(list(source)) == 70
    finally:
        new.deactivate()
        p.shutdown()
    assert not e.get_finished_spans()

    assert not inspected


@pytest.mark.parametrize(
    "veto", ["capture", "canonical", "traceloop", "environment", "respan_env"]
)
def test_initial_privacy_skips_all_payload_and_native_error_diagnostics(
    runtime, monkeypatch, veto
):
    r, p, e, inst = runtime
    token = None
    if veto == "capture":
        inst.deactivate()
        inst = TogetherInstrumentor(capture_content=False, tracer_provider=p)
        inst.activate()
    elif veto == "canonical":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    elif veto == "traceloop":
        token = context.attach(
            context.set_value("override_enable_content_tracing", False)
        )
    else:
        monkeypatch.setenv(
            "TRACELOOP_TRACE_CONTENT"
            if veto == "environment"
            else "RESPAN_TRACE_CONTENT",
            "false",
        )
    inspected = []

    def spy(*args):
        inspected.append(True)
        raise AssertionError("content inspection attempted")

    monkeypatch.setattr(adapter, "request_attributes", spy)
    try:
        with r.client() as client:
            assert (
                chat(client, "private").choices[0].message.content == "native response"
            )
            with pytest.raises(RateLimitError):
                chat(client, "failure")
    finally:
        if token is not None:
            context.detach(token)
        if veto == "capture":
            inst.deactivate()
    for span in e.get_finished_spans():
        assert (
            INPUT not in span.attributes
            and OUTPUT not in span.attributes
            and ERROR_MESSAGE not in span.attributes
            and not span.status.description
            and not span.events
        )
    assert len(r.requests) == 2

    assert not inspected


def test_native_late_stream_veto_scrubs_retention_and_never_widens(runtime):
    r, _, e, _ = runtime
    with r.client() as client:
        source = chat(client, stream=True)
        next(source)
        call = source.__dict__["_iterator"].call
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        try:
            next(source)
        finally:
            context.detach(token)
        assert not call.chunks and not call.priority and not call.propagated
        source.close()
    span = e.get_finished_spans()[0]
    assert (
        INPUT not in span.attributes
        and OUTPUT not in span.attributes
        and not span.status.description
    )


@pytest.mark.parametrize("finished", [False, True])
def test_observed_ancestor_initial_false_survives_restore_and_end(runtime, finished):
    r, p, e, _ = runtime
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    parent = p.get_tracer("app").start_span("ancestor")
    context.detach(token)
    if finished:
        parent.end()
    token = context.attach(
        trace.set_span_in_context(
            trace.NonRecordingSpan(parent.context) if finished else parent
        )
    )
    try:
        with r.client() as client:
            chat(client)
    finally:
        context.detach(token)
    if not finished:
        parent.end()
    span = next(span for span in e.get_finished_spans() if span.name == "together.chat")
    assert INPUT not in span.attributes


@pytest.mark.parametrize("finished", [False, True])
def test_ancestor_attribute_false_is_irreversible(runtime, finished):
    r, p, e, _ = runtime
    parent = p.get_tracer("app").start_span("ancestor")
    parent.set_attribute("traceloop.enable_content_tracing", False)
    if finished:
        parent.end()
    token = context.attach(
        trace.set_span_in_context(
            trace.NonRecordingSpan(parent.context) if finished else parent
        )
    )
    try:
        with r.client() as client:
            chat(client)
            if not finished:
                parent.set_attribute("traceloop.enable_content_tracing", True)
            chat(client)
    finally:
        context.detach(token)
    if not finished:
        parent.end()
    spans = [s for s in e.get_finished_spans() if s.name == "together.chat"]
    assert len(spans) == 2 and all(INPUT not in s.attributes for s in spans)


def test_generic_child_veto_denies_active_ancestor_stream_and_sibling(runtime):
    r, p, e, _ = runtime
    with r.client() as client, p.get_tracer("app").start_as_current_span("parent"):
        stream = chat(client, stream=True)
        next(stream)
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        child = p.get_tracer("app").start_span("child")
        child.end()
        context.detach(token)
        stream.close()
        chat(client)
    spans = [s for s in e.get_finished_spans() if s.name == "together.chat"]
    assert len(spans) == 2 and all(
        INPUT not in s.attributes and OUTPUT not in s.attributes for s in spans
    )


@pytest.mark.parametrize("carrier", ["active", "finished", "nonrecording"])
def test_late_provider_unknown_local_carrier_fails_closed(runtime, carrier):
    r, _, _, inst = runtime
    inst.deactivate()
    p = TracerProvider()
    e = InMemorySpanExporter()
    p.add_span_processor(SimpleSpanProcessor(e))
    parent = p.get_tracer("unknown").start_span("old")
    if carrier == "finished":
        parent.end()
    if carrier != "active":
        parent = trace.NonRecordingSpan(parent.context)
    new = TogetherInstrumentor(tracer_provider=p)
    new.activate()
    token = context.attach(trace.set_span_in_context(parent))
    try:
        with r.client() as client:
            chat(client)
    finally:
        context.detach(token)
        new.deactivate()
        p.shutdown()
    span = next(s for s in e.get_finished_spans() if s.name == "together.chat")
    assert INPUT not in span.attributes


def test_ambient_and_supplied_flags_both_veto(runtime):
    from respan_instrumentation_together._policy import content_allowed

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


@pytest.mark.parametrize(
    "fault", ["startup", "on_start", "request", "attributes", "output", "end", "detach"]
)
def test_telemetry_faults_preserve_native_result_context_and_cleanup(
    runtime, monkeypatch, fault
):
    r, p, _, _ = runtime
    ambient = context.get_current()

    def fail(*args, **kwargs):
        raise RuntimeError("telemetry")

    if fault == "startup":
        monkeypatch.setattr(adapter, "_policy", fail)
    elif fault == "on_start":

        class Fault(SpanProcessor):
            def on_start(self, span, parent_context=None):
                context.attach(context.set_value("poison", True))
                fail()

        p.add_span_processor(Fault())
    elif fault == "request":
        monkeypatch.setattr(adapter, "request_attributes", fail)
    elif fault == "attributes":
        monkeypatch.setattr(Span, "set_attribute", fail)
    elif fault == "output":
        monkeypatch.setattr(adapter, "response_attributes", fail)
    elif fault == "end":
        monkeypatch.setattr(Span, "end", fail)
    else:
        monkeypatch.setattr(context, "detach", fail)
        monkeypatch.setattr(context._RUNTIME_CONTEXT, "detach", fail)
    with r.client() as client:
        assert chat(client).choices[0].message.content == "native response"
        source = chat(client, stream=True)
        next(source)
        source.close()
        assert source.response.is_closed
    assert context.get_current() is ambient and len(r.requests) == 2


def test_two_pending_streams_under_parent_survive_sibling_export(runtime):
    r, p, e, _ = runtime
    with r.client() as client, p.get_tracer("app").start_as_current_span("parent"):
        first = chat(client, stream=True)
        second = chat(client, stream=True)
        next(first)
        next(second)
        first.close()
        next(second)
        second.close()
    spans = [s for s in e.get_finished_spans() if s.name == "together.chat"]
    assert (
        len(spans) == 2
        and all(INPUT in s.attributes and OUTPUT in s.attributes for s in spans)
        and spans[1].attributes["gen_ai.completion.0.content"] == "0,1,"
    )


def test_owned_shared_configuration_foreign_wrapper_and_processors(runtime):
    r, p, e, first = runtime
    second = TogetherInstrumentor(tracer_provider=p)
    second.activate()
    with pytest.raises(ValueError):
        TogetherInstrumentor(capture_content=False, tracer_provider=p).activate()
    patch = adapter._PATCHES[0]
    wrapped = patch.wrapper

    def foreign(*args, **kwargs):
        return wrapped(*args, **kwargs)

    setattr(patch.cls, patch.method_name, foreign)
    processor = SimpleSpanProcessor(InMemorySpanExporter())
    p.add_span_processor(processor)
    try:
        first.deactivate()
        with r.client() as client:
            chat(client)
        second.deactivate()
        assert (
            inspect.getattr_static(patch.cls, patch.method_name) is foreign
            and p._active_span_processor._span_processors[-1] is processor
        )
        assert not any(
            isinstance(x, adapter.AncestorPolicy)
            for x in p._active_span_processor._span_processors
        )
    finally:
        setattr(patch.cls, patch.method_name, patch.original)
    assert len(e.get_finished_spans()) == 1


def test_partial_activation_owned_policy_rollback(runtime, monkeypatch):
    _, p, _, first = runtime
    first.deactivate()
    before = p._active_span_processor._span_processors
    original = adapter._policy

    def fault():
        original()
        raise RuntimeError("install")

    monkeypatch.setattr(adapter, "_policy", fault)
    with pytest.raises(RuntimeError):
        TogetherInstrumentor(tracer_provider=p).activate()
    assert (
        not adapter._PATCHES
        and not adapter._POLICIES
        and p._active_span_processor._span_processors == before
    )


def test_full_75history_10k_text_schema_false_zero_json_and_128_native_bound(runtime):
    r, _, e, _ = runtime
    messages = [{"role": "user", "content": str(i) + "x" * 10000} for i in range(75)]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "api_key": {"type": "string", "default": "secret multi word"},
                        "flag": {"type": "boolean", "default": False},
                        "zero": {"type": "integer", "default": 0},
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    with r.client() as client:
        response = client.chat.completions.create(
            model="native-model",
            messages=messages,
            tools=tools,
            extra_body={"native_extra": {"zero": 0, "flag": False}},
        )
    span = e.get_finished_spans()[0]
    captured = json.loads(span.attributes[INPUT])
    schema = json.loads(span.attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0][
        "function"
    ]["parameters"]
    assert (
        len(captured["messages"]) == 75
        and len(captured["messages"][0]["content"]) == 10001
        and schema["properties"]["api_key"]
        == {"type": "string", "default": "[REDACTED]"}
    )
    assert (
        schema["properties"]["flag"]["default"] is False
        and schema["properties"]["zero"]["default"] == 0
        and span.attributes["respan.entity.log_type"] == "chat"
        and OUTPUT in span.attributes
    )
    assert (
        json.loads(span.attributes[OUTPUT])[0]["choices"][0]["message"]["tool_calls"][
            0
        ]["id"]
        == response.choices[0].message.tool_calls[0].id
    )


def test_native_extra_body_effective_model_and_messages_are_projected(runtime):
    r, _, e, _ = runtime
    with r.client() as client:
        response = chat(
            client,
            extra_body={
                "model": "native-override",
                "messages": [{"role": "user", "content": "effective"}],
            },
        )
    attrs = e.get_finished_spans()[0].attributes
    assert (
        response.model == attrs[SpanAttributes.LLM_REQUEST_MODEL] == "native-override"
        and attrs["gen_ai.prompt.0.content"] == "effective"
        and r.requests[0]["model"] == "native-override"
    )


def test_unknown_hooks_numeric_container_model_dump_and_native_propagation_are_omitted(
    runtime,
):
    class Unknown:
        def __getattribute__(self, name):
            raise AssertionError("unknown getter")

        def model_dump(self):
            raise AssertionError("unknown dump")

        def __str__(self):
            raise AssertionError("unknown str")

    class Number(int):
        def __int__(self):
            raise AssertionError("unknown int")

    class Mapping(dict):
        def items(self):
            raise AssertionError("unknown items")

    assert (
        native_value(Unknown()) == {"type": "Unknown"}
        and native_value(Number(1)) == {"type": "Number"}
        and native_value(Mapping()) == {"type": "Mapping"}
    )
    from respan_tracing.utils.span_factory import _PROPAGATED_ATTRIBUTES

    token = _PROPAGATED_ATTRIBUTES.set(
        {"metadata": {"opaque": Unknown(), "api_key": "controlled-secret", "zero": 0}}
    )
    r, _, e, _ = runtime
    try:
        with r.client() as client:
            chat(client)
    finally:
        _PROPAGATED_ATTRIBUTES.reset(token)
    attrs = e.get_finished_spans()[0].attributes
    assert (
        attrs["respan.metadata.api_key"] == "[REDACTED]"
        and attrs["respan.metadata.zero"] == "0"
    )


def test_credential_json_redaction_valid_idempotent_and_no_type_hooks():
    raw = json.dumps(
        {
            "arguments": json.dumps(
                {"api_key": "multi word secret", "zero": 0, "flag": False}
            ),
            "authorization": "Bearer opaque",
            "url": "https://user:secret@example.invalid/path?token=secret",
        }
    )
    value = json_dumps(json.loads(raw))
    assert (
        "multi word" not in value
        and "opaque" not in value
        and "user:secret" not in value
        and json_dumps(json.loads(value)) == value
    )
    assert safe_text('token="multi word secret"') == 'token="[REDACTED]"'

    class Meta(type):
        @property
        def __name__(cls):
            raise AssertionError("name descriptor")

    class Opaque(metaclass=Meta):
        pass

    assert native_value(Opaque()) == {"type": "Opaque"}


def test_unknown_error_property_hooks_and_empty_error_message_not_invented():
    class Unknown(Exception):
        @property
        def status_code(self):
            raise AssertionError("unknown status property")

        @property
        def response(self):
            raise AssertionError("unknown response property")

    assert provider_status_code(Unknown()) is None


def test_native_source_usage_missing_fields_never_infers_total(runtime):
    r, _, e, _ = runtime
    r.usage = lambda: {"prompt_tokens": 0}
    with r.client() as client:
        chat(client)
    attrs = e.get_finished_spans()[0].attributes
    assert (
        attrs["gen_ai.usage.input_tokens"] == 0
        and "gen_ai.usage.output_tokens" not in attrs
        and "llm.usage.total_tokens" not in attrs
    )


@pytest.mark.parametrize("scheme", ["Bearer", "Basic"])
def test_actual_native_quoted_authorization_message_and_error_redacted(runtime, scheme):
    import httpx

    r, _, e, _ = runtime
    original = r.respond

    def respond(request):
        body = json.loads(request.content)
        if body["messages"][0]["content"] == "failure":
            return httpx.Response(
                429,
                json={"error": {"message": scheme + ' "controlled-private"'}},
                request=request,
            )
        return original(request)

    r.respond = respond
    content = "Authorization: " + scheme + ' "controlled-private"'
    with r.client() as client:
        chat(client, content)
        with pytest.raises(RateLimitError):
            chat(client, "failure")
    assert r.requests[0]["messages"][0]["content"] == content
    assert all(
        "controlled-private" not in json.dumps(dict(span.attributes))
        and "controlled-private" not in (span.status.description or "")
        for span in e.get_finished_spans()
    )
    assert safe_text(content) == safe_text(safe_text(content))


def test_native_stream_fragment_secrets_redacted_without_native_chunk_mutation(runtime):
    from tests._native import Body

    r, _, e, _ = runtime
    parts = []
    args = ['{"city":"Tokyo","api_key":"multi ', 'word private","flag":false,"zero":0}']
    for index, argument in enumerate(args):
        function = {"arguments": argument}
        call = {"index": 0, "function": function}
        if index == 0:
            call.update({"id": "native-call", "type": "function"})
            function["name"] = "get_weather"
        chunk = {
            "id": "native-fragments",
            "created": 1,
            "model": "native-response-model",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "Bearer " if index == 0 else "controlledtoken",
                        "tool_calls": [call],
                    },
                }
            ],
        }
        parts.append(("data: " + json.dumps(chunk) + "\n\n").encode())
    parts.append(b"data: [DONE]\n\n")

    def stream(payload, text=False):
        body = Body(parts)
        r.bodies.append(body)
        return body

    r.stream = stream
    with r.client() as client:
        chunks = list(chat(client, stream=True))
    assert chunks[0].choices[0].delta.tool_calls[0].function.arguments == args[0]
    attrs = e.get_finished_spans()[0].attributes
    assert (
        "multi " not in attrs[OUTPUT]
        and "word private" not in attrs[OUTPUT]
        and "controlledtoken" not in attrs[OUTPUT]
    )
    call = json.loads(attrs["gen_ai.completion.0.tool_calls"])[0]
    arguments = json.loads(call["function"]["arguments"])
    assert (
        arguments
        == {"city": "Tokyo", "api_key": "[REDACTED]", "flag": False, "zero": 0}
        and call["id"] == "native-call"
    )
    assert attrs["gen_ai.response.model"] == "native-response-model"


@pytest.mark.parametrize("consumed", [0, 1])
def test_pending_deactivation_ends_exactly_once_without_closing_native_stream(
    runtime, consumed
):
    r, _, e, inst = runtime
    with r.client() as client:
        source = chat(client, stream=True)
        if consumed:
            next(source)
        inst.deactivate()
        assert (
            not source.response.is_closed
            and r.bodies[-1].closes == 0
            and len(e.get_finished_spans()) == 1
        )
        assert len(list(source)) == 70 - consumed and source.response.is_closed
        source.close()
    assert len(e.get_finished_spans()) == 1


def test_explicit_veto_detach_without_consumption_latches_pending_span(runtime):
    r, _, e, _ = runtime
    with r.client() as client:
        source = chat(client, stream=True)
        next(source)
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        context.detach(token)
        source.close()
    attrs = e.get_finished_spans()[0].attributes
    assert INPUT not in attrs and OUTPUT not in attrs


def test_native_on_start_fault_ends_owned_partial_span_and_preserves_context(runtime):
    r, p, e, inst = runtime
    inst.deactivate()
    ambient = context.get_current()

    class Fault(SpanProcessor):
        def on_start(self, span, parent_context=None):
            context.attach(context.set_value("startup-poison", True))
            raise RuntimeError("native processor startup")

    p.add_span_processor(Fault())
    new = TogetherInstrumentor(tracer_provider=p)
    new.activate()
    try:
        with r.client() as client:
            assert chat(client).choices[0].message.content == "native response"
    finally:
        new.deactivate()
    assert context.get_current() is ambient and len(e.get_finished_spans()) == 1
    span = e.get_finished_spans()[0]
    assert (
        span.status.status_code.name == "UNSET"
        and INPUT not in span.attributes
        and OUTPUT not in span.attributes
    )


def test_owned_processor_add_then_raise_rollback_preserves_foreign(
    runtime, monkeypatch
):
    _, p, _, inst = runtime
    inst.deactivate()
    before = p._active_span_processor._span_processors
    original = p.add_span_processor

    def fail(processor):
        original(processor)
        raise RuntimeError("added then failed")

    monkeypatch.setattr(p, "add_span_processor", fail)
    with pytest.raises(RuntimeError):
        TogetherInstrumentor(tracer_provider=p).activate()
    assert (
        p._active_span_processor._span_processors == before
        and not adapter._PATCHES
        and not adapter._POLICIES
    )


def test_actual_remote_parent_allowed_and_other_trace_local_carrier_unknown(runtime):
    r, p, e, _ = runtime
    parent = p.get_tracer("observed").start_span("observed")
    remote = trace.SpanContext(
        trace_id=123, span_id=456, is_remote=True, trace_flags=trace.TraceFlags(1)
    )
    token = context.attach(trace.set_span_in_context(trace.NonRecordingSpan(remote)))
    try:
        with r.client() as client:
            chat(client)
    finally:
        context.detach(token)
    foreign = trace.SpanContext(
        trace_id=parent.context.trace_id ^ 1234,
        span_id=parent.context.span_id,
        is_remote=False,
        trace_flags=trace.TraceFlags(1),
    )
    token = context.attach(trace.set_span_in_context(trace.NonRecordingSpan(foreign)))
    try:
        with r.client() as client:
            chat(client)
    finally:
        context.detach(token)
        parent.end()
    spans = [s for s in e.get_finished_spans() if s.name == "together.chat"]
    assert INPUT in spans[0].attributes and INPUT not in spans[1].attributes


def test_unknown_local_closed_veto_registry_bound(runtime):
    _, p, _, _ = runtime
    policy = adapter._POLICIES[p]
    for index in range(5000):
        policy._deny_chain((index + 1, index + 1))
    assert len(policy.ended) <= 4096 and all(
        state[2] is None for state in policy.ended.values()
    )


def test_native_embedding_envelope_extras_and_sensitive_schema_keys(runtime):
    import httpx

    r, _, e, _ = runtime
    original = r.respond

    def respond(request):
        response = original(request)
        if request.url.path.endswith("/embeddings"):
            data = response.json()
            data["native_extra"] = {
                "flag": False,
                "zero": 0,
                "private_key": "controlled-private",
                "credentials": "controlled-private",
            }
            return httpx.Response(200, json=data, request=request)
        return response

    r.respond = respond
    with r.client() as client:
        response = client.embeddings.create(model="native-embedding", input=["hello"])
        chat(
            client,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                key: {"type": "string", "default": "controlled-private"}
                                for key in [
                                    "access_key",
                                    "secret_key",
                                    "private_key",
                                    "credentials",
                                ]
                            },
                        },
                    },
                }
            ],
        )
    span = e.get_finished_spans()[0]
    envelope = json.loads(span.attributes["respan.metadata.together.result"])[0]
    assert (
        len(json.loads(span.attributes[OUTPUT])[0])
        == len(response.data[0].embedding)
        == 5001
    )
    assert "embedding" not in envelope["data"][0] and envelope["native_extra"] == {
        "flag": False,
        "zero": 0,
        "private_key": "[REDACTED]",
        "credentials": "[REDACTED]",
    }
    schema = json.loads(
        e.get_finished_spans()[1].attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS]
    )[0]["function"]["parameters"]
    assert all(
        schema["properties"][key]["default"] == "[REDACTED]"
        and schema["properties"][key]["type"] == "string"
        for key in ["access_key", "secret_key", "private_key", "credentials"]
    )


def test_private_foreign_native_diagnostics_cleared_including_events(runtime):
    r, p, e, _ = runtime

    class Diagnostics(SpanProcessor):
        def on_start(self, span, parent_context=None):
            span.record_exception(ValueError("controlled-private"))
            span.set_attribute(ERROR_MESSAGE, "controlled-private")
            span.set_status(trace.Status(trace.StatusCode.ERROR, "controlled-private"))

    p.add_span_processor(Diagnostics())
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        with r.client() as client:
            chat(client)
    finally:
        context.detach(token)
    span = e.get_finished_spans()[0]
    assert (
        not span.events
        and not span.status.description
        and ERROR_MESSAGE not in span.attributes
    )
