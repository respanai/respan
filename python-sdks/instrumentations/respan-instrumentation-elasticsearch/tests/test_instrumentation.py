"""Real released urllib3/aiohttp clients, HTTP parsing and OTel lifecycle."""

from __future__ import annotations

import asyncio
import functools
import json

import pytest
from elastic_transport import HeadApiResponse, NodeConfig, ObjectApiResponse, Transport
from elasticsearch import (
    AsyncElasticsearch,
    Elasticsearch,
    NotFoundError,
    SerializationError,
)
from elasticsearch.helpers import streaming_bulk
from opentelemetry import context, trace
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider, _Span
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import StatusCode
from respan_instrumentation_elasticsearch import ElasticsearchInstrumentor
from respan_instrumentation_elasticsearch import _instrumentation as adapter
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

from .native_http import server

INPUT = SpanAttributes.TRACELOOP_ENTITY_INPUT
OUTPUT = SpanAttributes.TRACELOOP_ENTITY_OUTPUT


@pytest.fixture
def tracing():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    owners = []

    def activate(**kwargs):
        owner = ElasticsearchInstrumentor(tracer_provider=provider, **kwargs)
        owner.activate()
        owners.append(owner)
        return owner

    yield provider, exporter, activate
    for owner in reversed(owners):
        owner.deactivate()
    provider.shutdown()


def payload():
    return {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "vector": [0.0] * 5001,
                        "history": [{"value": i} for i in range(75)],
                        "zero": 0,
                        "false": False,
                        "empty": "",
                        "api_key": "controlled-secret",
                    }
                }
            ]
        },
        "took": 0,
        "timed_out": False,
    }


def call(url, *, asynchronous=False, **kwargs):
    if asynchronous:

        async def run():
            async with AsyncElasticsearch(url, max_retries=0) as client:
                return await client.search(
                    index="controlled-index",
                    body={"query": {"match": {"text": "private-controlled"}}},
                    **kwargs,
                )

        return asyncio.run(run())
    with Elasticsearch(url, max_retries=0) as client:
        return client.search(
            index="controlled-index",
            body={"query": {"match": {"text": "private-controlled"}}},
            **kwargs,
        )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_full_response_and_request(tracing, asynchronous):
    _, exporter, activate = tracing
    activate()
    with server(payload()) as (url, requests):
        response = call(url, asynchronous=asynchronous, request_cache=False)
    assert type(response) is ObjectApiResponse
    assert len(response.body["hits"]["hits"][0]["_source"]["vector"]) == 5001
    assert response.body["hits"]["hits"][0]["_source"]["api_key"] == "controlled-secret"
    assert len(requests) == 1
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = span.attributes
    output = json.loads(attrs[OUTPUT])
    source = output["hits"]["hits"][0]["_source"]
    assert len(source["vector"]) == 5001 and len(source["history"]) == 75
    assert (source["zero"], source["false"], source["empty"]) == (0, False, "")
    assert source["api_key"] == "[REDACTED]"
    request = json.loads(attrs[INPUT])
    assert request["params"]["request_cache"] == False
    assert request["body"]["query"]["match"]["text"] == "private-controlled"
    assert request["transport"]["body"] == request["body"]
    assert (
        attrs["respan.entity.log_type"] == "task"
        and attrs["db.collection.name"] == "controlled-index"
    )
    assert (
        attrs["http.response.status_code"] == 200
        and attrs["db.response.status_code"] == "200"
    )
    assert (
        span.name == "search" and span.instrumentation_scope.name == "elasticsearch-api"
    )
    assert attrs["traceloop.entity.path"] == ""
    assert not any(k.startswith(("gen_ai.", "llm.")) for k in attrs)
    assert not any(
        k in attrs
        for k in (
            "status_code",
            "model",
            "traceloop.span.kind",
            "elasticsearch.status_code",
        )
    )


@pytest.mark.parametrize(
    "gate", ["capture", "canonical", "traceloop", "respan_env", "traceloop_env"]
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_private_bodyless(tracing, monkeypatch, gate, asynchronous):
    _, exporter, activate = tracing
    activate(capture_content=gate != "capture")
    token = None
    if gate in ("canonical", "traceloop"):
        token = context.attach(
            context.set_value(
                ENABLE_CONTENT_TRACING_KEY
                if gate == "canonical"
                else "override_enable_content_tracing",
                False,
            )
        )
    if gate.endswith("env"):
        monkeypatch.setenv(
            "RESPAN_TRACE_CONTENT"
            if gate == "respan_env"
            else "TRACELOOP_TRACE_CONTENT",
            "false",
        )
    try:
        with server(payload()) as (url, _requests):
            assert type(call(url, asynchronous=asynchronous)) is ObjectApiResponse
    finally:
        if token is not None:
            context.detach(token)
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert INPUT not in spans[0].attributes and OUTPUT not in spans[0].attributes
    assert not spans[0].events and spans[0].status.description is None
    assert "private-controlled" not in json.dumps(dict(spans[0].attributes))


@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_suppression_before_native_span(tracing, key, asynchronous):
    _, exporter, activate = tracing
    activate()
    token = context.attach(context.set_value(key, True))
    try:
        with server() as (url, requests):
            call(url, asynchronous=asynchronous)
    finally:
        context.detach(token)
    assert len(requests) == 1 and exporter.get_finished_spans() == ()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sampled_out_has_no_customer_conversion(monkeypatch, asynchronous):
    provider = TracerProvider(sampler=ALWAYS_OFF)
    owner = ElasticsearchInstrumentor(tracer_provider=provider)
    owner.activate()
    calls = []

    class Meta(type):
        def __eq__(self, other):
            calls.append("eq")
            return False

    class Opaque(metaclass=Meta):
        def model_dump(self, *args, **kwargs):
            calls.append("dump")
            return {}

        def __str__(self):
            calls.append("str")
            return "opaque"

        def __repr__(self):
            calls.append("repr")
            return "opaque"

    try:
        with server() as (url, _requests):

            def invoke():
                if asynchronous:

                    async def run():
                        async with AsyncElasticsearch(url, max_retries=0) as client:
                            await client.search(index="i", body={"query": Opaque()})

                    asyncio.run(run())
                else:
                    with Elasticsearch(url, max_retries=0) as client:
                        client.search(index="i", body={"query": Opaque()})

            with pytest.raises(SerializationError):
                invoke()
            sampled = list(calls)
            calls.clear()
            owner.deactivate()
            with pytest.raises(SerializationError):
                invoke()
            assert calls == sampled and "dump" not in calls and "eq" not in calls
    finally:
        owner.deactivate()
        provider.shutdown()


@pytest.mark.parametrize("hookname", ["request_hook", "response_hook"])
def test_hook_fault_preserves_native_response(tracing, hookname):
    _, exporter, activate = tracing

    def fail(*args):
        raise RuntimeError("controlled hook fault")

    activate(**{hookname: fail})
    with server() as (url, requests):
        response = call(url)
    assert type(response) is ObjectApiResponse and len(requests) == 1
    span = exporter.get_finished_spans()[0]
    assert INPUT not in span.attributes and OUTPUT not in span.attributes


def test_actual_not_found_no_synthetic_output(tracing):
    _, exporter, activate = tracing
    activate()
    with (
        server(
            {
                "error": {
                    "type": "index_not_found_exception",
                    "reason": "password=controlled",
                },
                "status": 404,
            },
            status=404,
        ) as (url, requests),
        Elasticsearch(url, max_retries=0) as client,
        pytest.raises(NotFoundError) as caught,
    ):
        client.get(index="missing", id="doc")
    assert caught.value.meta.status == 404 and len(requests) == 1
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert (
        span.attributes["error.type"] == "NotFoundError"
        and span.attributes["http.response.status_code"] == 404
    )
    assert OUTPUT not in span.attributes and not span.events
    assert "controlled" not in json.dumps(dict(span.attributes))


def test_ignored_status_native_output_and_false_head(tracing):
    _, exporter, activate = tracing
    activate()
    with (
        server({"error": "actual-missing", "status": 404}, status=404) as (
            url,
            _requests,
        ),
        Elasticsearch(url, max_retries=0) as client,
    ):
        response = client.options(ignore_status=404).get(index="missing", id="doc")
        head = client.exists(index="missing", id="doc")
    assert type(head) is HeadApiResponse and head.body is False
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert json.loads(spans[0].attributes[OUTPUT]) == response.body
    assert json.loads(spans[1].attributes[OUTPUT]) is False
    assert all(s.status.status_code is not StatusCode.ERROR for s in spans)


def test_native_retry_single_span(tracing):
    _, exporter, activate = tracing
    activate()
    with (
        server(sequence=[(503, {"error": "retry"}), (200, {"result": "actual"})]) as (
            url,
            requests,
        ),
        Elasticsearch(url, max_retries=1, retry_on_status=[503]) as client,
    ):
        response = client.search(index="i", query={"match_all": {}})
    assert response.body == {"result": "actual"} and len(requests) == 2
    spans = exporter.get_finished_spans()
    assert len(spans) == 1 and spans[0].attributes["http.response.status_code"] == 200


def test_direct_native_transport(tracing):
    _, exporter, activate = tracing
    activate()
    with server({"actual": 0}) as (url, _requests):
        from urllib.parse import urlsplit

        parsed = urlsplit(url)
        transport = Transport([NodeConfig("http", parsed.hostname, parsed.port)])
        try:
            response = transport.perform_request("GET", "/", max_retries=0)
        finally:
            transport.close()
    assert response.body == {"actual": 0}
    spans = exporter.get_finished_spans()
    assert len(spans) == 1 and json.loads(spans[0].attributes[OUTPUT]) == {"actual": 0}


@pytest.mark.parametrize(
    "attribute", [ENABLE_CONTENT_TRACING_KEY, "traceloop.enable_content_tracing"]
)
def test_initial_parent_attribute_irreversible(tracing, attribute):
    provider, exporter, activate = tracing
    activate()
    with provider.get_tracer("application").start_as_current_span(
        "parent", attributes={attribute: False}
    ) as parent:
        parent.set_attribute(attribute, True)
        with server() as (url, _):
            call(url)
    span = next(s for s in exporter.get_finished_spans() if s.name == "search")
    assert INPUT not in span.attributes and OUTPUT not in span.attributes


def test_active_parent_observed_by_generic_child(tracing):
    provider, exporter, activate = tracing
    activate()
    with provider.get_tracer("application").start_as_current_span("parent") as parent:
        parent.set_attribute(ENABLE_CONTENT_TRACING_KEY, False)
        with provider.get_tracer("application").start_as_current_span("generic"):
            pass
        parent.set_attribute(ENABLE_CONTENT_TRACING_KEY, True)
        with server() as (url, _):
            call(url)
    span = next(s for s in exporter.get_finished_spans() if s.name == "search")
    assert OUTPUT not in span.attributes


def test_late_response_hook_veto(tracing):
    _, exporter, activate = tracing
    token = []

    def veto(*args):
        token.append(
            context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        )

    activate(response_hook=veto)
    ambient = context.get_current()
    with server() as (url, _):
        call(url)
    assert context.get_current() is ambient
    span = exporter.get_finished_spans()[0]
    assert INPUT not in span.attributes and OUTPUT not in span.attributes


def test_late_end_readable_snapshot_scrub(tracing, monkeypatch):
    _, exporter, activate = tracing
    activate()
    original = _Span.end

    def end(self, *args, **kwargs):
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        try:
            return original(self, *args, **kwargs)
        finally:
            context.detach(token)

    monkeypatch.setattr(_Span, "end", end)
    with server() as (url, _):
        call(url)
    span = exporter.get_finished_spans()[0]
    assert INPUT not in span.attributes and OUTPUT not in span.attributes
    assert not span.events and span.status.description is None


def test_foreign_start_fault_native_survives(tracing):
    provider, exporter, activate = tracing

    class Foreign(SpanProcessor):
        def on_start(self, span, parent_context=None):
            if span.name == "search":
                raise RuntimeError("controlled startup")

    foreign = Foreign()
    provider.add_span_processor(foreign)
    activate()
    with server() as (url, requests):
        response = call(url)
    assert type(response) is ObjectApiResponse and len(requests) == 1
    spans = exporter.get_finished_spans()
    assert len(spans) == 1 and INPUT not in spans[0].attributes


def test_native_attribute_mutate_raise(tracing, monkeypatch):
    _, exporter, activate = tracing
    activate()
    original = _Span.set_attribute
    count = []

    def setter(self, key, item):
        original(self, key, item)
        if key == "http.request.method" and not count:
            count.append(True)
            raise RuntimeError("controlled attr")

    monkeypatch.setattr(_Span, "set_attribute", setter)
    with server() as (url, requests):
        response = call(url)
    assert type(response) is ObjectApiResponse and len(requests) == 1
    span = exporter.get_finished_spans()[0]
    assert INPUT not in span.attributes and OUTPUT not in span.attributes


def test_attach_mutates_then_raises_restores_context(tracing, monkeypatch):
    _, _, activate = tracing
    activate()
    original = context.attach
    count = []
    ambient = context.get_current()

    def attach(ctx):
        token = original(ctx)
        if not count:
            count.append(True)
            raise RuntimeError("controlled attach")
        return token

    monkeypatch.setattr(context, "attach", attach)
    with server() as (url, _):
        response = call(url)
    assert type(response) is ObjectApiResponse and context.get_current() is ambient


def test_detach_fault_restores_context(tracing, monkeypatch):
    _, _, activate = tracing
    activate()
    ambient = context.get_current()

    def detach(*args):
        raise RuntimeError("controlled detach")

    monkeypatch.setattr(context, "detach", detach)
    with server() as (url, _):
        call(url)
    assert context.get_current() is ambient


def test_remote_and_unknown_carrier_tuple_ids(tracing):
    provider, exporter, activate = tracing
    activate()
    with server() as (url, _):
        remote = trace.NonRecordingSpan(
            trace.SpanContext(123, 456, True, trace.TraceFlags(1))
        )
        token = context.attach(trace.set_span_in_context(remote))
        try:
            call(url)
        finally:
            context.detach(token)
        known = provider.get_tracer("application").start_span("known")
        key = known.get_span_context()
        known.end()
        unknown = trace.NonRecordingSpan(
            trace.SpanContext(key.trace_id + 1, key.span_id, False, trace.TraceFlags(1))
        )
        token = context.attach(trace.set_span_in_context(unknown))
        try:
            call(url)
        finally:
            context.detach(token)
    spans = [s for s in exporter.get_finished_spans() if s.name == "search"]
    assert INPUT in spans[0].attributes and INPUT not in spans[1].attributes


def test_native_raw_query_credentials_valid_schema(tracing, monkeypatch):
    _, exporter, activate = tracing
    activate()
    monkeypatch.setenv(
        "OTEL_PYTHON_INSTRUMENTATION_ELASTICSEARCH_CAPTURE_SEARCH_QUERY", "raw"
    )
    body = {
        "custom": {
            "properties": {
                "api_key": {"type": "string", "default": "controlled-schema"}
            }
        },
        "query": {"match": {"text": 'Bearer "controlled bearer secret"'}},
        "url": "https://example.invalid/?api%5Fkey=controlled-url",
    }
    with server(body) as (url, _), Elasticsearch(url, max_retries=0) as client:
        response = client.search(index="i", body=body)
    attrs = exporter.get_finished_spans()[0].attributes
    encoded = json.dumps(dict(attrs))
    assert not any(
        x in encoded
        for x in ["controlled-schema", "controlled bearer secret", "controlled-url"]
    )
    output = json.loads(attrs[OUTPUT])
    assert output["custom"]["properties"]["api_key"]["type"] == "string"
    assert output["custom"]["properties"]["api_key"]["default"] == "[REDACTED]"
    assert json.loads(attrs["db.query.text"])
    assert response.body == body


def test_shared_lifecycle_conflict_foreign_wrapper(tracing, monkeypatch):
    provider, _, activate = tracing
    first = activate()
    second = activate()
    first.deactivate()
    owner = ElasticsearchInstrumentor(tracer_provider=provider, capture_content=False)
    with pytest.raises(ValueError):
        owner.activate()
    from elasticsearch._sync.client._base import BaseClient

    installed = BaseClient._perform_request

    @functools.wraps(installed)
    def foreign(*args, **kwargs):
        return installed(*args, **kwargs)

    monkeypatch.setattr(BaseClient, "_perform_request", foreign)
    second.deactivate()
    assert BaseClient._perform_request is foreign
    assert all(
        type(p).__name__ != "AncestorPolicy"
        for p in provider._active_span_processor._span_processors
    )


def test_activation_mutate_raise_rollback(tracing, monkeypatch):
    provider, _, _ = tracing
    from elasticsearch._sync.client._base import BaseClient

    original = BaseClient._perform_request
    changes = []

    def setter(owner, name, value):
        setattr(owner, name, value)
        if owner is BaseClient and name == "_perform_request" and not changes:
            changes.append(True)
            raise RuntimeError("controlled activation")

    monkeypatch.setattr(adapter, "setattr", setter, raising=False)
    owner = ElasticsearchInstrumentor(tracer_provider=provider)
    with pytest.raises(RuntimeError):
        owner.activate()
    assert BaseClient._perform_request is original
    assert not adapter._PATCHES and not adapter._OWNERS
    assert all(
        type(p).__name__ != "AncestorPolicy"
        for p in provider._active_span_processor._span_processors
    )


def test_two_siblings_survive_processor_suppression(tracing):
    provider, exporter, activate = tracing
    activate()
    with (
        provider.get_tracer("application").start_as_current_span("parent"),
        server() as (url, _),
    ):
        call(url)
        call(url)
    spans = [s for s in exporter.get_finished_spans() if s.name == "search"]
    assert len(spans) == 2 and all(
        INPUT in s.attributes and OUTPUT in s.attributes for s in spans
    )


def test_streaming_bulk_real_native_generator(tracing):
    _, exporter, activate = tracing
    activate()
    reply = {
        "errors": False,
        "items": [
            {"index": {"_index": "i", "_id": "1", "status": 201, "result": "created"}}
        ],
    }
    with server(reply) as (url, requests), Elasticsearch(url, max_retries=0) as client:
        generator = streaming_bulk(
            client,
            [
                {
                    "_index": "i",
                    "_id": "1",
                    "_source": {
                        "text": "controlled",
                        "password": "controlled-bulk-secret",
                    },
                }
            ],
        )
        assert type(generator).__name__ == "generator"
        result = list(generator)
    assert result[0][0] is True and len(requests) == 1
    spans = exporter.get_finished_spans()
    assert any(s.name == "bulk" and OUTPUT in s.attributes for s in spans)
    assert all(s.attributes["respan.entity.log_type"] == "task" for s in spans)
    assert b"controlled-bulk-secret" in requests[0]["body"]
    import base64

    def credential(value):
        if type(value) is dict:
            if "base64" in value:
                return b"controlled-bulk-secret" in base64.b64decode(value["base64"])
            return any(credential(item) for item in value.values())
        if type(value) is list:
            return any(credential(item) for item in value)
        return type(value) is str and "controlled-bulk-secret" in value

    assert not any(
        credential(json.loads(s.attributes[INPUT]))
        for s in spans
        if INPUT in s.attributes
    )


@pytest.mark.parametrize(
    "body,content_type,expected_type",
    [
        ([{"actual": False, "zero": 0}], "application/json", "ListApiResponse"),
        ("native text password=controlled-text", "text/plain", "TextApiResponse"),
        (b"native\x00bytes", "application/vnd.mapbox-vector-tile", "BinaryApiResponse"),
        ({}, "application/json", "ObjectApiResponse"),
    ],
)
def test_native_typed_return_protocols(tracing, body, content_type, expected_type):
    _, exporter, activate = tracing
    activate()
    with (
        server(body, content_type=content_type) as (url, requests),
        Elasticsearch(url, max_retries=0) as client,
    ):
        response = client.perform_request(
            "GET", "/controlled", endpoint_id="controlled"
        )
    assert (
        type(response).__name__ == expected_type
        and response.body == body
        and len(requests) == 1
    )
    output = json.loads(exporter.get_finished_spans()[0].attributes[OUTPUT])
    if type(body) is bytes:
        import base64

        assert base64.b64decode(output["base64"]) == body
    elif type(body) is str:
        assert output == "native text password=[REDACTED]"
    else:
        assert output == body


def test_released_disabled_respan_tracer(tracing, monkeypatch):
    from respan_tracing.core.tracer import RespanTracer

    _, exporter, activate = tracing
    monkeypatch.setattr(RespanTracer, "_instance", None)
    RespanTracer(is_enabled=False)
    activate()
    with server() as (url, requests):
        call(url)
    assert len(requests) == 1 and not exporter.get_finished_spans()


def test_disabled_native_tracer_respected(tracing, monkeypatch):
    _, exporter, activate = tracing
    activate()
    monkeypatch.setenv("OTEL_PYTHON_INSTRUMENTATION_ELASTICSEARCH_ENABLED", "false")
    with server() as (url, requests):
        call(url)
    assert len(requests) == 1 and not exporter.get_finished_spans()


def test_private_foreign_diagnostics_and_marker(tracing):
    provider, exporter, activate = tracing

    class Marker(SpanProcessor):
        def on_start(self, span, parent_context=None):
            span.set_attribute("respan.metadata.run_id", "controlled-marker")
            span.set_attribute(
                "respan.metadata", json.dumps({"run_id": "controlled-marker"})
            )
            if span.name == "search":
                span.record_exception(RuntimeError("controlled diagnostic"))
                from opentelemetry.trace import Status

                span.set_status(Status(StatusCode.ERROR, "controlled diagnostic"))

    provider.add_span_processor(Marker())
    activate(capture_content=False)
    with server() as (url, _):
        call(url)
    span = exporter.get_finished_spans()[0]
    assert not span.events and span.status.description is None
    assert span.attributes["respan.metadata.run_id"] == "controlled-marker"
    assert json.loads(span.attributes["respan.metadata"]) == {
        "run_id": "controlled-marker"
    }
    assert (
        "error.type" not in span.attributes and "error.message" not in span.attributes
    )


def test_recording_opaque_metaclass_has_no_extra_hooks(tracing):
    _, _, activate = tracing
    owner = activate()
    calls = []

    class Meta(type):
        def __eq__(self, other):
            calls.append("eq")
            return False

    class Opaque(metaclass=Meta):
        def __repr__(self):
            calls.append("repr")
            return "opaque"

        def __str__(self):
            calls.append("str")
            return "opaque"

        def model_dump(self, *args, **kwargs):
            calls.append("dump")
            return {}

    from elasticsearch import SerializationError

    with server() as (url, _), Elasticsearch(url, max_retries=0) as client:
        with pytest.raises(SerializationError):
            client.search(index="i", body={"query": Opaque()})
        instrumented = list(calls)
        calls.clear()
        owner.deactivate()
        with pytest.raises(SerializationError):
            client.search(index="i", body={"query": Opaque()})
    assert calls == instrumented and "eq" not in calls and "dump" not in calls
