"""Actual released Ollama requests, HTTPX decoding and NDJSON resources."""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Iterator

import httpx
import ollama
import pytest
from opentelemetry import context, trace
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
)
from opentelemetry.semconv_ai import (
    SpanAttributes as A,
)
from opentelemetry.trace import NonRecordingSpan, SpanContext, StatusCode, TraceFlags
from respan_instrumentation_ollama import OllamaInstrumentor
from respan_instrumentation_ollama import _instrumentation as I
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.utils.span_factory import _PROPAGATED_ATTRIBUTES

IN = A.TRACELOOP_ENTITY_INPUT
OUT = A.TRACELOOP_ENTITY_OUTPUT
CHAT = {
    "model": "reported",
    "message": {"role": "assistant", "content": "native", "thinking": "reason"},
    "done": True,
    "prompt_eval_count": 0,
    "eval_count": 0,
    "prompt_eval_cached_count": 0,
}


class Body(httpx.SyncByteStream):
    def __init__(self, frames):
        self.data = [
            json.dumps(f).encode() + b"\n" if type(f) is dict else f for f in frames
        ]
        self.reads = 0
        self.closed = 0

    def __iter__(self):
        for b in self.data:
            self.reads += 1
            yield b

    def close(self):
        self.closed += 1


class ABody(httpx.AsyncByteStream):
    def __init__(self, frames):
        self.body = Body(frames)

    async def __aiter__(self):
        for b in self.body:
            self.body.reads += 0
            yield b

    async def aclose(self):
        self.body.close()


@pytest.fixture
def setup():
    ex = InMemorySpanExporter()
    pr = TracerProvider()
    pr.add_span_processor(SimpleSpanProcessor(ex))
    inst = OllamaInstrumentor(tracer_provider=pr)
    inst.activate()
    yield inst, pr, ex
    for owner in tuple(I._OWNERS):
        owner.deactivate()
    pr.shutdown()


def client(payload=CHAT, status=200, body=None, hooks=None):
    requests = []
    responses = []

    def handler(request):
        requests.append(request)
        r = (
            httpx.Response(status, stream=body)
            if body is not None
            else httpx.Response(status, json=payload)
        )
        responses.append(r)
        return r

    c = ollama.Client(
        transport=httpx.MockTransport(handler),
        event_hooks=hooks or {},
        headers={"Authorization": "Bearer fixture-header-secret"},
    )
    return c, requests, responses


def attrs(ex, index=-1):
    return dict(ex.get_finished_spans()[index].attributes)


def chat(c, **kw):
    return c.chat(
        model="requested", messages=[{"role": "user", "content": "prompt"}], **kw
    )


@pytest.mark.parametrize("mode", ["chat", "generate", "embed", "embeddings"])
def test_native_unary_and_source_fields(setup, mode):
    _, _, ex = setup
    payload = (
        CHAT
        if mode == "chat"
        else {
            "model": "reported",
            "response": "native",
            "thinking": "reason",
            "context": [0, 1],
            "prompt_eval_count": 0,
            "eval_count": 0,
            "prompt_eval_cached_count": 0,
        }
        if mode == "generate"
        else {
            "model": "reported",
            "embeddings": [[0.0] * 5001],
            "embedding": [0.0] * 5001,
            "prompt_eval_count": 0,
        }
    )
    c, req, _res = client(payload)
    kwargs = (
        {"messages": [{"role": "user", "content": "prompt"}]}
        if mode == "chat"
        else {"prompt": "prompt"}
        if mode in ("generate", "embeddings")
        else {"input": ["", "prompt"], "dimensions": 5001, "truncate": False}
    )
    result = getattr(c, mode)(model="requested", **kwargs)
    assert type(result) is getattr(
        ollama,
        {
            "chat": "ChatResponse",
            "generate": "GenerateResponse",
            "embed": "EmbedResponse",
            "embeddings": "EmbeddingsResponse",
        }[mode],
    )
    a = attrs(ex)
    assert a[A.LLM_REQUEST_MODEL] == "requested"
    assert a["gen_ai.response.model"] == "reported"
    assert json.loads(a[IN]) == json.loads(req[0].content)
    assert "fixture-header-secret" not in json.dumps(a)
    assert "gen_ai.usage.total_tokens" not in a
    assert a["http.response.status_code"] == 200
    if mode in ("embed", "embeddings"):
        assert (
            len(json.loads(a[OUT])[0] if mode == "embed" else json.loads(a[OUT]))
            == 5001
        )
    else:
        assert json.loads(a[OUT]) == payload
        assert (
            a["gen_ai.usage.input_tokens"] == 0 and a["gen_ai.usage.output_tokens"] == 0
        )
        assert a[A.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 0
    c._client.close()


@pytest.mark.parametrize("mode", ["chat", "generate"])
def test_full_native_settings_history_schema_and_empty(setup, mode):
    _, _, ex = setup
    c, req, _ = client(
        CHAT
        if mode == "chat"
        else {
            "model": "reported",
            "response": "",
            "thinking": "reason",
            "image": "aGVsbG8=",
            "completed": 0,
            "total": 0,
            "logprobs": [],
        }
    )
    schema = {
        "type": "object",
        "properties": {
            "api_key": {"type": "string", "default": "credential-secret"},
            "nested": {
                "type": "object",
                "properties": {"secret": {"const": "private-secret"}},
            },
        },
    }
    options = ollama.Options(temperature=0, num_predict=0, seed=0, top_p=0)
    if mode == "chat":
        history = [{"role": "user", "content": f"item{i}"} for i in range(75)]
        c.chat(
            model="requested",
            messages=history,
            format=schema,
            options=options,
            think=False,
            keep_alive=0,
        )
        assert len(json.loads(attrs(ex)[IN])["messages"]) == 75
    else:
        kw = (
            {"width": 256, "height": 256, "steps": 0}
            if "width" in inspect.signature(c.generate).parameters
            else {}
        )
        c.generate(
            model="requested",
            prompt="",
            system="",
            suffix="",
            context=[],
            raw=False,
            format=schema,
            options=options,
            think=False,
            keep_alive=0,
            **kw,
        )
    a = attrs(ex)
    native = json.loads(req[0].content)
    recorded = json.loads(a[IN])
    assert recorded["format"]["properties"]["api_key"]["type"] == "string"
    assert recorded["format"]["properties"]["api_key"]["default"] == "[REDACTED]"
    assert (
        recorded["format"]["properties"]["nested"]["properties"]["secret"]["const"]
        == "[REDACTED]"
    )
    assert recorded["options"] == native["options"]
    assert recorded["think"] is False
    assert a[A.LLM_REQUEST_TEMPERATURE] == 0 and a[A.LLM_REQUEST_MAX_TOKENS] == 0
    assert "credential-secret" not in json.dumps(
        a
    ) and "private-secret" not in json.dumps(a)
    assert json.loads(a[OUT])["model"] == "reported"
    c._client.close()


@pytest.mark.parametrize("mode", ["chat", "generate"])
def test_native_stream_long_content_full_frames_and_close(setup, mode):
    _, _, ex = setup
    frames = [
        {
            "model": "reported",
            "message": {
                "role": "assistant",
                "content": "x" * 10000,
                "thinking": "r" * 10000,
            },
        }
        if mode == "chat"
        else {"model": "reported", "response": "x" * 10000, "thinking": "r" * 10000},
        {
            "message": {"role": "assistant", "content": ""},
            "response": "",
            "done": True,
            "prompt_eval_count": 0,
            "eval_count": 0,
            "prompt_eval_cached_count": 0,
        },
    ]
    body = Body(frames)
    c, _, responses = client(body=body)
    stream = (
        chat(c, stream=True)
        if mode == "chat"
        else c.generate(model="requested", prompt="prompt", stream=True)
    )
    assert isinstance(stream, Iterator)
    assert body.reads == 0
    assert len(ex.get_finished_spans()) == 0
    chunks = list(stream)
    a = attrs(ex)
    assert type(chunks[0]) is (
        ollama.ChatResponse if mode == "chat" else ollama.GenerateResponse
    )
    assert len(a[f"{A.LLM_COMPLETIONS}.0.content"]) == 10000
    assert json.loads(a[OUT]) == frames
    assert a["gen_ai.usage.input_tokens"] == 0
    assert body.closed == 1 and responses[0].is_closed
    assert ex.get_finished_spans()[-1].status.status_code is StatusCode.UNSET
    assert "iter_lines" not in responses[0].__dict__
    c._client.close()


@pytest.mark.parametrize("first", [False, True])
def test_native_pre_first_and_partial_close(setup, first):
    _, _, ex = setup
    body = Body([CHAT, CHAT])
    c, req, _ = client(body=body)
    s = chat(c, stream=True)
    native = s.native
    if first:
        next(s)
    result = s.close()
    assert result is None and native.gi_frame is None
    assert len(req) == int(first)
    assert body.closed == int(first)
    assert len(ex.get_finished_spans()) == 1
    if first:
        assert len(json.loads(attrs(ex)[OUT])) == 1
    else:
        assert OUT not in attrs(ex) and "http.response.status_code" not in attrs(ex)
    c._client.close()


def test_native_send_throw_and_exception_identity(setup):
    _, _, ex = setup
    body = Body([CHAT, CHAT])
    c, _, _ = client(body=body)
    s = chat(c, stream=True)
    with pytest.raises(TypeError):
        s.send(1)
    assert len(ex.get_finished_spans()) == 0
    chunk = s.send(None)
    assert type(chunk) is ollama.ChatResponse
    failure = ValueError("native user failure")
    with pytest.raises(ValueError) as caught:
        s.throw(failure)
    assert caught.value is failure
    assert body.closed == 1
    assert ex.get_finished_spans()[-1].status.status_code is StatusCode.ERROR
    c._client.close()


@pytest.mark.parametrize("kind", ["http", "event", "json", "connection"])
def test_native_errors_and_sourced_status(setup, kind):
    _, _, ex = setup
    if kind == "connection":
        error = httpx.ConnectError("native connection failure")
        c = ollama.Client(
            transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(error))
        )
        with pytest.raises(ConnectionError) as caught:
            chat(c)
        assert type(caught.value) is ConnectionError
    elif kind == "http":
        c, _, _ = client({"error": "native failure"}, 429)
        with pytest.raises(ollama.ResponseError) as caught:
            chat(c)
        assert caught.value.status_code == 429
        assert attrs(ex)["http.response.status_code"] == 429
    else:
        body = Body(
            [CHAT, {"error": "native frame failure"}]
            if kind == "event"
            else [CHAT, b"{broken\n"]
        )
        c, _, _ = client(body=body)
        with pytest.raises(
            ollama.ResponseError if kind == "event" else json.JSONDecodeError
        ):
            list(chat(c, stream=True))
        assert body.closed == 1 and len(json.loads(attrs(ex)[OUT])) == 1
        assert attrs(ex)["http.response.status_code"] == 200
    assert ex.get_finished_spans()[-1].status.status_code is StatusCode.ERROR
    if kind == "connection":
        assert "http.response.status_code" not in attrs(ex)
    c._client.close()


@pytest.mark.parametrize(
    "key", [_SUPPRESS_INSTRUMENTATION_KEY, SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY]
)
def test_both_native_suppression_before_capture(setup, key):
    _, _, ex = setup
    c, req, _ = client()
    tok = context.attach(context.set_value(key, True))
    try:
        r = chat(c)
    finally:
        context.detach(tok)
    assert (
        type(r) is ollama.ChatResponse and len(req) == 1 and not ex.get_finished_spans()
    )
    c._client.close()


@pytest.mark.parametrize(
    "kind", ["respan", "traceloop", "env_respan", "env_traceloop", "setting"]
)
def test_initial_content_false_never_widens(setup, monkeypatch, kind):
    inst, pr, ex = setup
    tok = None
    if kind == "setting":
        inst.deactivate()
        inst = OllamaInstrumentor(capture_content=False, tracer_provider=pr)
        inst.activate()
    elif kind.startswith("env"):
        monkeypatch.setenv(
            "RESPAN_TRACE_CONTENT"
            if kind == "env_respan"
            else "TRACELOOP_TRACE_CONTENT",
            " FALSE ",
        )
    else:
        tok = context.attach(
            context.set_value(
                ENABLE_CONTENT_TRACING_KEY
                if kind == "respan"
                else "override_enable_content_tracing",
                False,
            )
        )
    body = Body([CHAT])
    c, _, _ = client(body=body)
    s = chat(c, stream=True)
    if tok is not None:
        context.detach(tok)
    monkeypatch.delenv("RESPAN_TRACE_CONTENT", raising=False)
    monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
    list(s)
    a = attrs(ex)
    assert IN not in a and OUT not in a
    assert "error.message" not in a
    inst.deactivate()
    c._client.close()


def test_late_context_veto_before_restore_and_partial_read(setup):
    _, pr, ex = setup
    c, _, _ = client(body=Body([CHAT, CHAT]))
    tr = pr.get_tracer("native-test")
    with tr.start_as_current_span("parent"):
        s = chat(c, stream=True)
        next(s)
        tok = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        context.detach(tok)
        list(s)
    a = attrs(ex, 0)
    assert IN not in a and OUT not in a
    c._client.close()


def test_active_child_and_finished_ancestor_veto_latches(setup):
    _, pr, ex = setup
    tr = pr.get_tracer("native-test")
    c, _, _ = client()
    with tr.start_as_current_span("parent") as parent:
        with tr.start_as_current_span("child") as child:
            child.set_attribute(ENABLE_CONTENT_TRACING_KEY, False)
            chat(c)
            child.set_attribute(ENABLE_CONTENT_TRACING_KEY, True)
        chat(c)
    calls = [s for s in ex.get_finished_spans() if s.name == "ollama.chat"]
    assert len(calls) == 2 and all(IN not in s.attributes for s in calls)
    with trace.use_span(parent, end_on_exit=False):
        chat(c)
    assert IN not in attrs(ex)
    c._client.close()


@pytest.mark.parametrize("remote", [False, True])
def test_unknown_local_carrier_conservative_remote_preserved(setup, remote):
    _, _, ex = setup
    c, _, _ = client()
    carrier = NonRecordingSpan(SpanContext(1, 2, remote, TraceFlags(1)))
    with trace.use_span(carrier, end_on_exit=False):
        chat(c)
    assert (IN in attrs(ex)) is remote
    assert ex.get_finished_spans()[-1].parent.span_id == 2
    c._client.close()


def test_native_sampler_skips_payload_extraction(monkeypatch):
    ex = InMemorySpanExporter()
    pr = TracerProvider(sampler=ALWAYS_OFF)
    pr.add_span_processor(SimpleSpanProcessor(ex))
    i = OllamaInstrumentor(tracer_provider=pr)
    i.activate()
    monkeypatch.setattr(I, "value", lambda _: pytest.fail("unsampled extraction"))
    c, req, _ = client()
    r = chat(c)
    assert (
        type(r) is ollama.ChatResponse and len(req) == 1 and not ex.get_finished_spans()
    )
    c._client.close()
    i.deactivate()
    pr.shutdown()


def test_two_pending_native_bodies_same_parent_export_guard(setup):
    _, pr, ex = setup
    c, _, _ = client(body=Body([CHAT]))
    d, _, _ = client(body=Body([CHAT]))
    tr = pr.get_tracer("native-test")
    with tr.start_as_current_span("parent") as parent:
        a = chat(c, stream=True)
        b = chat(d, stream=True)
        list(a)
        list(b)
    calls = [s for s in ex.get_finished_spans() if s.name == "ollama.chat"]
    assert len(calls) == 2 and all(
        IN in s.attributes and OUT in s.attributes for s in calls
    )
    assert all(s.parent.span_id == parent.get_span_context().span_id for s in calls)
    c._client.close()
    d._client.close()


def test_native_callable_tool_current_vs_historical_and_schema(setup):
    _, _, ex = setup

    def weather(city: str) -> str:
        """Weather for a city."""
        return city

    payload = {
        "model": "reported",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "weather",
                        "arguments": {"city": "Tokyo", "secret": "hidden"},
                    }
                },
                {"function": {"name": "weather", "arguments": {"city": "Paris"}}},
            ],
        },
    }
    c, _req, _ = client(payload)
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "historical", "arguments": {"city": "old"}}}
            ],
        },
        {"role": "tool", "tool_name": "historical", "content": ""},
    ]
    before = json.dumps(history)
    r = c.chat(model="requested", messages=history, tools=[weather])
    a = attrs(ex)
    assert before == json.dumps(history)
    assert r.message.tool_calls[0].function.arguments["secret"] == "hidden"
    defs = json.loads(a[A.LLM_REQUEST_FUNCTIONS])
    assert defs[0]["function"]["parameters"]["properties"]["city"]["type"] == "string"
    calls = json.loads(a[f"{A.LLM_COMPLETIONS}.0.tool_calls"])
    assert len(calls) == 2 and calls[0]["function"]["name"] == "weather"
    assert "historical" not in a[f"{A.LLM_COMPLETIONS}.0.tool_calls"]
    assert "hidden" not in json.dumps(a)
    assert "historical" in a[f"{A.LLM_PROMPTS}.0.tool_calls"]
    c._client.close()


def test_native_callback_order_request_bytes_and_identity(setup):
    _, _, _ex = setup
    events = []

    def first(r):
        events.append(("first", r))

    def second(r):
        events.append(("second", r))

    c, req, res = client(hooks={"response": [first, second]})
    chat(c)
    assert events == [("first", res[0]), ("second", res[0])]
    native = json.loads(req[0].content)
    assert native["messages"][0]["content"] == "prompt"
    assert c._client.event_hooks["response"][:2] == [first, second]
    assert "json" not in res[0].__dict__
    c._client.close()


def test_native_image_bytes_and_raw_complete_response(setup):
    _, _, ex = setup
    c, req, _ = client(
        {
            "model": "reported",
            "message": {"role": "assistant", "content": ""},
            "done_reason": "unload",
            "context": [0],
            "logprobs": [{"token": "a", "logprob": 0}],
        }
    )
    c.chat(
        model="requested",
        messages=[{"role": "user", "content": "", "images": [b"complete-image-bytes"]}],
    )
    a = attrs(ex)
    assert (
        json.loads(a[IN])["messages"][0]["images"][0]
        == json.loads(req[0].content)["messages"][0]["images"][0]
    )
    assert (
        json.loads(a[OUT])["done_reason"] == "unload"
        and json.loads(a[OUT])["logprobs"][0]["logprob"] == 0
    )
    c._client.close()


def test_empty_native_response_is_output_not_invented_completion(setup):
    _, _, ex = setup
    if ollama.GenerateResponse.model_fields["response"].is_required():
        pytest.skip("native SDK0.6.0 requires a generation response field")
    c, _, _ = client({})
    c.generate(model="requested", prompt="")
    a = attrs(ex)
    assert json.loads(a[OUT]) == {} and f"{A.LLM_COMPLETIONS}.0.content" not in a
    assert "gen_ai.response.model" not in a and "gen_ai.usage.input_tokens" not in a
    c._client.close()


@pytest.mark.parametrize("fault", ["startup", "attributes", "end", "detach", "serde"])
def test_telemetry_faults_preserve_native_and_exact_ambient(setup, monkeypatch, fault):
    _, _, _ex = setup
    c, req, _ = client()
    ambient = context.get_current()
    if fault == "startup":

        def broken(*a, **kw):
            context.attach(context.Context())
            raise ValueError("startup")

        monkeypatch.setattr(I, "_observer", broken)
    elif fault == "attributes":
        monkeypatch.setattr(
            Span,
            "set_attributes",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("attributes")),
        )
    elif fault == "end":

        def broken(*a, **kw):
            context.attach(context.Context())
            raise ValueError("end")

        monkeypatch.setattr(Span, "end", broken)
    elif fault == "detach":
        monkeypatch.setattr(
            context, "detach", lambda *a: (_ for _ in ()).throw(ValueError("detach"))
        )
    else:
        monkeypatch.setattr(
            I, "value", lambda *a, **k: (_ for _ in ()).throw(ValueError("serde"))
        )
    r = chat(c)
    assert (
        type(r) is ollama.ChatResponse
        and r.message.content == "native"
        and len(req) == 1
    )
    assert context.get_current() is ambient
    c._client.close()


def test_unknown_metadata_and_error_hooks_never_run(setup):
    _, _, ex = setup

    class Unknown:
        def __str__(self):
            raise AssertionError("unknown str")

        def model_dump(self):
            pytest.fail("unknown model_dump")

        def __iter__(self):
            pytest.fail("unknown iter")

    token = _PROPAGATED_ATTRIBUTES.set(
        {
            "metadata": {"safe": Unknown(), "api_key": "private"},
            "environment": Unknown(),
        }
    )
    try:
        c, _, _ = client()
        chat(c)
        assert json.loads(attrs(ex)["respan.metadata"]) == {
            "safe": None,
            "api_key": "[REDACTED]",
        }
        c._client.close()
    finally:
        _PROPAGATED_ATTRIBUTES.reset(token)

    class UnknownError(httpx.ConnectError):
        def __str__(self):
            raise AssertionError("unknown error str")

    err = UnknownError("private")
    # SDK itself converts ConnectError to ConnectionError without stringifying.
    c = ollama.Client(
        transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(err))
    )
    with pytest.raises(ConnectionError):
        chat(c)
    assert "private" not in json.dumps(attrs(ex))
    c._client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chat", "generate", "embed", "embeddings"])
async def test_native_async_unary(setup, mode):
    _, _, ex = setup
    requests = []

    async def handler(req):
        requests.append(req)
        return httpx.Response(
            200,
            json=CHAT
            if mode == "chat"
            else {
                "model": "reported",
                "response": "native",
                "thinking": "reason",
                "embeddings": [[0.0] * 5001],
                "embedding": [0.0] * 5001,
                "prompt_eval_count": 0,
            },
        )

    c = ollama.AsyncClient(transport=httpx.MockTransport(handler))
    kw = (
        {"messages": [{"role": "user", "content": "prompt"}]}
        if mode == "chat"
        else {"input": ""}
        if mode == "embed"
        else {"prompt": ""}
    )
    r = await getattr(c, mode)(model="requested", **kw)
    a = attrs(ex)
    assert type(r) is getattr(
        ollama,
        {
            "chat": "ChatResponse",
            "generate": "GenerateResponse",
            "embed": "EmbedResponse",
            "embeddings": "EmbeddingsResponse",
        }[mode],
    )
    assert json.loads(a[IN]) == json.loads(requests[0].content)
    assert a["http.response.status_code"] == 200
    await c._client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["all", "preclose", "partial", "asend", "athrow", "error"]
)
async def test_native_async_stream_protocol_resources(setup, kind):
    _, _, ex = setup
    body = ABody([CHAT, {"error": "failure"}] if kind == "error" else [CHAT, CHAT])
    requests = []

    async def handler(req):
        requests.append(req)
        return httpx.Response(200, stream=body)

    c = ollama.AsyncClient(transport=httpx.MockTransport(handler))
    s = await c.chat(model="requested", messages=[], stream=True)
    assert isinstance(s, AsyncIterator) and not requests
    if kind == "all":
        assert len([r async for r in s]) == 2
    elif kind == "preclose":
        await s.aclose()
        assert not requests and s.native.ag_frame is None and OUT not in attrs(ex)
    elif kind == "partial":
        await s.__anext__()
        await s.aclose()
        assert len(json.loads(attrs(ex)[OUT])) == 1
    elif kind == "asend":
        with pytest.raises(TypeError):
            await s.asend(1)
        assert not ex.get_finished_spans()
        assert type(await s.asend(None)) is ollama.ChatResponse
        await s.aclose()
    elif kind == "athrow":
        await s.__anext__()
        failure = ValueError("native")
        with pytest.raises(ValueError) as caught:
            await s.athrow(failure)
        assert caught.value is failure
    else:
        with pytest.raises(ollama.ResponseError):
            [r async for r in s]
        assert len(json.loads(attrs(ex)[OUT])) == 1
    assert body.body.closed == int(kind != "preclose")
    await c._client.aclose()


@pytest.mark.asyncio
async def test_native_async_concurrent_requests_and_callbacks(setup):
    import asyncio

    _, pr, ex = setup
    events = []

    async def hook(r):
        events.append(r)

    async def handler(req):
        await asyncio.sleep(0)
        return httpx.Response(200, json=CHAT)

    c = ollama.AsyncClient(
        transport=httpx.MockTransport(handler), event_hooks={"response": [hook]}
    )
    with pr.get_tracer("native-test").start_as_current_span("parent") as parent:
        results = await asyncio.gather(
            *[
                c.chat(
                    model="requested", messages=[{"role": "user", "content": str(i)}]
                )
                for i in range(2)
            ]
        )
    calls = [s for s in ex.get_finished_spans() if s.name == "ollama.chat"]
    assert len(events) == len(results) == len(calls) == 2
    assert {json.loads(s.attributes[IN])["messages"][0]["content"] for s in calls} == {
        "0",
        "1",
    }
    assert all(s.parent.span_id == parent.get_span_context().span_id for s in calls)
    await c._client.aclose()


@pytest.mark.parametrize("mode", ["chat", "generate"])
def test_fragmented_native_stream_secrets_redacted_without_changing_chunks(setup, mode):
    _, _, ex = setup
    if mode == "chat":
        frames = [
            {
                "message": {
                    "role": "assistant",
                    "content": 'Bearer "frag',
                    "thinking": 'private_key="frag',
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": 'ment-secret"',
                    "thinking": 'ment-secret"',
                }
            },
        ]
    else:
        frames = [
            {"response": 'Bearer "frag', "thinking": 'private_key="frag'},
            {"response": 'ment-secret"', "thinking": 'ment-secret"'},
        ]
    c, _body, _ = client(body=Body(frames))
    stream = (
        chat(c, stream=True)
        if mode == "chat"
        else c.generate(model="requested", prompt="", stream=True)
    )
    chunks = list(stream)
    a = attrs(ex)
    assert "ment-secret" not in json.dumps(a) and '"frag' not in json.dumps(a)
    assert "ment-secret" in (
        chunks[1].message.content if mode == "chat" else chunks[1].response
    )
    assert "[REDACTED]" in a[f"{A.LLM_COMPLETIONS}.0.content"]
    c._client.close()


def test_actual_native_on_start_processor_fault_ends_started_span_and_restores_context(
    setup, monkeypatch
):
    _, _pr, ex = setup
    observer = I._OBSERVERS[0][1]
    original = observer.on_start
    ambient = context.get_current()

    def broken(span, parent_context=None):
        original(span, parent_context)
        context.attach(context.Context())
        raise ValueError("native processor fault")

    monkeypatch.setattr(observer, "on_start", broken)
    c, req, _ = client()
    r = chat(c)
    assert (
        type(r) is ollama.ChatResponse
        and len(req) == 1
        and context.get_current() is ambient
    )
    assert (
        len(ex.get_finished_spans()) == 1
        and ex.get_finished_spans()[0].end_time is not None
    )
    assert not I._PENDING
    c._client.close()


def test_content_veto_during_native_attribute_fault_scrubs_before_end(
    setup, monkeypatch
):
    _, _, ex = setup
    original = Span.set_attributes
    ambient = context.get_current()

    def broken(span, values):
        original(span, values)
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        raise ValueError("attribute callback failure")

    monkeypatch.setattr(Span, "set_attributes", broken)
    c, _, _ = client()
    r = chat(c)
    a = attrs(ex)
    assert type(r) is ollama.ChatResponse and IN not in a and OUT not in a
    assert context.get_current() is ambient
    c._client.close()


def test_unknown_suppression_flag_bool_hook_not_invoked(setup):
    _, _, ex = setup

    class Unknown:
        def __bool__(self):
            raise AssertionError("unknown bool hook")

    token = context.attach(context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, Unknown()))
    c, _, _ = client()
    try:
        assert type(chat(c)) is ollama.ChatResponse
    finally:
        context.detach(token)
    assert not ex.get_finished_spans()
    c._client.close()


def test_native_pending_stream_survives_deactivation_and_foreign_tap(setup):
    inst, _, ex = setup
    body = Body([CHAT, CHAT])
    c, _, responses = client(body=body)
    s = chat(c, stream=True)
    next(s)
    tap = responses[0].iter_lines

    def foreign(*a, **kw):
        return tap(*a, **kw)

    responses[0].iter_lines = foreign
    inst.deactivate()
    assert responses[0].iter_lines is foreign
    assert len(list(s)) == 1 and body.closed and len(ex.get_finished_spans()) == 1
    c._client.close()


def test_native_unrecognized_error_preserved_without_customer_hooks(setup):
    _, _, ex = setup

    class UnknownError(RuntimeError):
        def __str__(self):
            raise AssertionError("unknown str hook")

    error = UnknownError("unknown diagnostic private")
    c = ollama.Client(
        transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(error))
    )
    with pytest.raises(UnknownError) as caught:
        chat(c)
    assert caught.value is error and "unknown diagnostic private" not in json.dumps(
        attrs(ex)
    )
    assert "error.message" not in attrs(ex)
    c._client.close()


def test_native_unknown_local_span_id_collision_does_not_widen(setup):
    _, pr, ex = setup
    c, _, _ = client()
    with pr.get_tracer("native-test").start_as_current_span(
        "observed-parent"
    ) as parent:
        sc = parent.get_span_context()
        unknown = NonRecordingSpan(
            SpanContext(sc.trace_id + 1, sc.span_id, False, TraceFlags(1))
        )
        with trace.use_span(unknown, end_on_exit=False):
            chat(c)
        assert IN not in attrs(ex)
        chat(c)
        assert IN in attrs(ex)
    c._client.close()


def test_native_unsampled_model_only_extraction_not_invoked(monkeypatch):
    pr = TracerProvider(sampler=ALWAYS_OFF)
    i = OllamaInstrumentor(tracer_provider=pr)
    i.activate()
    monkeypatch.setattr(
        I, "_model_only", lambda _: pytest.fail("unsampled model extraction")
    )
    c, _, _ = client()
    try:
        assert type(chat(c)) is ollama.ChatResponse
    finally:
        c._client.close()
        i.deactivate()
        pr.shutdown()


@pytest.mark.parametrize("capture", [True, False])
def test_native_embedding_vectors_and_extra_envelope_capture_gate(setup, capture):
    _, _, ex = setup
    payload = {
        "model": "reported",
        "embeddings": [[0.0] * 5001],
        "prompt_eval_count": 0,
        "total_duration": 0,
        "custom": {"flag": False, "api_key": "controlled-secret"},
    }
    c, _, _ = client(payload)
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, capture))
    try:
        r = c.embed(model="requested", input="")
    finally:
        context.detach(token)
    a = attrs(ex)
    assert len(r.embeddings[0]) == 5001
    if capture:
        assert len(json.loads(a[OUT])[0]) == 5001
        envelope = json.loads(a["respan.metadata.ollama.result"])
        assert (
            "embeddings" not in envelope
            and envelope["total_duration"] == 0
            and envelope["custom"]["flag"] is False
        )
        assert envelope["custom"]["api_key"] == "[REDACTED]"
    else:
        assert OUT not in a and "respan.metadata.ollama.result" not in a
    c._client.close()


@pytest.mark.parametrize("fault", [False, True])
def test_native_owned_diagnostic_events_scrubbed_on_veto(setup, monkeypatch, fault):
    from opentelemetry.sdk.trace import SpanProcessor

    _, pr, ex = setup

    class Diagnostics(SpanProcessor):
        def on_start(self, span, parent_context=None):
            span.record_exception(RuntimeError("controlled-private-diagnostic"))

        def on_end(self, span):
            pass

        def shutdown(self):
            pass

        def force_flush(self, *a, **kw):
            return True

    pr.add_span_processor(Diagnostics())
    if fault:
        original = Span.set_attributes

        def attribute_fault(span, values):
            original(span, values)
            context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
            raise ValueError("attribute fault veto")

        monkeypatch.setattr(Span, "set_attributes", attribute_fault)
    c, _, _ = client()
    ambient = context.get_current()
    token = None
    if not fault:
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        r = chat(c)
    finally:
        if token is not None:
            context.detach(token)
    span = ex.get_finished_spans()[-1]
    assert (
        type(r) is ollama.ChatResponse
        and not span.events
        and IN not in span.attributes
        and OUT not in span.attributes
    )
    assert context.get_current() is ambient
    c._client.close()
