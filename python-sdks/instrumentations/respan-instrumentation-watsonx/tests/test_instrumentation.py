"""Real current/minimum IBM SDK constructors, JSON/SSE parsing and resources."""

from __future__ import annotations

import gc
import json

import httpx
import pytest
from _native import runtime
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import (
    TextChatParameters,
)
from ibm_watsonx_ai.wml_client_error import WMLClientError
from opentelemetry import context, trace
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes as A
from opentelemetry.trace import NonRecordingSpan, SpanContext, StatusCode, TraceFlags
from respan_instrumentation_watsonx import WatsonxInstrumentor
from respan_instrumentation_watsonx import _instrumentation as I
from respan_instrumentation_watsonx._privacy import json_text, text, value
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.utils.span_factory import _PROPAGATED_ATTRIBUTES

IN = A.TRACELOOP_ENTITY_INPUT
OUT = A.TRACELOOP_ENTITY_OUTPUT
CHAT = {
    "model_id": "reported",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "native",
                "reasoning_content": "controlled reasoning",
                "tool_calls": [
                    {
                        "id": "native-id",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"city":"Tokyo"}',
                        },
                    }
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    "custom": {"flag": False},
}
GEN = {
    "model_id": "reported",
    "results": [
        {
            "generated_text": "native",
            "input_token_count": 0,
            "generated_token_count": 0,
            "stop_reason": "eos_token",
        }
    ],
    "custom": {"flag": False},
}
EMB = {
    "model_id": "reported",
    "results": [{"embedding": [0.0] * 5001, "input": "native"}],
    "input_token_count": 0,
    "custom": {"flag": False, "api_key": "controlled-secret"},
}


@pytest.fixture
def setup():
    ex = InMemorySpanExporter()
    pr = TracerProvider()
    pr.add_span_processor(SimpleSpanProcessor(ex))
    i = WatsonxInstrumentor(tracer_provider=pr)
    i.activate()
    yield i, pr, ex
    for owner in tuple(I._OWNERS):
        owner.deactivate()
    pr.shutdown()


def attrs(ex):
    return dict(ex.get_finished_spans()[-1].attributes)


def close(c):
    c.httpx_client.close()


@pytest.mark.parametrize(
    "method",
    [
        "generate",
        "generate_text",
        "chat",
        "generate_embedding",
        "embed_documents",
        "embed_query",
    ],
)
def test_real_native_unary_single_span_identity_body_and_source(setup, method):
    _, _, ex = setup
    payload = (
        CHAT
        if method == "chat"
        else EMB
        if method in ("generate_embedding", "embed_documents", "embed_query")
        else GEN
    )
    c, m, e, requests, responses, *_ = runtime(payload)
    if method == "chat":
        r = m.chat(messages=[{"role": "user", "content": "native"}])
    elif method == "generate_embedding":
        r = e.generate(inputs=["native"])
    elif method == "embed_documents":
        r = e.embed_documents(["native"])
    elif method == "embed_query":
        r = e.embed_query("native")
    else:
        r = getattr(m, method)(prompt="native")
    a = attrs(ex)
    assert len(ex.get_finished_spans()) == len(requests) == 1
    assert a["gen_ai.request.model"] == (
        "ibm/slate"
        if method in ("generate_embedding", "embed_documents", "embed_query")
        else "ibm/granite"
    )
    assert (
        a["gen_ai.response.model"] == "reported"
        and a["http.response.status_code"] == 200
    )
    request = json.loads(a[IN])
    assert request["native_request"] == json.loads(requests[0].content)
    assert "json" not in responses[0].__dict__
    if method in ("generate_embedding", "embed_documents", "embed_query"):
        assert len(json.loads(a[OUT])[0]) == 5001
        envelope = json.loads(a["respan.metadata.watsonx.result"])
        assert (
            "embedding" not in envelope["results"][0]
            and envelope["custom"]["api_key"] == "[REDACTED]"
        )
    else:
        assert (
            json.loads(a[OUT]) == payload
            and a["gen_ai.usage.input_tokens"] == 0
            and a["gen_ai.usage.output_tokens"] == 0
        )
        assert ("llm.usage.total_tokens" in a) == (method == "chat")
    if method == "generate_text":
        assert r == "native"
    elif method == "embed_query":
        assert len(r) == 5001
    close(c)


def test_native_long_history_schema_zero_settings_and_historical_current_tools(setup):
    _, _, ex = setup
    c, m, _e, _requests, *_ = runtime(CHAT)
    history = [{"role": "user", "content": f"item{n}"} for n in range(75)]
    history[0] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "old-id",
                "type": "function",
                "function": {"name": "old", "arguments": "{}"},
            }
        ],
    }
    before = json.dumps(history)
    schema = {
        "type": "object",
        "properties": {
            "api_key": {
                "type": "string",
                "example": "controlled-secret",
                "default": "controlled-secret",
            },
            "answer": {"type": "string"},
        },
    }
    r = m.chat(
        messages=history,
        params=TextChatParameters(
            temperature=0, max_tokens=0, response_format={"type": "json_object"}
        ),
        tools=[
            {"type": "function", "function": {"name": "weather", "parameters": schema}}
        ],
        tool_choice_option="auto",
    )
    a = attrs(ex)
    assert before == json.dumps(history)
    request = json.loads(a[IN])
    assert len(request["messages"]) == 75
    assert (
        request["native_request"]["temperature"] == 0
        and a[A.LLM_REQUEST_TEMPERATURE] == 0
    )
    assert (
        request["tools"][0]["function"]["parameters"]["properties"]["api_key"]["type"]
        == "string"
    )
    assert "controlled-secret" not in json.dumps(a)
    assert (
        "old-id" in json.dumps(request["messages"][0]["tool_calls"])
        and "old-id" not in a["gen_ai.completion.0.tool_calls"]
    )
    assert (
        "native-id" in a["gen_ai.completion.0.tool_calls"]
        and r["choices"][0]["message"]["reasoning_content"] == "controlled reasoning"
    )
    close(c)


@pytest.mark.parametrize("raw", [False, True])
def test_native_stream_300frames_fulloutput_originalbytes_and_resources(setup, raw):
    _, _, ex = setup
    frames = [
        {
            "model_id": "reported",
            "results": [
                {
                    "generated_text": str(n) + " ",
                    "input_token_count": 0,
                    "generated_token_count": 0,
                    "stop_reason": "not_finished",
                }
            ],
        }
        for n in range(300)
    ]
    c, m, _e, requests, responses, b, *_ = runtime(frames=frames)
    stream = m.generate_text_stream(prompt="native", raw_response=raw)
    assert not requests and not b.reads
    chunks = list(stream)
    a = attrs(ex)
    assert (
        len(chunks) == 300
        and len(json.loads(a[OUT])) == 300
        and len(a["gen_ai.completion.0.content"].split()) == 300
    )
    assert type(chunks[0]) is (dict if raw else str)
    assert b.closed == 1 and responses[0].is_closed
    assert a["gen_ai.usage.input_tokens"] == 0 and "llm.usage.total_tokens" not in a
    assert ex.get_finished_spans()[-1].status.status_code is StatusCode.UNSET
    close(c)


@pytest.mark.parametrize(
    "action", ["preclose", "partial", "gc", "deactivate", "sendthrow"]
)
def test_native_stream_protocol_pre_first_close_gc_deactivate(setup, action):
    i, _, ex = setup
    c, m, _e, requests, _responses, b, *_ = runtime(frames=[GEN, GEN])
    s = m.generate_text_stream(prompt="native", raw_response=True)
    if action == "preclose":
        s.close()
        assert not requests and s.native.gi_frame is None
    elif action == "partial":
        assert next(s) == GEN
        s.close()
        assert b.closed == 1
    elif action == "gc":
        del s
        gc.collect()
        assert not requests
    elif action == "deactivate":
        next(s)
        i.deactivate()
        assert len(list(s)) == 1 and b.closed == 1
    else:
        with pytest.raises(TypeError):
            s.send(1)
        assert not ex.get_finished_spans()
        assert s.send(None) == GEN
        error = ValueError("native caller error")
        with pytest.raises(ValueError) as caught:
            s.throw(error)
        assert caught.value is error and b.closed == 1
    assert len(ex.get_finished_spans()) == 1
    close(c)


@pytest.mark.parametrize("mode", ["chat", "generate"])
def test_native_fragmented_escaped_quoted_credentials(setup, mode):
    _, _, ex = setup
    pieces = [
        'Authorization: Bearer "frag',
        "ment-secret\" Basic 'frag",
        "ment-secret' password=\"frag",
        'ment-secret"',
    ]
    frames = (
        [
            {"choices": [{"index": 0, "delta": {"content": p, "reasoning_content": p}}]}
            for p in pieces
        ]
        if mode == "chat"
        else [{"results": [{"generated_text": p}]} for p in pieces]
    )
    c, m, _e, _, _, _b, *_ = runtime(frames=frames)
    s = (
        m.chat_stream(messages=[])
        if mode == "chat"
        else m.generate_text_stream(prompt="native", raw_response=True)
    )
    chunks = list(s)
    a = attrs(ex)
    assert "ment-secret" not in json.dumps(a) and '"frag' not in json.dumps(a)
    assert "ment-secret" in json.dumps(chunks)
    assert "[REDACTED]" in a["gen_ai.completion.0.content"]
    close(c)


@pytest.mark.parametrize(
    "key", [_SUPPRESS_INSTRUMENTATION_KEY, SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY]
)
def test_actual_native_suppression(setup, key):
    _, _, ex = setup
    c, m, *_ = runtime()
    tok = context.attach(context.set_value(key, True))
    try:
        assert m.generate_text(prompt="native") == "native"
    finally:
        context.detach(tok)
    assert not ex.get_finished_spans()
    close(c)


@pytest.mark.parametrize(
    "kind", ["setting", "respan", "traceloop", "env_respan", "env_traceloop"]
)
def test_initial_veto_cannot_widen(setup, monkeypatch, kind):
    i, pr, ex = setup
    tok = None
    if kind == "setting":
        i.deactivate()
        i = WatsonxInstrumentor(tracer_provider=pr, capture_content=False)
        i.activate()
    elif kind.startswith("env"):
        monkeypatch.setenv(
            "RESPAN_TRACE_CONTENT"
            if kind == "env_respan"
            else "TRACELOOP_TRACE_CONTENT",
            " false ",
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
    c, m, _e, *_ = runtime(frames=[GEN])
    s = m.generate_text_stream(prompt="private", raw_response=True)
    if tok is not None:
        context.detach(tok)
    monkeypatch.delenv("RESPAN_TRACE_CONTENT", raising=False)
    monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
    assert len(list(s)) == 1
    assert IN not in attrs(ex) and OUT not in attrs(ex)
    i.deactivate()
    close(c)


def test_actual_late_restore_child_ancestor_tupleid_and_remote(setup):
    _, pr, ex = setup
    tr = pr.get_tracer("native")
    c, m, _e, *_ = runtime(frames=[GEN, GEN])
    with tr.start_as_current_span("parent") as parent:
        s = m.generate_text_stream(prompt="lateprivate", raw_response=True)
        next(s)
        tok = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        context.detach(tok)
        list(s)
        assert IN not in attrs(ex) and OUT not in attrs(ex)
        with tr.start_as_current_span("child") as child:
            child.set_attribute(ENABLE_CONTENT_TRACING_KEY, False)
    with trace.use_span(parent, end_on_exit=False):
        assert (
            len(
                list(
                    m.generate_text_stream(prompt="finishedprivate", raw_response=True)
                )
            )
            == 2
        )
    assert IN not in attrs(ex)
    close(c)
    c, m, *_ = runtime()
    observed = tr.start_span("observed")
    sc = observed.get_span_context()
    unknown = NonRecordingSpan(
        SpanContext(sc.trace_id + 1, sc.span_id, False, TraceFlags(1))
    )
    with trace.use_span(unknown, end_on_exit=False):
        m.generate(prompt="unknown")
    assert IN not in attrs(ex)
    remote = NonRecordingSpan(SpanContext(9, 10, True, TraceFlags(1)))
    with trace.use_span(remote, end_on_exit=False):
        m.generate(prompt="remote")
    assert IN in attrs(ex)
    observed.end()
    close(c)


def test_actual_sampler_no_model_or_serde_inspection(monkeypatch):
    pr = TracerProvider(sampler=ALWAYS_OFF)
    i = WatsonxInstrumentor(tracer_provider=pr)
    i.activate()
    monkeypatch.setattr(I, "_snapshot", lambda *a: pytest.fail("unsampled snapshot"))
    monkeypatch.setattr(I, "_model_only", lambda *a: pytest.fail("unsampled model"))
    c, m, *_ = runtime()
    assert m.generate_text(prompt="native") == "native"
    close(c)
    i.deactivate()
    pr.shutdown()


def test_native_two_pending_streams_same_parent_transient_export_suppression(setup):
    _, pr, ex = setup
    c, m, *_ = runtime(frames=[GEN])
    d, n, *_ = runtime(frames=[GEN])
    with pr.get_tracer("native").start_as_current_span("parent") as parent:
        one = m.generate_text_stream(prompt="one", raw_response=True)
        two = n.generate_text_stream(prompt="two", raw_response=True)
        list(one)
        list(two)
    spans = [s for s in ex.get_finished_spans() if s.name == "watsonx.generate"]
    assert len(spans) == 2 and all(
        IN in s.attributes and OUT in s.attributes for s in spans
    )
    assert all(s.parent.span_id == parent.context.span_id for s in spans)
    close(c)
    close(d)


@pytest.mark.parametrize(
    "fault", ["startup", "serde", "attribute", "end", "detach", "attribute_veto"]
)
def test_native_telemetry_faults_do_not_change_results_ambient(
    setup, monkeypatch, fault
):
    _, _pr, ex = setup
    ambient = context.get_current()
    if fault == "startup":
        observer = I._OBSERVERS[0][1]
        original = observer.on_start

        def broken(span, parent_context=None):
            original(span, parent_context)
            context.attach(context.Context())
            raise ValueError("native onstart fault")

        monkeypatch.setattr(observer, "on_start", broken)
    elif fault == "serde":
        monkeypatch.setattr(
            I, "native_value", lambda *a: (_ for _ in ()).throw(ValueError("serde"))
        )
    elif fault in ("attribute", "attribute_veto"):
        original = Span.set_attributes

        def broken(span, attrs):
            original(span, attrs)
            if fault == "attribute_veto":
                context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
            raise ValueError("attribute fault")

        monkeypatch.setattr(Span, "set_attributes", broken)
    elif fault == "end":

        def broken(*a, **kw):
            context.attach(context.Context())
            raise ValueError("end fault")

        monkeypatch.setattr(Span, "end", broken)
    else:
        monkeypatch.setattr(
            context,
            "detach",
            lambda *a: (_ for _ in ()).throw(ValueError("detach fault")),
        )
    c, m, *_ = runtime()
    r = m.generate_text(prompt="native")
    assert r == "native" and context.get_current() is ambient
    if fault == "attribute_veto":
        assert IN not in attrs(ex) and OUT not in attrs(ex)
    close(c)


def test_actual_owned_diagnostic_events_and_unknown_hooks_scrub(setup):
    _, pr, ex = setup

    class Diagnostic(SpanProcessor):
        def on_start(self, span, parent_context=None):
            span.record_exception(ValueError("controlled-private-diagnostic"))

    pr.add_span_processor(Diagnostic())

    class Unknown:
        def __str__(self):
            raise AssertionError("str")

        def model_dump(self):
            raise AssertionError("modeldump")

        def __iter__(self):
            raise AssertionError("iter")

    t = _PROPAGATED_ATTRIBUTES.set(
        {"metadata": {"unknown": Unknown(), "api_key": "controlled-secret"}}
    )
    c, m, *_ = runtime()
    try:
        m.generate(prompt="native")
        assert json.loads(attrs(ex)["respan.metadata"]) == {
            "unknown": None,
            "api_key": "[REDACTED]",
        }
    finally:
        _PROPAGATED_ATTRIBUTES.reset(t)
    tok = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        m.generate(prompt="private")
    finally:
        context.detach(tok)
    assert not ex.get_finished_spans()[-1].events and IN not in attrs(ex)
    close(c)


@pytest.mark.parametrize("error_kind", ["http", "sse", "unknown"])
def test_actual_native_errors_identity_status_partial_resources(setup, error_kind):
    _, _, ex = setup
    if error_kind == "unknown":

        class UnknownError(RuntimeError):
            def __str__(self):
                raise AssertionError("unknownstr")

        error = UnknownError("controlled-secret")
        c, m, *_ = runtime()
        c.httpx_client._transport = httpx.MockTransport(
            lambda r: (_ for _ in ()).throw(error)
        )
        with pytest.raises(UnknownError) as caught:
            m.generate(prompt="native")
        assert caught.value is error and "controlled-secret" not in json.dumps(
            attrs(ex)
        )
    elif error_kind == "http":
        c, m, *_ = runtime(
            {"errors": [{"code": "controlled", "message": "native refusal"}]},
            status=429,
        )
        with pytest.raises(WMLClientError):
            m.generate(prompt="native")
        assert attrs(ex)["http.response.status_code"] == 429
    else:
        c, m, _e, _, _, b, *_ = runtime(
            frames=[GEN, b'event: error\ndata: {"message":"native failure"}\n\n']
        )
        with pytest.raises(WMLClientError):
            list(m.generate_text_stream(prompt="native", raw_response=True))
        assert (
            b.closed == 1
            and attrs(ex)["http.response.status_code"] == 200
            and len(json.loads(attrs(ex)[OUT])) == 1
        )
    assert ex.get_finished_spans()[-1].status.status_code is StatusCode.ERROR
    close(c)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method",
    [
        "agenerate",
        "achat",
        "agenerate_stream",
        "achat_stream",
        "aembed_documents",
        "aembed_query",
        "emb_agenerate",
    ],
)
async def test_actual_native_async_outcomes_and_resources(setup, method):
    _, _, ex = setup
    frames = (
        [CHAT]
        if method == "achat_stream"
        else [GEN]
        if method == "agenerate_stream"
        else None
    )
    payload = (
        CHAT
        if method == "achat"
        else EMB
        if method.startswith("aembed") or method == "emb_agenerate"
        else GEN
    )
    c, m, e, requests, _responses, _b, ab = runtime(payload, frames=frames)
    if method == "emb_agenerate":
        r = await e.agenerate(inputs=["native"])
    elif method == "aembed_documents":
        r = await e.aembed_documents(["native"])
    elif method == "aembed_query":
        r = await e.aembed_query("native")
    elif method.startswith("achat"):
        r = await getattr(m, method)(messages=[])
    else:
        r = await getattr(m, method)(prompt="native")
    if frames is not None:
        r = [chunk async for chunk in r]
        assert ab.body.closed == 1
    assert (
        len(ex.get_finished_spans()) == 1
        and len(requests) == 1
        and attrs(ex)["http.response.status_code"] == 200
    )
    assert OUT in attrs(ex)
    await c.async_httpx_client.aclose()
    close(c)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["preclose", "partial", "asend", "athrow"])
async def test_native_async_stream_protocol_cleanup(setup, action):
    _, _, ex = setup
    c, m, _e, requests, _responses, _b, ab = runtime(frames=[GEN, GEN])
    s = await m.agenerate_stream(prompt="native")
    if action == "preclose":
        await s.aclose()
        assert not requests
    elif action == "partial":
        await s.__anext__()
        await s.aclose()
        assert ab.body.closed == 1
    elif action == "asend":
        with pytest.raises(TypeError):
            await s.asend(1)
        assert not ex.get_finished_spans()
        assert await s.asend(None) == GEN
        await s.aclose()
    else:
        await s.__anext__()
        error = ValueError("native")
        with pytest.raises(ValueError) as caught:
            await s.athrow(error)
        assert caught.value is error and ab.body.closed == 1
    assert len(ex.get_finished_spans()) == 1
    await c.async_httpx_client.aclose()
    close(c)


def test_native_shared_config_foreign_restore_partial_activation_and_mutate_setter(
    setup, monkeypatch
):
    first, pr, _ex = setup
    second = WatsonxInstrumentor(tracer_provider=pr)
    second.activate()
    owned = ModelInference.generate
    with pytest.raises(ValueError):
        WatsonxInstrumentor(tracer_provider=pr, capture_content=False).activate()
    first.deactivate()
    assert ModelInference.generate is owned

    def foreign(*a, **kw):
        return owned(*a, **kw)

    ModelInference.generate = foreign
    second.deactivate()
    assert ModelInference.generate is foreign
    ModelInference.generate = owned.__wrapped__
    original = I.setattr if hasattr(I, "setattr") else setattr
    counter = []

    def mutate_raise(owner, name, replacement):
        original(owner, name, replacement)
        if owner is ModelInference and name == "generate" and not counter:
            counter.append(1)
            raise ValueError("setter aftermutation")

    monkeypatch.setattr(I, "setattr", mutate_raise, raising=False)
    fresh = WatsonxInstrumentor(tracer_provider=pr)
    before = ModelInference.generate
    fresh.activate()
    assert (
        not fresh._is_instrumented
        and ModelInference.generate is before
        and not I._PATCHES
    )


def test_builtin_sanitizer_idempotent_quoted_escaped_auth_and_schema():
    for data in (
        "password=controlled",
        'Bearer "controlled"',
        "Basic 'controlled'",
        'Bearer "con\\"trolled-secret"',
        "https://user:pass@example.com/?token=controlled",
    ):
        cleaned = text(data)
        assert (
            text(cleaned) == cleaned
            and "controlled" not in cleaned
            and "trolled-secret" not in cleaned
        )
    assert "controlled-secret" not in json_text(
        {
            "parameters": {
                "properties": {
                    "api_key": {
                        "type": "string",
                        "example": "controlled-secret",
                        "default": "controlled-secret",
                    }
                }
            }
        }
    )

    class Unknown(dict):
        def items(self):
            raise AssertionError("mapping hook")

    assert value(Unknown()) is None


@pytest.mark.parametrize(
    "kind", ["generate_batch", "generate_asyncmode", "embedding_batch"]
)
def test_actual_native_batch_full_results_and_no_duplicate_calls(setup, kind):
    _, _, ex = setup
    c, m, e, requests, *_ = runtime(EMB if kind == "embedding_batch" else GEN)
    if kind == "embedding_batch":
        r = e.generate(inputs=["one", "two", "three", "four"])
        assert len(r["results"]) == 2
    elif kind == "generate_asyncmode":
        r = list(m.generate(prompt=["one", "two"], async_mode=True))
        assert len(r) == 2
    else:
        r = m.generate(prompt=["one", "two"])
        assert len(r) == 2
    a = attrs(ex)
    out = json.loads(a[OUT])
    assert len(out) == 2
    request = json.loads(a[IN])
    assert a[A.LLM_REQUEST_MODEL] == (
        "ibm/slate" if kind == "embedding_batch" else "ibm/granite"
    )
    assert request["inputs" if kind == "embedding_batch" else "prompt"] == (
        ["one", "two", "three", "four"] if kind == "embedding_batch" else ["one", "two"]
    )
    if kind != "embedding_batch":
        assert (
            len(
                [
                    k
                    for k in a
                    if k.endswith(".content") and k.startswith(A.LLM_COMPLETIONS)
                ]
            )
            == 2
            and "gen_ai.usage.input_tokens" not in a
        )
    else:
        assert all(len(v) == 5001 for v in out)
    assert len(ex.get_finished_spans()) == 1 and len(requests) == 2
    close(c)


@pytest.mark.parametrize(
    "key", [ENABLE_CONTENT_TRACING_KEY, "traceloop.enable_content_tracing"]
)
@pytest.mark.parametrize("phase", ["initial", "child_start", "child_end"])
def test_native_parent_attribute_veto_never_widens(setup, key, phase):
    _, pr, ex = setup
    c, m, *_ = runtime()
    tracer = pr.get_tracer("application")
    with tracer.start_as_current_span(
        "parent", attributes={key: False} if phase == "initial" else {}
    ) as parent:
        if phase == "initial":
            parent.set_attribute(key, True)
        else:
            if phase == "child_start":
                parent.set_attribute(key, False)
            with tracer.start_as_current_span("generic-child"):
                if phase == "child_end":
                    parent.set_attribute(key, False)
            parent.set_attribute(key, True)
        m.generate(prompt="private-controlled")
    owned = next(
        span for span in ex.get_finished_spans() if span.name == "watsonx.generate"
    )
    assert IN not in owned.attributes and OUT not in owned.attributes
    close(c)


def test_native_foreign_start_fault_owned_span_cleanup_and_context_identity(
    monkeypatch,
):
    observed = []
    ex = InMemorySpanExporter()
    pr = TracerProvider()

    class Foreign(SpanProcessor):
        def on_start(self, span, parent_context=None):
            observed.append(span)
            context.attach(context.set_value("foreign", "mutation"))
            raise ValueError("foreign start fault")

        def on_end(self, span):
            pass

        def shutdown(self):
            pass

    foreign = Foreign()
    export = SimpleSpanProcessor(ex)
    pr.add_span_processor(foreign)
    pr.add_span_processor(export)
    i = WatsonxInstrumentor(tracer_provider=pr)
    i.activate()
    ambient = context.get_current()
    c, m, *_ = runtime()
    assert (
        m.generate(prompt="private-controlled")["results"][0]["generated_text"]
        == "native"
    )
    assert context.get_current() is ambient and observed[0].is_recording() is False
    assert not observed[0].events and IN not in observed[0].attributes
    observer = next(o for p, o in I._OBSERVERS if p is pr)
    assert pr._active_span_processor._span_processors == (observer, foreign, export)
    i.deactivate()
    assert pr._active_span_processor._span_processors == (foreign, export)
    close(c)
    pr.shutdown()


def test_opaque_metaclass_hooks_not_used_for_native_request_or_error(setup):
    _, _, _ex = setup
    calls = []

    class Meta(type):
        def __eq__(self, other):
            calls.append("eq")
            return False

        __hash__ = type.__hash__

    class Opaque(metaclass=Meta):
        pass

    from respan_instrumentation_watsonx._translator import native_value

    obj = Opaque()
    assert value(obj) is None and native_value(obj) is None
    c, m, *_ = runtime()
    with pytest.raises(TypeError) as bare:
        ModelInference.generate.__wrapped__(m, prompt="native", params={"unknown": obj})
    with pytest.raises(TypeError) as observed:
        m.generate(prompt="native", params={"unknown": obj})
    assert bare.value.args == observed.value.args and not calls
    close(c)


def test_actual_native_response_percent_encoded_url_credential_is_redacted(setup):
    _, _, ex = setup
    url = "https://example.invalid/path?api%5Fkey=controlled-url-secret&answer=0"
    c, m, *_ = runtime({"results": [{"generated_text": url}]})
    assert m.generate_text(prompt="Controlled URL fixture.") == url
    encoded = json.dumps(attrs(ex))
    assert "controlled-url-secret" not in encoded and "answer=0" in encoded
    assert text(text(url)) == text(url)
    close(c)
