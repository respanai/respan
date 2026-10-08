"""Released AgentOps APIs, controlled provider HTTP, and native semantics."""

import asyncio
import json

import httpx
import pytest
import respan_instrumentation_agentops._instrumentation as runtime
from agentops import agent, guardrail, task, tool, workflow
from agentops import trace as agentops_trace
from agentops.sdk.core import tracer
from agentops.semconv.span_kinds import SpanKind
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import StatusCode
from respan_instrumentation_agentops import AgentOpsInstrumentor
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

INPUT = SpanAttributes.TRACELOOP_ENTITY_INPUT
OUTPUT = SpanAttributes.TRACELOOP_ENTITY_OUTPUT


@pytest.fixture
def env(monkeypatch):
    p = TracerProvider()
    e = InMemorySpanExporter()
    p.add_span_processor(SimpleSpanProcessor(e))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", p)
    monkeypatch.setattr(runtime.RespanTracer, "_instance", None)
    monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
    yield e
    if runtime._RUNTIME:
        runtime._RUNTIME.restore()
        runtime._RUNTIME = None
    p.shutdown()


def owner(capture=True):
    o = AgentOpsInstrumentor(capture_content=capture)
    o.activate()
    return o


def no_content(s):
    a = dict(s.attributes)
    assert INPUT not in a and OUTPUT not in a and "PRIVATE" not in json.dumps(a)
    assert not any(k.startswith(("gen_ai.prompt.", "gen_ai.completion.")) for k in a)
    assert all("PRIVATE" not in json.dumps(dict(e.attributes)) for e in s.events)


@pytest.mark.parametrize(
    "decorator,logtype",
    [
        (task, "task"),
        (tool, "tool"),
        (agent, "agent"),
        (workflow, "workflow"),
        (guardrail, "guardrail"),
        (agentops_trace, "workflow"),
    ],
)
def test_native_decorator_return_and_kind(env, decorator, logtype):
    o = owner()
    value = {"actual": "result"}

    @decorator(name="source_operation")
    def call(value):
        return value

    assert call(value) is value
    o.deactivate()
    s = env.get_finished_spans()[0]
    assert (
        s.attributes[RESPAN_LOG_TYPE] == logtype
        and json.loads(s.attributes[OUTPUT]) == value
    )
    assert s.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in s.attributes


def test_actual_hierarchy_and_tool_shape(env):
    o = owner()

    @tool(name="lookup")
    def lookup(city):
        return city.upper()

    @agent(name="weather_agent")
    def weather(city):
        return lookup(city)

    @agentops_trace(name="workflow")
    def root(city):
        return weather(city)

    assert root("Paris") == "PARIS"
    o.deactivate()
    spans = env.get_finished_spans()
    assert len(spans) == 3
    bytype = {s.attributes[RESPAN_LOG_TYPE]: s for s in spans}
    assert (
        bytype["tool"].parent.span_id == bytype["agent"].context.span_id
        and bytype["agent"].parent.span_id == bytype["workflow"].context.span_id
    )
    assert json.loads(bytype["tool"].attributes[INPUT]) == {
        "name": "lookup",
        "arguments": {"args": ["Paris"], "kwargs": {}},
    }


@pytest.mark.parametrize("privacy", ["env", "context", "option"])
def test_initial_private_bound_no_serializer(env, monkeypatch, privacy):
    token = None
    if privacy == "env":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    if privacy == "context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    o = owner(privacy != "option")

    class Unsafe:
        def __str__(self):
            raise AssertionError("str called")

        def model_dump(self, *a, **kw):
            raise AssertionError("serializer called")

    value = Unsafe()

    @task
    def call(value):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        return value

    try:
        assert call(value) is value
    finally:
        if token:
            context.detach(token)
        o.deactivate()
    no_content(env.get_finished_spans()[0])


@pytest.mark.parametrize("privacy", ["env", "context"])
def test_end_veto_before_output(env, monkeypatch, privacy):
    o = owner()
    token = None

    @task
    def call(value):
        nonlocal token
        if privacy == "env":
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        else:
            token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        return value

    assert call("PRIVATE_INPUT") == "PRIVATE_INPUT"
    if token:
        context.detach(token)
    o.deactivate()
    no_content(env.get_finished_spans()[0])


def test_child_observed_veto_clears_ancestor(env, monkeypatch):
    o = owner()

    @task
    def child(value):
        return value

    @agentops_trace
    def root(value):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        child(value)
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        return value

    root("PRIVATE")
    o.deactivate()
    for s in env.get_finished_spans():
        no_content(s)


@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_suppression_no_spans_or_serializer(env, key):
    o = owner()
    token = context.attach(context.set_value(key, True))

    @task
    def call(value):
        return value

    assert call("PRIVATE") == "PRIVATE"
    context.detach(token)
    o.deactivate()
    assert not env.get_finished_spans()


def test_sampler_no_application_hooks(env, monkeypatch):
    p = TracerProvider(sampler=ALWAYS_OFF)
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", p)
    o = owner()

    class Unsafe:
        def __str__(self):
            raise AssertionError("str called")

        def model_dump(self, *a, **kw):
            raise AssertionError("serializer called")

    v = Unsafe()

    @task
    def call(value):
        return value

    assert call(v) is v
    o.deactivate()
    assert not env.get_finished_spans()
    p.shutdown()


def test_native_exception_identity_no_status_shortcuts(env):
    o = owner()
    error = ValueError("source failure")

    @task
    def fail():
        raise error

    with pytest.raises(ValueError) as exc:
        fail()
    assert exc.value is error
    o.deactivate()
    s = env.get_finished_spans()[0]
    assert s.status.status_code == StatusCode.ERROR and OUTPUT not in s.attributes
    assert (
        "status_code" not in s.attributes
        and "error.message" not in s.attributes
        and "http.status_code" not in s.attributes
    )
    assert not any("usage" in k for k in s.attributes)


@pytest.mark.parametrize(
    "end_state,expected",
    [
        ("Error", StatusCode.ERROR),
        ("Unknown", StatusCode.UNSET),
        ("Indeterminate", StatusCode.UNSET),
        ("Success", StatusCode.UNSET),
    ],
)
def test_explicit_native_trace_end_state(env, end_state, expected):
    o = owner()
    ctx = tracer.start_trace("manual")
    tracer.end_trace(ctx, end_state)
    o.deactivate()
    assert env.get_finished_spans()[0].status.status_code == expected


def test_native_async_preserves_result_and_context(env):
    o = owner()
    before = trace.get_current_span()

    @task
    async def call(value):
        return value

    async def run():
        v = {"actual": 1}
        assert await call(v) is v
        assert trace.get_current_span() is before

    asyncio.run(run())
    o.deactivate()
    assert len(env.get_finished_spans()) == 1


def test_full_dense_sparse_tool_result_and_no_application_hooks(env):
    o = owner()

    @tool
    def vector():
        return {
            "dense": list(range(5000)),
            "sparse": {i * 2: i / 5000 for i in range(5000)},
        }

    result = vector()
    o.deactivate()
    out = json.loads(env.get_finished_spans()[0].attributes[OUTPUT])
    assert (
        len(result["sparse"]) == 5000
        and len(out["dense"]) == 5000
        and len(out["sparse"]) == 5000
        and out["sparse"]["9998"] == 4999 / 5000
    )


def test_redaction(env):
    o = owner()

    @task
    def call(value):
        return value

    call(
        {
            "password": "two words",
            "text": 'https://user:password@example.org password="two words" Bearer abcdefghijk',
        }
    )
    o.deactivate()
    attrs = json.dumps(dict(env.get_finished_spans()[0].attributes))
    assert (
        "two words" not in attrs
        and "user:password" not in attrs
        and "abcdefghijk" not in attrs
    )


def test_native_generator_quirks_unchanged(env):
    # Released SDK attaches eagerly and does not end sync spans on close.
    o = owner()
    token = context.attach(context.get_current())
    before = trace.get_current_span()

    @task
    def generate():
        sent = yield "first"
        yield sent
        return "native return"

    g = generate()
    assert trace.get_current_span() is not before
    assert next(g) == "first" and g.send("second") == "second"
    with pytest.raises(StopIteration) as stop:
        next(g)
    assert stop.value.value is None  # released SDK consumes the generator return
    assert len(env.get_finished_spans()) == 1
    context.detach(token)
    o.deactivate()


def provider_response(request):
    body = json.loads(request.content)
    if request.url.path.endswith("/embeddings"):
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "fixture-embedding",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [i / 5000 for i in range(5000)],
                    }
                ],
                "usage": {"prompt_tokens": 13, "total_tokens": 13},
            },
        )
    if "FAIL_PROVIDER" in json.dumps(body.get("messages")):
        return httpx.Response(
            503,
            json={
                "error": {
                    "message": "controlled provider failure",
                    "type": "server_error",
                    "code": "fixture",
                }
            },
        )
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "prompt_tokens_details": {"cached_tokens": 3},
        "completion_tokens_details": {"reasoning_tokens": 2},
    }
    if body.get("stream"):
        parts = [
            {
                "id": "fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture-chat",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "answer"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture-chat",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            },
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join("data: " + json.dumps(p) + "\n\n" for p in parts)
            + "data: [DONE]\n\n",
        )
    message = {"role": "assistant", "content": "answer"}
    if body.get("tools"):
        message["content"] = ""
        message["tool_calls"] = [
            {
                "id": f"actual-call-{i}",
                "type": "function",
                "function": {
                    "name": "vector_tool",
                    "arguments": json.dumps(
                        {"label": str(i), "vector": list(range(5000))}
                    ),
                },
            }
            for i in range(2)
        ]
    return httpx.Response(
        200,
        json={
            "id": "fixture",
            "object": "chat.completion",
            "created": 1,
            "model": "fixture-chat",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if body.get("tools") else "stop",
                }
            ],
            "usage": usage,
        },
    )


@pytest.fixture
def provider(env):
    import openai
    from agentops.instrumentation.providers.openai import OpenaiInstrumentor

    o = owner()
    instrumentor = OpenaiInstrumentor()
    instrumentor.instrument()
    client = openai.OpenAI(
        api_key="fixture-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(provider_response)),
    )
    async_client = openai.AsyncOpenAI(
        api_key="fixture-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider_response)),
    )
    yield client, async_client
    instrumentor.uninstrument()
    client.close()
    asyncio.run(async_client.close())
    o.deactivate()


def llm(env):
    return next(
        s for s in env.get_finished_spans() if s.attributes[RESPAN_LOG_TYPE] == "chat"
    )


def test_actual_provider_usage_http(provider, env):
    c, _ = provider
    r = c.chat.completions.create(
        model="fixture-chat", messages=[{"role": "user", "content": "q"}]
    )
    assert r.choices[0].message.content == "answer"
    a = llm(env).attributes
    assert [
        a[k]
        for k in (
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        )
    ] == [11, 7, 18]
    assert (
        a[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 3
        and a[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 2
    )


@pytest.mark.parametrize("async_call", [False, True])
def test_actual_provider_full_embedding(provider, env, async_call):
    c, ac = provider
    r = (
        asyncio.run(
            ac.embeddings.create(model="fixture-embedding", input=["actual value"])
        )
        if async_call
        else c.embeddings.create(model="fixture-embedding", input=["actual value"])
    )
    assert len(r.data[0].embedding) == 5000
    a = env.get_finished_spans()[0].attributes
    assert a[RESPAN_LOG_TYPE] == "embedding" and len(json.loads(a[OUTPUT])[0]) == 5000
    assert (
        a[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 13
        and SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in a
    )


def test_current_calls_not_synthetic_executions_and_history(provider, env):
    c, _ = provider
    tools = [
        {
            "type": "function",
            "function": {
                "name": "vector_tool",
                "parameters": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                },
            },
        }
    ]
    history = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "history-only",
                "type": "function",
                "function": {"name": "old", "arguments": '{"x":1}'},
            }
        ],
    }
    r = c.chat.completions.create(
        model="fixture-chat",
        messages=[
            history,
            {"role": "tool", "content": "old result", "tool_call_id": "history-only"},
            {"role": "user", "content": "q"},
        ],
        tools=tools,
    )
    assert len(r.choices[0].message.tool_calls) == 2
    assert not any(
        s.attributes[RESPAN_LOG_TYPE] == "tool" for s in env.get_finished_spans()
    )
    a = llm(env).attributes
    calls = json.loads(a[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert len(calls) == 2 and all(x["id"] != "history-only" for x in calls)
    assert len(
        json.loads(calls[0]["function"]["arguments"])["vector"]
    ) == 5000 and isinstance(calls[0]["function"]["arguments"], str)
    assert json.loads(a[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == tools
    assert (
        json.loads(a[f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls"])[0]["id"]
        == "history-only"
    )


def test_provider_original_error_and_no_output(provider, env):
    import openai

    c, _ = provider
    with pytest.raises(openai.InternalServerError):
        c.chat.completions.create(
            model="fixture-chat",
            messages=[{"role": "user", "content": "FAIL_PROVIDER"}],
        )
    s = env.get_finished_spans()[0]
    assert (
        s.status.status_code == StatusCode.ERROR
        and OUTPUT not in s.attributes
        and not any("usage" in k for k in s.attributes)
    )


def test_provider_sse_native_result(provider, env):
    c, _ = provider
    r = c.chat.completions.create(
        model="fixture-chat",
        messages=[{"role": "user", "content": "q"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    assert "".join(x.choices[0].delta.content or "" for x in r if x.choices) == "answer"
    assert llm(env).attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 11


@pytest.mark.parametrize("privacy", ["env", "context"])
def test_provider_private_start_bound(provider, env, monkeypatch, privacy):
    c, _ = provider
    token = None
    if privacy == "env":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    else:
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        c.chat.completions.create(
            model="fixture-chat", messages=[{"role": "user", "content": "PRIVATE"}]
        )
    finally:
        if token:
            context.detach(token)
    no_content(env.get_finished_spans()[0])
    assert (
        env.get_finished_spans()[0].attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == 11
    )


def test_private_native_tags_and_update_metadata_not_passthrough(env):
    import agentops

    o = owner(False)

    @agentops_trace(name="private_metadata", tags={"note": "PRIVATE_TAG"})
    def run():
        assert agentops.update_trace_metadata({"note": "PRIVATE_DYNAMIC"})
        return "PRIVATE_OUTPUT"

    run()
    o.deactivate()
    no_content(env.get_finished_spans()[0])


def test_private_span_level_error_message_absent(env):
    o = owner(False)
    span, _, token = tracer.make_span(
        "source_error", SpanKind.TASK, attributes={"error.message": "PRIVATE_ERROR"}
    )
    tracer.finalize_span(span, token)
    o.deactivate()
    no_content(env.get_finished_spans()[0])


def test_no_start_serializer_with_sampling_or_content_off(env, monkeypatch):
    import respan_instrumentation_agentops._instrumentation as native

    calls = []
    original = native.json_value
    monkeypatch.setattr(
        native, "json_value", lambda *a, **kw: calls.append(1) or original(*a, **kw)
    )
    p = TracerProvider(sampler=ALWAYS_OFF)
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", p)
    o = owner()
    span = tracer.get_tracer().start_span(
        "sampled", attributes={"input": {"secret": "PRIVATE"}}
    )
    span.end()
    o.deactivate()
    p.shutdown()
    assert calls == []
    p = TracerProvider()
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", p)
    o = owner(False)
    span = tracer.get_tracer().start_span(
        "private", attributes={"input": {"secret": "PRIVATE"}}
    )
    span.end()
    o.deactivate()
    p.shutdown()
    assert calls == []


@pytest.mark.parametrize("target", ["json_value", "data", "allowed"])
def test_own_serializer_fault_preserves_actual_task_value_and_error(
    env, monkeypatch, target
):
    import respan_instrumentation_agentops._instrumentation as native

    o = owner()

    def fault(*a, **kw):
        raise RuntimeError("controlled telemetry fault")

    if target == "allowed":
        monkeypatch.setattr(o._runtime.processor, "allowed", fault)
    else:
        monkeypatch.setattr(native, target, fault)
        if target == "data":
            import respan_instrumentation_agentops._serialization as serialization

            monkeypatch.setattr(serialization, "data", fault)
    value = {"source": "result"}

    @task
    def run(value):
        return value

    assert run(value) is value
    error = ValueError("source error")

    @task
    def fail():
        raise error

    with pytest.raises(ValueError) as exc:
        fail()
    assert exc.value is error
    o.deactivate()


def test_source_numeric_text_not_parsed_as_number(env):
    o = owner()
    span, _, token = tracer.make_span(
        "text_source",
        SpanKind.LLM,
        attributes={
            "gen_ai.request.type": "chat",
            f"{SpanAttributes.LLM_COMPLETIONS}.0.role": "assistant",
            f"{SpanAttributes.LLM_COMPLETIONS}.0.content": "123",
        },
    )
    tracer.finalize_span(span, token)
    o.deactivate()
    assert (
        json.loads(env.get_finished_spans()[0].attributes[OUTPUT])["messages"][0][
            "content"
        ]
        == "123"
    )


@pytest.mark.parametrize("target", ["json_value", "data", "allowed"])
def test_own_observer_fault_preserves_actual_provider_tool_response(
    provider, env, monkeypatch, target
):
    import respan_instrumentation_agentops._instrumentation as native

    def fault(*a, **kw):
        raise RuntimeError("controlled telemetry fault")

    if target == "allowed":
        monkeypatch.setattr(native._RUNTIME.processor, "allowed", fault)
    else:
        monkeypatch.setattr(native, target, fault)
    c, _ = provider
    result = c.chat.completions.create(
        model="fixture-chat",
        messages=[{"role": "user", "content": "q"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "vector_tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}},
                    },
                },
            }
        ],
    )
    assert (
        len(result.choices[0].message.tool_calls) == 2
        and result.choices[0].message.tool_calls[0].id == "actual-call-0"
    )


def test_native_context_manager_veto_observed_before_detach(env):
    from agentops.sdk.decorators.utility import _record_entity_input

    o = owner()
    outer = context.attach(context.get_current())
    try:
        with tracer.get_tracer().start_as_current_span("manual_native") as span:
            _record_entity_input(span, ("PRIVATE_INPUT",), {})
            context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        assert context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
    finally:
        context.detach(outer)
        o.deactivate()
    no_content(env.get_finished_spans()[0])


def test_public_trace_metadata_and_guardrail_spec_canonical(env):
    import agentops
    from respan_sdk.constants.span_attributes import RESPAN_METADATA

    o = owner()

    @guardrail(spec="input")
    def check(value):
        return value

    @agentops_trace
    def run():
        assert agentops.update_trace_metadata({"stage": "controlled"})
        return check("value")

    run()
    o.deactivate()
    spans = env.get_finished_spans()
    assert (
        json.loads(spans[-1].attributes[RESPAN_METADATA])["agentops"]["trace_metadata"][
            "stage"
        ]
        == "controlled"
    )
    assert not any(k.startswith("trace.metadata.") for k in spans[-1].attributes)
    assert (
        json.loads(spans[0].attributes[RESPAN_METADATA])["agentops"]["spec"] == "input"
    )
