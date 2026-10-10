"""Actual released OpenLIT/OpenAI boundaries, with controlled HTTP only."""

import asyncio
import json
from importlib.metadata import version

import httpx
import openlit
import pytest
from openai import AsyncOpenAI, OpenAI, RateLimitError
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider, sampling
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_TOOL_CALL_ID,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import StatusCode
from pydantic import BaseModel
from respan_instrumentation_openlit import OpenLITInstrumentor
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


def response(request):
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
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "prompt_tokens_details": {"cached_tokens": 3},
        "completion_tokens_details": {"reasoning_tokens": 2},
    }
    calls = [
        {
            "id": f"actual-{i}",
            "type": "function",
            "function": {
                "name": "vector_tool",
                "arguments": json.dumps(
                    {
                        "dense": list(range(5000)),
                        "sparse": {j * 2: j / 256 for j in range(256)},
                    }
                ),
            },
        }
        for i in range(2)
    ]
    if request.url.path.endswith("/responses"):
        return httpx.Response(
            200,
            json={
                "id": "response-fixture",
                "object": "response",
                "created_at": 1.0,
                "model": "fixture",
                "status": "completed",
                "parallel_tool_calls": True,
                "tools": [],
                "output": [
                    {
                        "type": "message",
                        "id": "message-fixture",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"city":"Paris"}',
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                    "input_tokens_details": {"cached_tokens": 3},
                    "output_tokens_details": {"reasoning_tokens": 2},
                },
            },
        )
    if body.get("messages", [{}])[-1].get("content") == "fail":
        return httpx.Response(
            429,
            json={
                "error": {
                    "type": "rate_limit_error",
                    "message": "controlled source error",
                }
            },
        )
    message = {
        "role": "assistant",
        "content": '{"city":"Paris"}'
        if body.get("response_format")
        else "controlled answer",
    }
    if body.get("tools"):
        message["tool_calls"] = calls
    return httpx.Response(
        200,
        json={
            "id": "chat-fixture",
            "object": "chat.completion",
            "created": 1,
            "model": "fixture",
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
def native(monkeypatch):
    import respan_instrumentation_openlit._instrumentation as lifecycle

    assert lifecycle._REFCOUNT == 0
    provider = TracerProvider()
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    owner = OpenLITInstrumentor()
    owner.activate()
    client = OpenAI(
        api_key="fixture",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(response)),
    )
    async_client = AsyncOpenAI(
        api_key="fixture",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(response)),
    )
    try:
        yield provider, exporter, owner, client, async_client
    finally:
        owner.deactivate()
        client.close()
        asyncio.run(async_client.close())
        provider.shutdown()


def chat(client, **kwargs):
    return client.chat.completions.create(
        model="fixture",
        messages=[{"role": "user", "content": "PRIVATE_INPUT"}],
        **kwargs,
    )


@pytest.mark.parametrize("policy", ["env", "otel_env", "context", "constructor"])
def test_real_native_start_bounds_never_reenable_content(native, monkeypatch, policy):
    _, exporter, owner, client, _ = native
    token = None
    if policy in ("env", "otel_env"):
        monkeypatch.setenv(
            "TRACELOOP_TRACE_CONTENT"
            if policy == "env"
            else "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            "false",
        )
    elif policy == "context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    else:
        owner.deactivate()
        owner = OpenLITInstrumentor(capture_content=False)
        owner.activate()

    def private_transport(request):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, True))
        return response(request)

    client._client._transport = httpx.MockTransport(private_transport)
    try:
        assert chat(client).choices[0].message.content == "controlled answer"
        s = exporter.get_finished_spans()[-1]
        assert "PRIVATE_" not in json.dumps(dict(s.attributes))
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in s.attributes
        assert s.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 11
    finally:
        if token is not None:
            context.detach(token)
        if policy == "constructor":
            owner.deactivate()


def test_real_http_end_veto_clears_native_ancestor_before_detach(native):
    _, exporter, _, client, _ = native
    tokens = []

    def private_transport(request):
        tokens.append(
            context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        )
        return response(request)

    client._client._transport = httpx.MockTransport(private_transport)
    with openlit.start_trace("parent") as parent:
        parent.set_metadata({"gen_ai.workflow.input": "PRIVATE_PARENT"})
        chat(client)
        for token in reversed(tokens):
            context.detach(token)
        parent.set_result("PRIVATE_REENABLED")
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert "PRIVATE_" not in json.dumps([dict(s.attributes) for s in spans])
    assert all(
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in s.attributes for s in spans
    )


@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_real_suppression_skips_spans_and_own_serialization(native, monkeypatch, key):
    _, exporter, _, client, _ = native
    import respan_instrumentation_openlit._openai_hooks as hooks

    calls = []
    monkeypatch.setattr(hooks, "input_messages", lambda value: calls.append(value))
    token = context.attach(context.set_value(key, True))
    try:
        assert chat(client).choices[0].message.content == "controlled answer"
    finally:
        context.detach(token)
    assert not exporter.get_finished_spans() and not calls


def test_real_sampler_off_does_not_serialize(native, monkeypatch):
    _, _, owner, client, _ = native
    owner.deactivate()
    provider = TracerProvider(sampler=sampling.ALWAYS_OFF)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    owner.activate()
    import respan_instrumentation_openlit._openai_hooks as hooks

    calls = []
    monkeypatch.setattr(hooks, "input_messages", lambda value: calls.append(value))
    assert chat(client).choices[0].message.content == "controlled answer"
    assert not exporter.get_finished_spans() and not calls
    owner.deactivate()
    provider.shutdown()


def test_real_error_has_source_status_exception_and_no_invented_payload(native):
    _, exporter, _, client, _ = native
    with pytest.raises(RateLimitError) as raised:
        client.chat.completions.create(
            model="fixture", messages=[{"role": "user", "content": "fail"}]
        )
    assert raised.value.status_code == 429
    s = exporter.get_finished_spans()[0]
    a = s.attributes
    assert s.status.status_code is StatusCode.ERROR
    assert a["http.response.status_code"] == 429
    assert "status_code" not in a and "error.message" not in a
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in a
    assert not any("usage" in k for k in a)


def test_real_current_calls_history_schemas_and_actual_tool_vectors(native):
    _, exporter, _, client, _ = native
    tools = [
        {
            "type": "function",
            "function": {
                "name": "vector_tool",
                "parameters": {
                    "type": "object",
                    "properties": {f"field{i}": {"type": "number"} for i in range(120)},
                },
            },
        }
    ]
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "historical-only",
                    "type": "function",
                    "function": {"name": "old", "arguments": '{"x":1}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "historical-only",
            "content": json.dumps({"vector": list(range(5000))}),
        },
    ]
    result = client.chat.completions.create(
        model="fixture", messages=messages, tools=tools
    )
    a = exporter.get_finished_spans()[0].attributes
    current = json.loads(a[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert [c["id"] for c in current] == ["actual-0", "actual-1"]
    assert [
        c["id"] for c in json.loads(a[f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls"])
    ] == ["historical-only"]
    assert a[f"{SpanAttributes.LLM_PROMPTS}.1.tool_call_id"] == "historical-only"
    assert (
        len(json.loads(a[f"{SpanAttributes.LLM_PROMPTS}.1.content"])["vector"]) == 5000
    )
    assert (
        len(
            json.loads(a[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0]["function"][
                "parameters"
            ]["properties"]
        )
        == 120
    )
    assert a[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert a[SpanAttributes.LLM_USAGE_REASONING_TOKENS] == 2
    for c in result.choices[0].message.tool_calls:
        with openlit.start_trace("tool") as tool:
            tool.set_metadata(
                {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": c.function.name,
                    GEN_AI_TOOL_CALL_ID: c.id,
                    "gen_ai.tool.call.arguments": c.function.arguments,
                }
            )
            tool.set_result("")
            tool.set_metadata({"gen_ai.tool.call.result": c.function.arguments})
    spans = [
        s
        for s in exporter.get_finished_spans()
        if s.attributes[RESPAN_LOG_TYPE] == "tool"
    ]
    assert len(spans) == 2
    for s in spans:
        a = s.attributes
        i = json.loads(a[SpanAttributes.TRACELOOP_ENTITY_INPUT])
        o = json.loads(a[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
        assert i["name"] == "vector_tool" and len(i["arguments"]["dense"]) == 5000
        assert len(o["dense"]) == 5000 and len(o["sparse"]) == 256
        assert a[GEN_AI_TOOL_CALL_ID] in ["actual-0", "actual-1"]
        assert not any(
            k.startswith("gen_ai.tool.") and k != GEN_AI_TOOL_CALL_ID for k in a
        )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_embeddings_complete_with_actual_source_usage(native, asynchronous):
    _, exporter, _, client, async_client = native
    if asynchronous:
        result = asyncio.run(
            async_client.embeddings.create(
                model="fixture-embedding", input=["controlled"]
            )
        )
    else:
        result = client.embeddings.create(
            model="fixture-embedding", input=["controlled"]
        )
    assert len(result.data[0].embedding) == 5000
    a = exporter.get_finished_spans()[0].attributes
    assert json.loads(a[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == [
        result.data[0].embedding
    ]
    assert (
        a[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == a[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == 13
    )
    assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in a


class City(BaseModel):
    city: str


@pytest.mark.parametrize("asynchronous", [False, True])
def test_released_responses_parse_preserves_native_parsed_models(native, asynchronous):
    if version("openlit") < "1.45.0":
        pytest.skip("Released OpenLIT1.44 has no Responses.parse instrumentor")
    _, exporter, _, client, async_client = native
    if asynchronous:
        result = asyncio.run(
            async_client.responses.parse(
                model="fixture", input="controlled", text_format=City
            )
        )
    else:
        result = client.responses.parse(
            model="fixture", input="controlled", text_format=City
        )
    assert (
        isinstance(result.output_parsed, City) and result.output_parsed.city == "Paris"
    )
    assert len(exporter.get_finished_spans()) == 1
    a = exporter.get_finished_spans()[0].attributes
    assert (
        a[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 11
        and a[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 7
    )
    assert json.loads(a[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]) == {
        "city": "Paris"
    }


@pytest.mark.parametrize("observer", ["input_messages", "json_string"])
def test_actual_sdk_own_observer_fault_preserves_result_and_error(
    native, monkeypatch, observer
):
    _, _, _, client, _ = native
    import respan_instrumentation_openlit._openai_hooks as hooks

    def fail(*args, **kwargs):
        raise RuntimeError("controlled telemetry fault")

    monkeypatch.setattr(hooks, observer, fail)
    assert chat(client).choices[0].message.content == "controlled answer"
    with pytest.raises(RateLimitError):
        client.chat.completions.create(
            model="fixture", messages=[{"role": "user", "content": "fail"}]
        )


@pytest.mark.parametrize("counts", [False, 1.5, "9", -2, None, 0])
def test_raw_provider_usage_is_validated_before_native_sdk_coercion(native, counts):
    _, exporter, _, client, _ = native

    def transport(request):
        body = json.loads(response(request).content)
        if counts is None:
            body.pop("usage")
        else:
            body["usage"] = {
                "prompt_tokens": counts,
                "completion_tokens": counts,
                "total_tokens": counts,
            }
        return httpx.Response(200, json=body)

    client._client._transport = httpx.MockTransport(transport)
    assert chat(client).choices[0].message.content == "controlled answer"
    attrs = exporter.get_finished_spans()[0].attributes
    published = {
        k: v for k, v in attrs.items() if k.startswith(("gen_ai.usage.", "llm.usage."))
    }
    if type(counts) is int and counts >= 0:
        assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == counts
        assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == counts
        assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == counts
    else:
        assert not published


def test_delayed_native_stream_inherits_closed_private_external_parent(
    native, monkeypatch
):
    provider, exporter, _, client, _ = native

    def transport(request):
        chunks = [
            {
                "id": "fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "PRIVATE_STREAM"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
            + "data: [DONE]\n\n",
        )

    client._client._transport = httpx.MockTransport(transport)
    with provider.get_tracer("external").start_as_current_span("external-parent"):
        stream = chat(client, stream=True)
        assert next(stream).choices[0].delta.content == "PRIVATE_STREAM"
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    list(stream)
    child = next(
        s
        for s in exporter.get_finished_spans()
        if s.instrumentation_scope.name.startswith("openlit.")
    )
    assert "PRIVATE_" not in json.dumps(dict(child.attributes))
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in child.attributes


def test_private_inflight_native_call_keeps_bound_after_deactivation(
    native, monkeypatch
):
    _, exporter, owner, client, _ = native
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    def transport(request):
        owner.deactivate()
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        body = json.loads(response(request).content)
        body["choices"][0]["message"]["content"] = "PRIVATE_OUTPUT"
        return httpx.Response(200, json=body)

    client._client._transport = httpx.MockTransport(transport)
    assert chat(client).choices[0].message.content == "PRIVATE_OUTPUT"
    assert len(exporter.get_finished_spans()) == 1
    attrs = exporter.get_finished_spans()[0].attributes
    assert "PRIVATE_" not in json.dumps(dict(attrs))
    assert not any(k.startswith("openlit.") for k in attrs)


def test_foreign_native_config_and_provider_fields_restore_only_if_owned(native):
    provider, _, owner, _, _ = native
    from openlit._config import OpenlitConfig

    owner.deactivate()
    OpenlitConfig()
    OpenlitConfig.application_name = "foreign-app"
    owner.activate()
    foreign_getter = lambda *args, **kwargs: "foreign"
    provider.get_tracer = foreign_getter
    OpenlitConfig.openlit_url = "https://foreign.invalid"
    owner.deactivate()
    assert provider.get_tracer is foreign_getter
    assert OpenlitConfig.application_name == "foreign-app"
    assert OpenlitConfig.openlit_url == "https://foreign.invalid"


def test_own_policy_fault_preserves_actual_native_results_and_exceptions(
    native, monkeypatch
):
    _, _, _, client, _ = native
    import respan_instrumentation_openlit._instrumentation as lifecycle

    def fail(*args, **kwargs):
        raise RuntimeError("controlled policy fault")

    monkeypatch.setattr(lifecycle._PROCESSOR, "allowed", fail)
    assert chat(client).choices[0].message.content == "controlled answer"
    with pytest.raises(RateLimitError):
        client.chat.completions.create(
            model="fixture", messages=[{"role": "user", "content": "fail"}]
        )


def test_explicit_private_parent_context_is_inherited_with_ambient_opt_in(native):
    provider, exporter, _, client, _ = native
    parent = provider.get_tracer("caller").start_span(
        "private-parent", context=context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
    )
    token = context.attach(trace.set_span_in_context(parent))
    try:
        assert chat(client).choices[0].message.content == "controlled answer"
    finally:
        context.detach(token)
        parent.end()
    child = next(
        s
        for s in exporter.get_finished_spans()
        if s.instrumentation_scope.name.startswith("openlit.")
    )
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in child.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in child.attributes


def test_context_parent_detach_veto_clears_delayed_native_child(native):
    provider, exporter, _, client, _ = native

    def transport(request):
        chunks = [
            {
                "id": "fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "PRIVATE_STREAM"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
            + "data: [DONE]\n\n",
        )

    client._client._transport = httpx.MockTransport(transport)
    with provider.get_tracer("caller").start_as_current_span("external-parent"):
        stream = chat(client, stream=True)
        next(stream)
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    list(stream)
    child = next(
        s
        for s in exporter.get_finished_spans()
        if s.instrumentation_scope.name.startswith("openlit.")
    )
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in child.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in child.attributes


def test_actual_source_content_redacts_complete_quoted_credentials(native):
    _, exporter, _, client, _ = native
    original = '{"api_key":"synthetic secret with spaces"} password=\'another secret value\' https://user:password@host/path'

    def transport(request):
        body = json.loads(response(request).content)
        body["choices"][0]["message"]["content"] = original
        return httpx.Response(200, json=body)

    client._client._transport = httpx.MockTransport(transport)
    assert chat(client).choices[0].message.content == original
    exported = json.dumps(dict(exporter.get_finished_spans()[0].attributes))
    assert (
        "synthetic" not in exported
        and "another secret" not in exported
        and "user:password" not in exported
    )


def test_actual_tool_schema_sensitive_names_preserved_values_redacted(native):
    _, exporter, _, client, _ = native
    tools = [
        {
            "type": "function",
            "function": {
                "name": "credential_tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "api_key": {
                            "type": "string",
                            "default": "ACTUAL_DEFAULT",
                            "examples": ["ACTUAL_EXAMPLE"],
                        }
                    },
                },
            },
        }
    ]
    observed = []

    def transport(request):
        observed.append(json.loads(request.content))
        body = json.loads(response(request).content)
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            '{"api_key":"ACTUAL_ARGUMENT"}'
        )
        return httpx.Response(200, json=body)

    client._client._transport = httpx.MockTransport(transport)
    result = chat(client, tools=tools)
    assert (
        json.loads(result.choices[0].message.tool_calls[0].function.arguments)[
            "api_key"
        ]
        == "ACTUAL_ARGUMENT"
    )
    assert (
        observed[0]["tools"][0]["function"]["parameters"]["properties"]["api_key"][
            "default"
        ]
        == "ACTUAL_DEFAULT"
    )
    attrs = exporter.get_finished_spans()[0].attributes
    schema = json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0]["function"][
        "parameters"
    ]
    assert schema["properties"]["api_key"]["type"] == "string"
    assert schema["properties"]["api_key"]["examples"] == ["[REDACTED]"]
    assert "ACTUAL_" not in json.dumps(dict(attrs))
