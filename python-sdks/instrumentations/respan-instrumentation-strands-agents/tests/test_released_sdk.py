"""Released Strands agent/provider runtime behavior and adapter ownership."""

from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace

import httpx
import pytest
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import StatusCode
from respan_instrumentation_strands_agents import StrandsAgentsInstrumentor
from respan_instrumentation_strands_agents._processor import enrich_strands_agents_span
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.telemetry import tracer as tracer_module


class FixtureClient(httpx.AsyncClient):
    async def aclose(self):
        pass


def model(*, fail=False, tools=False, counts=None):
    count = 0

    def respond(request):
        nonlocal count
        count += 1
        if fail:
            raise httpx.ConnectError("controlled provider error", request=request)
        payload = json.loads(request.content)
        call_tool = tools and count == 1
        delta = {"role": "assistant", "content": "fixture answer"}
        if call_tool:
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "fixture-call-id",
                        "type": "function",
                        "function": {
                            "name": payload["tools"][0]["function"]["name"],
                            "arguments": '{"value":"fixture-input"}',
                        },
                    }
                ],
            }
        chunk = {
            "id": "fixture",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "fixture-model",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": "tool_calls" if call_tool else "stop",
                }
            ],
            "usage": counts
            or {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
                "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    return OpenAIModel(
        model_id="fixture-model",
        client_args={
            "api_key": "fixture-key",
            "base_url": "https://fixture.invalid",
            "max_retries": 0,
            "http_client": FixtureClient(transport=httpx.MockTransport(respond)),
        },
    )


@pytest.fixture
def runtime(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    native = tracer_module.Tracer()
    monkeypatch.setattr(tracer_module, "_tracer_instance", native)
    instrumentor = StrandsAgentsInstrumentor()
    instrumentor.activate()
    try:
        yield instrumentor, provider, exporter, native
    finally:
        instrumentor.deactivate()
        provider.shutdown()


def chats(exporter):
    return [
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get(RESPAN_LOG_TYPE) == "chat"
    ]


def test_released_openai_provider_usage_return_and_connected_tree(runtime):
    _, _, exporter, _ = runtime
    agent = Agent(name="released-agent", model=model(), callback_handler=None)
    result = agent("fixture prompt")
    assert result.message["role"] == "assistant"
    assert "fixture answer" in str(result)
    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    chat = chats(exporter)[0]
    assert chat.attributes[SpanAttributes.LLM_SYSTEM] == "openai"
    assert chat.attributes[SpanAttributes.GEN_AI_IS_STREAMING] is True
    assert chat.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 12
    assert chat.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 4
    assert chat.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 16
    assert chat.attributes[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert chat.attributes[SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] == 2
    assert chat.attributes[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 1
    ids = {span.context.span_id for span in spans}
    assert all(span.parent is None or span.parent.span_id in ids for span in spans)
    assert not StrandsAgentsInstrumentor._shared_processor._states


def test_released_tool_call_id_and_current_turn_only(runtime):
    _, _, exporter, _ = runtime

    @tool
    def lookup(value: str) -> str:
        """Return a controlled tool result."""
        return f"result:{value}"

    agent = Agent(
        name="tool-agent",
        model=model(tools=True),
        tools=[lookup],
        callback_handler=None,
    )
    agent("use lookup")
    tool_span = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get(RESPAN_LOG_TYPE) == "tool"
    )
    assert tool_span.attributes[gen_ai.GEN_AI_TOOL_CALL_ID] == "fixture-call-id"
    assert json.loads(tool_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])[
        "arguments"
    ] == {"value": "fixture-input"}
    first, second = chats(exporter)
    first_calls = json.loads(
        first.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
    )
    assert first_calls[0]["id"] == "fixture-call-id"
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls" not in second.attributes
    assert (
        json.loads(first.attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0][
            "function"
        ]["name"]
        == "lookup"
    )


@pytest.mark.parametrize("policy", ["env", "context"])
def test_released_start_disabled_never_captures_after_reenable(
    runtime, monkeypatch, policy
):
    _, _, exporter, native = runtime
    token = None
    if policy == "env":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    else:
        token = context_api.attach(
            context_api.set_value(ENABLE_CONTENT_TRACING_KEY, False)
        )
    span = native.start_agent_span(
        messages=[{"role": "user", "content": [{"text": "private-start"}]}],
        agent_name="private-agent",
    )
    if token is not None:
        context_api.detach(token)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    native.end_agent_span(
        span, response=SimpleNamespace(stop_reason="end_turn", content="private-end")
    )
    serialized = str(exporter.get_finished_spans()[0].attributes)
    assert "private-start" not in serialized
    assert "private-end" not in serialized
    assert (
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT
        not in exporter.get_finished_spans()[0].attributes
    )


def test_released_end_privacy_veto_clears_content_and_exceptions(runtime, monkeypatch):
    _, _, exporter, native = runtime
    span = native.start_model_invoke_span(
        messages=[{"role": "user", "content": [{"text": "private-before"}]}],
        model_id="fixture-model",
    )
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    native.end_span_with_error(
        span, "private-failure", RuntimeError("private-exception")
    )
    result = exporter.get_finished_spans()[0]
    assert result.status.status_code is StatusCode.ERROR
    assert result.status.description is None
    assert "private" not in str(result.attributes)
    assert "private" not in str([event.attributes for event in result.events])
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in result.attributes
    assert "status_code" not in result.attributes


@pytest.mark.parametrize(
    "suppression_key",
    [
        context_api._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_released_suppression_is_native_nonrecording(runtime, suppression_key):
    _, _, exporter, _ = runtime
    token = context_api.attach(context_api.set_value(suppression_key, True))
    try:
        result = Agent(model=model(), callback_handler=None)("suppressed prompt")
        assert "fixture answer" in str(result)
    finally:
        context_api.detach(token)
    assert exporter.get_finished_spans() == ()


def test_released_sampler_off(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=ALWAYS_OFF)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(tracer_module, "_tracer_instance", tracer_module.Tracer())
    owner = StrandsAgentsInstrumentor()
    owner.activate()
    try:
        assert "fixture answer" in str(
            Agent(model=model(), callback_handler=None)("sampled off")
        )
        assert not owner._shared_processor._states
        assert exporter.get_finished_spans() == ()
    finally:
        owner.deactivate()
        provider.shutdown()


def test_released_error_has_native_status_no_invented_http_or_output(runtime):
    _, _, exporter, _ = runtime
    with pytest.raises(Exception) as caught:
        Agent(model=model(fail=True), callback_handler=None)("error prompt")
    assert type(caught.value).__name__ == "APIConnectionError"
    errors = [
        span
        for span in exporter.get_finished_spans()
        if span.status.status_code is StatusCode.ERROR
    ]
    assert len(errors) >= 2
    for span in errors:
        assert "status_code" not in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
        assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in span.attributes


def test_released_native_async_generator_context_close_and_cancellation(runtime):
    _, _, exporter, _ = runtime

    async def run():
        agent = Agent(model=model(), callback_handler=None)
        stream = agent.stream_async("native stream")
        assert inspect.isasyncgen(stream)
        assert hasattr(stream, "asend") and hasattr(stream, "athrow")
        async for _event in stream:
            pass
        closing = agent.stream_async("close early")
        await anext(closing)
        await closing.aclose()
        cancelled = agent.stream_async("cancel early")
        await anext(cancelled)
        with pytest.raises(asyncio.CancelledError):
            await cancelled.athrow(asyncio.CancelledError())

    asyncio.run(run())
    assert chats(exporter)


def test_refcount_foreign_hooks_and_transaction_rollback(runtime, monkeypatch):
    owner, provider, _, native = runtime
    second = StrandsAgentsInstrumentor()
    second.activate()
    assert (
        len(
            [
                p
                for p in provider._active_span_processor._span_processors
                if p is owner._shared_processor
            ]
        )
        == 1
    )
    second.deactivate()
    assert owner._shared_processor is not None
    original = owner._original_tracer_methods["_start_span"]
    wrapper = tracer_module.Tracer._start_span

    def foreign(*args, **kwargs):
        return wrapper(*args, **kwargs)

    monkeypatch.setattr(tracer_module.Tracer, "_start_span", foreign)
    owner.deactivate()
    assert tracer_module.Tracer._start_span is foreign
    native.start_model_invoke_span(messages=[], model_id="inert-retained-wrapper").end()
    assert owner._activation_count == 0
    restored = tracer_module.Tracer._add_event
    broken = StrandsAgentsInstrumentor()

    def fail(*_args):
        raise RuntimeError("registration failure")

    monkeypatch.setattr(broken, "_register_processor", fail)
    with pytest.raises(RuntimeError, match="registration failure"):
        broken.activate()
    assert broken._activation_count == 0
    assert tracer_module.Tracer._add_event is restored
    assert not broken._shared_processor
    monkeypatch.setattr(tracer_module.Tracer, "_start_span", original)


def test_nested_tools_are_scoped_per_agent_span(runtime):
    _, _, exporter, native = runtime
    config = lambda name: {
        name: {"toolSpec": {"name": name, "inputSchema": {"json": {"type": "object"}}}}
    }
    outer = native.start_agent_span(
        messages=[], agent_name="outer", tools_config=config("outer-tool")
    )
    with trace.use_span(outer, end_on_exit=False):
        inner = native.start_agent_span(
            messages=[], agent_name="inner", tools_config=config("inner-tool")
        )
        child = native.start_model_invoke_span(
            messages=[], parent_span=inner, model_id="fixture-model"
        )
        native.end_span_with_error(child, "controlled error")
        native.end_agent_span(
            inner,
            response=SimpleNamespace(stop_reason="end_turn", content="inner answer"),
        )
        sibling = native.start_model_invoke_span(
            messages=[], parent_span=outer, model_id="fixture-model"
        )
        native.end_span_with_error(sibling, "controlled error")
    native.end_agent_span(
        outer, response=SimpleNamespace(stop_reason="end_turn", content="outer answer")
    )
    first, second = chats(exporter)
    assert (
        json.loads(first.attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0][
            "function"
        ]["name"]
        == "inner-tool"
    )
    assert (
        json.loads(second.attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0][
            "function"
        ]["name"]
        == "outer-tool"
    )


def test_unrelated_genai_scope_is_untouched(runtime):
    _, provider, exporter, _ = runtime
    span = provider.get_tracer("foreign-provider").start_span(
        "chat",
        attributes={
            gen_ai.GEN_AI_OPERATION_NAME: "chat",
            SpanAttributes.LLM_SYSTEM: "openai",
        },
    )
    span.add_event("native-content", {"content": "foreign"})
    span.end()
    result = exporter.get_finished_spans()[0]
    assert RESPAN_LOG_TYPE not in result.attributes
    assert result.events[0].attributes["content"] == "foreign"


def test_strict_usage_no_bool_negative_or_partial_invention():
    attrs = {
        SpanAttributes.LLM_SYSTEM: "strands-agents",
        gen_ai.GEN_AI_OPERATION_NAME: "chat",
        gen_ai.GEN_AI_USAGE_INPUT_TOKENS: 0,
        gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS: True,
        SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS: -1,
    }
    span = SimpleNamespace(name="chat", _attributes=attrs, events=())
    enrich_strands_agents_span(span)
    assert span._attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 0
    assert gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS not in span._attributes
    assert SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in span._attributes
    assert SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS not in span._attributes


def test_latest_message_conventions_system_input_and_tool_payloads(runtime):
    _, _, exporter, native = runtime
    if (
        "system_prompt"
        not in inspect.signature(native.start_model_invoke_span).parameters
    ):
        pytest.skip("System prompt native telemetry added after minimum Strands 1.20")
    native.use_latest_genai_conventions = True
    native._span_attributes_only = True
    span = native.start_model_invoke_span(
        messages=[{"role": "user", "content": [{"text": "latest prompt"}]}],
        model_id="fixture-model",
        system_prompt="latest system",
    )
    native.end_model_invoke_span(
        span,
        message={"role": "assistant", "content": [{"text": "latest answer"}]},
        usage={"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
        metrics={},
        stop_reason="end_turn",
    )
    chat = chats(exporter)[0]
    content = json.loads(chat.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert content[0]["role"] == "system"
    assert "latest system" in str(content[0])
    assert chat.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 0
    tool_span = native.start_tool_call_span(
        {"name": "lookup", "toolUseId": "latest-tool-id", "input": {"query": "fixture"}}
    )
    native.end_tool_call_span(
        tool_span,
        {
            "toolUseId": "latest-tool-id",
            "status": "success",
            "content": [{"text": "fixture result"}],
        },
    )
    result = exporter.get_finished_spans()[-1]
    assert result.attributes[gen_ai.GEN_AI_TOOL_CALL_ID] == "latest-tool-id"
    assert json.loads(result.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])[
        "arguments"
    ] == {"query": "fixture"}
    assert "fixture result" in result.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]


def test_latest_memory_spans_map_native_payloads_and_privacy(runtime, monkeypatch):
    _, _, exporter, native = runtime
    if not hasattr(native, "start_memory_search_span"):
        pytest.skip("Memory telemetry added after minimum Strands 1.20")
    span = native.start_memory_search_span("controlled memory query", ["fixture-store"])
    native.end_memory_search_span(
        span,
        [
            SimpleNamespace(
                content="controlled memory result",
                store_name="fixture-store",
                metadata={"source": "fixture"},
            )
        ],
    )
    result = exporter.get_finished_spans()[0]
    assert result.attributes[RESPAN_LOG_TYPE] == "task"
    assert (
        "controlled memory query"
        in result.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    )
    assert (
        "controlled memory result"
        in result.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    private = native.start_memory_search_span("private memory query", ["private-store"])
    native.end_memory_search_span(private, [])
    hidden = exporter.get_finished_spans()[-1]
    assert "private" not in str(hidden.attributes)
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in hidden.attributes


def test_refresh_failure_rolls_back_existing_singleton(monkeypatch):
    from respan_instrumentation_strands_agents import _instrumentation

    original_provider = object()
    original_tracer = object()
    native = SimpleNamespace(
        tracer_provider=original_provider,
        tracer=original_tracer,
        _include_tool_definitions=False,
        service_name="strands.telemetry.tracer",
    )

    def fail():
        raise RuntimeError("failure after provider mutation")

    native._parse_semconv_opt_in = fail
    monkeypatch.setattr(tracer_module, "_tracer_instance", native)
    provider = TracerProvider()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer_provider", lambda: provider)
    owner = StrandsAgentsInstrumentor()
    with pytest.raises(RuntimeError, match="failure after provider mutation"):
        owner.activate()
    assert native.tracer_provider is original_provider
    assert native.tracer is original_tracer
    assert not owner._activation_count
    assert provider._active_span_processor._span_processors == ()
    provider.shutdown()


def test_native_tool_vector_and_large_tool_lists_are_complete(runtime):
    _, _, exporter, native = runtime
    vectors = [float(index) for index in range(4096)]
    span = native.start_tool_call_span(
        {
            "name": "vectors",
            "toolUseId": "vector-tool-id",
            "input": {"vectors": vectors},
        }
    )
    native.end_tool_call_span(
        span,
        {
            "toolUseId": "vector-tool-id",
            "status": "success",
            "content": [{"json": {"vectors": vectors}}],
        },
    )
    result = exporter.get_finished_spans()[0]
    assert (
        json.loads(result.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])[
            "arguments"
        ]["vectors"]
        == vectors
    )
    assert (
        json.loads(result.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[0][
            "json"
        ]["vectors"]
        == vectors
    )
    tools = [
        {"name": f"tool-{index}", "inputSchema": {"json": {"type": "object"}}}
        for index in range(75)
    ]
    chat = native.start_model_invoke_span(
        messages=[], **{gen_ai.GEN_AI_TOOL_DEFINITIONS: json.dumps(tools)}
    )
    native.end_span_with_error(chat, "controlled failure")
    assert (
        len(
            json.loads(
                chats(exporter)[0].attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS]
            )
        )
        == 75
    )


def test_latest_responses_provider_real_sse_usage(runtime):
    try:
        from strands.models.openai_responses import OpenAIResponsesModel
    except ImportError:
        pytest.skip("Responses provider added after minimum Strands 1.20")
    _, _, exporter, _ = runtime

    def respond(request):
        assert json.loads(request.content)["stream"] is True
        body = {
            "id": "fixture-response",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 9,
                "output_tokens": 3,
                "total_tokens": 12,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens_details": {"reasoning_tokens": 1},
            },
        }
        events = [
            {"type": "response.created", "response": body},
            {
                "type": "response.output_text.delta",
                "delta": "Responses answer",
                "item_id": "fixture-item",
                "output_index": 0,
                "content_index": 0,
            },
            {"type": "response.completed", "response": body},
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        )

    provider = OpenAIResponsesModel(
        model_id="fixture-responses-model",
        client_args={
            "api_key": "fixture-key",
            "base_url": "https://fixture.invalid/v1",
            "http_client": FixtureClient(transport=httpx.MockTransport(respond)),
        },
    )
    assert "Responses answer" in str(
        Agent(model=provider, callback_handler=None)("Responses prompt")
    )
    chat = chats(exporter)[0]
    assert chat.attributes[SpanAttributes.LLM_SYSTEM] == "openai"
    assert chat.attributes[SpanAttributes.GEN_AI_IS_STREAMING] is True
    assert chat.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 9
    assert chat.attributes[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 2
    assert chat.attributes[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 1


def test_provider_missing_usage_not_replaced_by_native_defaults(runtime):
    try:
        from strands.models.openai_responses import OpenAIResponsesModel
    except ImportError:
        pytest.skip("Responses provider added after minimum Strands 1.20")
    _, _, exporter, native = runtime
    provider = OpenAIResponsesModel(model_id="fixture-model")
    span = native.start_model_invoke_span(messages=[], model_id="fixture-model")
    with trace.use_span(span, end_on_exit=False):
        result = provider._format_chunk(
            {"chunk_type": "metadata", "data": SimpleNamespace(output_tokens=3)}
        )
    assert result["metadata"]["usage"]["inputTokens"] == 0  # SDK's unmodified result
    native.end_model_invoke_span(
        span,
        message={"role": "assistant", "content": [{"text": "fixture answer"}]},
        usage=result["metadata"]["usage"],
        metrics={},
        stop_reason="end_turn",
    )
    chat = chats(exporter)[0]
    assert gen_ai.GEN_AI_USAGE_INPUT_TOKENS not in chat.attributes
    assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in chat.attributes
    assert SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in chat.attributes
    assert chat.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 3
