"""Released DSPy contracts under controlled provider/model boundaries."""

import asyncio
import json
from functools import wraps

import dspy
import numpy as np
import pytest
from litellm import ModelResponse
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_dspy import DSPyInstrumentor, _instrumentation
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


class ControlledLM(dspy.BaseLM):
    def __init__(self, replies, **kwargs):
        super().__init__("openai/fixture-dspy", cache=False, **kwargs)
        self.replies = iter(replies)

    def forward(self, prompt=None, messages=None, **kwargs):
        value = next(self.replies)
        if isinstance(value, BaseException):
            raise value
        return value

    async def aforward(self, prompt=None, messages=None, **kwargs):
        await asyncio.sleep(0)
        return self.forward(prompt, messages, **kwargs)


def reply(content="answer", *, calls=None, count=11):
    return ModelResponse(
        model="fixture-dspy",
        id="fixture-response",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": calls,
                },
            }
        ],
        usage={
            "prompt_tokens": count,
            "completion_tokens": 7,
            "total_tokens": count + 7,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    )


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    plugin = DSPyInstrumentor(tracer_provider=provider)
    plugin.activate()
    dspy.configure(disable_history=False)
    yield provider, exporter, plugin
    plugin.deactivate()
    provider.shutdown()


def spans(exporter, kind):
    return [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get(RESPAN_LOG_TYPE) == kind
    ]


def test_real_predict_lm_adapter_topology_usage_and_native_prediction(runtime):
    provider, exporter, _ = runtime
    model = ControlledLM([reply("[[ ## answer ## ]]\nParis\n[[ ## completed ## ]]")])
    with (
        dspy.context(lm=model),
        provider.get_tracer("test").start_as_current_span("root") as root,
    ):
        value = dspy.Predict("question -> answer")(question="Capital of France?")
        assert isinstance(value, dspy.Prediction) and value.answer == "Paris"
        assert trace.get_current_span() is root
    recorded = exporter.get_finished_spans()
    ids = {s.context.span_id for s in recorded}
    assert all(s.parent is None or s.parent.span_id in ids for s in recorded)
    assert spans(exporter, "task") and len(spans(exporter, "chat")) == 1
    attrs = spans(exporter, "chat")[0].attributes
    assert attrs[gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS] == 11
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 18
    assert attrs[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 2
    assert attrs[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert all(SpanAttributes.TRACELOOP_SPAN_KIND not in s.attributes for s in recorded)


def test_tool_native_result_and_failure_identity_without_invented_output(runtime):
    _, exporter, _ = runtime
    output = {"vector": [float(i) for i in range(5000)]}
    tool = dspy.Tool(lambda value: output, name="vector_tool")
    assert tool(value=2) is output
    attrs = spans(exporter, "tool")[0].attributes
    assert (
        len(json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])["vector"]) == 5000
    )
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "name": "vector_tool",
        "arguments": {"value": 2},
    }
    error = ValueError("controlled failure")

    def fail(value):
        raise error

    with pytest.raises(ValueError) as raised:
        dspy.Tool(fail)(value="input")
    assert raised.value is error
    failed = spans(exporter, "tool")[-1]
    assert failed.status.status_code is trace.StatusCode.ERROR
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in failed.attributes
    assert failed.events[0].attributes["exception.message"] == "controlled failure"


def test_model_error_has_no_completion_usage_or_fake_http_status(runtime):
    _, exporter, _ = runtime
    error = ValueError("controlled provider failure")
    with pytest.raises(ValueError) as raised:
        ControlledLM([error])("prompt")
    assert raised.value is error
    span = spans(exporter, "chat")[0]
    assert span.status.status_code is trace.StatusCode.ERROR
    assert not any(
        k.startswith(SpanAttributes.LLM_COMPLETIONS) for k in span.attributes
    )
    assert gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS not in span.attributes
    assert "status_code" not in span.attributes


@pytest.mark.parametrize("policy", ["environment", "context", "include_content"])
def test_privacy_start_bound_and_no_payload_serialization(runtime, monkeypatch, policy):
    provider, exporter, plugin = runtime

    class Hostile:
        def __repr__(self):
            raise AssertionError("content inspected")

        def __str__(self):
            raise AssertionError("content inspected")

    token = None
    if policy == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    elif policy == "context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    else:
        plugin.deactivate()
        plugin = DSPyInstrumentor(include_content=False, tracer_provider=provider)
        plugin.activate()
    try:
        assert dspy.Tool(lambda value: value)(value=Hostile()).__class__ is Hostile
    finally:
        if token is not None:
            context.detach(token)
        if policy == "include_content":
            plugin.deactivate()
    attrs = spans(exporter, "tool")[0].attributes
    assert (
        SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
        and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    )


def test_privacy_end_veto_and_initial_false_cannot_be_enabled(runtime, monkeypatch):
    _, exporter, _ = runtime

    def disable(value):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        return "private output"

    dspy.Tool(disable)(value="private input")
    assert "private" not in str(spans(exporter, "tool")[0].attributes)

    def enable(value):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        return "private output"

    dspy.Tool(enable)(value="private input")
    assert "private" not in str(spans(exporter, "tool")[-1].attributes)


def test_suppression_and_sampled_off_calls_never_export(runtime):
    _, exporter, plugin = runtime
    token = context.attach(
        context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    try:
        assert dspy.Tool(lambda value: value)(value="suppressed") == "suppressed"
    finally:
        context.detach(token)
    assert not exporter.get_finished_spans()
    plugin.deactivate()
    disabled = TracerProvider(sampler=ALWAYS_OFF)
    p = DSPyInstrumentor(tracer_provider=disabled)
    p.activate()
    try:
        assert dspy.Tool(lambda value: value)(value="off") == "off"
    finally:
        p.deactivate()
        disabled.shutdown()
    assert not exporter.get_finished_spans()


def test_shared_global_target_registration_foreign_callbacks_and_hooks(
    runtime, monkeypatch
):
    provider, exporter, plugin = runtime
    second = DSPyInstrumentor(tracer_provider=provider)
    second.activate()
    tool = dspy.Tool(lambda value: value)
    targeted = DSPyInstrumentor(tool, tracer_provider=provider)
    targeted.activate()
    assert tool(value="one") == "one" and len(spans(exporter, "tool")) == 1
    plugin.deactivate()
    tool(value="two")
    assert len(spans(exporter, "tool")) == 2
    from dspy.utils.callback import BaseCallback

    foreign_callback = BaseCallback()
    dspy.configure(callbacks=dspy.settings.callbacks + [foreign_callback])
    ours = dspy.BaseLM._process_lm_response

    @wraps(ours)
    def foreign(self, *args, **kwargs):
        return ours(self, *args, **kwargs)

    monkeypatch.setattr(dspy.BaseLM, "_process_lm_response", foreign)
    second.deactivate()
    targeted.deactivate()
    assert dspy.BaseLM._process_lm_response is foreign
    assert foreign_callback in dspy.settings.callbacks
    dspy.configure(
        callbacks=[c for c in dspy.settings.callbacks if c is not foreign_callback]
    )
    tool(value="untraced")
    assert len(spans(exporter, "tool")) == 2


def test_registration_failure_rolls_back_owned_callback_and_hooks(monkeypatch):
    original = dspy.BaseLM._process_lm_response
    actual = _instrumentation._Runtime.install_hooks

    def broken(self):
        actual(self)
        raise RuntimeError("partial activation")

    monkeypatch.setattr(_instrumentation._Runtime, "install_hooks", broken)
    provider = TracerProvider()
    plugin = DSPyInstrumentor(tracer_provider=provider)
    before = list(dspy.settings.callbacks)
    with pytest.raises(RuntimeError):
        plugin.activate()
    assert (
        list(dspy.settings.callbacks) == before
        and dspy.BaseLM._process_lm_response is original
    )
    assert _instrumentation._RUNTIME is None
    provider.shutdown()


@pytest.mark.asyncio
async def test_async_lm_and_tool_callbacks_detach_and_preserve_outputs(runtime):
    provider, exporter, _ = runtime

    async def echo(value):
        await asyncio.sleep(0)
        return value

    with provider.get_tracer("test").start_as_current_span("root") as root:
        assert await dspy.Tool(echo).acall(value="async tool") == "async tool"
        values = await ControlledLM([reply("async model")]).acall("async prompt")
        assert values == ["async model"]
        assert trace.get_current_span() is root
    assert len(spans(exporter, "chat")) == 1 and len(spans(exporter, "tool")) == 1


def test_full_embeddings_preserve_native_numpy_return_and_do_not_invent_usage(runtime):
    _, exporter, _ = runtime
    result = np.array([[float(i) for i in range(5000)]], dtype=np.float32)
    embedder = dspy.Embedder(lambda texts: result, caching=False)
    native = embedder(["vector fixture"])
    assert isinstance(native, np.ndarray) and native.shape == (1, 5000)
    attrs = spans(exporter, "embedding")[0].attributes
    assert len(json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[0]) == 5000
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
        "vector fixture"
    ]
    assert gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS not in attrs


def test_request_zero_override_and_incomplete_invalid_usage(runtime):
    _, exporter, _ = runtime
    lm = ControlledLM([reply()], temperature=0.8, max_tokens=33)
    lm("prompt", temperature=0, max_tokens=0)
    attrs = spans(exporter, "chat")[0].attributes
    assert (
        attrs[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0
        and attrs[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 0
    )
    from respan_instrumentation_dspy._utils import add_lm_usage_attributes

    attrs = {}
    add_lm_usage_attributes(attrs, {"prompt_tokens": -1, "completion_tokens": 3})
    assert (
        gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS not in attrs
        and SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in attrs
    )
    assert attrs[gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 3


def test_current_and_historical_tool_calls_keep_json_arguments_and_ids(runtime):
    _, exporter, _ = runtime
    previous = {
        "id": "previous-id",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"value":1}'},
    }
    current = {
        "id": "current-id",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"value":2}'},
    }
    lm = ControlledLM([reply(None, calls=[current])])
    lm(
        messages=[
            {"role": "assistant", "content": None, "tool_calls": [previous]},
            {"role": "tool", "content": "old result", "tool_call_id": "previous-id"},
        ]
    )
    attrs = spans(exporter, "chat")[0].attributes
    assert (
        json.loads(attrs[f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls"])[0]["id"]
        == "previous-id"
    )
    emitted = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert (
        len(emitted) == 1
        and emitted[0]["id"] == "current-id"
        and isinstance(emitted[0]["function"]["arguments"], str)
    )


def test_native_lm15_request_response_preserves_identity_and_source_usage_without_history(
    runtime,
):
    if not hasattr(dspy, "lm15"):
        pytest.skip("Native lm15 types were introduced after declared minimum DSPy3.0")
    _, exporter, _ = runtime
    from dspy.lm15 import FunctionTool, Message, Request, Response, ToolCallPart, Usage

    native = Response(
        id="native-response-1",
        model="fixture-model",
        message=Message.assistant(
            [ToolCallPart(id="native-tool-1", name="lookup", input={"value": 2})]
        ),
        finish_reason="tool_call",
        usage=Usage(
            input_tokens=11,
            output_tokens=7,
            total_tokens=18,
            cache_read_tokens=3,
            reasoning_tokens=2,
        ),
    )

    class Engine:
        def complete(self, request):
            return native

        def close(self):
            pass

    lm = dspy.LM("openai/fixture-model", engine=Engine(), cache=False, num_retries=0)
    request = Request(
        model="openai/fixture-model",
        messages=(Message.user("lookup fixture"),),
        tools=(
            FunctionTool(
                name="lookup",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                },
            ),
        ),
    )
    with dspy.context(disable_history=True):
        assert lm(request) is native
    attrs = spans(exporter, "chat")[0].attributes
    assert attrs[gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS] == 11
    assert attrs[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert attrs[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 2
    assert (
        json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])[0]["id"]
        == "native-tool-1"
    )
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "lookup fixture"
    assert "lookup" in attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS]


@pytest.mark.asyncio
async def test_native_streamify_preserves_prediction_and_detached_caller_context(
    runtime,
):
    if not hasattr(dspy, "lm15"):
        pytest.skip("Controlled native streaming engine requires lm15")
    provider, exporter, _ = runtime
    from dspy.lm15 import Message, Response, TextPart, Usage, response_to_events

    native = Response(
        id="stream-response",
        model="fixture-model",
        message=Message.assistant(
            [TextPart("[[ ## answer ## ]]\nstreamed\n[[ ## completed ## ]]")]
        ),
        finish_reason="stop",
        usage=Usage(input_tokens=4, output_tokens=2, total_tokens=6),
    )

    class Engine:
        async def complete(self, request):
            return native

        async def stream(self, request):
            for event in response_to_events(native):
                yield event

        async def aclose(self):
            pass

    class Sync:
        def complete(self, request):
            return native

        def stream(self, request):
            yield from response_to_events(native)

        def close(self):
            pass

    lm = dspy.LM(
        "openai/fixture-model",
        engine=Sync(),
        async_engine=Engine(),
        cache=False,
        num_retries=0,
    )
    with (
        dspy.context(lm=lm),
        provider.get_tracer("test").start_as_current_span("root") as root,
    ):
        stream = dspy.streamify(dspy.Predict("question -> answer"))
        outputs = []
        async for chunk in stream(question="stream fixture"):
            outputs.append(chunk)
            assert trace.get_current_span() is root
    assert isinstance(outputs[-1], dspy.Prediction) and outputs[-1].answer == "streamed"
    assert len(spans(exporter, "chat")) == 1
    assert (
        spans(exporter, "chat")[0].attributes[
            gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS
        ]
        == 4
    )


@pytest.mark.asyncio
async def test_async_embedder_and_embedding_error_keep_native_semantics(runtime):
    _, exporter, _ = runtime
    if not hasattr(dspy.Embedder, "acall"):
        pytest.skip("Async Embedder not available in this supported SDK")
    result = await dspy.Embedder(
        lambda texts: [[1.0, 2.0, 3.0] for text in texts], caching=False
    ).acall(["one", "two"])
    assert result.shape == (2, 3)
    assert len(spans(exporter, "embedding")) == 1
    error = ValueError("embedding failure")

    def broken(texts):
        raise error

    with pytest.raises(ValueError) as raised:
        dspy.Embedder(broken, caching=False)(["fixture"])
    assert raised.value is error
    assert (
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT
        not in spans(exporter, "embedding")[-1].attributes
    )


@pytest.mark.asyncio
async def test_react_v2_actual_dispatch_ids_and_ambiguous_identical_calls(runtime):
    if not hasattr(dspy, "ReActV2"):
        pytest.skip("ReActV2 introduced after supported minimum")
    _, exporter, _ = runtime

    async def lookup(value: int):
        return {"vector": [float(i) for i in range(5000)]}

    agent = dspy.ReActV2("question -> answer", tools=[lookup])
    calls = dspy.ToolCalls(
        tool_calls=[{"id": "source-a", "name": "lookup", "args": {"value": 2}}]
    )
    results, final = await agent._aexecute_tool_calls(calls)
    assert final is None and len(results.tool_call_results[0].value["vector"]) == 5000
    attrs = spans(exporter, "tool")[0].attributes
    assert attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_ID] == "source-a"
    assert (
        len(json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])["vector"]) == 5000
    )
    ambiguous = dspy.ToolCalls(
        tool_calls=[
            {"id": "source-b", "name": "lookup", "args": {"value": 2}},
            {"id": "source-c", "name": "lookup", "args": {"value": 2}},
        ]
    )
    await agent._aexecute_tool_calls(ambiguous)
    assert all(
        gen_ai_attributes.GEN_AI_TOOL_CALL_ID not in s.attributes
        for s in spans(exporter, "tool")[1:]
    )


def test_provider_usage_details_and_quoted_credential_redaction():
    from respan_instrumentation_dspy._serialization import redact_text
    from respan_instrumentation_dspy._utils import add_lm_usage_attributes

    attrs = {}
    add_lm_usage_attributes(
        attrs, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )
    assert attrs[gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS] == 0
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 0
    add_lm_usage_attributes(
        attrs,
        {
            "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 5},
            "output_tokens_details": {"reasoning_tokens": 6},
        },
    )
    assert attrs[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 4
    assert attrs[SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] == 5
    assert attrs[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 6
    for text in [
        "token='private phrase with spaces'",
        '"api_key": "private phrase with spaces"',
        "Bearer private-token",
        "https://user:password@host/path",
    ]:
        result = redact_text(text)
        assert "private" not in result and "password" not in result


def test_shutdown_clears_pending_payload_and_detaches_on_native_completion(runtime):
    provider, exporter, plugin = runtime

    def shutdown(value):
        plugin.deactivate()
        return value

    with provider.get_tracer("test").start_as_current_span("root") as root:
        assert dspy.Tool(shutdown)(value="pending content") == "pending content"
        assert trace.get_current_span() is root
    recorded = spans(exporter, "tool")
    assert (
        len(recorded) == 1
        and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in recorded[0].attributes
    )
