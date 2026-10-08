"""Exercise released smolagents behavior with deterministic model responses."""

import json
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

import pytest
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_smolagents import SmolagentsInstrumentor, _instrumentation
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from smolagents import CodeAgent, Model, ToolCallingAgent, tool
from smolagents.models import (
    ChatMessage,
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    ChatMessageToolCallStreamDelta,
    MessageRole,
)
from smolagents.monitoring import TokenUsage


class FixtureModel(Model):
    provider = "fixture"

    def __init__(self, replies):
        super().__init__(model_id="fixture-smolagents")
        self.replies = iter(replies)

    def generate(self, messages, tools_to_call_from=None, **kwargs):
        value = next(self.replies)
        if isinstance(value, BaseException):
            raise value
        return value


def message(content=None, calls=None):
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content=content,
        tool_calls=calls,
        token_usage=TokenUsage(11, 7),
    )


def call(name, arguments, identifier="call-1"):
    return ChatMessageToolCall(
        id=identifier,
        type="function",
        function=ChatMessageToolCallFunction(name=name, arguments=arguments),
    )


@tool
def add(left: int, right: int) -> int:
    """Add two integers.

    Args:
        left: First integer.
        right: Second integer.
    """
    return left + right


@tool
def fail(value: str) -> str:
    """Raise a controlled tool error.

    Args:
        value: Failure message.
    """
    raise RuntimeError(value)


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    plugin = SmolagentsInstrumentor(tracer_provider=provider)
    plugin.activate()
    yield provider, exporter, plugin
    plugin.deactivate()
    provider.shutdown()


def kinds(exporter, kind):
    return [
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get(RESPAN_LOG_TYPE) == kind
    ]


def test_tool_calling_real_agent_connected_ids_and_canonical_contract(runtime):
    provider, exporter, _ = runtime
    model = FixtureModel(
        [
            message(calls=[call("add", {"left": 2, "right": 3}, "add-1")]),
            message(calls=[call("final_answer", {"answer": "five"}, "answer-1")]),
        ]
    )
    agent = ToolCallingAgent(tools=[add], model=model, verbosity_level=0)
    with provider.get_tracer("test").start_as_current_span("root") as root:
        assert agent.run("Add 2 and 3") == "five"
        assert trace.get_current_span() is root
    spans = exporter.get_finished_spans()
    assert len(kinds(exporter, "agent")) == 1
    assert len(kinds(exporter, "task")) == 2
    assert len(kinds(exporter, "chat")) == 2
    tools = kinds(exporter, "tool")
    assert {
        span.attributes[gen_ai_attributes.GEN_AI_TOOL_CALL_ID] for span in tools
    } == {"add-1", "answer-1"}
    assert json.loads(tools[0].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "name": "add",
        "arguments": {"left": 2, "right": 3},
    }
    assert json.loads(tools[0].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == 5
    chats = kinds(exporter, "chat")
    assert (
        json.loads(
            chats[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
        )[0]["id"]
        == "add-1"
    )
    assert chats[0].attributes[gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS] == 11
    assert chats[0].attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 18
    ids = {span.context.span_id for span in spans}
    assert all(span.parent is None or span.parent.span_id in ids for span in spans)
    assert all(
        SpanAttributes.TRACELOOP_SPAN_KIND not in span.attributes for span in spans
    )
    assert all(
        not any(
            key in span.attributes
            for key in ("model", "tools", "tool_calls", "respan.span.tools")
        )
        for span in spans
    )
    assert (
        SpanAttributes.LLM_REQUEST_MODEL not in kinds(exporter, "agent")[0].attributes
    )


def test_code_agent_and_full_result_keep_native_return(runtime):
    _, exporter, _ = runtime
    from smolagents.agents import RunResult

    agent = CodeAgent(
        tools=[add],
        model=FixtureModel(
            [message("Thought: add.\n<code>final_answer(add(2, 3))</code>")]
        ),
        verbosity_level=0,
    )
    result = agent.run("Add", return_full_result=True)
    assert isinstance(result, RunResult)
    assert result.output == 5
    assert (
        json.loads(
            kinds(exporter, "agent")[0].attributes[
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT
            ]
        )
        == 5
    )
    assert len(kinds(exporter, "chat")) == 1
    assert kinds(exporter, "tool")


def test_streaming_agent_detaches_at_creation_each_yield_and_thread(runtime):
    provider, exporter, _ = runtime
    agent = ToolCallingAgent(
        tools=[],
        model=FixtureModel(
            [message(calls=[call("final_answer", {"answer": "streamed"})])]
        ),
        verbosity_level=0,
    )
    with provider.get_tracer("test").start_as_current_span("root") as root:
        iterator = agent.run("Stream", stream=True)
        assert isinstance(iterator, _instrumentation._Stream)
        assert trace.get_current_span() is root
        next(iterator)
        assert trace.get_current_span() is root
        with ThreadPoolExecutor(1) as executor:
            chunks = executor.submit(list, iterator).result()
        assert chunks[-1].output == "streamed"
        assert trace.get_current_span() is root
    assert (
        json.loads(
            kinds(exporter, "agent")[0].attributes[
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT
            ]
        )
        == "streamed"
    )
    assert (
        kinds(exporter, "chat")[0].parent.span_id
        == kinds(exporter, "task")[0].context.span_id
    )


def test_native_stream_send_throw_close_and_errors(runtime):
    provider, exporter, _ = runtime

    class StreamModel(Model):
        def generate_stream(self, messages, **kwargs):
            try:
                received = yield ChatMessageStreamDelta(content="first")
                yield ChatMessageStreamDelta(content=received)
            except ValueError:
                yield ChatMessageStreamDelta(content="handled")
            return "native-stop-value"

    model = StreamModel(model_id="stream-fixture")
    with provider.get_tracer("test").start_as_current_span("root") as root:
        stream = model.generate_stream([])
        assert trace.get_current_span() is root
        assert next(stream).content == "first"
        assert stream.send("second").content == "second"
        assert stream.throw(ValueError("controlled")).content == "handled"
        with pytest.raises(StopIteration) as stopped:
            next(stream)
        assert stopped.value.value == "native-stop-value"
        assert trace.get_current_span() is root
        closed = model.generate_stream([])
        next(closed)
        assert closed.close() is None
        assert trace.get_current_span() is root
    assert len(kinds(exporter, "chat")) == 2
    assert (
        kinds(exporter, "chat")[0].attributes[
            f"{SpanAttributes.LLM_COMPLETIONS}.0.content"
        ]
        == "firstsecondhandled"
    )


def test_streamed_tool_delta_and_usage_are_complete(runtime):
    _, exporter, _ = runtime

    class StreamModel(Model):
        def generate_stream(self, messages, **kwargs):
            yield ChatMessageStreamDelta(
                tool_calls=[
                    ChatMessageToolCallStreamDelta(
                        index=0,
                        id="tool-7",
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name="add", arguments='{"left":'
                        ),
                    )
                ]
            )
            yield ChatMessageStreamDelta(
                tool_calls=[
                    ChatMessageToolCallStreamDelta(
                        index=0,
                        function=ChatMessageToolCallFunction(
                            name="", arguments='2,"right":3}'
                        ),
                    )
                ],
                token_usage=TokenUsage(4, 2),
            )

    chunks = list(StreamModel(model_id="stream-fixture").generate_stream([]))
    assert len(chunks) == 2
    span = kinds(exporter, "chat")[0]
    assert json.loads(
        span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
    )[0] == {
        "id": "tool-7",
        "type": "function",
        "function": {"name": "add", "arguments": '{"left":2,"right":3}'},
    }
    assert span.attributes[gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS] == 4
    assert span.attributes[SpanAttributes.GEN_AI_IS_STREAMING] is True


def test_model_and_tool_failure_preserve_exception_identity_and_error_status(runtime):
    _, exporter, _ = runtime
    error = ValueError("controlled model failure")
    with pytest.raises(ValueError) as raised:
        FixtureModel([error]).generate([{"role": "user", "content": "hello"}])
    assert raised.value is error
    with pytest.raises(RuntimeError):
        fail("controlled tool failure")
    for span in exporter.get_finished_spans():
        assert span.status.status_code is trace.StatusCode.ERROR
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
        assert f"{SpanAttributes.LLM_COMPLETIONS}.0.content" not in span.attributes
        assert gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS not in span.attributes


def test_native_run_exception_finishes_span_and_detaches(runtime, monkeypatch):
    provider, exporter, _ = runtime
    agent = ToolCallingAgent(tools=[], model=FixtureModel([]), verbosity_level=0)
    error = RuntimeError("run failure")
    monkeypatch.setattr(
        agent, "initialize_system_prompt", lambda: (_ for _ in ()).throw(error)
    )
    with provider.get_tracer("test").start_as_current_span("root") as root:
        with pytest.raises(RuntimeError) as raised:
            agent.run("x")
        assert raised.value is error
        assert trace.get_current_span() is root
    assert kinds(exporter, "agent")[0].status.status_code is trace.StatusCode.ERROR


@pytest.mark.parametrize("policy", ["environment", "context"])
def test_privacy_initial_upper_bound_and_end_veto(runtime, monkeypatch, policy):
    _, exporter, _ = runtime

    class Hostile:
        def __iter__(self):
            raise AssertionError("content must not be read")

    def disable():
        if policy == "environment":
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
            return None
        return context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))

    def enable(token):
        if token is not None:
            context.detach(token)
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")

    token = disable()
    model = FixtureModel([message("private")])
    model.generate(Hostile())
    enable(token)
    span = kinds(exporter, "chat")[0]
    assert not any(
        key.startswith((SpanAttributes.LLM_PROMPTS, SpanAttributes.LLM_COMPLETIONS))
        for key in span.attributes
    )

    class FlipModel(Model):
        def generate(self, messages, **kwargs):
            nonlocal token
            token = disable()
            return message("private-output")

    FlipModel(model_id="flip").generate([{"role": "user", "content": "private-input"}])
    # The instrumentor restores the caller context after the internal veto.
    enable(None)
    span = kinds(exporter, "chat")[-1]
    assert "private" not in str(span.attributes)


def test_privacy_stream_creation_bound_and_consumer_veto(runtime, monkeypatch):
    _, exporter, _ = runtime

    class StreamModel(Model):
        def generate_stream(self, messages, **kwargs):
            yield ChatMessageStreamDelta(content="private-output")

    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    stream = StreamModel().generate_stream(
        [{"role": "user", "content": "private-input"}]
    )
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    list(stream)
    assert "private" not in str(kinds(exporter, "chat")[0].attributes)
    stream = StreamModel().generate_stream(
        [{"role": "user", "content": "private-input"}]
    )
    next(stream)
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        stream.close()
    finally:
        context.detach(token)
    assert "private" not in str(kinds(exporter, "chat")[-1].attributes)


def test_suppression_and_sampling_do_not_inspect_content(runtime):
    _, exporter, _ = runtime

    class Hostile:
        def __iter__(self):
            raise AssertionError("content accessed")

    token = context.attach(
        context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    try:
        FixtureModel([message("hidden")]).generate(Hostile())
    finally:
        context.detach(token)
    assert not exporter.get_finished_spans()
    runtime[2].deactivate()
    provider = TracerProvider(sampler=ALWAYS_OFF)
    plugin = SmolagentsInstrumentor(tracer_provider=provider)
    plugin.activate()
    try:
        FixtureModel([message("hidden")]).generate(Hostile())
    finally:
        plugin.deactivate()
        provider.shutdown()


def test_shared_lifecycle_foreign_hook_and_late_subclass(runtime, monkeypatch):
    provider, exporter, plugin = runtime
    second = SmolagentsInstrumentor(tracer_provider=provider)
    second.activate()
    plugin.deactivate()

    class LateModel(Model):
        def generate(self, messages, **kwargs):
            return message("late")

    assert LateModel().generate([]).content == "late"
    assert len(kinds(exporter, "chat")) == 1
    ours = LateModel.generate

    @wraps(ours)
    def foreign(self, *args, **kwargs):
        return ours(self, *args, **kwargs)

    monkeypatch.setattr(LateModel, "generate", foreign)
    second.deactivate()
    assert LateModel.generate is foreign
    LateModel().generate([])
    assert len(kinds(exporter, "chat")) == 1
    plugin.activate()
    LateModel().generate([])
    assert len(kinds(exporter, "chat")) == 2
    plugin.deactivate()
    assert LateModel.generate is foreign


def test_partial_activation_rolls_back_and_conflicting_provider_is_rejected(
    monkeypatch,
):
    original = ToolCallingAgent._step_stream
    original_patch = _instrumentation._Runtime.patch

    def broken(self, cls, name, kind):
        original_patch(self, cls, name, kind)
        if cls is ToolCallingAgent:
            raise RuntimeError("partial")

    monkeypatch.setattr(_instrumentation._Runtime, "patch", broken)
    provider = TracerProvider()
    with pytest.raises(RuntimeError):
        SmolagentsInstrumentor(tracer_provider=provider).activate()
    assert _instrumentation._RUNTIME is None
    assert ToolCallingAgent._step_stream is original
    monkeypatch.setattr(_instrumentation._Runtime, "patch", original_patch)
    plugin = SmolagentsInstrumentor(tracer_provider=provider)
    plugin.activate()
    try:
        with pytest.raises(ValueError):
            SmolagentsInstrumentor(tracer_provider=TracerProvider()).activate()
    finally:
        plugin.deactivate()
        provider.shutdown()


def test_planning_parallel_tools_and_managed_agent_ids(runtime):
    _, exporter, _ = runtime
    child = ToolCallingAgent(
        tools=[],
        name="researcher",
        description="Controlled fact",
        verbosity_level=0,
        model=FixtureModel(
            [
                message(
                    calls=[call("final_answer", {"answer": "managed"}, "child-final")]
                )
            ]
        ),
    )
    parent = ToolCallingAgent(
        tools=[add],
        managed_agents=[child],
        planning_interval=10,
        verbosity_level=0,
        model=FixtureModel(
            [
                message("Plan: use parallel sums, then delegate."),
                message(
                    calls=[
                        call("add", {"left": 1, "right": 2}, "parallel-1"),
                        call("add", {"left": 3, "right": 4}, "parallel-2"),
                    ]
                ),
                message(calls=[call("researcher", {"task": "Get fact"}, "delegate-1")]),
                message(
                    calls=[call("final_answer", {"answer": "managed"}, "parent-final")]
                ),
            ]
        ),
    )
    assert parent.run("Plan and delegate") == "managed"
    tools = kinds(exporter, "tool")
    assert {
        span.attributes[gen_ai_attributes.GEN_AI_TOOL_CALL_ID] for span in tools
    } == {"parallel-1", "parallel-2", "delegate-1", "child-final", "parent-final"}
    assert len(kinds(exporter, "agent")) == 2
    assert any(
        span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "smolagents.plan"
        for span in kinds(exporter, "task")
    )
    child_span = next(
        span
        for span in kinds(exporter, "agent")
        if span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "researcher"
    )
    delegate_span = next(
        span
        for span in tools
        if span.attributes.get(gen_ai_attributes.GEN_AI_TOOL_CALL_ID) == "delegate-1"
    )
    assert child_span.parent.span_id == delegate_span.context.span_id


def test_request_schema_parameters_cache_reasoning_and_full_tool_lists(runtime):
    _, exporter, _ = runtime
    reply = message(
        calls=[
            call("add", {"left": index, "right": 1}, f"complete-{index}")
            for index in range(75)
        ]
    )
    reply.raw = {
        "usage": {
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2},
        }
    }
    schema = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object"}},
    }
    FixtureModel([reply]).generate(
        [{"role": "user", "content": "x" * 20000}],
        tools_to_call_from=[add] * 75,
        response_format=schema,
        temperature=0.3,
        reasoning_effort="low",
    )
    attrs = kinds(exporter, "chat")[0].attributes
    assert (
        len(json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])) == 75
    )
    assert (
        json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])[-1]["id"]
        == "complete-74"
    )
    assert len(json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])) == 75
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "x" * 20000
    assert (
        json.loads(attrs[SpanAttributes.LLM_REQUEST_STRUCTURED_OUTPUT_SCHEMA]) == schema
    )
    assert attrs[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.3
    assert attrs[SpanAttributes.GEN_AI_REQUEST_REASONING_EFFORT] == "low"
    assert attrs[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert attrs[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 2


def test_stream_exception_preserves_identity_status_and_context(runtime):
    provider, exporter, _ = runtime
    error = ValueError("native stream failure")

    class BrokenModel(Model):
        def generate_stream(self, messages, **kwargs):
            yield ChatMessageStreamDelta(content="prefix")
            raise error

    with provider.get_tracer("test").start_as_current_span("root") as root:
        stream = BrokenModel().generate_stream([])
        next(stream)
        with pytest.raises(ValueError) as raised:
            next(stream)
        assert raised.value is error
        assert trace.get_current_span() is root
    assert kinds(exporter, "chat")[0].status.status_code is trace.StatusCode.ERROR


def test_privacy_false_to_true_does_not_read_content_and_keeps_usage(
    runtime, monkeypatch
):
    _, exporter, _ = runtime

    class HostileReply:
        token_usage = TokenUsage(5, 2)
        raw = None

        @property
        def content(self):
            raise AssertionError("private output accessed")

    class FlipModel(Model):
        def generate(self, messages, **kwargs):
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
            return HostileReply()

    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    native = FlipModel().generate([{"role": "user", "content": "private"}])
    assert isinstance(native, HostileReply)
    attrs = kinds(exporter, "chat")[0].attributes
    assert attrs[gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS] == 5
    assert not any(
        key.startswith((SpanAttributes.LLM_PROMPTS, SpanAttributes.LLM_COMPLETIONS))
        for key in attrs
    )


def test_private_error_has_only_class_and_telemetry_export_failure_is_fail_open(
    runtime, monkeypatch
):
    provider, exporter, _ = runtime
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    error = ValueError("private payload")
    with pytest.raises(ValueError):
        FixtureModel([error]).generate([])
    span = kinds(exporter, "chat")[0]
    assert "private payload" not in str(span.events)
    assert span.events[0].attributes == {"exception.type": "ValueError"}
    monkeypatch.setattr(
        exporter,
        "export",
        lambda spans: (_ for _ in ()).throw(RuntimeError("export failed")),
    )
    result = message("native result")
    assert FixtureModel([result]).generate([]) is result
    assert provider


def test_stream_error_before_first_chunk_and_unadvanced_close_do_not_invent_output(
    runtime,
):
    _, exporter, _ = runtime
    error = ValueError("before first chunk")

    class Broken(Model):
        def generate_stream(self, messages, **kwargs):
            if False:
                yield
            raise error

    stream = Broken().generate_stream([])
    with pytest.raises(ValueError) as raised:
        next(stream)
    assert raised.value is error
    assert kinds(exporter, "chat")[0].status.status_code is trace.StatusCode.ERROR

    class Empty(Model):
        def generate_stream(self, messages, **kwargs):
            yield ChatMessageStreamDelta(content="")

    Empty().generate_stream([]).close()
    for span in kinds(exporter, "chat"):
        assert not any(
            key.startswith(SpanAttributes.LLM_COMPLETIONS) for key in span.attributes
        )
        assert SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in span.attributes
    actual = Empty().generate_stream([])
    next(actual)
    actual.close()
    assert (
        kinds(exporter, "chat")[-1].attributes[
            f"{SpanAttributes.LLM_COMPLETIONS}.0.content"
        ]
        == ""
    )


def test_tool_arguments_dict_and_json_string_current_and_history(runtime):
    _, exporter, _ = runtime
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "history",
                    "type": "function",
                    "function": {"name": "add", "arguments": {"left": 1, "right": 2}},
                }
            ],
        }
    ]
    FixtureModel(
        [message(calls=[call("add", '{"left":3,"right":4}', "current")])]
    ).generate(history)
    attrs = kinds(exporter, "chat")[0].attributes
    previous = json.loads(attrs[f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls"])
    current = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert isinstance(previous[0]["function"]["arguments"], str)
    assert json.loads(previous[0]["function"]["arguments"]) == {"left": 1, "right": 2}
    assert current[0]["function"]["arguments"] == '{"left":3,"right":4}'
    assert current[0]["id"] == "current" and len(current) == 1


def test_redaction_quoted_spaces_bearer_url_and_hostile_error(runtime):
    _, exporter, _ = runtime
    from respan_instrumentation_smolagents._serialization import (
        json_string,
        redact_text,
    )

    secret = 'Authorization: Bearer controlled-credential; "api_key": "controlled credential with spaces"; https://user:controlled-password@example.test/path'
    redacted = redact_text(secret)
    assert "controlled" not in redacted
    assert "controlled" not in json_string(
        {"api_key": "controlled credential", "content": secret}
    )

    class HostileError(ValueError):
        def __str__(self):
            raise AssertionError("do not call exception __str__")

    error = HostileError("native failure")
    with pytest.raises(HostileError) as raised:
        FixtureModel([error]).generate(
            [{"role": "user", "content": "retained actual input"}]
        )
    assert raised.value is error
    span = kinds(exporter, "chat")[0]
    assert span.status.status_code is trace.StatusCode.ERROR
    assert span.events[0].attributes["exception.message"] == "native failure"
    assert (
        span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "retained actual input"
    )


def test_prior_config_privacy_kwargs_and_code_executor_context(runtime):
    provider, exporter, plugin = runtime
    from types import SimpleNamespace

    plugin.deactivate()
    config = SimpleNamespace(hide_llm_tools=True, hide_llm_invocation_parameters=True)
    compatible = SmolagentsInstrumentor(
        tracer_provider=provider, config=config, skip_dep_check=True
    )
    compatible.activate()
    try:
        native = message("hidden")
        assert (
            FixtureModel([native]).generate(
                [{"role": "user", "content": "hidden"}],
                tools_to_call_from=[add],
                temperature=0.3,
            )
            is native
        )
        assert "hidden" not in str(kinds(exporter, "chat")[0].attributes)
        assert (
            SpanAttributes.LLM_REQUEST_TEMPERATURE
            not in kinds(exporter, "chat")[0].attributes
        )
    finally:
        compatible.deactivate()
    plugin.activate()
    exporter.clear()
    with provider.get_tracer("test").start_as_current_span("root"):
        CodeAgent(
            tools=[add],
            model=FixtureModel([message("<code>final_answer(add(2, 3))</code>")]),
            verbosity_level=0,
        ).run("Add")
    spans = exporter.get_finished_spans()
    identifiers = {span.context.span_id for span in spans}
    assert all(
        span.parent is None or span.parent.span_id in identifiers for span in spans
    )
    assert all(span.parent is not None for span in kinds(exporter, "tool"))


def test_invalid_or_hostile_computed_usage_total_does_not_hide_actual_output(runtime):
    _, exporter, _ = runtime

    class HostileTotal:
        input_tokens = 4
        output_tokens = 3

        @property
        def total_tokens(self):
            raise ValueError("unavailable total")

    for usage in (TokenUsage(-1, 3), HostileTotal()):
        reply = message("actual output")
        reply.token_usage = usage
        assert (
            FixtureModel([reply]).generate(
                [{"role": "user", "content": "actual input"}]
            )
            is reply
        )
        attrs = kinds(exporter, "chat")[-1].attributes
        assert attrs[gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 3
        assert SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in attrs
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "actual output"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "actual input"


def test_identical_parallel_tool_calls_keep_model_ids_without_guessing_execution_id(
    runtime,
):
    _, exporter, _ = runtime
    agent = ToolCallingAgent(
        tools=[add],
        verbosity_level=0,
        model=FixtureModel(
            [
                message(
                    calls=[
                        call("add", {"left": 1, "right": 2}, "identical-1"),
                        call("add", {"left": 1, "right": 2}, "identical-2"),
                    ]
                ),
                message(
                    calls=[
                        call(
                            "final_answer", {"answer": "three twice"}, "identical-final"
                        )
                    ]
                ),
            ]
        ),
    )
    assert agent.run("Add the same values twice") == "three twice"
    executions = [
        s
        for s in kinds(exporter, "tool")
        if s.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "add"
    ]
    assert len(executions) == 2
    assert all(
        gen_ai_attributes.GEN_AI_TOOL_CALL_ID not in s.attributes for s in executions
    )
    emitted = json.loads(
        kinds(exporter, "chat")[0].attributes[
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"
        ]
    )
    assert {c["id"] for c in emitted} == {"identical-1", "identical-2"}
