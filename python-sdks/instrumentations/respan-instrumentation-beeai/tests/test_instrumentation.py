"""Exercise real released BeeAI APIs with controlled model boundaries."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from beeai_framework.backend import UserMessage
from beeai_framework.emitter import Emitter, EmitterOptions
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import tool
from beeai_framework.workflows import Workflow
from opentelemetry import context, trace
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from pydantic import BaseModel
from respan_instrumentation_beeai import BeeAIInstrumentor
from respan_instrumentation_beeai._serialization import MAX_CHARS, data, json_value
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer

from ._models import FixtureChatModel, FixtureEmbeddingModel


@pytest.fixture
def recording(monkeypatch):
    RespanTracer.reset_instance()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    instrumentor = BeeAIInstrumentor()
    instrumentor.activate()
    yield exporter, provider, instrumentor
    instrumentor.deactivate()
    provider.shutdown()
    RespanTracer.reset_instance()


def test_chat_content_usage_and_parent_preserve_result(recording):
    exporter, provider, _ = recording
    model = FixtureChatModel()

    async def call():
        with provider.get_tracer("caller").start_as_current_span("workflow") as parent:
            output = await model.run([UserMessage("Check traces.")])
            assert output.get_text_content() == "Check parent links and provider usage."
            assert trace.get_current_span() is parent

    asyncio.run(call())
    spans = exporter.get_finished_spans()
    chat = next(s for s in spans if s.attributes.get(RESPAN_LOG_TYPE) == "chat")
    parent = next(s for s in spans if s.name == "workflow")
    assert chat.parent.span_id == parent.context.span_id
    assert chat.attributes[SpanAttributes.LLM_REQUEST_MODEL] == model.model_id
    assert chat.attributes[SpanAttributes.LLM_SYSTEM] == "openai"
    assert chat.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 11
    assert chat.attributes["gen_ai.usage.input_tokens"] == 11
    from beeai_framework.backend.types import ChatModelUsage

    if "cached_prompt_tokens" in ChatModelUsage.model_fields:
        assert chat.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    else:
        assert SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS not in chat.attributes
    assert (
        chat.attributes["gen_ai.completion.0.content"]
        == "Check parent links and provider usage."
    )
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in chat.attributes
    assert not {
        "model",
        "status_code",
        "error.message",
        "respan.span.tool_calls",
    }.intersection(chat.attributes)


def test_real_stream_events_and_final_completion(recording):
    exporter, _, _ = recording

    async def call():
        run = FixtureChatModel().run([UserMessage("Stream checks.")], stream=True)
        events = [meta.name async for _, meta in run]
        assert events.count("new_token") == 2

    asyncio.run(call())
    (span,) = exporter.get_finished_spans()
    assert span.attributes[SpanAttributes.LLM_IS_STREAMING] is True
    assert (
        span.attributes["gen_ai.completion.0.content"]
        == "Check parent links and provider usage."
    )
    assert span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 18


def test_actual_embeddings_preserved(recording):
    exporter, _, _ = recording
    output = asyncio.run(_embed())
    (span,) = exporter.get_finished_spans()
    assert output.embeddings == [[index / 128 for index in range(128)]] * 2
    assert (
        json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
        == output.embeddings
    )
    assert span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 5
    assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in span.attributes


async def _embed():
    return await FixtureEmbeddingModel().create(["first", "second"])


def test_real_requirement_agent_connected(recording):
    from beeai_framework.agents.requirement import RequirementAgent

    exporter, _, _ = recording

    async def call():
        response = await RequirementAgent(llm=FixtureChatModel(mode="agent")).run(
            "Explain tracing."
        )
        assert response.last_message.text == "Tracing connects model calls and tools."

    asyncio.run(call())
    spans = exporter.get_finished_spans()
    agent = next(s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "agent")
    children = [
        s for s in spans if s.parent and s.parent.span_id == agent.context.span_id
    ]
    assert any(s.attributes[RESPAN_LOG_TYPE] == "chat" for s in children)
    assert len({s.context.span_id for s in spans}) == len(spans)


def test_tool_actual_id_input_and_output(recording):
    exporter, _, _ = recording

    @tool(name="city_summary", description="Summarize the city.")
    def city_summary(city: str) -> str:
        return f"{city}: summary"

    async def call():
        response = await FixtureChatModel(mode="tool").run(
            [UserMessage("Paris")], tools=[city_summary]
        )
        msg = response.get_tool_calls()[0]
        output = await city_summary.run(json.loads(msg.args)).context(
            {"tool_call_msg": msg}
        )
        assert output.result == "Paris: summary"

    asyncio.run(call())
    chat, execution = exporter.get_finished_spans()
    assert (
        json.loads(chat.attributes["gen_ai.completion.0.tool_calls"])[0]["id"]
        == "call_beeai_city"
    )
    assert execution.attributes["gen_ai.tool.call.id"] == "call_beeai_city"
    assert json.loads(execution.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "name": "city_summary",
        "arguments": {"city": "Paris"},
    }
    assert (
        json.loads(execution.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
        == "Paris: summary"
    )
    assert not any("tool_calls" in k for k in execution.attributes)


def test_tool_without_source_id_does_not_invent_one(recording):
    exporter, _, _ = recording

    @tool(name="echo", description="Echo a value.")
    def echo(value: str) -> str:
        return value

    asyncio.run(echo.run({"value": "result"})._run_tasks())
    (span,) = exporter.get_finished_spans()
    assert "gen_ai.tool.call.id" not in span.attributes


def test_real_workflow_child_and_state(recording):
    exporter, _, _ = recording

    class State(BaseModel):
        answer: str = ""

    async def call():
        flow = Workflow(State, name="TraceWorkflow")

        async def step(state):
            output = await FixtureChatModel().run([UserMessage("Check traces")])
            state.answer = output.get_text_content()
            return Workflow.END

        flow.add_step("model", step)
        return await flow.run(State())

    output = asyncio.run(call())
    spans = exporter.get_finished_spans()
    flow = next(s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "workflow")
    chat = next(s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "chat")
    assert chat.parent.span_id == flow.context.span_id
    assert (
        json.loads(flow.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])["answer"]
        == output.state.answer
    )


def test_failure_status_no_synthetic_output_or_http_status(recording):
    exporter, _, _ = recording

    async def call():
        with pytest.raises(FrameworkError) as caught:
            await FixtureChatModel(error=True).run([UserMessage("Fail")])
        assert (
            caught.value.__cause__ is not None or "Chat" in type(caught.value).__name__
        )

    asyncio.run(call())
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    assert "status_code" not in span.attributes
    assert not any(k.startswith("gen_ai.completion") for k in span.attributes)
    assert span.events[0].name == "exception"


@pytest.mark.parametrize("switch", ["false", "0", "off", "no"])
def test_start_privacy_is_not_reenabled(recording, monkeypatch, switch):
    exporter, _, _ = recording
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", switch)

    async def call():
        async def enable(event, meta):
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")

        await FixtureChatModel().run([UserMessage("private input")]).on("start", enable)

    asyncio.run(call())
    (span,) = exporter.get_finished_spans()
    assert not any(
        k.startswith(
            (
                "gen_ai.prompt",
                "gen_ai.completion",
                "traceloop.entity.input",
                "traceloop.entity.output",
            )
        )
        for k in span.attributes
    )
    assert span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 18


def test_end_privacy_veto(recording, monkeypatch):
    exporter, _, _ = recording

    async def call():
        async def disable(event, meta):
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

        await (
            FixtureChatModel()
            .run([UserMessage("private input")])
            .on("success", disable)
        )

    asyncio.run(call())
    (span,) = exporter.get_finished_spans()
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert "gen_ai.completion.0.content" not in span.attributes


def test_private_error_diagnostic_contains_only_type(recording, monkeypatch):
    exporter, _, _ = recording
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    async def call():
        with pytest.raises(FrameworkError):
            await FixtureChatModel(error=True).run([UserMessage("private input")])

    asyncio.run(call())
    (span,) = exporter.get_finished_spans()
    assert set(span.events[0].attributes) == {"exception.type"}
    assert span.status.status_code == StatusCode.ERROR


def test_explicit_privacy_option(recording):
    exporter, _, instrumentor = recording
    instrumentor.deactivate()
    private = BeeAIInstrumentor(trace_content=False)
    private.activate()
    try:
        asyncio.run(_chat())
    finally:
        private.deactivate()
    (span,) = exporter.get_finished_spans()
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes


async def _chat():
    return await FixtureChatModel().run([UserMessage("checks")])


def test_suppression_tree_is_skipped(recording):
    exporter, _, _ = recording
    token = context.attach(context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
    try:
        asyncio.run(_chat())
    finally:
        context.detach(token)
    assert exporter.get_finished_spans() == ()
    assert BeeAIInstrumentor._listener.runs == {}


def test_nonrecording_sampler_does_not_serialize(recording, monkeypatch):
    exporter, provider, _ = recording
    provider.sampler = ALWAYS_OFF
    BeeAIInstrumentor._listener.tracer = provider.get_tracer("nonrecording")
    monkeypatch.setattr(
        "respan_instrumentation_beeai._instrumentation.json_value",
        lambda value: pytest.fail("nonrecording input serialized"),
    )
    asyncio.run(_chat())
    assert exporter.get_finished_spans() == ()


def test_shared_owners_and_foreign_listener_survive(recording):
    exporter, _, owner = recording
    foreign_events = []

    async def foreign(event, meta):
        foreign_events.append(meta.name)

    cleanup = Emitter.root().on(
        "*.*", foreign, EmitterOptions(match_nested=True, is_blocking=True)
    )
    second = BeeAIInstrumentor()
    second.activate()
    owner.deactivate()
    try:
        asyncio.run(_chat())
        second.deactivate()
        asyncio.run(_chat())
    finally:
        cleanup()
        second.deactivate()
    assert len(exporter.get_finished_spans()) == 1
    assert foreign_events.count("success") >= 4
    assert BeeAIInstrumentor._owners == 0


def test_activation_rollback_and_retry(recording, monkeypatch):
    _, _, owner = recording
    owner.deactivate()
    original = Emitter.on
    monkeypatch.setattr(
        Emitter,
        "on",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("registration failure")),
    )
    with pytest.raises(RuntimeError):
        owner.activate()
    assert BeeAIInstrumentor._listener is None
    assert BeeAIInstrumentor._owners == 0
    monkeypatch.setattr(Emitter, "on", original)
    owner.activate()
    assert owner._is_instrumented


def test_disabled_respan_skips_listener(recording):
    _, _, owner = recording
    owner.deactivate()
    RespanTracer._instance = SimpleNamespace(is_enabled=False)
    owner.activate()
    assert not owner._is_instrumented
    RespanTracer._instance = None


def test_serializer_bounds_redacts_and_skips_hooks():
    class Hostile:
        def __repr__(self):
            raise AssertionError("repr called")

        def __str__(self):
            raise AssertionError("str called")

    value = {
        "api_key": "private",
        "nested": {"authorization": "Bearer testsecretvalue"},
        "unknown": Hostile(),
        "long": "x" * 100_000,
    }
    encoded = json_value(value)
    assert len(encoded) <= MAX_CHARS
    assert "private" not in encoded
    json.loads(encoded)
    assert data(Hostile()) == "[UNSUPPORTED:Hostile]"


def test_default_usage_is_not_invented(recording):
    from beeai_framework.backend import AssistantMessage
    from beeai_framework.backend.types import ChatModelOutput

    exporter, _, _ = recording
    model = FixtureChatModel()

    async def create(input, run):
        return ChatModelOutput(output=[AssistantMessage("No provider usage.")])

    model._create = create
    asyncio.run(model.run([UserMessage("checks")])._run_tasks())
    (span,) = exporter.get_finished_spans()
    assert not any("usage" in key for key in span.attributes)


def test_parallel_calls_have_separate_spans(recording):
    exporter, _, _ = recording

    async def call():
        results = await asyncio.gather(*(_chat() for _ in range(3)))
        assert len(results) == 3

    asyncio.run(call())
    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    assert len({s.context.trace_id for s in spans}) == 3
    assert BeeAIInstrumentor._listener.runs == {}


def test_run_iteration_close_keeps_native_protocol_and_cleans_spans(recording):
    exporter, _, _ = recording

    async def call():
        iterator = (
            FixtureChatModel().run([UserMessage("checks")], stream=True).__aiter__()
        )
        await anext(iterator)
        await iterator.aclose()
        await asyncio.sleep(0)

    asyncio.run(call())
    assert BeeAIInstrumentor._listener.runs == {}
    assert len(exporter.get_finished_spans()) == 1


def test_partial_registration_rollback_keeps_foreign_hook(recording, monkeypatch):
    _, _, owner = recording
    owner.deactivate()
    original = Emitter.on
    foreign_events = []

    async def foreign(event, meta):
        foreign_events.append(meta.name)

    cleanup = Emitter.root().on(
        "*.*", foreign, EmitterOptions(match_nested=True, is_blocking=True)
    )
    before = len(Emitter.root()._listeners)

    def partial(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("failure after listener addition")

    monkeypatch.setattr(Emitter, "on", partial)
    with pytest.raises(RuntimeError):
        owner.activate()
    assert len(Emitter.root()._listeners) == before
    monkeypatch.setattr(Emitter, "on", original)
    try:
        asyncio.run(_chat())
        assert foreign_events
    finally:
        cleanup()


def test_stream_without_source_usage_omits_defaults(recording):
    from beeai_framework.backend import AssistantMessage
    from beeai_framework.backend.types import ChatModelOutput

    exporter, _, _ = recording
    model = FixtureChatModel()

    async def stream(input, run):
        yield ChatModelOutput(output=[AssistantMessage("No usage in chunks.")])

    model._create_stream = stream

    async def call():
        return await model.run([UserMessage("checks")], stream=True)

    asyncio.run(call())
    (span,) = exporter.get_finished_spans()
    assert not any("usage" in key for key in span.attributes)


def test_source_tool_result_id_is_kept_only_in_prompt(recording):
    from beeai_framework.backend import (
        AssistantMessage,
        MessageToolCallContent,
        MessageToolResultContent,
        ToolMessage,
    )

    exporter, _, _ = recording

    async def call():
        return await FixtureChatModel().run(
            [
                UserMessage("Use the result."),
                AssistantMessage(
                    MessageToolCallContent(
                        id="historic_call",
                        tool_name="city_summary",
                        args='{"city":"Paris"}',
                    )
                ),
                ToolMessage(
                    MessageToolResultContent(
                        tool_call_id="historic_call",
                        tool_name="city_summary",
                        result="Paris summary",
                    )
                ),
            ]
        )

    asyncio.run(call())
    (span,) = exporter.get_finished_spans()
    assert (
        json.loads(span.attributes["gen_ai.prompt.1.tool_calls"])[0]["id"]
        == "historic_call"
    )
    assert span.attributes["gen_ai.prompt.2.tool_call_id"] == "historic_call"
    assert "gen_ai.completion.0.tool_calls" not in span.attributes


def test_multiple_current_tool_calls_single_encoded_and_redacted(recording):
    from beeai_framework.backend import AssistantMessage, MessageToolCallContent
    from beeai_framework.backend.types import ChatModelOutput

    exporter, _, _ = recording
    model = FixtureChatModel()

    async def create(input, run):
        return ChatModelOutput(
            output=[
                AssistantMessage(
                    [
                        MessageToolCallContent(
                            id="first",
                            tool_name="one",
                            args='{"value":1,"api_key":"private"}',
                        ),
                        MessageToolCallContent(
                            id="second", tool_name="two", args='{"value":2}'
                        ),
                    ]
                )
            ]
        )

    model._create = create

    async def call():
        return await model.run([UserMessage("Call tools")])

    output = asyncio.run(call())
    assert output.get_tool_calls()[0].args == '{"value":1,"api_key":"private"}'
    (span,) = exporter.get_finished_spans()
    calls = json.loads(span.attributes["gen_ai.completion.0.tool_calls"])
    assert [call["id"] for call in calls] == ["first", "second"]
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "value": 1,
        "api_key": "[REDACTED]",
    }
    assert json.loads(calls[1]["function"]["arguments"]) == {"value": 2}
    assert "private" not in json.dumps(dict(span.attributes))


def test_context_privacy_skips_serialization_at_start(recording, monkeypatch):
    from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

    exporter, _, _ = recording
    monkeypatch.setattr(
        "respan_instrumentation_beeai._instrumentation.json_value",
        lambda *a, **k: pytest.fail("private input serialized"),
    )
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        output = asyncio.run(_chat())
        assert output.get_text_content()
    finally:
        context.detach(token)
    (span,) = exporter.get_finished_spans()
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert "gen_ai.completion.0.content" not in span.attributes


def test_context_late_privacy_veto_is_irreversible(recording):
    from beeai_framework.backend import AssistantMessage
    from beeai_framework.backend.types import ChatModelOutput
    from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

    exporter, _, _ = recording
    model = FixtureChatModel()

    async def create(input, run):
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        return ChatModelOutput(output=[AssistantMessage("Private result")])

    model._create = create

    async def call():
        return await model.run([UserMessage("private input")])

    asyncio.run(call())
    (span,) = exporter.get_finished_spans()
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert "gen_ai.completion.0.content" not in span.attributes


def test_language_model_suppression(recording):
    from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY

    exporter, _, _ = recording
    token = context.attach(
        context.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
    )
    try:
        asyncio.run(_chat())
    finally:
        context.detach(token)
    assert exporter.get_finished_spans() == ()


def test_complete_known_payloads_preserve_counts_lengths_and_vectors():
    from beeai_framework.backend import AssistantMessage, MessageToolCallContent
    from respan_instrumentation_beeai._instrumentation import _messages

    call_id = "call_" + "a" * 600
    calls = [
        MessageToolCallContent(
            id=call_id + str(i),
            tool_name="many_calls",
            args=json.dumps({"values": list(range(200)), "text": "x" * 20000}),
        )
        for i in range(120)
    ]
    mapped = _messages([AssistantMessage(calls)])
    values = mapped[0]["tool_calls"]
    assert len(values) == 120
    assert values[0]["id"] == call_id + "0"
    arguments = json.loads(values[0]["function"]["arguments"])
    assert len(arguments["values"]) == 200
    assert len(arguments["text"]) == 20000
    vector = list(range(5000))
    assert json.loads(json_value([vector], complete=True)) == [vector]


def test_complete_tool_result_in_prompt_keeps_vector_tail(recording):
    from beeai_framework.backend import MessageToolResultContent, ToolMessage

    exporter, _, _ = recording
    vector = list(range(5000))

    async def call():
        return await FixtureChatModel().run(
            [
                UserMessage("Use result"),
                ToolMessage(
                    MessageToolResultContent(
                        tool_name="vector_tool",
                        tool_call_id="actual_result_id",
                        result=vector,
                    )
                ),
            ]
        )

    asyncio.run(call())
    (span,) = exporter.get_finished_spans()
    assert json.loads(span.attributes["gen_ai.prompt.1.content"])[0] == vector
    assert span.attributes["gen_ai.prompt.1.tool_call_id"] == "actual_result_id"


def test_error_str_hook_is_never_called(recording):
    from respan_instrumentation_beeai._instrumentation import _Run

    exporter, provider, _ = recording

    class HostileError(RuntimeError):
        def __str__(self):
            raise AssertionError("Application str hook called")

    span = provider.get_tracer("error-test").start_span("error")
    run = _Run(span, "chat", True, "error")
    BeeAIInstrumentor._listener._error(
        run,
        HostileError(
            "https://user:password@example.test/path", 'api_key="private value"'
        ),
    )
    span.end()
    (ended,) = exporter.get_finished_spans()
    assert ended.status.status_code == StatusCode.ERROR
    serialized = json.dumps(dict(ended.events[0].attributes))
    assert "private value" not in serialized
    assert "user:password" not in serialized


def test_redaction_handles_url_credentials_and_quoted_assignments():
    from respan_instrumentation_beeai._serialization import text

    assert "user:password" not in text("https://user:password@example.test/path")
    assert "private value" not in text('Debug "api_key": "private value"')
    assert (
        json.loads(json_value('{"api_key":"private value"}'))
        == '{"api_key":"[REDACTED]"}'
    )


def test_conflicting_shared_owner_cannot_override_privacy(recording):
    _, _, first = recording
    second = BeeAIInstrumentor(trace_content=False)
    with pytest.raises(ValueError, match="different settings"):
        second.activate()
    assert not second._is_instrumented
    assert first._is_instrumented
    assert BeeAIInstrumentor._owners == 1


@pytest.mark.parametrize(
    "flag",
    [
        "hide_input_text",
        "hide_output_text",
        "hide_embedding_vectors",
        "hide_llm_tools",
        "hide_llm_invocation_parameters",
    ],
)
def test_config_privacy_flags_suppress_payloads(recording, flag):
    exporter, _, owner = recording
    owner.deactivate()
    private = BeeAIInstrumentor(config=SimpleNamespace(**{flag: True}))
    private.activate()
    try:
        asyncio.run(_chat())
    finally:
        private.deactivate()
    (span,) = exporter.get_finished_spans()
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert "gen_ai.completion.0.content" not in span.attributes
    assert SpanAttributes.LLM_REQUEST_FUNCTIONS not in span.attributes


def test_complete_string_tool_result_in_prompt(recording):
    from beeai_framework.backend import MessageToolResultContent, ToolMessage

    exporter, _, _ = recording
    result = "result-" * 5000

    async def call():
        return await FixtureChatModel().run(
            [
                UserMessage("Use result"),
                ToolMessage(
                    MessageToolResultContent(
                        tool_name="text_tool",
                        tool_call_id="actual_string_result_id",
                        result=result,
                    )
                ),
            ]
        )

    asyncio.run(call())
    (span,) = exporter.get_finished_spans()
    assert span.attributes["gen_ai.prompt.1.content"] == result
