import asyncio
import json
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
from opentelemetry.semconv_ai import SpanAttributes

from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_instrumentation_agentscope import AgentScopeInstrumentor
from respan_instrumentation_agentscope import _instrumentation
from respan_instrumentation_agentscope._instrumentation import (
    AGENTSCOPE_AGENT_MODULE,
    AGENTSCOPE_MODEL_MODULE,
    AGENTSCOPE_TOOL_MODULE,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer

PROMPT_PREFIX = f"{SpanAttributes.LLM_PROMPTS}."
COMPLETION_PREFIX = f"{SpanAttributes.LLM_COMPLETIONS}."
BANNED_ALIASES = {
    "respan.span.tools",
    "respan.span.tool_calls",
    "respan.span.handoffs",
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "span_tools",
    "has_tool_calls",
    "parallel_tool_calls",
    SpanAttributes.TRACELOOP_SPAN_KIND,
}


class FakeState:
    session_id = "session_123"
    reply_id = "reply_123"


class FakeAgent:
    name = "Planner"
    state = FakeState()

    async def reply(self, inputs=None):
        return FakeMsg(role="assistant", name="Planner", content="Plan accepted.")

    def reply_stream(self, inputs=None):
        async def stream():
            yield SimpleNamespace(type="reply_start")
            yield FakeMsg(role="assistant", name="Planner", content="Streamed plan.")

        return stream()


class FakeFailingAgent(FakeAgent):
    async def reply(self, inputs=None):
        raise RuntimeError("agent failed")


class FakeNestedAgent(FakeAgent):
    def __init__(self, model, toolkit):
        self.model = model
        self.toolkit = toolkit

    async def reply(self, inputs=None):
        await self.model([FakeMsg(content="Nested model.")])
        async for _ in self.toolkit.call_tool(FakeToolCall(), object()):
            pass
        return FakeMsg(role="assistant", name="Planner", content="Nested done.")


class FakeUsage:
    input_tokens = 7
    output_tokens = 5
    cache_input_tokens = 2


class FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeToolCallBlock:
    type = "tool_call"

    def __init__(self):
        self.id = "call_1"
        self.name = "lookup_weather"
        self.input = '{"city":"Tokyo"}'
        self.state = "pending"


class FakeMsg:
    def __init__(self, role="user", name="user", content="hello"):
        self.role = role
        self.name = name
        self.content = content

    def get_content_blocks(self):
        if isinstance(self.content, list):
            return self.content
        return [FakeTextBlock(self.content)]

    def get_text_content(self):
        if isinstance(self.content, str):
            return self.content
        return ""


class FakeChatResponse:
    is_last = True
    usage = FakeUsage()

    def __init__(self, content=None):
        self.content = content if content is not None else [FakeTextBlock("Sunny.")]

    def get_content_blocks(self):
        return list(self.content)


class FakeKeyErrorProxy:
    __slots__ = ("_values",)

    def __init__(self, **values):
        object.__setattr__(self, "_values", values)

    def __getattr__(self, key):
        values = object.__getattribute__(self, "_values")
        if key in values:
            return values[key]
        raise KeyError(key)


class FakeChatModelBase:
    model = "gpt-4o-mini"
    provider = "OpenAI"

    async def __call__(self, messages, tools=None, tool_choice=None, **kwargs):
        return FakeChatResponse()


class FakeStreamingChatModel(FakeChatModelBase):
    async def __call__(self, messages, tools=None, tool_choice=None, **kwargs):
        async def stream():
            yield FakeChatResponse(content=[FakeTextBlock("Sun")])
            yield FakeChatResponse(content=[FakeTextBlock("Sunny.")])

        return stream()


class FakeToolCallModel(FakeChatModelBase):
    async def __call__(self, messages, tools=None, tool_choice=None, **kwargs):
        return FakeChatResponse(content=[FakeToolCallBlock()])


class FakeKeyErrorModel(FakeChatModelBase):
    async def __call__(self, messages, tools=None, tool_choice=None, **kwargs):
        return FakeKeyErrorProxy(
            content=[FakeKeyErrorProxy(type="text", text="Proxy response.")],
            usage=FakeKeyErrorProxy(input_tokens=3, output_tokens=4),
        )


class FakeToolCall:
    id = "call_1"
    name = "lookup_weather"
    input = '{"city":"Tokyo"}'


class FakeToolChunk:
    def __init__(self, content, state="running"):
        self.content = [FakeTextBlock(content)]
        self.state = state


class FakeToolResponse(FakeToolChunk):
    pass


class FakeToolkit:
    async def call_tool(self, tool_call, state):
        yield FakeToolChunk("weather:")
        yield FakeToolResponse("sunny", state="success")


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


@pytest.fixture
def captured_spans(monkeypatch):
    spans = []

    def fake_build_readable_span(**kwargs):
        span = SimpleNamespace(**kwargs)
        spans.append(span)
        return span

    monkeypatch.setattr(_instrumentation, "build_readable_span", fake_build_readable_span)
    monkeypatch.setattr(_instrumentation, "inject_span", lambda span: True)
    return spans


def _install_fake_agentscope_modules(monkeypatch):
    agentscope_module = ModuleType("agentscope")
    agent_module = ModuleType(AGENTSCOPE_AGENT_MODULE)
    model_module = ModuleType(AGENTSCOPE_MODEL_MODULE)
    tool_module = ModuleType(AGENTSCOPE_TOOL_MODULE)

    agent_module.Agent = FakeAgent
    model_module.ChatModelBase = FakeChatModelBase
    tool_module.Toolkit = FakeToolkit

    agentscope_module.agent = agent_module
    agentscope_module.model = model_module
    agentscope_module.tool = tool_module

    monkeypatch.setitem(sys.modules, "agentscope", agentscope_module)
    monkeypatch.setitem(sys.modules, AGENTSCOPE_AGENT_MODULE, agent_module)
    monkeypatch.setitem(sys.modules, AGENTSCOPE_MODEL_MODULE, model_module)
    monkeypatch.setitem(sys.modules, AGENTSCOPE_TOOL_MODULE, tool_module)
    return SimpleNamespace(
        agent_class=FakeAgent,
        model_class=FakeChatModelBase,
        toolkit_class=FakeToolkit,
    )


def _assert_no_banned_aliases(attrs):
    assert BANNED_ALIASES.isdisjoint(attrs.keys())


def test_activate_patches_and_deactivates_agentscope_classes(monkeypatch):
    fake_modules = _install_fake_agentscope_modules(monkeypatch)
    original_agent_reply = fake_modules.agent_class.reply
    original_model_call = fake_modules.model_class.__call__
    original_tool_call = fake_modules.toolkit_class.call_tool

    instrumentor = AgentScopeInstrumentor()
    instrumentor.activate()

    assert fake_modules.agent_class.reply is not original_agent_reply
    assert fake_modules.model_class.__call__ is not original_model_call
    assert fake_modules.toolkit_class.call_tool is not original_tool_call
    assert instrumentor._is_instrumented is True

    instrumentor.deactivate()

    assert fake_modules.agent_class.reply is original_agent_reply
    assert fake_modules.model_class.__call__ is original_model_call
    assert fake_modules.toolkit_class.call_tool is original_tool_call
    assert instrumentor._is_instrumented is False


def test_activate_specific_instances_does_not_patch_classes(monkeypatch):
    fake_modules = _install_fake_agentscope_modules(monkeypatch)
    original_agent_reply = fake_modules.agent_class.reply
    agent = FakeAgent()

    instrumentor = AgentScopeInstrumentor(agent=agent, instrument_models=False, instrument_tools=False)
    instrumentor.activate()

    assert agent.reply is not original_agent_reply
    assert fake_modules.agent_class.reply is original_agent_reply

    instrumentor.deactivate()

    assert fake_modules.agent_class.reply is original_agent_reply


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake_modules = _install_fake_agentscope_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = AgentScopeInstrumentor()
    with caplog.at_level("INFO"):
        instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert fake_modules.agent_class.reply is FakeAgent.reply
    assert "AgentScope instrumentation skipped" in caplog.text


def test_agent_reply_emits_agent_span(captured_spans):
    agent = FakeAgent()
    instrumentor = AgentScopeInstrumentor(agent=agent, instrument_models=False, instrument_tools=False)
    instrumentor.activate()

    result = asyncio.run(agent.reply(FakeMsg(content="Draft a plan.")))

    assert result.content == "Plan accepted."
    assert len(captured_spans) == 1
    attrs = captured_spans[0].attributes
    assert captured_spans[0].name == "agentscope.agent"
    assert attrs[RESPAN_LOG_TYPE] == "agent"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "Planner"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] == "Draft a plan."
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == "Plan accepted."
    assert attrs["agentscope.session.id"] == "session_123"
    _assert_no_banned_aliases(attrs)


def test_agent_reply_stream_emits_after_consumption(captured_spans):
    agent = FakeAgent()
    instrumentor = AgentScopeInstrumentor(agent=agent, instrument_models=False, instrument_tools=False)
    instrumentor.activate()

    async def collect():
        return [item async for item in agent.reply_stream(FakeMsg(content="Stream."))]

    result = asyncio.run(collect())

    assert len(result) == 2
    assert captured_spans[0].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == "Streamed plan."


def test_agent_failure_emits_error_span(captured_spans):
    agent = FakeFailingAgent()
    instrumentor = AgentScopeInstrumentor(agent=agent, instrument_models=False, instrument_tools=False)
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="agent failed"):
        asyncio.run(agent.reply(FakeMsg(content="Fail.")))

    assert len(captured_spans) == 1
    assert captured_spans[0].status_code == 500
    assert captured_spans[0].error_message == "agent failed"
    assert captured_spans[0].attributes["status_code"] == 500
    assert captured_spans[0].attributes[ERROR_MESSAGE_ATTR] == "agent failed"
    _assert_no_banned_aliases(captured_spans[0].attributes)


def test_model_call_emits_canonical_chat_span(captured_spans):
    model = FakeToolCallModel()
    instrumentor = AgentScopeInstrumentor(model=model, agent=None, toolkit=None)
    instrumentor.activate()

    result = asyncio.run(
        model(
            [FakeMsg(content="Use a tool.")],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "lookup_weather", "parameters": {}},
                }
            ],
        )
    )

    assert result.content[0].name == "lookup_weather"
    attrs = captured_spans[0].attributes
    assert captured_spans[0].name == "agentscope.model_call"
    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert attrs[SpanAttributes.LLM_SYSTEM] == "openai"
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 7
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 5
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 12
    assert attrs[f"{PROMPT_PREFIX}0.role"] == "user"
    assert attrs[f"{PROMPT_PREFIX}0.content"] == "Use a tool."
    assert attrs[f"{COMPLETION_PREFIX}0.role"] == "assistant"
    assert json.loads(attrs[f"{COMPLETION_PREFIX}0.tool_calls"])[0]["function"]["name"] == "lookup_weather"
    assert SpanAttributes.LLM_REQUEST_FUNCTIONS in attrs
    _assert_no_banned_aliases(attrs)


def test_model_stream_emits_after_consumption(captured_spans):
    model = FakeStreamingChatModel()
    instrumentor = AgentScopeInstrumentor(model=model, instrument_tools=False)
    instrumentor.activate()

    async def collect():
        stream = await model([FakeMsg(content="Stream.")])
        return [chunk async for chunk in stream]

    result = asyncio.run(collect())

    assert len(result) == 2
    assert captured_spans[0].attributes[f"{COMPLETION_PREFIX}0.content"] == "Sunny."


def test_model_call_handles_keyerror_getattr_sdk_objects(captured_spans):
    model = FakeKeyErrorModel()
    instrumentor = AgentScopeInstrumentor(model=model, instrument_tools=False)
    instrumentor.activate()

    result = asyncio.run(
        model([FakeKeyErrorProxy(role="user", content="Proxy prompt.")])
    )

    assert result.content[0].text == "Proxy response."
    attrs = captured_spans[0].attributes
    assert attrs[f"{PROMPT_PREFIX}0.content"] == "Proxy prompt."
    assert attrs[f"{COMPLETION_PREFIX}0.content"] == "Proxy response."
    assert attrs["gen_ai.usage.input_tokens"] == 3
    assert attrs["gen_ai.usage.output_tokens"] == 4

    opaque = FakeKeyErrorProxy()
    assert _instrumentation._object_to_dict(opaque) == {"value": opaque}


def test_toolkit_call_tool_emits_tool_span(captured_spans):
    toolkit = FakeToolkit()
    tool_call = FakeToolCall()
    instrumentor = AgentScopeInstrumentor(toolkit=toolkit, instrument_models=False)
    instrumentor.activate()

    async def collect():
        return [item async for item in toolkit.call_tool(tool_call, state=object())]

    result = asyncio.run(collect())

    assert len(result) == 2
    attrs = captured_spans[0].attributes
    assert captured_spans[0].name == "agentscope.tool"
    assert attrs[RESPAN_LOG_TYPE] == "tool"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "lookup_weather"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] == (
        '{"name":"lookup_weather","arguments":"{\\"city\\":\\"Tokyo\\"}"}'
    )
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == "sunny"
    assert attrs["agentscope.tool.call.id"] == "call_1"
    _assert_no_banned_aliases(attrs)


def test_model_and_tool_spans_parent_to_active_agent_context(captured_spans):
    model = FakeChatModelBase()
    toolkit = FakeToolkit()
    agent = FakeNestedAgent(model=model, toolkit=toolkit)
    instrumentor = AgentScopeInstrumentor(
        agent=agent,
        model=model,
        toolkit=toolkit,
    )
    instrumentor.activate()

    asyncio.run(agent.reply(FakeMsg(content="Start.")))

    agent_span = next(span for span in captured_spans if span.name == "agentscope.agent")
    model_span = next(span for span in captured_spans if span.name == "agentscope.model_call")
    tool_span = next(span for span in captured_spans if span.name == "agentscope.tool")
    assert model_span.trace_id == agent_span.trace_id
    assert tool_span.trace_id == agent_span.trace_id
    assert model_span.parent_id == agent_span.span_id
    assert tool_span.parent_id == agent_span.span_id
