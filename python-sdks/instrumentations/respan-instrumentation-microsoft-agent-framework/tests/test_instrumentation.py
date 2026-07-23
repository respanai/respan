import json
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_microsoft_agent_framework import (
    MicrosoftAgentFrameworkInstrumentor,
)
from respan_instrumentation_microsoft_agent_framework import _instrumentation
from respan_instrumentation_microsoft_agent_framework._constants import (
    ATTR_GEN_AI_INPUT_MESSAGES,
    ATTR_GEN_AI_OUTPUT_MESSAGES,
    ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
    ATTR_GEN_AI_TOOL_DEFINITIONS,
    TOP_LEVEL_ALIAS_ATTRS,
)
from respan_instrumentation_microsoft_agent_framework._processor import (
    AgentFrameworkSpanProcessor,
)
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_tracing.core.tracer import RespanTracer


class FakeActiveProcessor:
    def __init__(self, processors=()):
        self._span_processors = processors


class FakeTracerProvider:
    def __init__(self):
        self.export_processor = object()
        self._active_span_processor = FakeActiveProcessor((self.export_processor,))
        self.added_processors = []

    def add_span_processor(self, processor):
        self.added_processors.append(processor)


class FakeSpan:
    def __init__(self, attrs, name="chat gpt-4.1-nano", scope="agent_framework"):
        self.name = name
        self.attributes = dict(attrs)
        self._attributes = dict(attrs)
        self.instrumentation_scope = SimpleNamespace(name=scope)


@pytest.fixture(autouse=True)
def reset_respan_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


@pytest.fixture
def fake_tracer_provider(monkeypatch):
    provider = FakeTracerProvider()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: provider,
    )
    return provider


def _install_fake_agent_framework(monkeypatch):
    calls = []
    agent_framework_module = ModuleType("agent_framework")
    observability_module = ModuleType("agent_framework.observability")
    observability_module.OBSERVABILITY_SETTINGS = SimpleNamespace(
        is_user_disabled=False,
        enable_sensitive_data=False,
    )

    def enable_instrumentation(**kwargs):
        calls.append(kwargs)
        observability_module.OBSERVABILITY_SETTINGS.enable_sensitive_data = kwargs.get(
            "enable_sensitive_data",
            False,
        )

    observability_module.enable_instrumentation = enable_instrumentation
    agent_framework_module.observability = observability_module
    monkeypatch.setitem(sys.modules, "agent_framework", agent_framework_module)
    monkeypatch.setitem(
        sys.modules,
        "agent_framework.observability",
        observability_module,
    )
    return calls, observability_module


def _assert_no_off_contract_aliases(attrs):
    banned = TOP_LEVEL_ALIAS_ATTRS | {
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
        RESPAN_SPAN_HANDOFFS,
    }
    for key in banned:
        assert key not in attrs


def test_activate_registers_processor_and_enables_framework_observability(
    monkeypatch,
    fake_tracer_provider,
):
    calls, observability_module = _install_fake_agent_framework(monkeypatch)

    instrumentor = MicrosoftAgentFrameworkInstrumentor(capture_content=True)
    instrumentor.activate()

    processors = fake_tracer_provider._active_span_processor._span_processors
    assert isinstance(processors[0], AgentFrameworkSpanProcessor)
    assert processors[1] is fake_tracer_provider.export_processor
    assert calls == [{"enable_sensitive_data": True}]
    assert observability_module.OBSERVABILITY_SETTINGS.enable_sensitive_data is True
    assert instrumentor._is_instrumented is True

    instrumentor.deactivate()

    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider.export_processor,
    )
    assert instrumentor._is_instrumented is False


def test_activate_is_idempotent(monkeypatch, fake_tracer_provider):
    _install_fake_agent_framework(monkeypatch)

    instrumentor = MicrosoftAgentFrameworkInstrumentor()
    instrumentor.activate()
    instrumentor.activate()

    processors = fake_tracer_provider._active_span_processor._span_processors
    assert sum(isinstance(item, AgentFrameworkSpanProcessor) for item in processors) == 1


def test_activate_skips_when_dependency_missing(
    monkeypatch,
    caplog,
    fake_tracer_provider,
):
    def import_module_raises(module_name):
        if module_name == "agent_framework.observability":
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(_instrumentation.importlib, "import_module", import_module_raises)

    instrumentor = MicrosoftAgentFrameworkInstrumentor()
    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "missing dependency" in caplog.text
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider.export_processor,
    )


def test_activate_skips_when_respan_tracing_disabled(
    monkeypatch,
    caplog,
    fake_tracer_provider,
):
    _install_fake_agent_framework(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = MicrosoftAgentFrameworkInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert "Respan tracing is disabled" in caplog.text
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider.export_processor,
    )


def test_processor_maps_chat_span_and_removes_aliases():
    span = FakeSpan(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "gpt-4.1-nano",
            "gen_ai.usage.input_tokens": 12,
            "gen_ai.usage.output_tokens": 7,
            ATTR_GEN_AI_SYSTEM_INSTRUCTIONS: json.dumps(
                [{"type": "text", "content": "Use concise answers."}]
            ),
            ATTR_GEN_AI_INPUT_MESSAGES: json.dumps(
                [{"role": "user", "content": "Use the weather tool for Seattle."}]
            ),
            ATTR_GEN_AI_OUTPUT_MESSAGES: json.dumps(
                [
                    {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "tool_call",
                                "id": "call_1",
                                "name": "lookup_weather",
                                "arguments": {"city": "Seattle"},
                            }
                        ],
                    }
                ]
            ),
            ATTR_GEN_AI_TOOL_DEFINITIONS: json.dumps(
                [{"name": "lookup_weather", "description": "Return weather."}]
            ),
            "model": "bad-alias",
            "tool_calls": "bad-alias",
            RESPAN_SPAN_TOOLS: "bad-alias",
            SpanAttributes.TRACELOOP_SPAN_KIND: "llm",
        }
    )

    AgentFrameworkSpanProcessor().on_end(span)
    attrs = span._attributes

    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4.1-nano"
    assert attrs[SpanAttributes.LLM_SYSTEM] == "openai"
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 12
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 7
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 19
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Use concise answers."
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "user"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == (
        "Use the weather tool for Seattle."
    )
    tool_calls = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert tool_calls == [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city": "Seattle"}',
            },
            "id": "call_1",
        }
    ]
    functions = json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])
    assert functions[0]["function"]["name"] == "lookup_weather"

    for raw_key in (
        ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
        ATTR_GEN_AI_INPUT_MESSAGES,
        ATTR_GEN_AI_OUTPUT_MESSAGES,
        ATTR_GEN_AI_TOOL_DEFINITIONS,
    ):
        assert raw_key not in attrs
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs
    _assert_no_off_contract_aliases(attrs)


def test_processor_maps_tool_span_and_strips_gen_ai_tool_attrs():
    span = FakeSpan(
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "lookup_weather",
            "gen_ai.tool.call.id": "call_1",
            "gen_ai.tool.call.arguments": json.dumps({"city": "Seattle"}),
            "gen_ai.tool.call.result": "Sunny and 72F.",
            "tool_calls": "bad-alias",
        },
        name="execute_tool lookup_weather",
    )

    AgentFrameworkSpanProcessor().on_end(span)
    attrs = span._attributes

    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "lookup_weather"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == "lookup_weather"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "name": "lookup_weather",
        "arguments": {"city": "Seattle"},
        "id": "call_1",
    }
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == "Sunny and 72F."
    assert not any(key.startswith("gen_ai.tool.") for key in attrs)
    _assert_no_off_contract_aliases(attrs)


def test_processor_marks_error_type_as_backend_error_status():
    span = FakeSpan(
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "always_fail",
            "error.type": "RuntimeError",
        },
        name="execute_tool always_fail",
    )

    AgentFrameworkSpanProcessor().on_end(span)
    attrs = span._attributes

    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert attrs["status_code"] == 500
    assert attrs[ERROR_MESSAGE_ATTR] == "RuntimeError"


def test_processor_maps_agent_and_workflow_spans():
    agent_span = FakeSpan(
        {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "weather_agent",
            ATTR_GEN_AI_INPUT_MESSAGES: json.dumps([{"role": "user", "content": "Hi"}]),
            ATTR_GEN_AI_OUTPUT_MESSAGES: json.dumps(
                [{"role": "assistant", "content": "Hello"}]
            ),
        },
        name="invoke_agent weather_agent",
    )
    workflow_span = FakeSpan(
        {
            "workflow.name": "weather_workflow",
            "workflow.id": "wf_123",
        },
        name="workflow.run weather_workflow",
    )

    processor = AgentFrameworkSpanProcessor()
    processor.on_end(agent_span)
    processor.on_end(workflow_span)

    assert agent_span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_AGENT
    assert (
        agent_span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "weather_agent"
    )
    assert json.loads(agent_span._attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
        {"role": "user", "content": "Hi"}
    ]
    assert workflow_span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_WORKFLOW
    assert (
        workflow_span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME]
        == "weather_workflow"
    )
    assert workflow_span._attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
