import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

from respan_instrumentation_watson_orchestrate_adk import (
    WatsonOrchestrateADKInstrumentor,
)
from respan_instrumentation_watson_orchestrate_adk import _instrumentation
from respan_instrumentation_watson_orchestrate_adk._constants import (
    AGENT_BUILDER_CLIENT_MODULE,
    CPE_CLIENT_MODULE,
    OFF_CONTRACT_ALIASES,
    PYTHON_TOOL_MODULE,
    RUN_CLIENT_MODULE,
    WATSONX_AI_CLIENT_MODULE,
)
from respan_instrumentation_watson_orchestrate_adk._otel_emitter import (
    build_agent_run_attrs,
    build_chat_attrs,
    build_tool_attrs,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_THREADS_ID
from respan_tracing.core.tracer import RespanTracer


def _install_module(monkeypatch, module_name, **attrs):
    parts = module_name.split(".")
    for index in range(1, len(parts)):
        parent_name = ".".join(parts[:index])
        module = sys.modules.get(parent_name)
        if module is None:
            module = ModuleType(parent_name)
            monkeypatch.setitem(sys.modules, parent_name, module)
        if index > 1:
            grandparent_name = ".".join(parts[: index - 1])
            setattr(sys.modules[grandparent_name], parts[index - 1], module)

    module = ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    if len(parts) > 1:
        setattr(sys.modules[".".join(parts[:-1])], parts[-1], module)
    return module


@pytest.fixture(autouse=True)
def reset_instrumentation():
    RespanTracer.reset_instance()
    _instrumentation._restore_methods()
    yield
    _instrumentation._restore_methods()
    RespanTracer.reset_instance()


def _assert_no_aliases(attrs):
    for alias in OFF_CONTRACT_ALIASES:
        assert alias not in attrs


def test_build_chat_attrs_maps_canonical_fields_without_aliases():
    attrs = build_chat_attrs(
        method_name="generate_response",
        call_kwargs={
            "input": "Summarize the ticket.",
            "model": "watsonx/meta-llama/llama-3-3-70b-instruct",
            "tools": [
                {
                    "name": "lookup_ticket",
                    "description": "Lookup a ticket.",
                    "input_schema": {"type": "object"},
                }
            ],
        },
        response={
            "choices": [
                {"message": {"role": "assistant", "content": "Ticket summarized."}}
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
            },
        },
    )

    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[TLSpanAttributes.LLM_SYSTEM] == "watsonx_orchestrate"
    assert attrs[TLSpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        attrs[TLSpanAttributes.LLM_REQUEST_MODEL]
        == "watsonx/meta-llama/llama-3-3-70b-instruct"
    )
    assert attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.content"] == "Summarize the ticket."
    assert attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    assert (
        attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"]
        == "Ticket summarized."
    )
    assert attrs[TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 12
    assert attrs[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 4
    assert attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 16
    assert json.loads(attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS]) == [
        {
            "type": "function",
            "function": {
                "name": "lookup_ticket",
                "description": "Lookup a ticket.",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert TLSpanAttributes.TRACELOOP_SPAN_KIND not in attrs
    _assert_no_aliases(attrs)


def test_build_tool_attrs_maps_execution_without_tool_call_aliases():
    attrs = build_tool_attrs(
        tool_name="lookup_ticket",
        args=(),
        kwargs={"ticket_id": "INC-7"},
        response=SimpleNamespace(content={"status": "open"}),
    )

    assert attrs[RESPAN_LOG_TYPE] == "tool"
    assert attrs[TLSpanAttributes.TRACELOOP_ENTITY_NAME] == "lookup_ticket"
    assert json.loads(attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "name": "lookup_ticket",
        "arguments": {"ticket_id": "INC-7"},
    }
    assert json.loads(attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "content": {"status": "open"}
    }
    _assert_no_aliases(attrs)


def test_build_agent_run_attrs_records_thread_and_run_without_aliases():
    attrs = build_agent_run_attrs(
        method_name="create_run",
        call_kwargs={
            "message": "hello",
            "agent_id": "agent-123",
            "thread_id": "thread-1",
        },
        response={"run_id": "run-1", "thread_id": "thread-1", "status": "queued"},
    )

    assert attrs[RESPAN_LOG_TYPE] == "agent"
    assert attrs[RESPAN_THREADS_ID] == "thread-1"
    assert attrs[TLSpanAttributes.TRACELOOP_ENTITY_NAME] == "agent-123"
    assert "create_run" in attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT]
    _assert_no_aliases(attrs)


def test_activate_patches_tool_and_run_clients(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        _instrumentation._otel_emitter,
        "emit_tool_span",
        lambda **kwargs: emitted.append(("tool", kwargs)),
    )
    monkeypatch.setattr(
        _instrumentation._otel_emitter,
        "emit_agent_run_span",
        lambda **kwargs: emitted.append(("run", kwargs)),
    )

    class PythonTool:
        name = "lookup_ticket"

        def __call__(self, **kwargs):
            return SimpleNamespace(content={"ok": True, "kwargs": kwargs})

    class RunClient:
        def create_run(self, message, agent_id=None, thread_id=None, capture_logs=False):
            return {
                "run_id": "run-1",
                "thread_id": thread_id or "thread-1",
                "agent_id": agent_id,
            }

    _install_module(monkeypatch, PYTHON_TOOL_MODULE, PythonTool=PythonTool)
    _install_module(monkeypatch, RUN_CLIENT_MODULE, RunClient=RunClient)

    original_tool_call = PythonTool.__call__
    original_create_run = RunClient.create_run

    instrumentor = WatsonOrchestrateADKInstrumentor()
    instrumentor.activate()

    assert instrumentor._is_instrumented is True
    assert PythonTool.__call__ is not original_tool_call
    assert RunClient.create_run is not original_create_run

    assert PythonTool()(ticket_id="INC-7").content["ok"] is True
    assert RunClient().create_run(
        "hello",
        agent_id="agent-123",
        thread_id="thread-1",
    )["run_id"] == "run-1"

    assert emitted[0][0] == "tool"
    assert emitted[0][1]["tool_name"] == "lookup_ticket"
    assert emitted[0][1]["kwargs"] == {"ticket_id": "INC-7"}
    assert emitted[1][0] == "run"
    assert emitted[1][1]["method_name"] == "create_run"
    assert emitted[1][1]["call_kwargs"]["message"] == "hello"
    assert emitted[1][1]["call_kwargs"]["agent_id"] == "agent-123"

    instrumentor.deactivate()

    assert PythonTool.__call__ is original_tool_call
    assert RunClient.create_run is original_create_run
    assert instrumentor._is_instrumented is False


def test_chat_client_failure_emits_error_span(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        _instrumentation._otel_emitter,
        "emit_chat_span",
        lambda **kwargs: emitted.append(kwargs),
    )

    class WatsonxAIClient:
        model = "watsonx/ibm/granite-3-8b-instruct"

        def generate_response(self, input, model=None, **kwargs):
            raise RuntimeError("provider unavailable")

    _install_module(
        monkeypatch,
        WATSONX_AI_CLIENT_MODULE,
        WatsonxAIClient=WatsonxAIClient,
    )

    instrumentor = WatsonOrchestrateADKInstrumentor()
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="provider unavailable"):
        WatsonxAIClient().generate_response("hello")

    assert emitted == [
        {
            "method_name": "generate_response",
            "call_kwargs": {"input": "hello"},
            "start_ns": emitted[0]["start_ns"],
            "error_message": "provider unavailable",
            "instance": emitted[0]["instance"],
        }
    ]


def test_activate_is_idempotent(monkeypatch):
    class AgentBuilderClient:
        def submit_chat(self, chat_llm, user_message=None, agent_id=None):
            return {"formatted_message": {"content": "ok"}}

    _install_module(
        monkeypatch,
        AGENT_BUILDER_CLIENT_MODULE,
        AgentBuilderClient=AgentBuilderClient,
    )

    original = AgentBuilderClient.submit_chat

    instrumentor = WatsonOrchestrateADKInstrumentor()
    instrumentor.activate()
    first_patch = AgentBuilderClient.submit_chat
    instrumentor.activate()

    assert AgentBuilderClient.submit_chat is first_patch
    assert AgentBuilderClient.submit_chat is not original


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch):
    class CPEClient:
        def submit_chat_with_agent_architect(self, chat_llm, user_message=None):
            return {}

    _install_module(monkeypatch, CPE_CLIENT_MODULE, CPEClient=CPEClient)
    original = CPEClient.submit_chat_with_agent_architect

    RespanTracer(is_enabled=False)
    instrumentor = WatsonOrchestrateADKInstrumentor()
    instrumentor.activate()

    assert CPEClient.submit_chat_with_agent_architect is original
    assert instrumentor._is_instrumented is False
