"""Exercise IBM SDK implementations; replace only their transport boundary."""

from __future__ import annotations

import json

import pytest
from opentelemetry import context, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv_ai import SpanAttributes as A
from respan_instrumentation_watson_orchestrate_adk import (
    WatsonOrchestrateADKInstrumentor,
)
from respan_instrumentation_watson_orchestrate_adk import _instrumentation as runtime
from respan_instrumentation_watson_orchestrate_adk import _otel_emitter as emitter
from respan_sdk.constants.span_attributes import RESPAN_LOG_ID, RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


@pytest.fixture
def captured(monkeypatch):
    runtime._restore_methods()
    spans = []
    monkeypatch.setattr(emitter, "inject_span", lambda span: spans.append(span))
    instrumentor = WatsonOrchestrateADKInstrumentor()
    instrumentor.activate()
    yield spans
    instrumentor.deactivate()
    runtime._restore_methods()


def run_client(monkeypatch, response=None):
    from ibm_watsonx_orchestrate_clients.chat.run_client import RunClient

    client = object.__new__(RunClient)
    client.base_endpoint = "/runs"
    client.base_url = "https://watson.invalid/v1"
    client.api_key = "fixture-key"
    client.authenticator = None
    client.verify = True
    calls = []

    def post(path, data):
        calls.append((path, data))
        return response or {
            "run_id": "provider-run",
            "thread_id": "provider-thread",
            "status": "queued",
        }

    monkeypatch.setattr(client, "_post", post)
    return client, calls


def test_run_files_polling_and_unique_span_ids(captured, monkeypatch):
    client, calls = run_client(monkeypatch)
    files = [
        {
            "url": "https://storage.invalid/report?signature=secret",
            "filename": "report.txt",
        }
    ]
    client.create_run_with_files("", files, agent_id="support", capture_logs=True)
    monkeypatch.setattr(
        client, "_get", lambda path: {"run_id": "provider-run", "status": "completed"}
    )
    result = client.wait_for_run_completion(
        "provider-run", poll_interval=0, max_retries=1
    )
    assert result["status"] == "completed"
    assert calls[0][1]["context"]["data"][0]["files"] == files
    assert len(captured) == 2
    assert captured[0].context.span_id != captured[1].context.span_id
    assert all(RESPAN_LOG_ID not in span.attributes for span in captured)
    assert "signature=secret" not in str(dict(captured[0].attributes))


@pytest.mark.parametrize("state", ["failed", "cancelled", "canceled", "error"])
def test_returned_failure_is_error_without_changing_return(
    captured, monkeypatch, state
):
    response = {
        "run_id": "provider-run",
        "status": state,
        "error": "controlled failure",
        "diagnostics": {"step": "last"},
    }
    client, _ = run_client(monkeypatch, response)
    assert client.create_run("hello") is response
    assert captured[0].status.status_code.name == "ERROR"
    assert "http.response.status_code" not in captured[0].attributes
    assert json.loads(captured[0].attributes[A.TRACELOOP_ENTITY_OUTPUT]) == response


@pytest.mark.parametrize(
    "module_name,class_name,expected_path",
    [
        (
            "watsonx_ai.watsonx_ai_client",
            "WatsonxAIClient",
            "/text/chat?version=2023-10-25",
        ),
        ("groq.groq_client", "GroqClient", "/openai/v1/chat/completions"),
        (
            "ai_gateway.ai_gateway_client",
            "AIGatewayClient",
            "/gateway/model/chat/completions",
        ),
    ],
)
def test_real_inference_requests_current_tool_call_and_zero_usage(
    captured, monkeypatch, module_name, class_name, expected_path
):
    module = pytest.importorskip(
        "ibm_watsonx_orchestrate.client.autodiscover." + module_name
    )
    cls = getattr(module, class_name)
    client = object.__new__(cls)
    client.model = "model-request"
    client.space_id = "fixture-space"
    calls = []
    response = {
        "model": "model-response",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-current",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"id":7}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    def post(path, data):
        calls.append((path, data))
        return response

    monkeypatch.setattr(client, "_post", post)
    assert (
        client.generate_response("question", instructions="system instruction")
        is response
    )
    assert calls[0][0] == expected_path
    assert calls[0][1]["messages"][0]["content"] == "system instruction"
    attrs = captured[0].attributes
    assert attrs[A.LLM_REQUEST_MODEL] == "model-request"
    assert attrs[A.LLM_RESPONSE_MODEL] == "model-response"
    assert attrs["gen_ai.prompt.0.role"] == "system"
    assert attrs["gen_ai.prompt.1.content"] == "question"
    assert (
        json.loads(attrs["gen_ai.completion.0.tool_calls"])[0]["id"] == "call-current"
    )
    assert "gen_ai.completion.0.content" not in attrs
    assert attrs[A.LLM_USAGE_PROMPT_TOKENS] == 0
    assert attrs[A.LLM_USAGE_COMPLETION_TOKENS] == 0


@pytest.mark.parametrize(
    "value", [-1, 0.5, float("inf"), float("nan"), True, "invalid"]
)
def test_invalid_usage_is_not_reported(value):
    attrs = emitter.build_chat_attrs(
        method_name="generate_response",
        call_kwargs={},
        response={"usage": {"prompt_tokens": value}},
    )
    assert A.LLM_USAGE_PROMPT_TOKENS not in attrs


def test_history_tool_calls_do_not_become_current_output():
    attrs = emitter.build_chat_attrs(
        method_name="submit_chat",
        call_kwargs={
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "old", "function": {"name": "lookup", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "old", "content": "done"},
            ]
        },
        response={"choices": [{"message": {"content": "final"}}]},
    )
    assert json.loads(attrs["gen_ai.prompt.0.tool_calls"])[0]["id"] == "old"
    assert attrs["gen_ai.prompt.1.tool_call_id"] == "old"
    assert "gen_ai.completion.0.tool_calls" not in attrs


def test_flow_methods_are_sync_workflows(captured, monkeypatch):
    from ibm_watsonx_orchestrate_clients.tools.tempus_client import TempusClient

    client = object.__new__(TempusClient)
    paths = []

    def post(path, data):
        paths.append(path)
        return {"result": data}

    monkeypatch.setattr(client, "_post", post)
    assert client.run_flow("flow-one", {"x": 1}) == {"result": {"x": 1}}
    assert client.arun_flow("flow-two", {"x": 2}) == {"result": {"x": 2}}
    assert paths == [
        "/flows/flow-one/versions/TIP/run",
        "/flows/flow-two/versions/TIP/run/async",
    ]
    assert all(s.attributes[RESPAN_LOG_TYPE] == "workflow" for s in captured)


def test_real_architect_clients(captured, monkeypatch):
    from ibm_watsonx_orchestrate_clients.ai_builder.agent_builder_client import (
        AgentBuilderClient,
    )
    from ibm_watsonx_orchestrate_clients.ai_builder.cpe.cpe_client import CPEClient

    client = object.__new__(AgentBuilderClient)
    client.chat_id = "old-thread"
    monkeypatch.setattr(
        client,
        "_post_nd_json",
        lambda path, data: [
            {
                "event": "message.created",
                "data": {
                    "thread_id": "next-thread",
                    "message": {
                        "content": "architect result",
                        "additional_properties": {"architect_conversational_state": {}},
                    },
                },
            }
        ],
    )
    assert (
        client.submit_chat("architect-model", "create agent")["formatted_message"][
            "content"
        ]
        == "architect result"
    )
    assert client.chat_id == "next-thread"
    cpe = object.__new__(CPEClient)
    cpe.chat_id = "cpe-thread"
    monkeypatch.setattr(
        cpe, "_post_nd_json", lambda path, data: [{"content": "refined result"}]
    )
    cpe.submit_chat_with_agent_architect("architect-model", "create agent")
    cpe.submit_refine_agent_with_chats(
        "refine instruction",
        "architect-model",
        {},
        {},
        {},
        [],
        model="target-agent-model",
    )
    assert len(captured) == 3
    assert captured[-1].attributes[A.LLM_REQUEST_MODEL] == "architect-model"


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_real_websocket_callbacks_failure_and_cleanup(
    captured, monkeypatch, failed
):
    from ibm_watsonx_orchestrate_clients.chat import run_client as module

    client, _ = run_client(monkeypatch)
    events = []
    closed = []

    class Socket:
        def __init__(self, **kwargs):
            self.handlers = {}

        def register_handler(self, name, fn):
            self.handlers[name] = fn

        async def connect(self, *args):
            pass

        async def listen(self):
            event = "run.failed" if failed else "run.completed"
            self.handlers["message.created"](
                {"event": "message.created", "data": {"message": "hello"}}
            )
            self.handlers[event](
                {
                    "event": event,
                    "data": {
                        "run_id": "provider-run",
                        "status": "failed" if failed else "completed",
                    },
                }
            )

        async def disconnect(self):
            closed.append(True)

    monkeypatch.setattr(module, "WebSocketClient", Socket)
    result = await client.stream_run_with_websocket(
        "agent", "thread", "run", on_message=events.append
    )
    assert len(events) == 1 and closed == [True]
    assert result["status"] == ("failed" if failed else "completed")
    assert captured[0].status.status_code.name == ("ERROR" if failed else "OK")


@pytest.mark.parametrize("flag", ["environment", "context"])
def test_content_disabled_and_suppression(captured, monkeypatch, flag):
    client, _ = run_client(monkeypatch)
    token = None
    if flag == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    else:
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        client.create_run("private prompt")
    finally:
        if token is not None:
            context.detach(token)
    assert A.TRACELOOP_ENTITY_INPUT not in captured[0].attributes
    assert A.TRACELOOP_ENTITY_OUTPUT not in captured[0].attributes
    token = context.attach(context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
    try:
        client.create_run("suppressed")
    finally:
        context.detach(token)
    assert len(captured) == 1


def test_nested_tool_parent_context_and_falsy_result(captured, monkeypatch):
    from ibm_watsonx_orchestrate.agent_builder.tools import tool

    client, _ = run_client(monkeypatch)

    @tool
    def lookup() -> int:
        """Look up a controlled fixture."""
        client.create_run("nested")
        return 0

    provider = TracerProvider()
    with provider.get_tracer(__name__).start_as_current_span("outer") as parent:
        assert lookup().content == 0
        assert trace.get_current_span() is parent
    assert len(captured) == 2
    assert captured[0].parent.span_id == captured[1].context.span_id
    assert captured[1].parent.span_id == parent.context.span_id
    assert json.loads(captured[1].attributes[A.TRACELOOP_ENTITY_OUTPUT])["content"] == 0


def test_retained_foreign_wrapper_inert_after_deactivate(monkeypatch):
    runtime._restore_methods()
    from ibm_watsonx_orchestrate_clients.chat.run_client import RunClient

    spans = []
    monkeypatch.setattr(emitter, "inject_span", lambda span: spans.append(span))
    original = RunClient.create_run
    first = WatsonOrchestrateADKInstrumentor()
    first.activate()
    inner = RunClient.create_run

    def foreign(self, *args, **kwargs):
        return inner(self, *args, **kwargs)

    monkeypatch.setattr(RunClient, "create_run", foreign)
    first.deactivate()
    client, _ = run_client(monkeypatch)
    client.create_run("inactive")
    assert not spans
    second = WatsonOrchestrateADKInstrumentor()
    second.activate()
    client.create_run("active once")
    assert len(spans) == 1
    second.deactivate()
    monkeypatch.setattr(RunClient, "create_run", original)


def test_activation_rolls_back_partial_failure(monkeypatch):
    runtime._restore_methods()
    from ibm_watsonx_orchestrate.agent_builder.tools.python_tool import PythonTool

    original = PythonTool.__call__
    load = runtime._optional_class

    def broken(module, cls):
        if cls == "RunClient":
            raise RuntimeError("controlled install failure")
        return load(module, cls)

    monkeypatch.setattr(runtime, "_optional_class", broken)
    with pytest.raises(RuntimeError):
        WatsonOrchestrateADKInstrumentor().activate()
    assert PythonTool.__call__ is original
    assert runtime._ACTIVATION_COUNT == 0


@pytest.mark.asyncio
async def test_websocket_cancellation_preserves_exception_and_cleans_up(
    captured, monkeypatch
):
    import asyncio

    from ibm_watsonx_orchestrate_clients.chat import run_client as module

    client, _ = run_client(monkeypatch)
    connected = asyncio.Event()
    closed = []

    class Socket:
        def __init__(self, **kwargs):
            pass

        def register_handler(self, *args):
            pass

        async def connect(self, *args):
            connected.set()

        async def listen(self):
            await asyncio.Event().wait()

        async def disconnect(self):
            closed.append(True)

    monkeypatch.setattr(module, "WebSocketClient", Socket)
    task = asyncio.create_task(
        client.stream_run_with_websocket("agent", "thread", "run")
    )
    await connected.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == [True]
    assert len(captured) == 1
    assert captured[0].status.status_code.name == "ERROR"
    assert captured[0].attributes["error.type"] == "CancelledError"


def test_unsampled_parent_does_not_emit(captured, monkeypatch):
    client, _ = run_client(monkeypatch)
    parent = trace.NonRecordingSpan(trace.SpanContext(1, 2, False, trace.TraceFlags(0)))
    with trace.use_span(parent):
        client.create_run("not sampled")
    assert not captured


def test_capture_policy_snapshot_survives_provider_change(captured, monkeypatch):
    client, _ = run_client(monkeypatch)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    def post(path, data):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        return {"status": "failed", "error": "private-provider-detail"}

    monkeypatch.setattr(client, "_post", post)
    client.create_run("private-input")
    assert "private" not in str(dict(captured[0].attributes))
    assert "private" not in captured[0].status.description


def test_failed_chat_does_not_manufacture_completion():
    attrs = emitter.build_chat_attrs(
        method_name="generate_response",
        call_kwargs={"input": "question"},
        error_message="ProviderError: failed",
    )
    assert A.TRACELOOP_ENTITY_OUTPUT not in attrs
    assert not any(key.startswith("gen_ai.completion.") for key in attrs)
    assert A.LLM_USAGE_PROMPT_TOKENS not in attrs


def test_always_off_sampler_drops_root_and_nested_calls(captured, monkeypatch):
    from ibm_watsonx_orchestrate.agent_builder.tools import tool
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

    provider = TracerProvider(sampler=ALWAYS_OFF)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    client, _ = run_client(monkeypatch)

    @tool
    def run_nested() -> int:
        """Run a controlled nested call."""
        client.create_run("not sampled")
        return 1

    assert run_nested().content == 1
    assert not captured
