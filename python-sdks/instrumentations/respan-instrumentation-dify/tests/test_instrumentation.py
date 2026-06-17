import json
from types import SimpleNamespace

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_dify import _instrumentation as instrumentation
from respan_instrumentation_dify._constants import OFF_CONTRACT_ALIASES
from respan_instrumentation_dify._translator import build_dify_span_data
from respan_sdk.constants.span_attributes import (
    RESPAN_CUSTOMER_PARAMS_ID,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload
        self.closed = False

    def json(self):
        return self._payload

    def iter_lines(self, *args, **kwargs):
        for item in self._payload:
            yield f"data: {json.dumps(item)}".encode()

    def iter_content(self, *args, **kwargs):
        for item in self._payload:
            yield f"data: {json.dumps(item)}\n\n".encode()

    def close(self):
        self.closed = True


def test_build_chat_span_data_sets_canonical_attributes():
    span_name, attrs = build_dify_span_data(
        method="POST",
        endpoint="/chat-messages",
        request_json={
            "inputs": {"city": "Paris"},
            "query": "What is the weather?",
            "user": "user-123",
            "response_mode": "blocking",
            "conversation_id": "conv-123",
        },
        response=FakeResponse(
            {
                "event": "message",
                "task_id": "task-123",
                "message_id": "message-123",
                "conversation_id": "conv-123",
                "mode": "chat",
                "answer": "Sunny.",
                "metadata": {
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 2,
                        "total_tokens": 10,
                        "latency": 0.25,
                    }
                },
            }
        ),
    )

    assert span_name == "dify.chat"
    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[RESPAN_LOG_METHOD] == "tracing_integration"
    assert attrs[RESPAN_CUSTOMER_PARAMS_ID] == "user-123"
    assert attrs[RESPAN_THREADS_ID] == "conv-123"
    assert attrs[SpanAttributes.LLM_SYSTEM] == "dify"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.CHAT.value
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "What is the weather?"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Sunny."
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 8
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 2
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 10
    assert attrs[f"{RESPAN_METADATA}.dify.endpoint"] == "/chat-messages"
    assert attrs[f"{RESPAN_METADATA}.dify.task_id"] == "task-123"
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)


def test_build_completion_span_data_uses_inputs_query():
    _, attrs = build_dify_span_data(
        method="POST",
        endpoint="/completion-messages",
        request_json={
            "inputs": {"query": "Translate hello to Spanish."},
            "user": "user-123",
        },
        response=FakeResponse({"answer": "Hola.", "metadata": {"usage": {}}}),
    )

    assert attrs[RESPAN_LOG_TYPE] == "text"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.CHAT.value
    assert (
        attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Translate hello to Spanish."
    )
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Hola."
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)


def test_build_workflow_span_data_sets_workflow_log_type_without_llm_aliases():
    _, attrs = build_dify_span_data(
        method="POST",
        endpoint="/workflows/run",
        request_json={"inputs": {"query": "Summarize this"}, "user": "user-123"},
        response=FakeResponse(
            {
                "workflow_run_id": "run-123",
                "data": {
                    "workflow_id": "workflow-123",
                    "status": "succeeded",
                    "outputs": {"result": "Short summary."},
                    "total_tokens": 42,
                },
            }
        ),
    )

    assert attrs[RESPAN_LOG_TYPE] == "workflow"
    assert SpanAttributes.LLM_REQUEST_TYPE not in attrs
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == '{"result":"Short summary."}'
    assert attrs[f"{RESPAN_METADATA}.dify.workflow_run_id"] == "run-123"
    assert attrs[f"{RESPAN_METADATA}.dify.total_tokens"] == "42"
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)


def test_streaming_events_accumulate_answer_and_usage():
    _, attrs = build_dify_span_data(
        method="POST",
        endpoint="/chat-messages",
        request_json={"query": "Hi", "user": "user-123"},
        stream_events=[
            {"event": "message", "answer": "Hel"},
            {"event": "message", "answer": "lo"},
            {
                "event": "message_end",
                "metadata": {
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    }
                },
            },
        ],
    )

    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Hello"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == "Hello"
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 2
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)


def test_respan_params_override_span_context():
    span_name, attrs = build_dify_span_data(
        method="GET",
        endpoint="/parameters",
        request_params={"user": "user-123"},
        response=FakeResponse({"opening_statement": "Hello"}),
        respan_params={
            "span_name": "custom.dify.parameters",
            "workflow_name": "dify_custom.workflow",
            "customer_identifier": "customer-override",
            "metadata": {"example": "params"},
        },
    )

    assert span_name == "custom.dify.parameters"
    assert attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "dify_custom.workflow"
    assert attrs[RESPAN_TRACE_GROUP_ID] == "dify_custom.workflow"
    assert attrs[RESPAN_CUSTOMER_PARAMS_ID] == "customer-override"
    assert attrs[f"{RESPAN_METADATA}.example"] == "params"
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)


def test_instrumentor_patches_send_request_and_streaming(monkeypatch):
    emitted = []

    class DifyClient:
        def _send_request(self, method, endpoint, json=None, params=None, stream=False):
            if stream:
                return FakeResponse(
                    [
                        {"event": "message", "answer": "Hel"},
                        {"event": "message", "answer": "lo"},
                    ]
                )
            return FakeResponse({"answer": "Blocking"})

        def _send_request_with_files(self, method, endpoint, data, files):
            return FakeResponse({"id": "file-123"})

        def file_upload(self, user, files):
            return self._send_request_with_files(
                "POST", "/files/upload", data={"user": user}, files=files
            )

    class ChatClient(DifyClient):
        def create_chat_message(self, inputs, query, user, response_mode="blocking"):
            return self._send_request(
                "POST",
                "/chat-messages",
                {"inputs": inputs, "query": query, "user": user},
                stream=response_mode == "streaming",
            )

        def get_conversation_messages(self, user, conversation_id=None):
            return self._send_request("GET", "/messages", params={"user": user})

        def get_conversations(self, user):
            return self._send_request("GET", "/conversations", params={"user": user})

        def rename_conversation(self, conversation_id, name, user):
            return self._send_request("POST", "/conversations/id/name", {"user": user})

    class CompletionClient(DifyClient):
        def create_completion_message(self, inputs, response_mode, user):
            return self._send_request(
                "POST",
                "/completion-messages",
                {"inputs": inputs, "user": user},
                stream=response_mode == "streaming",
            )

    fake_module = SimpleNamespace(
        DifyClient=DifyClient,
        ChatClient=ChatClient,
        CompletionClient=CompletionClient,
    )
    monkeypatch.setattr(
        instrumentation.importlib,
        "import_module",
        lambda module_name: fake_module,
    )
    monkeypatch.setattr(
        instrumentation,
        "emit_dify_span",
        lambda **kwargs: emitted.append(kwargs),
    )

    instrumentor = instrumentation.DifyInstrumentor()
    instrumentor.activate()

    client = ChatClient()
    response = client.create_chat_message(
        inputs={},
        query="Hi",
        user="user-123",
        response_mode="streaming",
        respan_params={"span_name": "custom.stream"},
    )

    assert emitted == []
    assert [line for line in response.iter_lines()]
    assert len(emitted) == 1
    assert emitted[0]["call_context"].endpoint == "/chat-messages"
    assert emitted[0]["call_context"].respan_params["span_name"] == "custom.stream"
    assert emitted[0]["stream_events"][0]["answer"] == "Hel"

    instrumentor.deactivate()
