import asyncio
import json
from types import SimpleNamespace

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_instrumentation_dify import _instrumentation as instrumentation
from respan_instrumentation_dify._constants import OFF_CONTRACT_ALIASES
from respan_instrumentation_dify._translator import build_dify_span_data
from respan_sdk.constants.span_attributes import (
    RESPAN_CUSTOMER_PARAMS_ID,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_PROMPT,
    RESPAN_PROPERTIES,
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


def _metadata(attributes):
    return json.loads(attributes[RESPAN_METADATA])


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
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
    assert _metadata(attrs)["dify.endpoint"] == "/chat-messages"
    assert _metadata(attrs)["dify.task_id"] == "task-123"
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


def test_build_chat_span_data_maps_only_provider_exposed_model():
    _, attrs = build_dify_span_data(
        method="POST",
        endpoint="/chat-messages",
        request_json={"query": "Hi", "user": "user-123"},
        response=FakeResponse(
            {
                "answer": "Hello",
                "metadata": {
                    "usage": {
                        "model": "anthropic/claude-sonnet-4",
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                    }
                },
            }
        ),
    )

    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "anthropic/claude-sonnet-4"
    assert "model" not in attrs


def test_build_chat_span_data_does_not_invent_missing_model():
    _, attrs = build_dify_span_data(
        method="POST",
        endpoint="/chat-messages",
        request_json={"query": "Hi", "user": "user-123"},
        response=FakeResponse({"answer": "Hello", "metadata": {"usage": {}}}),
    )

    assert SpanAttributes.LLM_REQUEST_MODEL not in attrs
    assert "model" not in attrs


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
    assert (
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == '{"result":"Short summary."}'
    )
    assert _metadata(attrs)["dify.workflow_run_id"] == "run-123"
    assert _metadata(attrs)["dify.total_tokens"] == 42
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)


def test_streaming_events_accumulate_answer_and_usage():
    _, attrs = build_dify_span_data(
        method="POST",
        endpoint="/chat-messages",
        request_json={"query": "Hi", "user": "user-123"},
        stream_events=[
            {
                "event": "message",
                "task_id": "task-stream",
                "message_id": "message-stream",
                "conversation_id": "conversation-stream",
                "answer": "Hel",
            },
            {
                "event": "message",
                "task_id": "task-stream",
                "message_id": "message-stream",
                "conversation_id": "conversation-stream",
                "answer": "lo",
            },
            {
                "event": "message_end",
                "task_id": "task-stream",
                "message_id": "message-stream",
                "conversation_id": "conversation-stream",
                "metadata": {
                    "usage": {
                        "model": "dify/stream-model",
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
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "dify/stream-model"
    assert _metadata(attrs)["dify.event"] == "message_end"
    assert _metadata(attrs)["dify.task_id"] == "task-stream"
    assert _metadata(attrs)["dify.message_id"] == "message-stream"
    assert _metadata(attrs)["dify.conversation_id"] == "conversation-stream"
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
    assert _metadata(attrs)["example"] == "params"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)


def test_metadata_uses_only_the_canonical_json_attribute():
    _, attrs = build_dify_span_data(
        method="GET",
        endpoint="/parameters",
        request_params={"user": "user-123"},
        response=FakeResponse({"opening_statement": "Hello"}),
    )

    metadata = json.loads(attrs[RESPAN_METADATA])
    assert metadata["dify.method"] == "GET"
    assert metadata["dify.endpoint"] == "/parameters"
    assert not any(key.startswith(f"{RESPAN_METADATA}.") for key in attrs)


def test_propagated_metadata_merges_into_the_canonical_json_attribute():
    _, attrs = build_dify_span_data(
        method="GET",
        endpoint="/parameters",
        response=FakeResponse({"opening_statement": "Hello"}),
        propagated_attributes={
            f"{RESPAN_METADATA}.run_id": "run-123",
            f"{RESPAN_METADATA}.example": "dify-loopback",
            RESPAN_METADATA: json.dumps(
                {
                    "suite": "integration",
                    "dify.endpoint": "must-not-replace-call-metadata",
                }
            ),
        },
    )

    metadata = _metadata(attrs)
    assert metadata["dify.endpoint"] == "/parameters"
    assert metadata["dify.method"] == "GET"
    assert metadata["run_id"] == "run-123"
    assert metadata["example"] == "dify-loopback"
    assert metadata["suite"] == "integration"
    assert not any(key.startswith(f"{RESPAN_METADATA}.") for key in attrs)


def test_child_entity_path_uses_real_parent_workflow_context():
    _, attrs = build_dify_span_data(
        method="GET",
        endpoint="/parameters",
        response=FakeResponse({"opening_statement": "Hello"}),
        current_workflow_name="parent.workflow",
        parent_id="parent-span-id",
    )

    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == "parent.workflow"


def test_structured_respan_params_are_json_serialized_for_otel():
    _, attrs = build_dify_span_data(
        method="GET",
        endpoint="/parameters",
        response=FakeResponse({"opening_statement": "Hello"}),
        respan_params={
            "properties": {"region": "us", "attempts": [1, 2]},
            "prompt": {"id": "prompt-1", "variables": {"city": "Paris"}},
        },
    )

    assert json.loads(attrs[RESPAN_PROPERTIES]) == {
        "attempts": [1, 2],
        "region": "us",
    }
    assert json.loads(attrs[RESPAN_PROMPT]) == {
        "id": "prompt-1",
        "variables": {"city": "Paris"},
    }


def test_include_content_false_omits_request_and_response_content():
    _, attrs = build_dify_span_data(
        method="POST",
        endpoint="/chat-messages",
        request_json={"query": "private prompt", "user": "user-123"},
        response=FakeResponse(
            {
                "answer": "private answer",
                "metadata": {"usage": {"prompt_tokens": 2, "completion_tokens": 3}},
            }
        ),
        include_content=False,
    )

    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    assert f"{SpanAttributes.LLM_PROMPTS}.0.content" not in attrs
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.content" not in attrs
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 2
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 3
    assert "private prompt" not in json.dumps(attrs, default=str)
    assert "private answer" not in json.dumps(attrs, default=str)


def test_workspace_credentials_are_recursively_redacted_without_mutation():
    credentials = {
        "openai_api_key": "python-provider-key",
        "apiKey": "python-camel-key",
        "Authorization": "Bearer python-authorization",
        "nested": {
            "refresh_token": "python-refresh-token",
            "password": "python-password",
            "password_hash": "python-password-hash",
            "aws_secret_access_key": "python-aws-secret-access-key",
            "safe_region": "us-east-1",
        },
    }
    _, attrs = build_dify_span_data(
        method="POST",
        endpoint=("/workspaces/current/model-providers/openai/credentials/validate"),
        request_json=credentials,
        response=FakeResponse(
            {
                "result": "success",
                "provider": {
                    "credentials": {"access_token": "python-response-access-token"}
                },
            }
        ),
        respan_params={"metadata": {"session_token": "python-metadata-session-token"}},
    )

    serialized_attributes = json.dumps(attrs, default=str)
    for secret in (
        "python-provider-key",
        "python-camel-key",
        "python-authorization",
        "python-refresh-token",
        "python-password",
        "python-password-hash",
        "python-aws-secret-access-key",
        "python-response-access-token",
        "python-metadata-session-token",
    ):
        assert secret not in serialized_attributes
    assert "[REDACTED]" in serialized_attributes
    assert "us-east-1" in serialized_attributes
    assert credentials["openai_api_key"] == "python-provider-key"
    assert credentials["nested"]["refresh_token"] == "python-refresh-token"


def test_rag_pipeline_is_classified_as_workflow():
    span_name, attrs = build_dify_span_data(
        method="POST",
        endpoint="/datasets/dataset-1/pipeline/run",
        request_json={"inputs": {"source": "docs"}, "response_mode": "blocking"},
        response=FakeResponse(
            {"data": {"status": "succeeded", "outputs": {"documents": 2}}}
        ),
    )

    assert span_name == "dify.workflow"
    assert attrs[RESPAN_LOG_TYPE] == "workflow"
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


def test_concurrent_instrumentors_preserve_first_content_policy(monkeypatch):
    emitted = []

    class DifyClient:
        def _send_request(self, method, endpoint, json=None, params=None, stream=False):
            return FakeResponse({"answer": "private answer"})

        def _send_request_with_files(self, method, endpoint, data, files):
            return FakeResponse({"id": "file-123"})

    class ChatClient(DifyClient):
        pass

    class CompletionClient(DifyClient):
        pass

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

    first = instrumentation.DifyInstrumentor(include_content=False)
    conflicting = instrumentation.DifyInstrumentor(include_content=True)
    first.activate()
    conflicting.activate()
    try:
        ChatClient()._send_request(
            "POST",
            "/chat-messages",
            {"query": "private prompt", "user": "user-123"},
        )
        assert emitted[-1]["include_content"] is False

        conflicting.deactivate()
        ChatClient()._send_request(
            "POST",
            "/chat-messages",
            {"query": "still private", "user": "user-123"},
        )
        assert emitted[-1]["include_content"] is False
    finally:
        conflicting.deactivate()
        first.deactivate()

    replacement = instrumentation.DifyInstrumentor(include_content=True)
    replacement.activate()
    try:
        ChatClient()._send_request(
            "POST",
            "/chat-messages",
            {"query": "visible prompt", "user": "user-123"},
        )
        assert emitted[-1]["include_content"] is True
    finally:
        replacement.deactivate()


def test_deactivation_preserves_a_later_foreign_patch(monkeypatch):
    class DifyClient:
        def _send_request(self, method, endpoint, json=None, params=None, stream=False):
            return FakeResponse({"answer": "original"})

        def _send_request_with_files(self, method, endpoint, data, files):
            return FakeResponse({"id": "file-123"})

    class ChatClient(DifyClient):
        pass

    class CompletionClient(DifyClient):
        pass

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

    original = DifyClient._send_request
    first = instrumentation.DifyInstrumentor()
    second = instrumentation.DifyInstrumentor()
    first.activate()
    second.activate()
    assert DifyClient._send_request is not original

    def foreign_patch(self, *args, **kwargs):
        return FakeResponse({"answer": "foreign"})

    DifyClient._send_request = foreign_patch
    first.deactivate()
    assert DifyClient._send_request is foreign_patch
    second.deactivate()

    assert DifyClient._send_request is foreign_patch
    assert instrumentation._ACTIVE_INSTANCES == 0
    assert instrumentation._PATCHED_METHODS == {}


def test_sync_stream_early_close_emits_partial_events_once(monkeypatch):
    emitted = []
    call_context = SimpleNamespace(endpoint="/chat-messages")
    response = FakeResponse(
        [
            {"event": "message", "answer": "partial"},
            {"event": "message_end"},
        ]
    )
    wrapped = instrumentation._InstrumentedStreamingResponse(
        response=response,
        call_context=call_context,
    )
    monkeypatch.setattr(
        instrumentation,
        "emit_dify_span",
        lambda **kwargs: emitted.append(kwargs),
    )

    lines = wrapped.iter_lines()
    assert next(lines)
    lines.close()
    wrapped.close()

    assert len(emitted) == 1
    assert emitted[0]["stream_events"][0]["answer"] == "partial"
    assert response.closed is True


def test_async_clients_are_patched_and_stream_on_response_mode(monkeypatch):
    emitted = []

    class DifyClient:
        def _send_request(self, method, endpoint, json=None, params=None, stream=False):
            return FakeResponse({"result": "success"})

        def _send_request_with_files(self, method, endpoint, data, files):
            return FakeResponse({"id": "file-sync"})

    class ChatClient(DifyClient):
        def create_chat_message(self, **kwargs):
            return self._send_request("POST", "/chat-messages", kwargs)

    class CompletionClient(DifyClient):
        def create_completion_message(self, **kwargs):
            return self._send_request("POST", "/completion-messages", kwargs)

    class AsyncResponse:
        status_code = 200

        def __init__(self, events):
            self.events = events
            self.closed = False

        def json(self):
            return {}

        async def aiter_lines(self):
            for event in self.events:
                yield f"data: {json.dumps(event)}"

        async def aclose(self):
            self.closed = True

    class AsyncDifyClient:
        async def _send_request(
            self, method, endpoint, json=None, params=None, stream=False
        ):
            return AsyncResponse(
                [
                    {"event": "message", "answer": "Hel"},
                    {"event": "message", "answer": "lo"},
                ]
            )

        async def _send_request_with_files(self, method, endpoint, data, files):
            return AsyncResponse([])

    class AsyncChatClient(AsyncDifyClient):
        async def create_chat_message(
            self, inputs, query, user, response_mode="blocking"
        ):
            return await self._send_request(
                "POST",
                "/chat-messages",
                {
                    "inputs": inputs,
                    "query": query,
                    "user": user,
                    "response_mode": response_mode,
                },
            )

    class AsyncCompletionClient(AsyncDifyClient):
        async def create_completion_message(self, inputs, response_mode, user):
            return await self._send_request(
                "POST",
                "/completion-messages",
                {
                    "inputs": inputs,
                    "response_mode": response_mode,
                    "user": user,
                },
            )

    sync_module = SimpleNamespace(
        DifyClient=DifyClient,
        ChatClient=ChatClient,
        CompletionClient=CompletionClient,
    )
    async_module = SimpleNamespace(
        AsyncDifyClient=AsyncDifyClient,
        AsyncChatClient=AsyncChatClient,
        AsyncCompletionClient=AsyncCompletionClient,
    )

    def import_module(name):
        if name == instrumentation.DIFY_ASYNC_CLIENT_MODULE:
            return async_module
        return sync_module

    monkeypatch.setattr(instrumentation.importlib, "import_module", import_module)
    monkeypatch.setattr(
        instrumentation,
        "emit_dify_span",
        lambda **kwargs: emitted.append(kwargs),
    )

    async def run():
        instrumentor = instrumentation.DifyInstrumentor()
        instrumentor.activate()
        try:
            response = await AsyncChatClient().create_chat_message(
                inputs={},
                query="Hi",
                user="user-123",
                response_mode="streaming",
                respan_params={"span_name": "custom.async.stream"},
            )
            assert emitted == []
            lines = [line async for line in response.aiter_lines()]
            assert len(lines) == 2
            assert len(emitted) == 1
            assert emitted[0]["call_context"].stream is True
            assert emitted[0]["call_context"].respan_params["span_name"] == (
                "custom.async.stream"
            )
            assert emitted[0]["stream_events"][1]["answer"] == "lo"
            await response.aclose()
            assert len(emitted) == 1
        finally:
            instrumentor.deactivate()

    asyncio.run(run())


def test_async_stream_error_emits_once(monkeypatch):
    emitted = []

    class ErrorResponse:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"event":"message","answer":"partial"}'
            raise RuntimeError("stream failed")

    wrapped = instrumentation._InstrumentedAsyncStreamingResponse(
        response=ErrorResponse(),
        call_context=SimpleNamespace(endpoint="/chat-messages"),
    )
    monkeypatch.setattr(
        instrumentation,
        "emit_dify_span",
        lambda **kwargs: emitted.append(kwargs),
    )

    async def run():
        try:
            async for _ in wrapped.aiter_lines():
                pass
        except RuntimeError as exc:
            assert str(exc) == "stream failed"
        else:
            raise AssertionError("expected stream error")

    asyncio.run(run())
    assert len(emitted) == 1
    assert str(emitted[0]["error"]) == "stream failed"
