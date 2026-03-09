from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dify_client import AsyncClient, Client, models
from dify_client.models import ResponseMode
from respan_sdk.respan_types import RespanParams
from respan_sdk.utils.time import now_utc
from respan_exporter_dify.exporter import create_async_client, create_client
from respan_exporter_dify.gateway import RespanGatewayClient
from respan_exporter_dify.utils import export_dify_call


class AssistantMessage:
    def __init__(self, *, message_id: str, usage: dict, content: str):
        self.id = message_id
        self.type = "assistant"
        self.usage = usage
        self.content = content


class ResultMessage:
    def __init__(self, *, messages: list, usage: dict):
        self.messages = messages
        self.usage = usage


class FakeRequestsResponse:
    def __init__(self, *, json_data=None, lines=None, status_code=200, text=""):
        self._json_data = json_data or {}
        self._lines = lines or []
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.text or f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=True):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_create_client():
    client = Client(api_key="test-dify-key")
    respan_client = create_client(client=client, api_key="test-respan-key")
    assert respan_client.api_key == "test-respan-key"
    assert respan_client._client == client


def test_create_async_client():
    client = AsyncClient(api_key="test-dify-key")
    respan_client = create_async_client(client=client, api_key="test-respan-key")
    assert respan_client.api_key == "test-respan-key"
    assert respan_client._client == client


def test_create_client_with_dify_key():
    respan_client = create_client(dify_api_key="test-dify-key", api_key="test-respan-key")
    assert respan_client.api_key == "test-respan-key"
    assert respan_client._client.api_key == "test-dify-key"


def test_create_client_without_dify_key_uses_gateway_mode():
    respan_client = create_client(api_key="test-respan-key")
    assert respan_client.api_key == "test-respan-key"
    assert isinstance(respan_client._client, RespanGatewayClient)
    assert respan_client._client.api_key == "test-respan-key"


def test_sync_client_inherits_active_trace_context():
    fake_result = {"messages": [{"type": "assistant", "content": "hi"}]}
    mock_client = MagicMock(spec=Client)
    mock_client.chat_messages = MagicMock(return_value=fake_result)

    with patch(
        "respan_exporter_dify.exporter._get_active_trace_context",
        return_value=("a" * 32, "b" * 16),
    ):
        with patch("respan_exporter_dify.exporter.export_dify_call") as export_mock:
            wrapper = create_client(client=mock_client, api_key="respan-key")
            req = MagicMock()
            req.response_mode = None
            wrapper.chat_messages(req, respan_params=None)

    params = export_mock.call_args.kwargs["params"]
    assert params.trace_unique_id == "a" * 32
    assert params.span_parent_id == "b" * 16


def test_sync_client_preserves_explicit_trace_context():
    fake_result = {"messages": [{"type": "assistant", "content": "hi"}]}
    mock_client = MagicMock(spec=Client)
    mock_client.chat_messages = MagicMock(return_value=fake_result)
    explicit_params = RespanParams(trace_unique_id="c" * 32, span_parent_id="d" * 16)

    with patch(
        "respan_exporter_dify.exporter._get_active_trace_context",
        return_value=("a" * 32, "b" * 16),
    ):
        with patch("respan_exporter_dify.exporter.export_dify_call") as export_mock:
            wrapper = create_client(client=mock_client, api_key="respan-key")
            req = MagicMock()
            req.response_mode = None
            wrapper.chat_messages(req, respan_params=explicit_params)

    params = export_mock.call_args.kwargs["params"]
    assert params.trace_unique_id == "c" * 32
    assert params.span_parent_id == "d" * 16


def test_gateway_blocking_chat_uses_respan_key_only():
    req = models.ChatRequest(
        query="What is the capital of France?",
        inputs={},
        user="user-123",
        response_mode=ResponseMode.BLOCKING,
    )
    lines = [
        'data: {"id":"chatcmpl-gateway-1","created":1730000000,"model":"gpt-4o-mini","choices":[{"delta":{"content":"Par"}}]}',
        'data: {"id":"chatcmpl-gateway-1","created":1730000001,"model":"gpt-4o-mini","choices":[{"delta":{"content":"is"}}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}',
        "data: [DONE]",
    ]

    with patch("respan_exporter_dify.gateway.requests.post", return_value=FakeRequestsResponse(lines=lines)) as post_mock:
        with patch("respan_exporter_dify.exporter.export_dify_call") as export_mock:
            wrapper = create_client(
                api_key="respan-key",
                gateway_base_url="https://api.respan.ai/api",
                gateway_model="gpt-4o-mini",
            )
            result = wrapper.chat_messages(req, respan_params=None)

    assert result.answer == "Paris"
    assert result.message_id == "chatcmpl-gateway-1"
    assert post_mock.call_args.kwargs["headers"]["Authorization"] == "Bearer respan-key"
    assert post_mock.call_args.kwargs["json"]["model"] == "gpt-4o-mini"
    assert post_mock.call_args.kwargs["json"]["disable_log"] is True
    assert post_mock.call_args.kwargs["json"]["stream"] is True
    assert post_mock.call_args.kwargs["json"]["messages"][-1]["content"] == "What is the capital of France?"
    export_mock.assert_called_once()


def test_gateway_streaming_chat_maps_openai_stream_to_dify_events():
    req = models.ChatRequest(
        query="Say hello",
        inputs={},
        user="user-123",
        response_mode=ResponseMode.STREAMING,
    )
    lines = [
        'data: {"id":"chatcmpl-stream-1","created":1730000000,"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"id":"chatcmpl-stream-1","created":1730000001,"choices":[{"delta":{"content":"lo"}}]}',
        'data: {"id":"chatcmpl-stream-1","created":1730000002,"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}',
        "data: [DONE]",
    ]

    with patch("respan_exporter_dify.gateway.requests.post", return_value=FakeRequestsResponse(lines=lines)) as post_mock:
        with patch("respan_exporter_dify.exporter.export_dify_call") as export_mock:
            wrapper = create_client(api_key="respan-key")
            events = list(wrapper.chat_messages(req, respan_params=None))

    assert len(events) == 3
    assert events[0].event == models.StreamEvent.MESSAGE
    assert events[0].answer == "Hel"
    assert events[1].answer == "lo"
    assert events[2].event == models.StreamEvent.MESSAGE_END
    assert events[2].metadata.usage.total_tokens == 3
    assert post_mock.call_args.kwargs["json"]["stream"] is True
    export_mock.assert_called_once()


def test_build_export_payloads_prefers_message_usage_and_session():
    start_time = now_utc()
    end_time = now_utc()
    params = RespanParams()
    result = [
        {
            "event": "message",
            "conversation_id": "session-from-message",
            "usage": {"input_tokens": 11, "output_tokens": 5},
            "response": {"answer": "hello world"},
        }
    ]

    with patch("respan_exporter_dify.utils.send_payloads") as send_mock:
        export_dify_call(
            api_key="test-key",
            endpoint="https://test",
            timeout=10,
            method_name="chat_messages",
            start_time=start_time,
            end_time=end_time,
            status="success",
            kwargs={"req": {"conversation_id": "session-from-hook"}},
            result=result,
            error_message=None,
            params=params,
        )
        send_mock.assert_called_once()
        payloads = send_mock.call_args.kwargs["payloads"]

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["span_name"] == "dify.message"
    assert payload["session_identifier"] == "session-from-message"
    assert payload["prompt_tokens"] == 11
    assert payload["completion_tokens"] == 5
    assert payload["usage"]["prompt_tokens"] == 11
    assert payload["usage"]["completion_tokens"] == 5
    assert "total_request_tokens" not in payload
    assert "total_tokens" not in payload["usage"]
    assert "hello world" in payload["output"]
    assert "completion_message" not in payload
    assert "completion_messages" not in payload


def test_build_export_payloads_falls_back_to_result_usage_and_hook_session():
    start_time = now_utc()
    end_time = now_utc()
    params = RespanParams()
    result = {
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        "messages": [{"type": "assistant", "content": "final answer"}],
    }

    with patch("respan_exporter_dify.utils.send_payloads") as send_mock:
        export_dify_call(
            api_key="test-key",
            endpoint="https://test",
            timeout=10,
            method_name="chat_messages",
            start_time=start_time,
            end_time=end_time,
            status="success",
            kwargs={"req": {"conversation_id": "session-from-hook"}},
            result=result,
            error_message=None,
            params=params,
        )
        send_mock.assert_called_once()
        payloads = send_mock.call_args.kwargs["payloads"]

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["span_name"] == "dify.assistant"
    assert payload["session_identifier"] == "session-from-hook"
    assert payload["prompt_tokens"] == 2
    assert payload["completion_tokens"] == 3
    assert payload["total_request_tokens"] == 5
    assert payload["usage"]["prompt_tokens"] == 2
    assert payload["usage"]["completion_tokens"] == 3
    assert payload["usage"]["total_tokens"] == 5


def test_build_export_payloads_uses_assistant_message_usage_and_dedupes_by_id():
    start_time = now_utc()
    end_time = now_utc()
    params = RespanParams()
    result = ResultMessage(
        messages=[
            AssistantMessage(
                message_id="turn-1",
                usage={"input_tokens": 7, "output_tokens": 4, "cache_read_input_tokens": 2},
                content="first",
            ),
            AssistantMessage(
                message_id="turn-1",
                usage={"input_tokens": 99, "output_tokens": 99},
                content="duplicate same turn",
            ),
            AssistantMessage(
                message_id="turn-2",
                usage={"input_tokens": 3, "output_tokens": 1},
                content="second",
            ),
        ],
        usage={"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200},
    )

    with patch("respan_exporter_dify.utils.send_payloads") as send_mock:
        export_dify_call(
            api_key="test-key",
            endpoint="https://test",
            timeout=10,
            method_name="chat_messages",
            start_time=start_time,
            end_time=end_time,
            status="success",
            kwargs={"req": {"conversation_id": "session-from-hook"}},
            result=result,
            error_message=None,
            params=params,
        )
        send_mock.assert_called_once()
        payloads = send_mock.call_args.kwargs["payloads"]

    assert len(payloads) == 2

    first_payload = payloads[0]
    assert first_payload["span_name"] == "dify.assistant"
    assert first_payload["prompt_tokens"] == 7
    assert first_payload["completion_tokens"] == 4
    assert first_payload["usage"]["cache_read_input_tokens"] == 2
    assert "total_request_tokens" not in first_payload
    assert first_payload["metadata"]["message_id"] == "turn-1"

    second_payload = payloads[1]
    assert second_payload["prompt_tokens"] == 3
    assert second_payload["completion_tokens"] == 1
    assert "total_request_tokens" not in second_payload
    assert second_payload["metadata"]["message_id"] == "turn-2"


def test_build_export_payloads_generates_child_span_for_linked_trace():
    start_time = now_utc()
    end_time = now_utc()
    params = RespanParams(trace_unique_id="a" * 32, span_parent_id="b" * 16)
    result = ResultMessage(
        messages=[
            AssistantMessage(
                message_id="turn-1",
                usage={"input_tokens": 2, "output_tokens": 1},
                content="linked response",
            )
        ],
        usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    )

    with patch("respan_exporter_dify.utils.send_payloads") as send_mock:
        export_dify_call(
            api_key="test-key",
            endpoint="https://test",
            timeout=10,
            method_name="chat_messages",
            start_time=start_time,
            end_time=end_time,
            status="success",
            kwargs={"req": {"conversation_id": "session-from-hook"}},
            result=result,
            error_message=None,
            params=params,
        )
        payloads = send_mock.call_args.kwargs["payloads"]

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["trace_unique_id"] == "a" * 32
    assert payload["span_parent_id"] == "b" * 16
    assert payload["span_unique_id"] != "b" * 16
    assert len(payload["span_unique_id"]) == 16
    assert "trace_name" not in payload


# --- RespanAsyncDifyClient behavior ---


@pytest.mark.asyncio
async def test_async_client_non_streaming_calls_export_success():
    fake_result = {"messages": [{"role": "assistant", "content": "hi"}]}
    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.achat_messages = AsyncMock(return_value=fake_result)

    with patch("respan_exporter_dify.exporter.export_dify_call") as export_mock:
        wrapper = create_async_client(client=mock_client, api_key="respan-key")
        req = MagicMock()
        req.response_mode = None
        result = await wrapper.achat_messages(req, respan_params=None)

    assert result == fake_result
    export_mock.assert_called_once()
    call_kw = export_mock.call_args.kwargs
    assert call_kw["status"] == "success"
    assert call_kw["result"] == fake_result
    assert call_kw["error_message"] is None


# --- Streaming generator export ---


def test_sync_streaming_unconsumed_does_not_export():
    """Streaming export runs in generator finally; unconsumed stream → no export."""
    chunks = [{"event": "message"}]
    mock_client = MagicMock(spec=Client)
    mock_client.chat_messages = MagicMock(return_value=iter(chunks))

    with patch("respan_exporter_dify.exporter.export_dify_call") as export_mock:
        wrapper = create_client(client=mock_client, api_key="respan-key")
        req = MagicMock()
        req.response_mode = ResponseMode.STREAMING
        _ = wrapper.chat_messages(req, respan_params=None)
        # Never iterate the stream

    export_mock.assert_not_called()


def test_sync_streaming_exports_collected_events():
    chunks = [{"event": "message"}, {"event": "message_end"}]
    mock_client = MagicMock(spec=Client)
    mock_client.chat_messages = MagicMock(return_value=iter(chunks))

    with patch("respan_exporter_dify.exporter.export_dify_call") as export_mock:
        wrapper = create_client(client=mock_client, api_key="respan-key")
        req = MagicMock()
        req.response_mode = ResponseMode.STREAMING
        stream = wrapper.chat_messages(req, respan_params=None)
        collected = list(stream)

    assert collected == chunks
    export_mock.assert_called_once()
    call_kw = export_mock.call_args.kwargs
    assert call_kw["status"] == "success"
    assert call_kw["result"] == chunks
    assert call_kw["error_message"] is None


@pytest.mark.asyncio
async def test_async_streaming_unconsumed_does_not_export():
    """Streaming export runs in async generator finally; unconsumed stream → no export."""
    async def async_iter_chunks():
        yield {"event": "message"}

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.achat_messages = AsyncMock(return_value=async_iter_chunks())

    with patch("respan_exporter_dify.exporter.export_dify_call") as export_mock:
        wrapper = create_async_client(client=mock_client, api_key="respan-key")
        req = MagicMock()
        req.response_mode = ResponseMode.STREAMING
        _ = await wrapper.achat_messages(req, respan_params=None)
        # Never iterate the stream

    export_mock.assert_not_called()


@pytest.mark.asyncio
async def test_async_streaming_exports_collected_events():
    chunks = [{"event": "message"}, {"event": "message_end"}]

    async def async_iter_chunks():
        for c in chunks:
            yield c

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.achat_messages = AsyncMock(return_value=async_iter_chunks())

    with patch("respan_exporter_dify.exporter.export_dify_call") as export_mock:
        wrapper = create_async_client(client=mock_client, api_key="respan-key")
        req = MagicMock()
        req.response_mode = ResponseMode.STREAMING
        collected = []
        async for chunk in await wrapper.achat_messages(req, respan_params=None):
            collected.append(chunk)

    assert collected == chunks
    export_mock.assert_called_once()
    call_kw = export_mock.call_args.kwargs
    assert call_kw["status"] == "success"
    assert call_kw["result"] == chunks
    assert call_kw["error_message"] is None


# --- Error path: exception → _export(status="error") ---


def test_sync_error_calls_export_with_error_status():
    mock_client = MagicMock(spec=Client)
    mock_client.chat_messages = MagicMock(side_effect=ValueError("dify failed"))

    with patch("respan_exporter_dify.exporter.export_dify_call") as export_mock:
        wrapper = create_client(client=mock_client, api_key="respan-key")
        req = MagicMock()
        req.response_mode = None
        with pytest.raises(ValueError, match="dify failed"):
            wrapper.chat_messages(req, respan_params=None)

    export_mock.assert_called_once()
    call_kw = export_mock.call_args.kwargs
    assert call_kw["status"] == "error"
    assert call_kw["error_message"] == "dify failed"
    assert call_kw["result"] is None


@pytest.mark.asyncio
async def test_async_error_calls_export_with_error_status():
    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.achat_messages = AsyncMock(side_effect=RuntimeError("async boom"))

    with patch("respan_exporter_dify.exporter.export_dify_call") as export_mock:
        wrapper = create_async_client(client=mock_client, api_key="respan-key")
        req = MagicMock()
        req.response_mode = None
        with pytest.raises(RuntimeError, match="async boom"):
            await wrapper.achat_messages(req, respan_params=None)

    export_mock.assert_called_once()
    call_kw = export_mock.call_args.kwargs
    assert call_kw["status"] == "error"
    assert call_kw["error_message"] == "async boom"
    assert call_kw["result"] is None
