from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dify_client import AsyncClient, Client
from dify_client.models import ResponseMode
from respan_sdk.respan_types import RespanParams
from respan_sdk.utils.time import now_utc
from respan_exporter_dify.exporter import create_async_client, create_client
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
