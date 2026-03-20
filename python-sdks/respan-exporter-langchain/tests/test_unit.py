"""Unit tests for the LangChain exporter - no external API calls needed."""

import json
import os
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from respan_exporter_langchain import RespanCallbackHandler, RespanLangchainExporter


class _InlineThread:
    """Replacement for threading.Thread that runs target inline on start()."""

    def __init__(self, target=None, args=None, kwargs=None, **_):
        self._target = target
        self._args = args or ()
        self._kwargs = kwargs or {}

    def start(self):
        if self._target:
            self._target(*self._args, **self._kwargs)

    def join(self, timeout=None):
        pass


def _patch_inline_thread():
    """Patch threading.Thread in the callback handler to run inline."""
    return patch(
        "respan_exporter_langchain.callback_handler.threading.Thread",
        side_effect=_InlineThread,
    )


class TestRespanLangchainExporter:
    """Tests for the exporter payload building."""

    def test_build_payload_basic(self):
        exporter = RespanLangchainExporter(api_key="test-key")
        spans = [
            {
                "span_id": "root-1",
                "parent_id": None,
                "name": "TestChain",
                "span_type": "workflow",
                "input": "hello",
                "output": "world",
                "start_time": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "end_time": datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                "metadata": {},
            },
        ]
        payloads = exporter.build_payload(spans=spans)
        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["trace_name"] == "TestChain"
        assert payload["log_type"] == "workflow"
        assert payload["span_name"] == "TestChain"
        assert payload["status_code"] == 200
        assert payload["latency"] == 1.0

    def test_build_payload_with_generation(self):
        exporter = RespanLangchainExporter(api_key="test-key")
        spans = [
            {
                "span_id": "root-1",
                "parent_id": None,
                "name": "RunnableSequence",
                "span_type": "workflow",
                "input": "hello",
                "output": None,
                "start_time": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "end_time": datetime(2025, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
                "metadata": {},
            },
            {
                "span_id": "llm-1",
                "parent_id": "root-1",
                "name": "ChatOpenAI",
                "span_type": "generation",
                "model": "gpt-4o-mini",
                "input": "hello",
                "output": "Hi there!",
                "start_time": datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                "end_time": datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_messages": [{"role": "user", "content": "hello"}],
                "completion_message": {"role": "assistant", "content": "Hi there!"},
                "metadata": {},
            },
        ]
        payloads = exporter.build_payload(spans=spans)
        assert len(payloads) == 2

        workflow_payload = payloads[0]
        llm_payload = payloads[1]

        assert workflow_payload["log_type"] == "workflow"
        # Output should be propagated from generation to workflow
        assert workflow_payload["output"] is not None

        assert llm_payload["log_type"] == "generation"
        assert llm_payload["model"] == "gpt-4o-mini"
        assert llm_payload["prompt_tokens"] == 10
        assert llm_payload["completion_tokens"] == 5
        assert llm_payload["total_request_tokens"] == 15

    def test_build_payload_error_span(self):
        exporter = RespanLangchainExporter(api_key="test-key")
        spans = [
            {
                "span_id": "root-1",
                "parent_id": None,
                "name": "TestChain",
                "span_type": "workflow",
                "input": "hello",
                "output": None,
                "error": "Something went wrong",
                "start_time": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "end_time": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "metadata": {},
            },
        ]
        payloads = exporter.build_payload(spans=spans)
        assert len(payloads) == 1
        assert payloads[0]["status_code"] == 500
        assert payloads[0]["error_message"] == "Something went wrong"

    def test_endpoint_building(self):
        e1 = RespanLangchainExporter(base_url="https://example.com/api")
        assert e1.endpoint == "https://example.com/api/v1/traces/ingest"

        e2 = RespanLangchainExporter(base_url="https://example.com/api/v1/traces/ingest")
        assert e2.endpoint == "https://example.com/api/v1/traces/ingest"

        e3 = RespanLangchainExporter(base_url="https://example.com")
        assert e3.endpoint == "https://example.com/api/v1/traces/ingest"

    def test_id_normalization(self):
        exporter = RespanLangchainExporter(api_key="test-key")
        spans = [
            {
                "span_id": "some-uuid-string",
                "parent_id": None,
                "name": "Test",
                "span_type": "workflow",
                "input": "x",
                "start_time": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "end_time": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "metadata": {},
            },
        ]
        payloads = exporter.build_payload(spans=spans)
        # IDs should be normalized to hex strings
        assert len(payloads[0]["trace_unique_id"]) == 32
        assert len(payloads[0]["span_unique_id"]) == 16


class TestRespanCallbackHandler:
    """Tests for the callback handler."""

    @patch("respan_exporter_langchain.exporter.requests.post")
    def test_chain_lifecycle(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        with _patch_inline_thread():
            handler = RespanCallbackHandler(api_key="test-key")

            root_id = uuid.uuid4()
            handler.on_chain_start(
                serialized={"name": "TestChain", "id": ["langchain", "chains", "TestChain"]},
                inputs={"input": "hello"},
                run_id=root_id,
            )

            handler.on_chain_end(
                outputs={"output": "world"},
                run_id=root_id,
            )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        payloads = call_args.kwargs.get("json") or call_args[1].get("json")
        assert len(payloads) == 1
        assert payloads[0]["log_type"] == "workflow"
        assert payloads[0]["span_name"] == "TestChain"

    @patch("respan_exporter_langchain.exporter.requests.post")
    def test_llm_span(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        with _patch_inline_thread():
            handler = RespanCallbackHandler(api_key="test-key")

            root_id = uuid.uuid4()
            llm_id = uuid.uuid4()

            handler.on_chain_start(
                serialized={"name": "RunnableSequence", "id": ["langchain", "RunnableSequence"]},
                inputs={"input": "hello"},
                run_id=root_id,
            )

            handler.on_llm_start(
                serialized={"name": "ChatOpenAI", "id": ["langchain", "ChatOpenAI"], "kwargs": {"model_name": "gpt-4o-mini"}},
                prompts=["hello"],
                run_id=llm_id,
                parent_run_id=root_id,
            )

            # Simulate LLMResult
            from langchain_core.outputs import LLMResult, Generation
            result = LLMResult(
                generations=[[Generation(text="Hi there!")]],
                llm_output={"token_usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
            )
            handler.on_llm_end(response=result, run_id=llm_id, parent_run_id=root_id)
            handler.on_chain_end(outputs={"output": "Hi there!"}, run_id=root_id)

        mock_post.assert_called_once()
        payloads = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert len(payloads) == 2

        # Find generation payload
        gen_payload = next(p for p in payloads if p["log_type"] == "generation")
        assert gen_payload["model"] == "gpt-4o-mini"
        assert gen_payload["prompt_tokens"] == 5
        assert gen_payload["completion_tokens"] == 3

    @patch("respan_exporter_langchain.exporter.requests.post")
    def test_tool_span(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        with _patch_inline_thread():
            handler = RespanCallbackHandler(api_key="test-key")

            root_id = uuid.uuid4()
            tool_id = uuid.uuid4()

            handler.on_chain_start(
                serialized={"name": "AgentExecutor", "id": ["langchain", "agents", "AgentExecutor"]},
                inputs={"input": "what's the weather?"},
                run_id=root_id,
            )

            handler.on_tool_start(
                serialized={"name": "weather_tool"},
                input_str='{"city": "SF"}',
                run_id=tool_id,
                parent_run_id=root_id,
            )

            handler.on_tool_end(output="Sunny, 72F", run_id=tool_id, parent_run_id=root_id)
            handler.on_chain_end(outputs={"output": "It's sunny"}, run_id=root_id)

        mock_post.assert_called_once()
        payloads = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")

        tool_payload = next(p for p in payloads if p["log_type"] == "tool")
        assert tool_payload["span_name"] == "weather_tool"

    @patch("respan_exporter_langchain.exporter.requests.post")
    def test_error_handling(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        with _patch_inline_thread():
            handler = RespanCallbackHandler(api_key="test-key")

            root_id = uuid.uuid4()
            handler.on_chain_start(
                serialized={"name": "TestChain"},
                inputs={"input": "hello"},
                run_id=root_id,
            )
            handler.on_chain_error(
                error=ValueError("test error"),
                run_id=root_id,
            )

        mock_post.assert_called_once()
        payloads = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payloads[0]["status_code"] == 500
        assert "test error" in payloads[0]["error_message"]

    @patch("respan_exporter_langchain.exporter.requests.post")
    def test_retriever_span(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        with _patch_inline_thread():
            handler = RespanCallbackHandler(api_key="test-key")

            root_id = uuid.uuid4()
            retriever_id = uuid.uuid4()

            handler.on_chain_start(
                serialized={"name": "RAGChain", "id": ["langchain", "RAGChain"]},
                inputs={"input": "what is respan?"},
                run_id=root_id,
            )
            handler.on_retriever_start(
                serialized={"name": "VectorStoreRetriever"},
                query="what is respan?",
                run_id=retriever_id,
                parent_run_id=root_id,
            )

            # Simulate Document objects
            class FakeDoc:
                def __init__(self, content, metadata):
                    self.page_content = content
                    self.metadata = metadata

            handler.on_retriever_end(
                documents=[FakeDoc("Respan is an observability platform", {"source": "docs"})],
                run_id=retriever_id,
                parent_run_id=root_id,
            )
            handler.on_chain_end(outputs={"output": "Respan is..."}, run_id=root_id)

        mock_post.assert_called_once()
        payloads = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")

        retriever_payload = next(p for p in payloads if p["span_name"] == "VectorStoreRetriever")
        assert retriever_payload["log_type"] == "tool"

    @patch("respan_exporter_langchain.exporter.requests.post")
    def test_tool_span_has_span_tools(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        with _patch_inline_thread():
            handler = RespanCallbackHandler(api_key="test-key")

            root_id = uuid.uuid4()
            tool_id = uuid.uuid4()

            handler.on_chain_start(
                serialized={"name": "Chain"},
                inputs={"input": "test"},
                run_id=root_id,
            )
            handler.on_tool_start(
                serialized={"name": "calculator"},
                input_str="2+2",
                run_id=tool_id,
                parent_run_id=root_id,
            )
            handler.on_tool_end(output="4", run_id=tool_id, parent_run_id=root_id)
            handler.on_chain_end(outputs={"output": "4"}, run_id=root_id)

        payloads = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        tool_payload = next(p for p in payloads if p["log_type"] == "tool")
        assert tool_payload["span_tools"] == ["calculator"]

    @patch("respan_exporter_langchain.exporter.requests.post")
    def test_token_usage_from_usage_metadata(self, mock_post):
        """Token usage extracted from AIMessage.usage_metadata should populate payload."""
        mock_post.return_value = MagicMock(status_code=200)
        with _patch_inline_thread():
            handler = RespanCallbackHandler(api_key="test-key")

            root_id = uuid.uuid4()
            llm_id = uuid.uuid4()

            handler.on_chain_start(
                serialized={"name": "Chain", "id": ["langchain", "Chain"]},
                inputs={"input": "hi"},
                run_id=root_id,
            )
            handler.on_chat_model_start(
                serialized={"name": "ChatOpenAI", "id": ["langchain", "ChatOpenAI"], "kwargs": {"model_name": "gpt-4o"}},
                messages=[[MagicMock(type="human", content="hi")]],
                run_id=llm_id,
                parent_run_id=root_id,
            )

            # Build an LLMResult whose AIMessage carries usage_metadata
            from langchain_core.outputs import LLMResult, ChatGeneration
            from langchain_core.messages import AIMessage

            ai_msg = AIMessage(content="hello back")
            ai_msg.usage_metadata = {
                "input_tokens": 12,
                "output_tokens": 8,
                "total_tokens": 20,
            }
            result = LLMResult(
                generations=[[ChatGeneration(message=ai_msg)]],
                llm_output={},  # no token_usage here
            )
            handler.on_llm_end(response=result, run_id=llm_id, parent_run_id=root_id)
            handler.on_chain_end(outputs={"output": "hello back"}, run_id=root_id)

        mock_post.assert_called_once()
        payloads = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        gen_payload = next(p for p in payloads if p["log_type"] == "generation")
        assert gen_payload["prompt_tokens"] == 12
        assert gen_payload["completion_tokens"] == 8
        assert gen_payload["total_request_tokens"] == 20

    @patch.dict(os.environ, {}, clear=False)
    @patch("respan_exporter_langchain.exporter.requests.post")
    def test_no_api_key_skips_export(self, mock_post):
        # Ensure RESPAN_API_KEY is not set so the handler truly has no key
        env = os.environ.copy()
        env.pop("RESPAN_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with _patch_inline_thread():
                handler = RespanCallbackHandler()  # no api key
                root_id = uuid.uuid4()
                handler.on_chain_start(
                    serialized={"name": "Test"},
                    inputs={"input": "hello"},
                    run_id=root_id,
                )
                handler.on_chain_end(outputs={"output": "world"}, run_id=root_id)
            # Should not have made any network call
            mock_post.assert_not_called()
