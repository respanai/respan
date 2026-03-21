"""Unit tests for Google ADK → Respan span conversion.

Uses a lightweight MockSpan that mimics OTel ReadableSpan — no dependency
on google-adk or a running OTel pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.trace import StatusCode

# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------

@dataclass
class MockContext:
    trace_id: int = 0xABCDEF1234567890ABCDEF1234567890
    span_id: int = 0x1234567890ABCDEF


@dataclass
class MockParent:
    span_id: int = 0xFEDCBA0987654321


@dataclass
class MockStatus:
    status_code: StatusCode = StatusCode.OK
    description: Optional[str] = None


@dataclass
class MockSpan:
    name: str = "test_span"
    context: MockContext = field(default_factory=MockContext)
    parent: Optional[MockParent] = field(default_factory=MockParent)
    start_time: int = 1_700_000_000_000_000_000  # ~2023-11-14 in ns
    end_time: int = 1_700_000_001_500_000_000    # 1.5s later
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: MockStatus = field(default_factory=MockStatus)


# ---------------------------------------------------------------------------
# Import the code under test
# ---------------------------------------------------------------------------

from respan_exporter_google_adk.respan_google_adk_exporter import (
    convert_span_to_respan_log,
)
from respan_exporter_google_adk.utils import (
    build_metadata,
    coerce_int,
    extract_span_type,
    extract_tool_calls_from_parts,
    gemini_request_to_input_text,
    gemini_request_to_prompt_messages,
    gemini_response_to_completion_message,
    message_to_text,
    messages_to_text,
    safe_json_parse,
    serialize,
)


# ===================================================================
# Span type routing
# ===================================================================

class TestExtractSpanType:
    def test_operation_name_invoke_agent(self):
        span = MockSpan(attributes={"gen_ai.operation_name": "invoke_agent"})
        assert extract_span_type(span) == "invoke_agent"

    def test_operation_name_generate_content(self):
        span = MockSpan(attributes={"gen_ai.operation_name": "generate_content"})
        assert extract_span_type(span) == "generate_content"

    def test_operation_name_execute_tool(self):
        span = MockSpan(attributes={"gen_ai.operation_name": "execute_tool"})
        assert extract_span_type(span) == "execute_tool"

    def test_name_prefix_invocation(self):
        span = MockSpan(name="invocation [agent_name]")
        assert extract_span_type(span) == "invocation"

    def test_name_prefix_call_llm(self):
        span = MockSpan(name="call_llm gemini-2.0-flash")
        assert extract_span_type(span) == "call_llm"

    def test_name_prefix_send_data(self):
        span = MockSpan(name="send_data")
        assert extract_span_type(span) == "send_data"

    def test_name_prefix_handle_context_caching(self):
        span = MockSpan(name="handle_context_caching")
        assert extract_span_type(span) == "handle_context_caching"

    def test_name_prefix_create_cache(self):
        span = MockSpan(name="create_cache")
        assert extract_span_type(span) == "create_cache"

    def test_unknown_span(self):
        span = MockSpan(name="something_else")
        assert extract_span_type(span) == "unknown"


# ===================================================================
# Per-type field mapping
# ===================================================================

class TestInvocationSpan:
    def test_basic_fields(self):
        span = MockSpan(name="invocation [weather_agent]")
        result = convert_span_to_respan_log(span)
        assert result is not None
        assert result["log_type"] == "agent"
        assert result["span_name"] == "invocation"
        assert result["error_bit"] == 0
        assert result["status_code"] == 200


class TestInvokeAgentSpan:
    def test_agent_fields(self):
        span = MockSpan(
            name="invoke_agent weather_agent",
            attributes={
                "gen_ai.operation_name": "invoke_agent",
                "gen_ai.agent.name": "weather_agent",
                "gen_ai.agent.description": "Gets weather info",
                "gen_ai.agent.version": "1.0",
                "gen_ai.conversation.id": "conv-123",
                "gcp.vertex.agent.session_id": "sess-456",
                "gcp.vertex.agent.invocation_id": "inv-789",
            },
        )
        result = convert_span_to_respan_log(span)
        assert result is not None
        assert result["log_type"] == "agent"
        assert result["span_name"] == "invoke_agent weather_agent"
        assert result["span_workflow_name"] == "weather_agent"
        assert result["session_identifier"] == "sess-456"
        meta = result["metadata"]
        assert meta["agent_name"] == "weather_agent"
        assert meta["agent_description"] == "Gets weather info"
        assert meta["agent_version"] == "1.0"
        assert meta["conversation_id"] == "conv-123"
        assert meta["session_id"] == "sess-456"
        assert meta["invocation_id"] == "inv-789"


class TestGenerateContentSpan:
    def _make_span(self, **extra_attrs):
        llm_request = {
            "config": {
                "systemInstruction": {
                    "parts": [{"text": "You are a helpful assistant."}],
                },
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "What's the weather?"}],
                },
            ],
        }
        llm_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "It's sunny!"},
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "NYC"},
                                },
                            },
                        ],
                    },
                },
            ],
        }
        attrs = {
            "gen_ai.operation_name": "generate_content",
            "gen_ai.request.model": "gemini-2.0-flash",
            "gen_ai.usage.input_tokens": "100",
            "gen_ai.usage.output_tokens": "50",
            "gen_ai.request.max_tokens": "1024",
            "gen_ai.system": "vertex_ai",
            "gen_ai.response.finish_reasons": '["STOP"]',
            "gen_ai.usage.experimental.reasoning_tokens": "10",
            "gen_ai.usage.experimental.reasoning_tokens_limit": "500",
            "gen_ai.usage.experimental.system_instruction_tokens": "20",
            "gen_ai.request.top_p": 0.95,
            "gcp.vertex.agent.llm_request": json.dumps(llm_request),
            "gcp.vertex.agent.llm_response": json.dumps(llm_response),
            "gen_ai.conversation.id": "conv-1",
            "gcp.vertex.agent.session_id": "sess-1",
            "gcp.vertex.agent.invocation_id": "inv-1",
            "gcp.vertex.agent.event_id": "evt-1",
            **extra_attrs,
        }
        return MockSpan(name="generate_content", attributes=attrs)

    def test_model_and_tokens(self):
        span = self._make_span()
        result = convert_span_to_respan_log(span)
        assert result["log_type"] == "generation"
        assert result["model"] == "gemini-2.0-flash"
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_request_tokens"] == 150
        assert result["max_tokens"] == 1024

    def test_prompt_messages(self):
        span = self._make_span()
        result = convert_span_to_respan_log(span)
        msgs = result["prompt_messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "helpful assistant" in msgs[0]["content"]
        assert msgs[1]["role"] == "user"
        assert result["input"] == "What's the weather?"
        assert result["full_request"]["config"]["systemInstruction"]["parts"][0]["text"] == "You are a helpful assistant."
        assert "contents" in result["full_request"]

    def test_completion_message_and_tool_calls(self):
        span = self._make_span()
        result = convert_span_to_respan_log(span)
        comp = result["completion_message"]
        assert comp["role"] == "assistant"
        assert "sunny" in comp["content"]
        assert len(comp["tool_calls"]) == 1
        assert comp["tool_calls"][0]["function"]["name"] == "get_weather"
        # tool_calls also on top level
        assert result["tool_calls"] is not None
        assert result["output"] == "It's sunny!"
        assert result["full_response"]["candidates"][0]["content"]["parts"][0]["text"] == "It's sunny!"

    def test_metadata(self):
        span = self._make_span()
        result = convert_span_to_respan_log(span)
        meta = result["metadata"]
        assert meta["gen_ai_system"] == "vertex_ai"
        assert meta["reasoning_tokens"] == 10
        assert meta["reasoning_tokens_limit"] == 500
        assert meta["system_instruction_tokens"] == 20
        assert meta["top_p"] == 0.95
        assert meta["conversation_id"] == "conv-1"
        assert meta["session_id"] == "sess-1"
        assert meta["event_id"] == "evt-1"


class TestCallLLMSpan:
    def test_same_as_generate_content(self):
        span = MockSpan(
            name="call_llm gemini-2.0-flash",
            attributes={
                "gen_ai.request.model": "gemini-2.0-flash",
                "gen_ai.usage.input_tokens": "200",
                "gen_ai.usage.output_tokens": "80",
            },
        )
        result = convert_span_to_respan_log(span)
        assert result["log_type"] == "generation"
        assert result["model"] == "gemini-2.0-flash"
        assert result["prompt_tokens"] == 200
        assert result["completion_tokens"] == 80
        assert result["total_request_tokens"] == 280


class TestExecuteToolSpan:
    def test_tool_fields(self):
        tool_args = {"location": "San Francisco"}
        tool_resp = {"temperature": 72, "unit": "F"}
        span = MockSpan(
            name="execute_tool get_weather",
            attributes={
                "gen_ai.operation_name": "execute_tool",
                "gen_ai.tool.name": "get_weather",
                "gen_ai.tool.description": "Fetches weather data",
                "gen_ai.tool.type": "function",
                "gen_ai.tool.call_id": "call-123",
                "gcp.vertex.agent.tool_call_args": json.dumps(tool_args),
                "gcp.vertex.agent.tool_response": json.dumps(tool_resp),
                "gcp.vertex.agent.event_id": "evt-5",
            },
        )
        result = convert_span_to_respan_log(span)
        assert result["log_type"] == "tool"
        assert result["span_tools"] == ["get_weather"]
        # input/output are serialized
        assert result["input"] is not None
        assert result["output"] is not None
        meta = result["metadata"]
        assert meta["tool_name"] == "get_weather"
        assert meta["tool_description"] == "Fetches weather data"
        assert meta["tool_type"] == "function"
        assert meta["tool_call_id"] == "call-123"
        assert meta["event_id"] == "evt-5"


class TestExecuteToolMergedSpan:
    def test_merged_tools(self):
        span = MockSpan(
            name="execute_tool (merged)",
            attributes={
                "gen_ai.operation_name": "execute_tool",
                "gen_ai.tool.name": "(merged tools)",
                "gcp.vertex.agent.tool_call_args": '{"a": 1}',
                "gcp.vertex.agent.tool_response": '{"b": 2}',
            },
        )
        result = convert_span_to_respan_log(span)
        assert result["log_type"] == "tool"
        assert result["span_tools"] == ["(merged tools)"]


class TestSendDataSpan:
    def test_send_data(self):
        span = MockSpan(
            name="send_data",
            attributes={
                "gcp.vertex.agent.data": '{"key": "value"}',
                "gcp.vertex.agent.invocation_id": "inv-1",
                "gcp.vertex.agent.event_id": "evt-1",
            },
        )
        result = convert_span_to_respan_log(span)
        assert result["log_type"] == "custom"
        assert result["span_name"] == "send_data"
        meta = result["metadata"]
        assert meta["data"] == {"key": "value"}
        assert meta["invocation_id"] == "inv-1"


class TestCreateCacheSpan:
    def test_cache_metadata(self):
        span = MockSpan(
            name="create_cache",
            attributes={
                "cache_contents_count": 5,
                "model": "gemini-2.0-flash",
                "ttl_seconds": 3600,
                "cache_name": "my-cache",
            },
        )
        result = convert_span_to_respan_log(span)
        assert result["log_type"] == "custom"
        assert result["span_name"] == "create_cache"
        meta = result["metadata"]
        assert meta["cache_contents_count"] == 5
        assert meta["model"] == "gemini-2.0-flash"
        assert meta["ttl_seconds"] == 3600
        assert meta["cache_name"] == "my-cache"


class TestUnknownSpan:
    def test_unknown_fallback(self):
        span = MockSpan(name="some_custom_thing")
        result = convert_span_to_respan_log(span)
        assert result["log_type"] == "custom"
        assert result["span_name"] == "some_custom_thing"


# ===================================================================
# Error handling
# ===================================================================

class TestErrorHandling:
    def test_error_span_status(self):
        span = MockSpan(
            name="generate_content",
            attributes={"gen_ai.operation_name": "generate_content"},
            status=MockStatus(status_code=StatusCode.ERROR, description="Rate limited"),
        )
        result = convert_span_to_respan_log(span)
        assert result["error_bit"] == 1
        assert result["status_code"] == 400
        assert result["error_message"] == "Rate limited"

    def test_error_span_attribute(self):
        span = MockSpan(
            name="execute_tool fail",
            attributes={
                "gen_ai.operation_name": "execute_tool",
                "error.type": "TimeoutError",
            },
        )
        result = convert_span_to_respan_log(span)
        assert result["error_bit"] == 1
        assert result["status_code"] == 400
        assert result["error_message"] == "TimeoutError"

    def test_missing_attributes_graceful(self):
        """Span with no attributes should still convert without crashing."""
        span = MockSpan(name="invocation [agent]", attributes={})
        result = convert_span_to_respan_log(span)
        assert result is not None
        assert result["error_bit"] == 0

    def test_none_context_returns_none(self):
        span = MockSpan()
        span.context = None
        result = convert_span_to_respan_log(span)
        assert result is None


# ===================================================================
# Gemini message conversion
# ===================================================================

class TestGeminiRequestConversion:
    def test_with_system_instruction(self):
        req = {
            "config": {
                "systemInstruction": {
                    "parts": [{"text": "Be helpful."}],
                },
            },
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
            ],
        }
        msgs = gemini_request_to_prompt_messages(req)
        assert msgs[0] == {"role": "system", "content": "Be helpful."}
        assert msgs[1] == {"role": "user", "content": "Hello"}

    def test_with_function_calls(self):
        req = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "test"}}},
                    ],
                },
            ],
        }
        msgs = gemini_request_to_prompt_messages(req)
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "search"

    def test_none_input(self):
        assert gemini_request_to_prompt_messages(None) is None

    def test_empty_contents(self):
        assert gemini_request_to_prompt_messages({"contents": []}) is None

    def test_input_text_only_includes_user_messages(self):
        req = {
            "config": {
                "systemInstruction": {
                    "parts": [{"text": "Be helpful."}],
                },
            },
            "contents": [
                {"role": "model", "parts": [{"text": "Previous answer"}]},
                {"role": "user", "parts": [{"text": "Hello"}]},
            ],
        }
        assert gemini_request_to_input_text(req) == "Hello"


class TestGeminiResponseConversion:
    def test_text_response(self):
        resp = {
            "candidates": [
                {"content": {"parts": [{"text": "Hello!"}]}},
            ],
        }
        msg = gemini_response_to_completion_message(resp)
        assert msg == {"role": "assistant", "content": "Hello!"}

    def test_with_tool_calls(self):
        resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "calc", "args": {"x": 1}}},
                        ],
                    },
                },
            ],
        }
        msg = gemini_response_to_completion_message(resp)
        assert msg["tool_calls"][0]["function"]["name"] == "calc"

    def test_adk_llm_response_shape(self):
        resp = {
            "content": {
                "role": "model",
                "parts": [{"text": "Hello from ADK"}],
            },
            "partial": False,
        }
        msg = gemini_response_to_completion_message(resp)
        assert msg == {"role": "assistant", "content": "Hello from ADK"}

    def test_empty_candidates(self):
        assert gemini_response_to_completion_message({"candidates": []}) is None

    def test_none_input(self):
        assert gemini_response_to_completion_message(None) is None


class TestMessageTextConversion:
    def test_messages_to_text(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
        assert messages_to_text(messages) == "Be helpful.\nHello"

    def test_message_to_text(self):
        message = {"role": "assistant", "content": "Done"}
        assert message_to_text(message) == "Done"


class TestExtractToolCalls:
    def test_function_call_parts(self):
        parts = [
            {"functionCall": {"name": "f1", "args": {"a": 1}}},
            {"text": "hello"},
            {"functionCall": {"name": "f2", "args": {}}},
        ]
        result = extract_tool_calls_from_parts(parts)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "f1"
        assert result[1]["function"]["name"] == "f2"

    def test_no_function_calls(self):
        parts = [{"text": "just text"}]
        assert extract_tool_calls_from_parts(parts) is None

    def test_empty_parts(self):
        assert extract_tool_calls_from_parts([]) is None


# ===================================================================
# Utility functions
# ===================================================================

class TestUtilities:
    def test_coerce_int(self):
        assert coerce_int("42") == 42
        assert coerce_int(42) == 42
        assert coerce_int(None) is None
        assert coerce_int("not_a_number") is None

    def test_safe_json_parse(self):
        assert safe_json_parse('{"a": 1}') == {"a": 1}
        assert safe_json_parse("not json") == "not json"
        assert safe_json_parse(None) is None
        assert safe_json_parse({"already": "dict"}) == {"already": "dict"}

    def test_serialize(self):
        assert serialize(None) is None
        assert serialize("hello") == "hello"
        assert serialize(42) == 42
        assert serialize({"a": [1, 2]}) == {"a": [1, 2]}


# ===================================================================
# Base field mapping
# ===================================================================

class TestBaseFields:
    def test_trace_and_span_ids(self):
        span = MockSpan(name="invocation [test]")
        result = convert_span_to_respan_log(span)
        assert result["trace_unique_id"] == format(0xABCDEF1234567890ABCDEF1234567890, "032x")
        assert result["span_unique_id"] == format(0x1234567890ABCDEF, "016x")
        assert result["span_parent_id"] == format(0xFEDCBA0987654321, "016x")

    def test_no_parent_falls_back_to_trace_id(self):
        span = MockSpan(name="invocation [test]", parent=None)
        result = convert_span_to_respan_log(span)
        assert result["span_parent_id"] == result["trace_unique_id"]

    def test_customer_and_session(self):
        span = MockSpan(
            name="invocation [test]",
            attributes={
                "user.id": "user-42",
                "gcp.vertex.agent.session_id": "sess-99",
            },
        )
        result = convert_span_to_respan_log(span)
        assert result["customer_identifier"] == "user-42"
        assert result["session_identifier"] == "sess-99"

    def test_latency(self):
        span = MockSpan(name="invocation [test]")
        result = convert_span_to_respan_log(span)
        assert abs(result["latency"] - 1.5) < 0.01


# ===================================================================
# Exporter HTTP
# ===================================================================

class TestExporterHTTP:
    def test_export_posts_payload(self):
        from respan_exporter_google_adk.respan_google_adk_exporter import (
            RespanGoogleADKExporter,
        )
        from opentelemetry.sdk.trace.export import SpanExportResult

        exporter = RespanGoogleADKExporter(api_key="test-key")

        span = MockSpan(
            name="invocation [test]",
            attributes={},
        )

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(exporter._client, "post", return_value=mock_response) as mock_post:
            result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer test-key"
        payload = call_kwargs.kwargs["json"]
        assert "data" in payload
        assert len(payload["data"]) == 1
        assert payload["data"][0]["log_type"] == "agent"

    def test_export_applies_customer_defaults(self):
        from respan_exporter_google_adk.respan_google_adk_exporter import (
            RespanGoogleADKExporter,
        )

        exporter = RespanGoogleADKExporter(
            api_key="test-key",
            customer_identifier="cust-1",
            customer_name="Test Customer",
            customer_email="test@example.com",
        )

        span = MockSpan(name="invocation [test]", attributes={})

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(exporter._client, "post", return_value=mock_response) as mock_post:
            exporter.export([span])

        payload = mock_post.call_args.kwargs["json"]
        log = payload["data"][0]
        assert log["customer_identifier"] == "cust-1"
        assert log["customer_name"] == "Test Customer"
        assert log["customer_email"] == "test@example.com"

    def test_export_no_api_key(self):
        from respan_exporter_google_adk.respan_google_adk_exporter import (
            RespanGoogleADKExporter,
        )
        from opentelemetry.sdk.trace.export import SpanExportResult

        exporter = RespanGoogleADKExporter(api_key=None)
        # Ensure env var is not set
        with patch.dict("os.environ", {}, clear=True):
            exporter.api_key = None
            result = exporter.export([MockSpan()])
        assert result == SpanExportResult.FAILURE
