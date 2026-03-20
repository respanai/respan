"""Tests for smolagents exporter."""
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


class TestRespanSmolagentsExporterInit:
    """Test exporter initialization."""

    def test_default_endpoint(self):
        from respan_exporter_smolagents.exporter import RespanSmolagentsExporter

        exporter = RespanSmolagentsExporter(api_key="test")
        assert "v1/traces/ingest" in exporter.endpoint

    def test_custom_base_url(self):
        from respan_exporter_smolagents.exporter import RespanSmolagentsExporter

        exporter = RespanSmolagentsExporter(
            api_key="test",
            base_url="https://custom.api.com/api",
        )
        assert exporter.endpoint == "https://custom.api.com/api/v1/traces/ingest"

    def test_default_environment(self):
        from respan_exporter_smolagents.exporter import RespanSmolagentsExporter

        exporter = RespanSmolagentsExporter(api_key="test")
        assert exporter.environment == "production"


class TestBuildPayload:
    """Test payload construction from spans."""

    def _make_span_dict(self, **overrides):
        base = {
            "trace_id": "abcdef1234567890abcdef1234567890",
            "span_id": "1234567890abcdef",
            "parent_id": None,
            "name": "smolagents.agent.run",
            "span_type": "AGENT",
            "kind": "AGENT",
            "span_path": None,
            "start_time": datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "end_time": datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            "attributes": {},
            "status_code": 200,
            "error": None,
        }
        base.update(overrides)
        return base

    def test_builds_payload_from_single_span(self):
        from respan_exporter_smolagents.exporter import RespanSmolagentsExporter

        exporter = RespanSmolagentsExporter(api_key="test")
        span = self._make_span_dict()
        payloads = exporter.build_payload(trace_or_spans=[span])

        assert len(payloads) == 1
        p = payloads[0]
        assert p["trace_unique_id"] is not None
        assert p["span_unique_id"] is not None
        assert p["span_name"] == "smolagents.agent.run"
        assert p["environment"] == "production"

    def test_maps_agent_span(self):
        from respan_exporter_smolagents.exporter import RespanSmolagentsExporter

        exporter = RespanSmolagentsExporter(api_key="test")
        span = self._make_span_dict(span_type="AGENT", kind="AGENT")
        payloads = exporter.build_payload(trace_or_spans=[span])
        assert payloads[0]["log_type"] == "agent"

    def test_maps_tool_span(self):
        from respan_exporter_smolagents.exporter import RespanSmolagentsExporter

        exporter = RespanSmolagentsExporter(api_key="test")
        span = self._make_span_dict(
            span_type="TOOL",
            kind="TOOL",
            parent_id="abcdef1234567890",
            attributes={"tool.name": "web_search"},
        )
        payloads = exporter.build_payload(trace_or_spans=[span])
        assert payloads[0]["log_type"] == "tool"
        assert payloads[0].get("span_tools") == ["web_search"]

    def test_maps_llm_span(self):
        from respan_exporter_smolagents.exporter import RespanSmolagentsExporter

        exporter = RespanSmolagentsExporter(api_key="test")
        span = self._make_span_dict(
            span_type="LLM",
            kind="LLM",
            parent_id="abcdef1234567890",
            attributes={"llm.model_name": "gpt-4"},
        )
        payloads = exporter.build_payload(trace_or_spans=[span])
        assert payloads[0]["log_type"] == "generation"
        assert payloads[0]["model"] == "gpt-4"

    def test_extracts_input_output(self):
        from respan_exporter_smolagents.exporter import RespanSmolagentsExporter

        exporter = RespanSmolagentsExporter(api_key="test")
        span = self._make_span_dict(
            attributes={
                "input.value": "Search for AI news",
                "output.value": "Here are the results...",
            },
        )
        payloads = exporter.build_payload(trace_or_spans=[span])
        assert "Search for AI news" in payloads[0]["input"]
        assert "Here are the results" in payloads[0]["output"]

    def test_handles_error_span(self):
        from respan_exporter_smolagents.exporter import RespanSmolagentsExporter

        exporter = RespanSmolagentsExporter(api_key="test")
        span = self._make_span_dict(error="Tool execution failed", status_code=500)
        payloads = exporter.build_payload(trace_or_spans=[span])
        assert payloads[0]["error_message"] == "Tool execution failed"
        assert payloads[0]["status_code"] == 500

    def test_empty_spans_returns_empty(self):
        from respan_exporter_smolagents.exporter import RespanSmolagentsExporter

        exporter = RespanSmolagentsExporter(api_key="test")
        assert exporter.build_payload(trace_or_spans=[]) == []
        assert exporter.build_payload(trace_or_spans=None) == []

    def test_stores_smolagents_ids_in_metadata(self):
        from respan_exporter_smolagents.exporter import RespanSmolagentsExporter

        exporter = RespanSmolagentsExporter(api_key="test")
        span = self._make_span_dict()
        payloads = exporter.build_payload(trace_or_spans=[span])
        metadata = payloads[0].get("metadata", {})
        assert "smolagents_trace_id" in metadata
        assert "smolagents_span_id" in metadata

    def test_sets_customer_identifier(self):
        from respan_exporter_smolagents.exporter import RespanSmolagentsExporter

        exporter = RespanSmolagentsExporter(api_key="test", customer_identifier="user-42")
        span = self._make_span_dict()
        payloads = exporter.build_payload(trace_or_spans=[span])
        assert payloads[0]["customer_identifier"] == "user-42"


class TestExport:
    """Test export functionality."""

    def test_export_sends_payloads(self):
        from respan_exporter_smolagents.exporter import RespanSmolagentsExporter

        exporter = RespanSmolagentsExporter(api_key="test-key")
        span = {
            "trace_id": "abcdef1234567890abcdef1234567890",
            "span_id": "1234567890abcdef",
            "parent_id": None,
            "name": "Test",
            "span_type": "AGENT",
            "kind": "AGENT",
            "span_path": None,
            "start_time": datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "end_time": datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            "attributes": {},
            "status_code": 200,
            "error": None,
        }

        with patch("respan_exporter_smolagents.exporter.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            payloads = exporter.export(trace_or_spans=[span])
            assert len(payloads) > 0
            mock_post.assert_called_once()

    def test_export_skips_without_api_key(self):
        from respan_exporter_smolagents.exporter import RespanSmolagentsExporter

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RESPAN_API_KEY", None)
            exporter = RespanSmolagentsExporter(api_key=None)

            with patch("respan_exporter_smolagents.exporter.requests.post") as mock_post:
                exporter.export(trace_or_spans=[{
                    "trace_id": "abcdef1234567890abcdef1234567890",
                    "span_id": "1234567890abcdef",
                    "parent_id": None, "name": "T", "span_type": "AGENT",
                    "kind": "AGENT", "span_path": None,
                    "start_time": datetime(2024, 1, 1, tzinfo=timezone.utc),
                    "end_time": datetime(2024, 1, 1, tzinfo=timezone.utc),
                    "attributes": {}, "status_code": 200, "error": None,
                }])
                mock_post.assert_not_called()
