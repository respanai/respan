"""Tests for CrewAI exporter."""
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestRespanCrewAIExporterInit:
    """Test exporter initialization."""

    def test_uses_env_api_key(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        with patch.dict(os.environ, {"RESPAN_API_KEY": "test-key-123"}):
            exporter = RespanCrewAIExporter()
            assert exporter.api_key == "test-key-123"

    def test_explicit_api_key_overrides_env(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        with patch.dict(os.environ, {"RESPAN_API_KEY": "env-key"}):
            exporter = RespanCrewAIExporter(api_key="explicit-key")
            assert exporter.api_key == "explicit-key"

    def test_default_endpoint(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        assert "v1/traces/ingest" in exporter.endpoint

    def test_custom_base_url(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(
            api_key="test",
            base_url="https://custom.api.com/api",
        )
        assert exporter.endpoint == "https://custom.api.com/api/v1/traces/ingest"

    def test_default_environment(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        assert exporter.environment == "production"

    def test_custom_environment(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test", environment="staging")
        assert exporter.environment == "staging"


class TestBuildPayload:
    """Test payload construction from spans."""

    def _make_span_dict(self, **overrides):
        """Create a minimal span dict for testing."""
        base = {
            "trace_id": "abcdef1234567890abcdef1234567890",
            "span_id": "1234567890abcdef",
            "parent_id": None,
            "name": "CrewAI Crew",
            "span_type": "CHAIN",
            "kind": "CHAIN",
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
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        span = self._make_span_dict()
        payloads = exporter.build_payload(trace_or_spans=[span])

        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["trace_unique_id"] is not None
        assert payload["span_unique_id"] is not None
        assert payload["span_name"] == "CrewAI Crew"
        assert payload["log_type"] == "workflow"  # root span with no parent
        assert payload["environment"] == "production"

    def test_builds_payload_with_generation_span(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        span = self._make_span_dict(
            name="LLM Call",
            span_type="LLM",
            kind="LLM",
            parent_id="abcdef1234567890",
            attributes={
                "llm.model_name": "gpt-4",
                "llm.token_count.prompt": 100,
                "llm.token_count.completion": 50,
                "llm.token_count.total": 150,
            },
        )
        payloads = exporter.build_payload(trace_or_spans=[span])

        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["log_type"] == "generation"
        assert payload["model"] == "gpt-4"
        assert payload["prompt_tokens"] == 100
        assert payload["completion_tokens"] == 50
        assert payload["total_request_tokens"] == 150

    def test_builds_payload_with_agent_span(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        span = self._make_span_dict(
            name="Research Agent",
            span_type="AGENT",
            kind="AGENT",
            parent_id="abcdef1234567890",
        )
        payloads = exporter.build_payload(trace_or_spans=[span])

        assert len(payloads) == 1
        assert payloads[0]["log_type"] == "agent"

    def test_builds_payload_with_tool_span(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        span = self._make_span_dict(
            name="Search Tool",
            span_type="TOOL",
            kind="TOOL",
            parent_id="abcdef1234567890",
            attributes={"tool.name": "web_search"},
        )
        payloads = exporter.build_payload(trace_or_spans=[span])

        assert len(payloads) == 1
        assert payloads[0]["log_type"] == "tool"
        assert payloads[0].get("span_tools") == ["web_search"]

    def test_builds_payload_with_task_span(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        span = self._make_span_dict(
            name="Research Task",
            span_type="CHAIN",
            kind="CHAIN",
            parent_id="abcdef1234567890",
        )
        payloads = exporter.build_payload(trace_or_spans=[span])

        assert len(payloads) == 1
        assert payloads[0]["log_type"] == "task"  # child CHAIN = task

    def test_extracts_input_output(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        span = self._make_span_dict(
            attributes={
                "input.value": "Research AI trends",
                "output.value": "AI is growing rapidly...",
            },
        )
        payloads = exporter.build_payload(trace_or_spans=[span])

        assert len(payloads) == 1
        payload = payloads[0]
        assert "Research AI trends" in payload["input"]
        assert "AI is growing rapidly" in payload["output"]

    def test_calculates_latency(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        span = self._make_span_dict(
            start_time=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
        )
        payloads = exporter.build_payload(trace_or_spans=[span])

        assert payloads[0]["latency"] == 5.0

    def test_handles_error_span(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        span = self._make_span_dict(
            error="Task execution failed",
            status_code=500,
        )
        payloads = exporter.build_payload(trace_or_spans=[span])

        assert payloads[0]["error_message"] == "Task execution failed"
        assert payloads[0]["status_code"] == 500

    def test_empty_spans_returns_empty(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        assert exporter.build_payload(trace_or_spans=[]) == []
        assert exporter.build_payload(trace_or_spans=None) == []

    def test_multiple_spans_in_trace(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        crew_span = self._make_span_dict(name="My Crew")
        agent_span = self._make_span_dict(
            name="Agent",
            span_id="aaaaaaaaaaaaaaaa",
            parent_id="1234567890abcdef",
            span_type="AGENT",
            kind="AGENT",
        )
        payloads = exporter.build_payload(trace_or_spans=[crew_span, agent_span])

        assert len(payloads) == 2

    def test_stores_crewai_ids_in_metadata(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        span = self._make_span_dict()
        payloads = exporter.build_payload(trace_or_spans=[span])

        metadata = payloads[0].get("metadata", {})
        assert "crewai_trace_id" in metadata
        assert "crewai_span_id" in metadata

    def test_sets_customer_identifier(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test", customer_identifier="user-42")
        span = self._make_span_dict()
        payloads = exporter.build_payload(trace_or_spans=[span])

        assert payloads[0]["customer_identifier"] == "user-42"

    def test_parent_id_fallback_when_parent_not_in_trace(self):
        """parent_id should fall back to normalize_span_id when the parent span is not in trace_or_spans."""
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        # Child span whose parent_id does NOT appear as any span's span_id
        child_span = self._make_span_dict(
            span_id="aaaaaaaaaaaaaaaa",
            parent_id="missing_parent_id_123",
            name="Orphan Task",
            span_type="CHAIN",
            kind="CHAIN",
        )
        payloads = exporter.build_payload(trace_or_spans=[child_span])

        assert len(payloads) == 1
        # parent_id must not be None; it should be a normalized hex value
        assert payloads[0]["parent_id"] is not None
        assert len(payloads[0]["parent_id"]) > 0


class TestExport:
    """Test the export (send) functionality."""

    def test_export_sends_payloads(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test-key")
        span = {
            "trace_id": "abcdef1234567890abcdef1234567890",
            "span_id": "1234567890abcdef",
            "parent_id": None,
            "name": "Test Crew",
            "span_type": "CHAIN",
            "kind": "CHAIN",
            "span_path": None,
            "start_time": datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "end_time": datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            "attributes": {},
            "status_code": 200,
            "error": None,
        }

        with patch("respan_exporter_crewai.exporter.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            payloads = exporter.export(trace_or_spans=[span])

            assert len(payloads) > 0
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert "Bearer test-key" in call_kwargs.kwargs.get("headers", {}).get("Authorization", "")

    def test_export_skips_when_no_api_key(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        with patch.dict(os.environ, {}, clear=True):
            env_vars_to_remove = ["RESPAN_API_KEY"]
            for var in env_vars_to_remove:
                os.environ.pop(var, None)

            exporter = RespanCrewAIExporter(api_key=None)
            span = {
                "trace_id": "abcdef1234567890abcdef1234567890",
                "span_id": "1234567890abcdef",
                "parent_id": None,
                "name": "Test",
                "span_type": "CHAIN",
                "kind": "CHAIN",
                "span_path": None,
                "start_time": datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                "end_time": datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                "attributes": {},
                "status_code": 200,
                "error": None,
            }

            with patch("respan_exporter_crewai.exporter.requests.post") as mock_post:
                payloads = exporter.export(trace_or_spans=[span])
                mock_post.assert_not_called()
                assert len(payloads) > 0  # payloads are built but not sent


class TestPropagateTraceOutput:
    """Test output propagation from generation spans to workflow/agent/task spans."""

    def test_propagates_generation_output_to_workflow(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        payloads = [
            {"log_type": "workflow", "output": None},
            {"log_type": "generation", "output": "Generated answer"},
        ]
        exporter._propagate_trace_output(payloads=payloads)
        assert payloads[0]["output"] == "Generated answer"

    def test_does_not_overwrite_existing_output(self):
        from respan_exporter_crewai.exporter import RespanCrewAIExporter

        exporter = RespanCrewAIExporter(api_key="test")
        payloads = [
            {"log_type": "workflow", "output": "Existing output"},
            {"log_type": "generation", "output": "Generated answer"},
        ]
        exporter._propagate_trace_output(payloads=payloads)
        assert payloads[0]["output"] == "Existing output"
