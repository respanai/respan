"""Tests for Microsoft Agents exporter."""
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


class TestRespanMicrosoftAgentsExporterInit:
    def test_default_endpoint(self):
        from respan_exporter_microsoft_agents.exporter import RespanMicrosoftAgentsExporter
        exporter = RespanMicrosoftAgentsExporter(api_key="test")
        assert "v1/traces/ingest" in exporter.endpoint

    def test_custom_base_url(self):
        from respan_exporter_microsoft_agents.exporter import RespanMicrosoftAgentsExporter
        exporter = RespanMicrosoftAgentsExporter(api_key="test", base_url="https://custom.api.com/api")
        assert exporter.endpoint == "https://custom.api.com/api/v1/traces/ingest"

    def test_default_environment(self):
        from respan_exporter_microsoft_agents.exporter import RespanMicrosoftAgentsExporter
        exporter = RespanMicrosoftAgentsExporter(api_key="test")
        assert exporter.environment == "production"


class TestBuildPayload:
    def _make_span(self, **overrides):
        base = {
            "trace_id": "abcdef1234567890abcdef1234567890",
            "span_id": "1234567890abcdef",
            "parent_id": None,
            "name": "autogen.agent.run",
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

    def test_builds_payload(self):
        from respan_exporter_microsoft_agents.exporter import RespanMicrosoftAgentsExporter
        exporter = RespanMicrosoftAgentsExporter(api_key="test")
        payloads = exporter.build_payload(trace_or_spans=[self._make_span()])
        assert len(payloads) == 1
        assert payloads[0]["span_name"] == "autogen.agent.run"

    def test_maps_agent_span(self):
        from respan_exporter_microsoft_agents.exporter import RespanMicrosoftAgentsExporter
        exporter = RespanMicrosoftAgentsExporter(api_key="test")
        payloads = exporter.build_payload(trace_or_spans=[self._make_span(span_type="AGENT")])
        assert payloads[0]["log_type"] == "agent"

    def test_maps_llm_span(self):
        from respan_exporter_microsoft_agents.exporter import RespanMicrosoftAgentsExporter
        exporter = RespanMicrosoftAgentsExporter(api_key="test")
        payloads = exporter.build_payload(trace_or_spans=[
            self._make_span(span_type="LLM", kind="LLM", parent_id="abc", attributes={"llm.model_name": "gpt-4"})
        ])
        assert payloads[0]["log_type"] == "generation"
        assert payloads[0]["model"] == "gpt-4"

    def test_maps_tool_span(self):
        from respan_exporter_microsoft_agents.exporter import RespanMicrosoftAgentsExporter
        exporter = RespanMicrosoftAgentsExporter(api_key="test")
        payloads = exporter.build_payload(trace_or_spans=[
            self._make_span(span_type="TOOL", kind="TOOL", parent_id="abc", attributes={"tool.name": "calculator"})
        ])
        assert payloads[0]["log_type"] == "tool"

    def test_handles_error(self):
        from respan_exporter_microsoft_agents.exporter import RespanMicrosoftAgentsExporter
        exporter = RespanMicrosoftAgentsExporter(api_key="test")
        payloads = exporter.build_payload(trace_or_spans=[self._make_span(error="failed")])
        assert payloads[0]["error_message"] == "failed"
        assert payloads[0]["status_code"] == 500

    def test_empty_returns_empty(self):
        from respan_exporter_microsoft_agents.exporter import RespanMicrosoftAgentsExporter
        exporter = RespanMicrosoftAgentsExporter(api_key="test")
        assert exporter.build_payload(trace_or_spans=[]) == []
        assert exporter.build_payload(trace_or_spans=None) == []

    def test_stores_framework_ids(self):
        from respan_exporter_microsoft_agents.exporter import RespanMicrosoftAgentsExporter
        exporter = RespanMicrosoftAgentsExporter(api_key="test")
        payloads = exporter.build_payload(trace_or_spans=[self._make_span()])
        metadata = payloads[0].get("metadata", {})
        assert "microsoft_agents_trace_id" in metadata

    def test_sets_customer_identifier(self):
        from respan_exporter_microsoft_agents.exporter import RespanMicrosoftAgentsExporter
        exporter = RespanMicrosoftAgentsExporter(api_key="test", customer_identifier="u42")
        payloads = exporter.build_payload(trace_or_spans=[self._make_span()])
        assert payloads[0]["customer_identifier"] == "u42"


class TestExport:
    def test_sends_payloads(self):
        from respan_exporter_microsoft_agents.exporter import RespanMicrosoftAgentsExporter
        exporter = RespanMicrosoftAgentsExporter(api_key="test-key")
        span = {
            "trace_id": "abcdef1234567890abcdef1234567890", "span_id": "1234567890abcdef",
            "parent_id": None, "name": "T", "span_type": "AGENT", "kind": "AGENT",
            "span_path": None,
            "start_time": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "end_time": datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            "attributes": {}, "status_code": 200, "error": None,
        }
        with patch("respan_exporter_microsoft_agents.exporter.requests.post") as mock:
            mock.return_value = MagicMock(status_code=200)
            payloads = exporter.export(trace_or_spans=[span])
            assert len(payloads) > 0
            mock.assert_called_once()

    def test_skips_without_api_key(self):
        from respan_exporter_microsoft_agents.exporter import RespanMicrosoftAgentsExporter
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RESPAN_API_KEY", None)
            exporter = RespanMicrosoftAgentsExporter(api_key=None)
            with patch("respan_exporter_microsoft_agents.exporter.requests.post") as mock:
                exporter.export(trace_or_spans=[{
                    "trace_id": "a" * 32, "span_id": "b" * 16, "parent_id": None,
                    "name": "T", "span_type": "AGENT", "kind": "AGENT", "span_path": None,
                    "start_time": datetime.now(timezone.utc), "end_time": datetime.now(timezone.utc),
                    "attributes": {}, "status_code": 200, "error": None,
                }])
                mock.assert_not_called()
