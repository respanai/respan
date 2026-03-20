"""Tests for LlamaIndex exporter."""
import os
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


class TestRespanLlamaIndexExporterInit:
    """Test exporter initialization."""

    def test_default_endpoint(self):
        from respan_exporter_llamaindex.exporter import RespanLlamaIndexExporter

        exporter = RespanLlamaIndexExporter(api_key="test")
        assert "v1/traces/ingest" in exporter.endpoint

    def test_custom_base_url(self):
        from respan_exporter_llamaindex.exporter import RespanLlamaIndexExporter

        exporter = RespanLlamaIndexExporter(
            api_key="test",
            base_url="https://custom.api.com/api",
        )
        assert exporter.endpoint == "https://custom.api.com/api/v1/traces/ingest"

    def test_default_environment(self):
        from respan_exporter_llamaindex.exporter import RespanLlamaIndexExporter

        exporter = RespanLlamaIndexExporter(api_key="test")
        assert exporter.environment == "production"


class TestBuildPayload:
    """Test payload construction from span records."""

    def _make_span_record(self, **overrides):
        defaults = {
            "span_id": str(uuid.uuid4()),
            "parent_id": None,
            "span_name": "query_engine.query",
            "instance_class": "RetrieverQueryEngine",
            "input": "What is AI?",
            "result": "AI is artificial intelligence.",
            "start_time": datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "end_time": datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
            "error": None,
            "metadata": {},
        }
        defaults.update(overrides)
        return defaults

    def test_builds_workflow_for_root_span(self):
        from respan_exporter_llamaindex.exporter import RespanLlamaIndexExporter

        exporter = RespanLlamaIndexExporter(api_key="test")
        spans = [self._make_span_record()]

        payloads = exporter.build_payloads(
            spans=spans,
            trace_id="trace-123",
            trace_name="LlamaIndex Query",
        )

        assert len(payloads) == 1
        p = payloads[0]
        assert p["log_type"] == "workflow"
        assert p["span_name"] == "query_engine.query"
        assert "What is AI?" in p["input"]
        assert "artificial intelligence" in p["output"]
        assert p["latency"] == 5.0
        assert p["status_code"] == 200

    def test_builds_task_for_child_span(self):
        from respan_exporter_llamaindex.exporter import RespanLlamaIndexExporter

        exporter = RespanLlamaIndexExporter(api_key="test")
        parent_id = str(uuid.uuid4())
        child_id = str(uuid.uuid4())
        spans = [
            self._make_span_record(span_id=parent_id),
            self._make_span_record(
                span_id=child_id,
                parent_id=parent_id,
                span_name="retriever.retrieve",
                instance_class="VectorIndexRetriever",
            ),
        ]

        payloads = exporter.build_payloads(
            spans=spans,
            trace_id="trace-123",
        )

        assert len(payloads) == 2
        child_payload = next(p for p in payloads if p["span_name"] == "retriever.retrieve")
        assert child_payload["log_type"] == "task"
        assert child_payload["span_parent_id"] is not None

    def test_maps_llm_span_to_generation(self):
        from respan_exporter_llamaindex.exporter import RespanLlamaIndexExporter

        exporter = RespanLlamaIndexExporter(api_key="test")
        spans = [
            self._make_span_record(
                instance_class="OpenAI",
                span_name="llm.chat",
                parent_id="some-parent",
                metadata={"model": "gpt-4"},
            ),
        ]

        payloads = exporter.build_payloads(spans=spans, trace_id="trace-123")
        assert payloads[0]["log_type"] == "generation"
        assert payloads[0]["model"] == "gpt-4"

    def test_maps_retriever_span_to_task(self):
        from respan_exporter_llamaindex.exporter import RespanLlamaIndexExporter

        exporter = RespanLlamaIndexExporter(api_key="test")
        spans = [
            self._make_span_record(
                instance_class="VectorIndexRetriever",
                span_name="retriever.retrieve",
                parent_id="some-parent",
            ),
        ]

        payloads = exporter.build_payloads(spans=spans, trace_id="trace-123")
        assert payloads[0]["log_type"] == "task"

    def test_handles_error_span(self):
        from respan_exporter_llamaindex.exporter import RespanLlamaIndexExporter

        exporter = RespanLlamaIndexExporter(api_key="test")
        spans = [
            self._make_span_record(
                error="Index not found",
                result=None,
            ),
        ]

        payloads = exporter.build_payloads(spans=spans, trace_id="trace-123")
        assert payloads[0]["error_message"] == "Index not found"
        assert payloads[0]["status_code"] == 500

    def test_empty_spans_returns_empty(self):
        from respan_exporter_llamaindex.exporter import RespanLlamaIndexExporter

        exporter = RespanLlamaIndexExporter(api_key="test")
        assert exporter.build_payloads(spans=[], trace_id="trace-123") == []

    def test_sets_customer_identifier(self):
        from respan_exporter_llamaindex.exporter import RespanLlamaIndexExporter

        exporter = RespanLlamaIndexExporter(
            api_key="test",
            customer_identifier="user-42",
        )
        spans = [self._make_span_record()]
        payloads = exporter.build_payloads(spans=spans, trace_id="trace-123")
        assert payloads[0]["customer_identifier"] == "user-42"


class TestExport:
    """Test the export (send) functionality."""

    def test_export_sends_payloads(self):
        from respan_exporter_llamaindex.exporter import RespanLlamaIndexExporter

        exporter = RespanLlamaIndexExporter(api_key="test-key")
        spans = [
            {
                "span_id": str(uuid.uuid4()),
                "parent_id": None,
                "span_name": "query",
                "instance_class": "QueryEngine",
                "input": "hello",
                "result": "world",
                "start_time": datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                "end_time": datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                "error": None,
                "metadata": {},
            }
        ]

        with patch("respan_exporter_llamaindex.exporter.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            exporter.export(spans=spans, trace_id="t-1", trace_name="test")
            mock_post.assert_called_once()

    def test_export_skips_without_api_key(self):
        from respan_exporter_llamaindex.exporter import RespanLlamaIndexExporter

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RESPAN_API_KEY", None)
            exporter = RespanLlamaIndexExporter(api_key=None)

            with patch("respan_exporter_llamaindex.exporter.requests.post") as mock_post:
                exporter.export(
                    spans=[{
                        "span_id": "s1", "parent_id": None,
                        "span_name": "q", "instance_class": "Q",
                        "input": "a", "result": "b",
                        "start_time": datetime.now(timezone.utc),
                        "end_time": datetime.now(timezone.utc),
                        "error": None, "metadata": {},
                    }],
                    trace_id="t-1",
                )
                mock_post.assert_not_called()
