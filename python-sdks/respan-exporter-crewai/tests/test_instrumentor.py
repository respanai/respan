"""Tests for CrewAI OTel instrumentor."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestRespanCrewAIInstrumentor:
    """Test instrumentor setup and configuration."""

    def test_instrumentation_dependencies(self):
        from respan_exporter_crewai.instrumentor import RespanCrewAIInstrumentor

        instrumentor = RespanCrewAIInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert any("crewai" in dep for dep in deps)
        assert any("openinference-instrumentation-crewai" in dep for dep in deps)

    def test_instrument_creates_exporter(self):
        from respan_exporter_crewai.instrumentor import RespanCrewAIInstrumentor

        instrumentor = RespanCrewAIInstrumentor()

        with patch("respan_exporter_crewai.instrumentor.RespanCrewAIExporter") as mock_cls:
            with patch.object(instrumentor, "_patch_span_processors"):
                instrumentor._instrument(
                    api_key="test-key",
                    environment="staging",
                )
                mock_cls.assert_called_once_with(
                    api_key="test-key",
                    endpoint=None,
                    base_url=None,
                    environment="staging",
                    customer_identifier=None,
                    timeout=10,
                )

    def test_instrument_with_passthrough(self):
        from respan_exporter_crewai.instrumentor import RespanCrewAIInstrumentor

        instrumentor = RespanCrewAIInstrumentor()

        with patch("respan_exporter_crewai.instrumentor.RespanCrewAIExporter"):
            with patch.object(instrumentor, "_patch_span_processors"):
                instrumentor._instrument(
                    api_key="test-key",
                    passthrough=True,
                )
                assert instrumentor._passthrough is True


class TestExportCrewAISpans:
    """Test the span export function used by wrappers."""

    def test_filters_non_crewai_spans(self):
        from respan_exporter_crewai import instrumentor as mod

        non_crewai_span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="openai"),
            attributes={"http.method": "POST"},
            context=SimpleNamespace(trace_id=123, span_id=456),
            parent=None,
            name="OpenAI Call",
            kind=SimpleNamespace(name="CLIENT"),
            start_time=1700000000000000000,
            end_time=1700000001000000000,
            status=SimpleNamespace(
                status_code=SimpleNamespace(name="OK"),
                description=None,
            ),
        )

        mock_exporter = MagicMock()
        original_exporter = mod._ACTIVE_EXPORTER
        mod._ACTIVE_EXPORTER = mock_exporter
        try:
            mod._export_crewai_spans(spans=[non_crewai_span])
            mock_exporter.build_payload.assert_not_called()
        finally:
            mod._ACTIVE_EXPORTER = original_exporter

    def test_exports_crewai_spans(self):
        from respan_exporter_crewai import instrumentor as mod

        crewai_span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.crewai"),
            attributes={"openinference.span.kind": "CHAIN"},
            context=SimpleNamespace(trace_id=123, span_id=456),
            parent=None,
            name="CrewAI Crew",
            kind=SimpleNamespace(name="INTERNAL"),
            start_time=1700000000000000000,
            end_time=1700000001000000000,
            status=SimpleNamespace(
                status_code=SimpleNamespace(name="OK"),
                description=None,
            ),
        )

        mock_exporter = MagicMock()
        mock_exporter.api_key = "test"
        mock_exporter.build_payload.return_value = [{"trace_unique_id": "abc"}]
        original_exporter = mod._ACTIVE_EXPORTER
        mod._ACTIVE_EXPORTER = mock_exporter
        try:
            mod._export_crewai_spans(spans=[crewai_span])
            mock_exporter.build_payload.assert_called_once()
            mock_exporter._send.assert_called_once()
        finally:
            mod._ACTIVE_EXPORTER = original_exporter


class TestSpanDedupeCache:
    """Test span deduplication cache."""

    def test_deduplicates_same_span(self):
        from respan_exporter_crewai.instrumentor import _SpanDedupeCache

        cache = _SpanDedupeCache(max_size=10)
        assert cache.add("trace1", "span1") is True
        assert cache.add("trace1", "span1") is False

    def test_allows_different_spans(self):
        from respan_exporter_crewai.instrumentor import _SpanDedupeCache

        cache = _SpanDedupeCache(max_size=10)
        assert cache.add("trace1", "span1") is True
        assert cache.add("trace1", "span2") is True

    def test_evicts_oldest_when_full(self):
        from respan_exporter_crewai.instrumentor import _SpanDedupeCache

        cache = _SpanDedupeCache(max_size=2)
        cache.add("t1", "s1")
        cache.add("t1", "s2")
        cache.add("t1", "s3")  # evicts s1
        assert cache.add("t1", "s1") is True  # s1 was evicted, can add again
