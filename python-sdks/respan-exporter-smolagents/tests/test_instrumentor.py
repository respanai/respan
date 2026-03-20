"""Tests for smolagents OTel instrumentor."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestRespanSmolagentsInstrumentor:
    """Test instrumentor setup."""

    def test_instrumentation_dependencies(self):
        from respan_exporter_smolagents.instrumentor import RespanSmolagentsInstrumentor

        instrumentor = RespanSmolagentsInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert any("smolagents" in dep for dep in deps)
        assert any("openinference-instrumentation-smolagents" in dep for dep in deps)

    def test_instrument_creates_exporter(self):
        from respan_exporter_smolagents.instrumentor import RespanSmolagentsInstrumentor

        instrumentor = RespanSmolagentsInstrumentor()

        with patch("respan_exporter_smolagents.instrumentor.RespanSmolagentsExporter") as mock_cls:
            with patch.object(instrumentor, "_patch_span_processors"):
                instrumentor._instrument(api_key="test-key", environment="staging")
                mock_cls.assert_called_once()


class TestExportSmolagentsSpans:
    """Test span export function."""

    def test_filters_non_smolagents_spans(self):
        from respan_exporter_smolagents import instrumentor as mod

        non_smolagents_span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="openai"),
            attributes={"http.method": "POST"},
            context=SimpleNamespace(trace_id=123, span_id=456),
            parent=None,
            name="OpenAI Call",
            kind=SimpleNamespace(name="CLIENT"),
            start_time=1700000000000000000,
            end_time=1700000001000000000,
            status=SimpleNamespace(status_code=SimpleNamespace(name="OK"), description=None),
        )

        mock_exporter = MagicMock()
        original = mod._ACTIVE_EXPORTER
        mod._ACTIVE_EXPORTER = mock_exporter
        try:
            mod._export_smolagents_spans(spans=[non_smolagents_span])
            mock_exporter.build_payload.assert_not_called()
        finally:
            mod._ACTIVE_EXPORTER = original

    def test_exports_smolagents_spans(self):
        from respan_exporter_smolagents import instrumentor as mod

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.smolagents"),
            attributes={"openinference.span.kind": "AGENT"},
            context=SimpleNamespace(trace_id=123, span_id=456),
            parent=None,
            name="Agent Run",
            kind=SimpleNamespace(name="INTERNAL"),
            start_time=1700000000000000000,
            end_time=1700000001000000000,
            status=SimpleNamespace(status_code=SimpleNamespace(name="OK"), description=None),
        )

        mock_exporter = MagicMock()
        mock_exporter.api_key = "test"
        mock_exporter.build_payload.return_value = [{"trace_unique_id": "abc"}]
        original = mod._ACTIVE_EXPORTER
        mod._ACTIVE_EXPORTER = mock_exporter
        try:
            mod._export_smolagents_spans(spans=[span])
            mock_exporter.build_payload.assert_called_once()
            mock_exporter._send.assert_called_once()
        finally:
            mod._ACTIVE_EXPORTER = original


class TestSpanDedupeCache:
    """Test deduplication cache."""

    def test_deduplicates_same_span(self):
        from respan_exporter_smolagents.instrumentor import _SpanDedupeCache

        cache = _SpanDedupeCache(max_size=10)
        assert cache.add("t1", "s1") is True
        assert cache.add("t1", "s1") is False

    def test_allows_different_spans(self):
        from respan_exporter_smolagents.instrumentor import _SpanDedupeCache

        cache = _SpanDedupeCache(max_size=10)
        assert cache.add("t1", "s1") is True
        assert cache.add("t1", "s2") is True
