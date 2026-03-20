"""Tests for Microsoft Agents instrumentor."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestRespanMicrosoftAgentsInstrumentor:
    def test_instrumentation_dependencies(self):
        from respan_exporter_microsoft_agents.instrumentor import RespanMicrosoftAgentsInstrumentor
        instrumentor = RespanMicrosoftAgentsInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert any("opentelemetry" in dep for dep in deps)

    def test_instrument_creates_exporter(self):
        from respan_exporter_microsoft_agents.instrumentor import RespanMicrosoftAgentsInstrumentor
        instrumentor = RespanMicrosoftAgentsInstrumentor()
        with patch("respan_exporter_microsoft_agents.instrumentor.RespanMicrosoftAgentsExporter") as mock:
            with patch.object(instrumentor, "_patch_span_processors"):
                instrumentor._instrument(api_key="test-key")
                mock.assert_called_once()


class TestExportMicrosoftAgentsSpans:
    def test_filters_non_microsoft_spans(self):
        from respan_exporter_microsoft_agents import instrumentor as mod
        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="openai"),
            attributes={"http.method": "POST"},
            context=SimpleNamespace(trace_id=123, span_id=456),
            parent=None, name="Call", kind=SimpleNamespace(name="CLIENT"),
            start_time=1700000000000000000, end_time=1700000001000000000,
            status=SimpleNamespace(status_code=SimpleNamespace(name="OK"), description=None),
        )
        mock_exporter = MagicMock()
        original = mod._ACTIVE_EXPORTER
        mod._ACTIVE_EXPORTER = mock_exporter
        try:
            mod._export_microsoft_agents_spans(spans=[span])
            mock_exporter.build_payload.assert_not_called()
        finally:
            mod._ACTIVE_EXPORTER = original

    def test_exports_autogen_spans(self):
        from respan_exporter_microsoft_agents import instrumentor as mod
        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="autogen.runtime"),
            attributes={"autogen.agent.name": "assistant"},
            context=SimpleNamespace(trace_id=123, span_id=456),
            parent=None, name="Agent Run", kind=SimpleNamespace(name="INTERNAL"),
            start_time=1700000000000000000, end_time=1700000001000000000,
            status=SimpleNamespace(status_code=SimpleNamespace(name="OK"), description=None),
        )
        mock_exporter = MagicMock()
        mock_exporter.api_key = "test"
        mock_exporter.build_payload.return_value = [{"trace_unique_id": "abc"}]
        original = mod._ACTIVE_EXPORTER
        mod._ACTIVE_EXPORTER = mock_exporter
        try:
            mod._export_microsoft_agents_spans(spans=[span])
            mock_exporter.build_payload.assert_called_once()
            mock_exporter._send.assert_called_once()
        finally:
            mod._ACTIVE_EXPORTER = original


class TestSpanDedupeCache:
    def test_deduplicates(self):
        from respan_exporter_microsoft_agents.instrumentor import _SpanDedupeCache
        cache = _SpanDedupeCache(max_size=10)
        assert cache.add("t1", "s1") is True
        assert cache.add("t1", "s1") is False

    def test_allows_different(self):
        from respan_exporter_microsoft_agents.instrumentor import _SpanDedupeCache
        cache = _SpanDedupeCache(max_size=10)
        assert cache.add("t1", "s1") is True
        assert cache.add("t1", "s2") is True
