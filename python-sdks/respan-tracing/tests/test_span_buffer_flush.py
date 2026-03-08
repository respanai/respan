"""Tests for SpanBuffer auto-flush on exit.

The bug: SpanBuffer.__exit__() didn't flush. Spans went into _local_queue,
buffer deactivated, spans got GC'd. process_spans() existed but nobody
called it. This test ensures auto-flush happens when tracer_provider is set.
"""
import pytest
from unittest.mock import MagicMock, patch, call

from respan_tracing import RespanTelemetry, get_client
from respan_tracing.processors.base import SpanBuffer


class TestSpanBufferAutoFlush:
    """Tests for SpanBuffer auto-flush on __exit__."""

    def setup_method(self):
        self.telemetry = RespanTelemetry(
            app_name="test-buffer-flush",
            api_key="test-key",
            is_enabled=True,
        )
        self.client = get_client()

    def test_auto_flush_on_exit(self):
        """Spans are auto-flushed through processor pipeline on context exit."""
        with self.client.get_span_buffer("trace-123") as buffer:
            buffer.create_span("step_1", {"status": "ok"})
            buffer.create_span("step_2", {"status": "ok"})
            span_count = buffer.get_span_count()
            assert span_count == 2

        # After exit, spans should still be readable (queue not cleared)
        assert len(buffer.get_all_spans()) == 2

    def test_auto_flush_calls_process_spans(self):
        """__exit__ calls process_spans when tracer_provider is set."""
        buffer = SpanBuffer(trace_id="trace-456", tracer_provider=MagicMock())
        buffer._local_queue = [MagicMock(), MagicMock()]

        with patch.object(buffer, 'process_spans', return_value=2) as mock_process:
            buffer.__enter__()
            buffer.__exit__(None, None, None)
            mock_process.assert_called_once_with(buffer._tracer_provider)

    def test_no_auto_flush_without_tracer_provider(self):
        """Without tracer_provider, no auto-flush (backward compatible)."""
        buffer = SpanBuffer(trace_id="trace-789")
        buffer._local_queue = [MagicMock()]

        with patch.object(buffer, 'process_spans') as mock_process:
            buffer.__enter__()
            buffer.__exit__(None, None, None)
            mock_process.assert_not_called()

    def test_no_auto_flush_on_empty_queue(self):
        """No process_spans call when queue is empty."""
        buffer = SpanBuffer(trace_id="trace-000", tracer_provider=MagicMock())

        with patch.object(buffer, 'process_spans') as mock_process:
            buffer.__enter__()
            buffer.__exit__(None, None, None)
            mock_process.assert_not_called()

    def test_get_all_spans_works_after_exit(self):
        """get_all_spans() still returns spans after auto-flush (read-only use case)."""
        with self.client.get_span_buffer("trace-read") as buffer:
            buffer.create_span("s1", {"x": 1})
            buffer.create_span("s2", {"x": 2})

        # Post-exit: spans still readable for unified log conversion
        spans = buffer.get_all_spans()
        assert len(spans) == 2
        assert spans[0].name == "s1"
        assert spans[1].name == "s2"

    def test_client_get_span_buffer_passes_tracer_provider(self):
        """client.get_span_buffer() passes tracer_provider for auto-flush."""
        buffer = self.client.get_span_buffer("trace-auto")
        assert buffer._tracer_provider is not None
        assert buffer._tracer_provider is self.client._tracer.tracer_provider

    def test_buffering_deactivated_before_flush(self):
        """_is_buffering is False before flush to prevent re-entry."""
        flush_order = []

        buffer = SpanBuffer(trace_id="trace-order", tracer_provider=MagicMock())
        original_process = buffer.process_spans

        def tracking_process(tp):
            flush_order.append(("process", buffer._is_buffering))
            return 0

        buffer.process_spans = tracking_process
        buffer._local_queue = [MagicMock()]

        buffer.__enter__()
        buffer.__exit__(None, None, None)

        # process_spans should have been called with _is_buffering=False
        assert flush_order == [("process", False)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
