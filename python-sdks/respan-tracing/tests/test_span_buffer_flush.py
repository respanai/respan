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


class TestSpanBufferDedup:
    """Tests for SpanBuffer deduplication.

    The bug: N BufferingSpanProcessor instances (one per processor chain) all
    share the same SpanBuffer via ContextVar. Each one calls buffer.buffer_span()
    when a span ends. Without dedup, the same span gets appended N times,
    causing N duplicate exports on replay.
    """

    def test_buffer_span_deduplicates_by_span_id(self):
        """Same span appended multiple times → only first is kept."""
        buffer = SpanBuffer(trace_id="trace-dedup")
        buffer.__enter__()

        mock_span = MagicMock()
        mock_span.get_span_context.return_value.span_id = 0xDEADBEEF
        mock_span.name = "test_span"

        assert buffer.buffer_span(mock_span) is True
        assert buffer.buffer_span(mock_span) is False
        assert buffer.buffer_span(mock_span) is False
        assert buffer.get_span_count() == 1

        buffer.__exit__(None, None, None)

    def test_different_spans_not_deduped(self):
        """Spans with different span_ids are all kept."""
        buffer = SpanBuffer(trace_id="trace-multi")
        buffer.__enter__()

        spans = []
        for i in range(5):
            mock_span = MagicMock()
            mock_span.get_span_context.return_value.span_id = i + 1
            mock_span.name = f"span_{i}"
            spans.append(mock_span)
            assert buffer.buffer_span(mock_span) is True

        assert buffer.get_span_count() == 5
        buffer.__exit__(None, None, None)

    def test_clear_spans_resets_dedup_tracking(self):
        """After clear_spans(), the same span_id can be buffered again."""
        buffer = SpanBuffer(trace_id="trace-clear")

        mock_span = MagicMock()
        mock_span.get_span_context.return_value.span_id = 0xCAFE
        mock_span.name = "test_span"

        buffer.buffer_span(mock_span)
        assert buffer.get_span_count() == 1

        buffer.clear_spans()
        assert buffer.get_span_count() == 0

        assert buffer.buffer_span(mock_span) is True
        assert buffer.get_span_count() == 1

    def test_multiple_processors_single_buffer(self):
        """Simulate N BufferingSpanProcessors sharing one buffer via ContextVar.

        This is the exact scenario that caused the 3x duplicate root spans:
        3 processor chains → 3 BufferingSpanProcessor instances → each calls
        buffer.buffer_span() → only the first should succeed.
        """
        from respan_tracing.processors.base import (
            BufferingSpanProcessor,
            _active_span_buffer,
        )

        # Create 3 processors (simulating production/debug/dogfood chains)
        processors = [
            BufferingSpanProcessor(MagicMock()) for _ in range(3)
        ]

        buffer = SpanBuffer(trace_id="trace-multi-proc")
        buffer.__enter__()

        # Create a span that ends → all 3 processors call on_end
        mock_span = MagicMock()
        mock_span.get_span_context.return_value.span_id = 0xBEEF
        mock_span.name = "root_span"

        for proc in processors:
            proc.on_end(mock_span)

        # Only 1 copy should be in the buffer, not 3
        assert buffer.get_span_count() == 1

        buffer.__exit__(None, None, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
