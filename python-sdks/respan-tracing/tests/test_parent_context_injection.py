"""Tests for SpanBuffer parent context injection (trace continuation).

When a workflow pauses and resumes, we want the post-resume spans to
share the pre-pause trace_id. SpanBuffer achieves this by injecting a
NonRecordingSpan parent context so all child spans inherit the parent's
trace_id.

This is the SDK side of Phase 6 (Continue-Trace-on-Resume) from
span_mv_and_trace_evolution.md.
"""
import pytest
from unittest.mock import MagicMock

from opentelemetry import trace
from respan_tracing import RespanTelemetry, get_client
from respan_tracing.processors.base import SpanBuffer


# Valid OTel hex IDs for testing
PARENT_TRACE_ID = "0123456789abcdef0123456789abcdef"
PARENT_SPAN_ID = "fedcba9876543210"


class TestParentContextInjection:
    """Tests for SpanBuffer with parent_trace_id/parent_span_id."""

    def setup_method(self):
        self.telemetry = RespanTelemetry(
            app_name="test-parent-ctx",
            api_key="test-key",
            is_enabled=True,
        )
        self.client = get_client()

    def test_spans_inherit_parent_trace_id(self):
        """Spans created in a buffer with parent context inherit the parent trace_id."""
        with self.client.get_span_buffer(
            trace_id="run-abc",
            parent_trace_id=PARENT_TRACE_ID,
            parent_span_id=PARENT_SPAN_ID,
        ) as buffer:
            buffer.create_span("resumed_step", {"status": "running"})
            buffer.create_span("resumed_step_2", {"status": "done"})

        spans = buffer.get_all_spans()
        assert len(spans) == 2

        expected_trace_id = int(PARENT_TRACE_ID, 16)
        for span in spans:
            actual_trace_id = span.get_span_context().trace_id
            assert actual_trace_id == expected_trace_id, (
                f"Span '{span.name}' has trace_id {actual_trace_id:#034x}, "
                f"expected {expected_trace_id:#034x}"
            )

    def test_spans_are_children_of_parent_span(self):
        """Spans created in buffer are children of the injected parent span."""
        with self.client.get_span_buffer(
            trace_id="run-def",
            parent_trace_id=PARENT_TRACE_ID,
            parent_span_id=PARENT_SPAN_ID,
        ) as buffer:
            buffer.create_span("child_step", {"x": 1})

        spans = buffer.get_all_spans()
        assert len(spans) == 1
        span = spans[0]

        expected_parent = int(PARENT_SPAN_ID, 16)
        assert span.parent is not None, "Span should have a parent"
        actual_parent = span.parent.span_id
        assert actual_parent == expected_parent, (
            f"Span parent_span_id is {actual_parent:#018x}, "
            f"expected {expected_parent:#018x}"
        )

    def test_no_parent_context_creates_new_trace(self):
        """Without parent context, spans get a new auto-generated trace_id (backward compat)."""
        with self.client.get_span_buffer(trace_id="run-ghi") as buffer:
            buffer.create_span("independent_step", {"status": "ok"})

        spans = buffer.get_all_spans()
        assert len(spans) == 1

        # Should NOT have the parent trace_id
        actual_trace_id = spans[0].get_span_context().trace_id
        assert actual_trace_id != int(PARENT_TRACE_ID, 16)
        assert actual_trace_id != 0  # Should have some valid trace_id

    def test_parent_context_detached_after_exit(self):
        """Parent context is properly detached after buffer exits."""
        # Get trace_id before entering buffer
        pre_span = trace.get_current_span()

        with self.client.get_span_buffer(
            trace_id="run-jkl",
            parent_trace_id=PARENT_TRACE_ID,
            parent_span_id=PARENT_SPAN_ID,
        ) as buffer:
            buffer.create_span("step", {"x": 1})

        # After exit, current span should be back to what it was before
        post_span = trace.get_current_span()
        # The parent NonRecordingSpan should not leak into outer context
        post_trace_id = post_span.get_span_context().trace_id
        assert post_trace_id != int(PARENT_TRACE_ID, 16), (
            "Parent context leaked — trace_id still active after buffer exit"
        )

    def test_partial_parent_context_ignored(self):
        """If only parent_trace_id is provided (no parent_span_id), no injection happens."""
        buffer = SpanBuffer(
            trace_id="run-mno",
            parent_trace_id=PARENT_TRACE_ID,
            # parent_span_id intentionally omitted
        )
        assert buffer._parent_trace_id == PARENT_TRACE_ID
        assert buffer._parent_span_id is None

        # __enter__ should not inject parent context
        buffer.__enter__()
        assert buffer._parent_context_token is None
        buffer.__exit__(None, None, None)

    def test_client_get_span_buffer_passes_parent_params(self):
        """client.get_span_buffer() correctly passes parent_trace_id/parent_span_id."""
        buffer = self.client.get_span_buffer(
            trace_id="run-pqr",
            parent_trace_id=PARENT_TRACE_ID,
            parent_span_id=PARENT_SPAN_ID,
        )
        assert buffer._parent_trace_id == PARENT_TRACE_ID
        assert buffer._parent_span_id == PARENT_SPAN_ID
        assert buffer._tracer_provider is not None

    def test_multiple_spans_share_trace_but_have_unique_span_ids(self):
        """All spans in a parent-context buffer share trace_id but have distinct span_ids."""
        with self.client.get_span_buffer(
            trace_id="run-stu",
            parent_trace_id=PARENT_TRACE_ID,
            parent_span_id=PARENT_SPAN_ID,
        ) as buffer:
            buffer.create_span("step_a", {})
            buffer.create_span("step_b", {})
            buffer.create_span("step_c", {})

        spans = buffer.get_all_spans()
        assert len(spans) == 3

        trace_ids = {s.get_span_context().trace_id for s in spans}
        span_ids = {s.get_span_context().span_id for s in spans}

        # All same trace_id
        assert len(trace_ids) == 1
        assert trace_ids.pop() == int(PARENT_TRACE_ID, 16)

        # All different span_ids
        assert len(span_ids) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
