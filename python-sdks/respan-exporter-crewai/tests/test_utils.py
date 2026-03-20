"""Tests for CrewAI exporter utility functions."""
import pytest
from types import SimpleNamespace


class TestIsCrewAISpan:
    """Test is_crewai_span detection logic."""

    def test_detects_crewai_instrumentation_scope(self):
        from respan_exporter_crewai.utils import is_crewai_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.crewai"),
            attributes={},
        )
        assert is_crewai_span(span=span) is True

    def test_detects_crewai_attribute_prefix(self):
        from respan_exporter_crewai.utils import is_crewai_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="other"),
            attributes={"crewai.crew.id": "123", "crewai.task.id": "456"},
        )
        assert is_crewai_span(span=span) is True

    def test_rejects_non_crewai_span(self):
        from respan_exporter_crewai.utils import is_crewai_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="openai"),
            attributes={"http.method": "POST"},
        )
        assert is_crewai_span(span=span) is False

    def test_detects_crewai_via_openinference_span_kind(self):
        """CrewAI spans from openinference have openinference.span.kind attribute."""
        from respan_exporter_crewai.utils import is_crewai_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.crewai"),
            attributes={"openinference.span.kind": "CHAIN"},
        )
        assert is_crewai_span(span=span) is True

    def test_handles_missing_instrumentation_scope(self):
        from respan_exporter_crewai.utils import is_crewai_span

        span = SimpleNamespace(
            instrumentation_scope=None,
            attributes={"crewai.crew.id": "abc"},
        )
        assert is_crewai_span(span=span) is True

    def test_handles_no_attributes(self):
        from respan_exporter_crewai.utils import is_crewai_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="other"),
            attributes=None,
        )
        assert is_crewai_span(span=span) is False


class TestOtelSpanToDict:
    """Test conversion of OTel spans to dict format."""

    def test_converts_basic_span(self):
        from respan_exporter_crewai.utils import otel_span_to_dict

        span_context = SimpleNamespace(trace_id=1234567890, span_id=9876543210)
        span = SimpleNamespace(
            context=span_context,
            parent=None,
            name="CrewAI Task",
            kind=SimpleNamespace(name="INTERNAL"),
            start_time=1700000000000000000,
            end_time=1700000001000000000,
            attributes={"openinference.span.kind": "CHAIN"},
            status=SimpleNamespace(
                status_code=SimpleNamespace(name="OK"),
                description=None,
            ),
            instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.crewai"),
        )

        result = otel_span_to_dict(span=span)
        assert result["name"] == "CrewAI Task"
        assert result["trace_id"] is not None
        assert result["span_id"] is not None
        assert result["parent_id"] is None
        assert result["start_time"] is not None
        assert result["end_time"] is not None
        assert result["status_code"] == 200

    def test_converts_span_with_parent(self):
        from respan_exporter_crewai.utils import otel_span_to_dict

        parent_context = SimpleNamespace(span_id=1111111111)
        span_context = SimpleNamespace(trace_id=1234567890, span_id=9876543210)
        span = SimpleNamespace(
            context=span_context,
            parent=parent_context,
            name="CrewAI Agent",
            kind=SimpleNamespace(name="INTERNAL"),
            start_time=1700000000000000000,
            end_time=1700000001000000000,
            attributes={"openinference.span.kind": "AGENT"},
            status=SimpleNamespace(
                status_code=SimpleNamespace(name="OK"),
                description=None,
            ),
            instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.crewai"),
        )

        result = otel_span_to_dict(span=span)
        assert result["parent_id"] is not None

    def test_converts_error_span(self):
        from respan_exporter_crewai.utils import otel_span_to_dict

        span_context = SimpleNamespace(trace_id=1234567890, span_id=9876543210)
        span = SimpleNamespace(
            context=span_context,
            parent=None,
            name="Failed Task",
            kind=SimpleNamespace(name="INTERNAL"),
            start_time=1700000000000000000,
            end_time=1700000001000000000,
            attributes={},
            status=SimpleNamespace(
                status_code=SimpleNamespace(name="ERROR"),
                description="task failed",
            ),
            instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.crewai"),
        )

        result = otel_span_to_dict(span=span)
        assert result["status_code"] == 500
        assert result["error"] == "task failed"


class TestGroupSpansByTrace:
    """Test grouping spans by trace ID."""

    def test_groups_correctly(self):
        from respan_exporter_crewai.utils import group_spans_by_trace

        spans = [
            {"trace_id": "trace1", "span_id": "a"},
            {"trace_id": "trace1", "span_id": "b"},
            {"trace_id": "trace2", "span_id": "c"},
        ]
        grouped = group_spans_by_trace(spans=spans)
        assert len(grouped) == 2
        assert len(grouped["trace1"]) == 2
        assert len(grouped["trace2"]) == 1

    def test_skips_missing_trace_id(self):
        from respan_exporter_crewai.utils import group_spans_by_trace

        spans = [
            {"trace_id": "trace1", "span_id": "a"},
            {"span_id": "b"},
            {"trace_id": "", "span_id": "c"},
        ]
        grouped = group_spans_by_trace(spans=spans)
        assert len(grouped) == 1
