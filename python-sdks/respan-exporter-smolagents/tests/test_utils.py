"""Tests for smolagents exporter utility functions."""
from types import SimpleNamespace


class TestIsSmolagentsSpan:
    """Test is_smolagents_span detection logic."""

    def test_detects_smolagents_instrumentation_scope(self):
        from respan_exporter_smolagents.utils import is_smolagents_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.smolagents"),
            attributes={},
        )
        assert is_smolagents_span(span=span) is True

    def test_detects_smolagents_attribute_prefix(self):
        from respan_exporter_smolagents.utils import is_smolagents_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="other"),
            attributes={"smolagents.agent.type": "ToolCallingAgent"},
        )
        assert is_smolagents_span(span=span) is True

    def test_rejects_non_smolagents_span(self):
        from respan_exporter_smolagents.utils import is_smolagents_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="openai"),
            attributes={"http.method": "POST"},
        )
        assert is_smolagents_span(span=span) is False

    def test_handles_missing_instrumentation_scope(self):
        from respan_exporter_smolagents.utils import is_smolagents_span

        span = SimpleNamespace(
            instrumentation_scope=None,
            attributes={"smolagents.tool.name": "web_search"},
        )
        assert is_smolagents_span(span=span) is True

    def test_handles_no_attributes(self):
        from respan_exporter_smolagents.utils import is_smolagents_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="other"),
            attributes=None,
        )
        assert is_smolagents_span(span=span) is False


class TestOtelSpanToDict:
    """Test conversion of OTel spans to dict format."""

    def test_converts_basic_span(self):
        from respan_exporter_smolagents.utils import otel_span_to_dict

        span_context = SimpleNamespace(trace_id=1234567890, span_id=9876543210)
        span = SimpleNamespace(
            context=span_context,
            parent=None,
            name="smolagents.agent.run",
            kind=SimpleNamespace(name="INTERNAL"),
            start_time=1700000000000000000,
            end_time=1700000001000000000,
            attributes={"openinference.span.kind": "AGENT"},
            status=SimpleNamespace(
                status_code=SimpleNamespace(name="OK"),
                description=None,
            ),
            instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.smolagents"),
        )

        result = otel_span_to_dict(span=span)
        assert result["name"] == "smolagents.agent.run"
        assert result["trace_id"] is not None
        assert result["span_id"] is not None
        assert result["status_code"] == 200

    def test_converts_error_span(self):
        from respan_exporter_smolagents.utils import otel_span_to_dict

        span_context = SimpleNamespace(trace_id=1234567890, span_id=9876543210)
        span = SimpleNamespace(
            context=span_context,
            parent=None,
            name="Failed",
            kind=SimpleNamespace(name="INTERNAL"),
            start_time=1700000000000000000,
            end_time=1700000001000000000,
            attributes={},
            status=SimpleNamespace(
                status_code=SimpleNamespace(name="ERROR"),
                description="agent failed",
            ),
            instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.smolagents"),
        )

        result = otel_span_to_dict(span=span)
        assert result["status_code"] == 500
        assert result["error"] == "agent failed"


class TestGroupSpansByTrace:
    """Test grouping spans by trace ID."""

    def test_groups_correctly(self):
        from respan_exporter_smolagents.utils import group_spans_by_trace

        spans = [
            {"trace_id": "t1", "span_id": "a"},
            {"trace_id": "t1", "span_id": "b"},
            {"trace_id": "t2", "span_id": "c"},
        ]
        grouped = group_spans_by_trace(spans=spans)
        assert len(grouped) == 2
        assert len(grouped["t1"]) == 2
