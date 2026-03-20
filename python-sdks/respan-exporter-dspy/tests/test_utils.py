"""Tests for dspy exporter utility functions."""
from types import SimpleNamespace


class TestIsDSPySpan:
    """Test is_dspy_span detection logic."""

    def test_detects_dspy_instrumentation_scope(self):
        from respan_exporter_dspy.utils import is_dspy_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.dspy"),
            attributes={},
        )
        assert is_dspy_span(span=span) is True

    def test_detects_dspy_attribute_prefix(self):
        from respan_exporter_dspy.utils import is_dspy_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="other"),
            attributes={"dspy.module.name": "ToolCallingAgent"},
        )
        assert is_dspy_span(span=span) is True

    def test_rejects_non_dspy_span(self):
        from respan_exporter_dspy.utils import is_dspy_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="openai"),
            attributes={"http.method": "POST"},
        )
        assert is_dspy_span(span=span) is False

    def test_handles_missing_instrumentation_scope(self):
        from respan_exporter_dspy.utils import is_dspy_span

        span = SimpleNamespace(
            instrumentation_scope=None,
            attributes={"dspy.predict.name": "web_search"},
        )
        assert is_dspy_span(span=span) is True

    def test_handles_no_attributes(self):
        from respan_exporter_dspy.utils import is_dspy_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="other"),
            attributes=None,
        )
        assert is_dspy_span(span=span) is False


class TestOtelSpanToDict:
    """Test conversion of OTel spans to dict format."""

    def test_converts_basic_span(self):
        from respan_exporter_dspy.utils import otel_span_to_dict

        span_context = SimpleNamespace(trace_id=1234567890, span_id=9876543210)
        span = SimpleNamespace(
            context=span_context,
            parent=None,
            name="dspy.agent.run",
            kind=SimpleNamespace(name="INTERNAL"),
            start_time=1700000000000000000,
            end_time=1700000001000000000,
            attributes={"openinference.span.kind": "AGENT"},
            status=SimpleNamespace(
                status_code=SimpleNamespace(name="OK"),
                description=None,
            ),
            instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.dspy"),
        )

        result = otel_span_to_dict(span=span)
        assert result["name"] == "dspy.agent.run"
        assert result["trace_id"] is not None
        assert result["span_id"] is not None
        assert result["status_code"] == 200

    def test_converts_error_span(self):
        from respan_exporter_dspy.utils import otel_span_to_dict

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
            instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.dspy"),
        )

        result = otel_span_to_dict(span=span)
        assert result["status_code"] == 500
        assert result["error"] == "agent failed"


class TestGroupSpansByTrace:
    """Test grouping spans by trace ID."""

    def test_groups_correctly(self):
        from respan_exporter_dspy.utils import group_spans_by_trace

        spans = [
            {"trace_id": "t1", "span_id": "a"},
            {"trace_id": "t1", "span_id": "b"},
            {"trace_id": "t2", "span_id": "c"},
        ]
        grouped = group_spans_by_trace(spans=spans)
        assert len(grouped) == 2
        assert len(grouped["t1"]) == 2
