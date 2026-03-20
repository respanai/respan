"""Tests for Microsoft Agents exporter utility functions."""
from datetime import datetime, timezone
from types import SimpleNamespace


class TestNsToDatetime:
    """Test nanosecond timestamp conversion."""

    def test_returns_none_for_none(self):
        from respan_exporter_microsoft_agents.utils import ns_to_datetime

        assert ns_to_datetime(value=None) is None

    def test_converts_zero_to_epoch(self):
        from respan_exporter_microsoft_agents.utils import ns_to_datetime

        result = ns_to_datetime(value=0)
        assert result == datetime(1970, 1, 1, tzinfo=timezone.utc)

    def test_converts_valid_nanoseconds(self):
        from respan_exporter_microsoft_agents.utils import ns_to_datetime

        result = ns_to_datetime(value=1_700_000_000_000_000_000)
        assert result is not None
        assert result.year == 2023

    def test_returns_none_for_non_numeric(self):
        from respan_exporter_microsoft_agents.utils import ns_to_datetime

        assert ns_to_datetime(value="not_a_number") is None


class TestIsMicrosoftAgentsSpan:
    """Test span detection for AutoGen and Semantic Kernel."""

    def test_detects_autogen_scope(self):
        from respan_exporter_microsoft_agents.utils import is_microsoft_agents_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="autogen.runtime"),
            attributes={},
        )
        assert is_microsoft_agents_span(span=span) is True

    def test_detects_semantic_kernel_scope(self):
        from respan_exporter_microsoft_agents.utils import is_microsoft_agents_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="semantic_kernel"),
            attributes={},
        )
        assert is_microsoft_agents_span(span=span) is True

    def test_detects_autogen_attributes(self):
        from respan_exporter_microsoft_agents.utils import is_microsoft_agents_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="other"),
            attributes={"autogen.agent.name": "assistant"},
        )
        assert is_microsoft_agents_span(span=span) is True

    def test_detects_semantic_kernel_attributes(self):
        from respan_exporter_microsoft_agents.utils import is_microsoft_agents_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="other"),
            attributes={"semantic_kernel.function.name": "chat"},
        )
        assert is_microsoft_agents_span(span=span) is True

    def test_rejects_non_microsoft_span(self):
        from respan_exporter_microsoft_agents.utils import is_microsoft_agents_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="openai"),
            attributes={"http.method": "POST"},
        )
        assert is_microsoft_agents_span(span=span) is False

    def test_handles_missing_scope(self):
        from respan_exporter_microsoft_agents.utils import is_microsoft_agents_span

        span = SimpleNamespace(
            instrumentation_scope=None,
            attributes={"autogen.message.type": "text"},
        )
        assert is_microsoft_agents_span(span=span) is True

    def test_handles_no_attributes(self):
        from respan_exporter_microsoft_agents.utils import is_microsoft_agents_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="other"),
            attributes=None,
        )
        assert is_microsoft_agents_span(span=span) is False

    def test_detects_semantic_hyphen_kernel_scope(self):
        from respan_exporter_microsoft_agents.utils import is_microsoft_agents_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="semantic-kernel.ai"),
            attributes={},
        )
        assert is_microsoft_agents_span(span=span) is True

    def test_detects_microsoft_scope(self):
        from respan_exporter_microsoft_agents.utils import is_microsoft_agents_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="microsoft.agents"),
            attributes={},
        )
        assert is_microsoft_agents_span(span=span) is True

    def test_detects_ag_prefix_attributes(self):
        from respan_exporter_microsoft_agents.utils import is_microsoft_agents_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="other"),
            attributes={"ag.runtime.id": "abc"},
        )
        assert is_microsoft_agents_span(span=span) is True

    def test_detects_sk_prefix_attributes(self):
        from respan_exporter_microsoft_agents.utils import is_microsoft_agents_span

        span = SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name="other"),
            attributes={"sk.kernel.id": "xyz"},
        )
        assert is_microsoft_agents_span(span=span) is True


class TestOtelSpanToDict:
    """Test OTel span conversion."""

    def test_converts_basic_span(self):
        from respan_exporter_microsoft_agents.utils import otel_span_to_dict

        span_context = SimpleNamespace(trace_id=1234567890, span_id=9876543210)
        span = SimpleNamespace(
            context=span_context,
            parent=None,
            name="autogen.agent.run",
            kind=SimpleNamespace(name="INTERNAL"),
            start_time=1700000000000000000,
            end_time=1700000001000000000,
            attributes={"openinference.span.kind": "AGENT"},
            status=SimpleNamespace(status_code=SimpleNamespace(name="OK"), description=None),
            instrumentation_scope=SimpleNamespace(name="autogen"),
        )

        result = otel_span_to_dict(span=span)
        assert result["name"] == "autogen.agent.run"
        assert result["trace_id"] is not None
        assert result["status_code"] == 200


class TestGroupSpansByTrace:
    """Test trace grouping."""

    def test_groups_correctly(self):
        from respan_exporter_microsoft_agents.utils import group_spans_by_trace

        spans = [
            {"trace_id": "t1", "span_id": "a"},
            {"trace_id": "t1", "span_id": "b"},
            {"trace_id": "t2", "span_id": "c"},
        ]
        grouped = group_spans_by_trace(spans=spans)
        assert len(grouped) == 2
        assert len(grouped["t1"]) == 2
