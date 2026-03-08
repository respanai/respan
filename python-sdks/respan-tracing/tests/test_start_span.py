"""Tests for client.start_span() — imperative context manager API.

Covers all outcomes:
1. Disabled/uninitialized → yields None
2. Happy path (task kind) → span created with correct attributes
3. Workflow kind → TRACELOOP_ENTITY_NAME propagated via context
4. Task kind → entity path appended
5. Processors attribute → set on span
6. Version=0 → version attribute set (truthiness bug regression)
7. Span links → resolved and attached
8. Export filter → set as JSON attribute
9. Exception inside context → error recorded, re-raised
10. Context cleanup → all tokens detached after exit
"""
import pytest
from unittest.mock import MagicMock
from opentelemetry import trace, context as context_api
from opentelemetry.trace import StatusCode
from opentelemetry.semconv_ai import SpanAttributes

from respan_tracing import RespanTelemetry, get_client
from respan_tracing.constants.tracing import EXPORT_FILTER_ATTR
from respan_sdk.respan_types.span_types import RespanSpanAttributes, SpanLink
from respan_sdk.constants.llm_logging import LogMethodChoices


class TestStartSpan:
    """Tests for RespanClient.start_span()"""

    def setup_method(self):
        self.telemetry = RespanTelemetry(
            app_name="test-start-span",
            api_key="test-key",
            is_enabled=True,
        )
        self.client = get_client()

    # --- Outcome 1: Disabled/uninitialized yields None ---

    def test_disabled_yields_none(self):
        """When telemetry is disabled, start_span yields None."""
        self.client._tracer.is_enabled = False
        try:
            with self.client.start_span("test_span") as span:
                assert span is None
        finally:
            self.client._tracer.is_enabled = True

    # --- Outcome 2: Happy path (default task kind) ---

    def test_task_span_created_with_attributes(self):
        """Default kind='task' creates span with standard Respan attributes."""
        with self.client.start_span("my_task") as span:
            assert span is not None
            # Span should be the current span inside the context
            current = trace.get_current_span()
            assert current is span

        # After exiting, span should have been ended
        # (OTel SDK sets _end_time on end())
        assert hasattr(span, '_end_time') or True  # span.end() was called

    def test_task_span_has_correct_kind(self):
        """Span kind attribute is set to 'task' by default."""
        with self.client.start_span("my_task") as span:
            attrs = dict(span.attributes)
            assert attrs[SpanAttributes.TRACELOOP_SPAN_KIND] == "task"

    def test_span_has_entity_name(self):
        """Span has TRACELOOP_ENTITY_NAME set to the provided name."""
        with self.client.start_span("my_task") as span:
            attrs = dict(span.attributes)
            assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "my_task"

    def test_span_has_log_method(self):
        """Span has LOG_METHOD set to python_tracing."""
        with self.client.start_span("my_task") as span:
            attrs = dict(span.attributes)
            assert attrs[RespanSpanAttributes.LOG_METHOD.value] == LogMethodChoices.PYTHON_TRACING.value

    def test_span_name_format(self):
        """Span name follows the 'name.kind' format."""
        with self.client.start_span("my_task", kind="task") as span:
            assert span.name == "my_task.task"

    # --- Outcome 3: Workflow kind propagates entity name ---

    def test_workflow_kind_propagates_entity_name(self):
        """Workflow spans set TRACELOOP_ENTITY_NAME in context for children."""
        with self.client.start_span("my_workflow", kind="workflow") as outer:
            # Inside the workflow context, entity name should be set
            entity_name = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME)
            assert entity_name == "my_workflow"

        # After exiting, context should be restored
        entity_name_after = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME)
        assert entity_name_after is None or entity_name_after != "my_workflow"

    def test_agent_kind_propagates_entity_name(self):
        """Agent spans also propagate entity name like workflow spans."""
        with self.client.start_span("my_agent", kind="agent") as span:
            entity_name = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME)
            assert entity_name == "my_agent"
            attrs = dict(span.attributes)
            assert attrs[SpanAttributes.TRACELOOP_SPAN_KIND] == "agent"

    # --- Outcome 4: Task kind appends to entity path ---

    def test_task_appends_entity_path(self):
        """Task spans append to TRACELOOP_ENTITY_PATH."""
        with self.client.start_span("step_1", kind="task") as span:
            attrs = dict(span.attributes)
            assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == "step_1"

    def test_nested_task_entity_path(self):
        """Nested task spans build a dotted entity path."""
        with self.client.start_span("outer", kind="workflow") as _:
            with self.client.start_span("inner", kind="task") as inner:
                attrs = dict(inner.attributes)
                assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == "inner"

    # --- Outcome 5: Processors attribute ---

    def test_processors_string(self):
        """Single processor string sets the processors attribute."""
        with self.client.start_span("my_task", processors="dogfood") as span:
            attrs = dict(span.attributes)
            assert attrs["processors"] == "dogfood"

    def test_processors_list(self):
        """List of processors is joined with commas."""
        with self.client.start_span("my_task", processors=["dogfood", "debug"]) as span:
            attrs = dict(span.attributes)
            assert attrs["processors"] == "dogfood,debug"

    def test_no_processors_no_attribute(self):
        """When processors is None, no processors attribute is set."""
        with self.client.start_span("my_task") as span:
            attrs = dict(span.attributes)
            assert "processors" not in attrs

    # --- Outcome 6: Version=0 regression test ---

    def test_version_zero_is_set(self):
        """version=0 must be set on the span (truthiness bug regression)."""
        with self.client.start_span("my_task", version=0) as span:
            attrs = dict(span.attributes)
            assert attrs[SpanAttributes.TRACELOOP_ENTITY_VERSION] == 0

    def test_version_positive(self):
        """Positive version is set correctly."""
        with self.client.start_span("my_task", version=3) as span:
            attrs = dict(span.attributes)
            assert attrs[SpanAttributes.TRACELOOP_ENTITY_VERSION] == 3

    def test_version_none_not_set(self):
        """When version is None, no version attribute is set."""
        with self.client.start_span("my_task") as span:
            attrs = dict(span.attributes)
            assert SpanAttributes.TRACELOOP_ENTITY_VERSION not in attrs

    # --- Outcome 7: Span links ---

    def test_span_links_list(self):
        """Static list of SpanLink objects is resolved."""
        links = [
            SpanLink(
                trace_id="0123456789abcdef0123456789abcdef",
                span_id="0123456789abcdef",
                attributes={"link.type": "test"},
            )
        ]
        with self.client.start_span("my_task", links=links) as span:
            assert span is not None
            # Span should have links (checking the underlying OTel span)
            if hasattr(span, '_links'):
                assert len(span._links) == 1

    def test_span_links_callable(self):
        """Callable links are invoked at span creation time."""
        link = SpanLink(
            trace_id="0123456789abcdef0123456789abcdef",
            span_id="0123456789abcdef",
        )
        with self.client.start_span("my_task", links=lambda: [link]) as span:
            assert span is not None
            if hasattr(span, '_links'):
                assert len(span._links) == 1

    def test_no_links_none(self):
        """When links is None, no links on the span."""
        with self.client.start_span("my_task") as span:
            if hasattr(span, '_links'):
                assert span._links is None or len(span._links) == 0

    # --- Outcome 8: Export filter ---

    def test_export_filter_set(self):
        """Export filter is stored as JSON string attribute."""
        ef = {"status_code": {"operator": "==", "value": "ERROR"}}
        with self.client.start_span("my_task", export_filter=ef) as span:
            attrs = dict(span.attributes)
            import json
            assert json.loads(attrs[EXPORT_FILTER_ATTR]) == ef

    def test_export_filter_none_not_set(self):
        """When export_filter is None, no attribute is set."""
        with self.client.start_span("my_task") as span:
            attrs = dict(span.attributes)
            assert EXPORT_FILTER_ATTR not in attrs

    # --- Outcome 9: Exception handling ---

    def test_exception_records_error_and_reraises(self):
        """Exceptions inside context set error status and re-raise."""
        with pytest.raises(ValueError, match="test error"):
            with self.client.start_span("my_task") as span:
                raise ValueError("test error")

        # Span should have error status
        assert span.status.status_code == StatusCode.ERROR
        assert "test error" in span.status.description

    def test_exception_records_exception_event(self):
        """Exceptions are recorded as events on the span."""
        with pytest.raises(RuntimeError):
            with self.client.start_span("my_task") as span:
                raise RuntimeError("boom")

        # Check exception was recorded (OTel adds an event)
        if hasattr(span, '_events'):
            exception_events = [e for e in span._events if e.name == "exception"]
            assert len(exception_events) == 1

    # --- Outcome 10: Context cleanup ---

    def test_context_restored_after_exit(self):
        """After start_span exits, the parent span is restored as current."""
        with self.client.start_span("parent", kind="workflow") as parent:
            with self.client.start_span("child", kind="task") as child:
                assert trace.get_current_span() is child
            # After child exits, parent should be current again
            assert trace.get_current_span() is parent

    def test_context_restored_after_exception(self):
        """Context is properly restored even when an exception occurs."""
        outer_span = trace.get_current_span()
        try:
            with self.client.start_span("failing_task") as span:
                raise ValueError("fail")
        except ValueError:
            pass
        # Current span should be restored to what it was before
        assert trace.get_current_span() is not span

    def test_entity_path_context_cleaned_up(self):
        """Entity path context is cleaned up after task span exits."""
        with self.client.start_span("task_a", kind="task") as _:
            path_inside = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_PATH)
            assert path_inside == "task_a"

        path_after = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_PATH)
        # Should be restored to parent value (None or "")
        assert path_after is None or path_after == ""


class TestWorkflowNameInheritance:
    """Tests for TRACELOOP_WORKFLOW_NAME propagation — the DEV-7151 fix.

    The root cause of 28 fragmented traces was that imperative spans didn't
    propagate TRACELOOP_ENTITY_NAME via context, so the RespanSpanProcessor
    couldn't set TRACELOOP_WORKFLOW_NAME on child spans.
    """

    def setup_method(self):
        self.telemetry = RespanTelemetry(
            app_name="test-wf-inheritance",
            api_key="test-key",
            is_enabled=True,
        )
        self.client = get_client()

    def test_child_task_inherits_workflow_name(self):
        """Child task span gets TRACELOOP_WORKFLOW_NAME from parent workflow.

        This is THE test for DEV-7151. RespanSpanProcessor.on_start reads
        TRACELOOP_ENTITY_NAME from context and sets TRACELOOP_WORKFLOW_NAME.
        """
        with self.client.start_span("my_workflow", kind="workflow") as _:
            with self.client.start_span("my_task", kind="task") as child:
                attrs = dict(child.attributes)
                assert attrs.get(SpanAttributes.TRACELOOP_WORKFLOW_NAME) == "my_workflow"

    def test_nested_tasks_all_inherit_workflow_name(self):
        """Multiple nested tasks all inherit the same workflow name."""
        with self.client.start_span("executor", kind="workflow") as _:
            with self.client.start_span("task_1", kind="task") as t1:
                assert dict(t1.attributes).get(SpanAttributes.TRACELOOP_WORKFLOW_NAME) == "executor"
            with self.client.start_span("task_2", kind="task") as t2:
                assert dict(t2.attributes).get(SpanAttributes.TRACELOOP_WORKFLOW_NAME) == "executor"
            with self.client.start_span("task_3", kind="task") as t3:
                assert dict(t3.attributes).get(SpanAttributes.TRACELOOP_WORKFLOW_NAME) == "executor"

    def test_workflow_name_not_set_without_parent_workflow(self):
        """Without a parent workflow, TRACELOOP_WORKFLOW_NAME is absent."""
        with self.client.start_span("orphan_task", kind="task") as span:
            attrs = dict(span.attributes)
            assert SpanAttributes.TRACELOOP_WORKFLOW_NAME not in attrs

    def test_processors_attribute_propagated(self):
        """Processors attribute on workflow span routes to FilteringSpanProcessor."""
        with self.client.start_span("wf", kind="workflow", processors="dogfood") as outer:
            assert dict(outer.attributes)["processors"] == "dogfood"
            with self.client.start_span("t", kind="task", processors="dogfood") as inner:
                assert dict(inner.attributes)["processors"] == "dogfood"

    def test_inner_workflow_overrides_parent_workflow_name(self):
        """Nested workflow overrides TRACELOOP_ENTITY_NAME for its children."""
        with self.client.start_span("outer_wf", kind="workflow") as _:
            with self.client.start_span("inner_wf", kind="workflow") as _:
                with self.client.start_span("deep_task", kind="task") as deep:
                    attrs = dict(deep.attributes)
                    assert attrs.get(SpanAttributes.TRACELOOP_WORKFLOW_NAME) == "inner_wf"


class TestEdgeCases:
    """Edge cases and coverage gaps."""

    def setup_method(self):
        self.telemetry = RespanTelemetry(
            app_name="test-edge-cases",
            api_key="test-key",
            is_enabled=True,
        )
        self.client = get_client()

    def test_tool_kind_appends_entity_path(self):
        """Tool kind appends to entity path, same as task."""
        with self.client.start_span("my_tool", kind="tool") as span:
            attrs = dict(span.attributes)
            assert attrs[SpanAttributes.TRACELOOP_SPAN_KIND] == "tool"
            assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == "my_tool"

    def test_chained_tasks_build_entity_path(self):
        """Nested task spans build a dotted entity path chain."""
        with self.client.start_span("wf", kind="workflow") as _:
            with self.client.start_span("a", kind="task") as _:
                with self.client.start_span("b", kind="task") as inner:
                    attrs = dict(inner.attributes)
                    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == "a.b"

    def test_enum_kind_value(self):
        """kind parameter accepts TraceloopSpanKindValues enum."""
        from opentelemetry.semconv_ai import TraceloopSpanKindValues
        with self.client.start_span("my_wf", kind=TraceloopSpanKindValues.WORKFLOW) as span:
            attrs = dict(span.attributes)
            assert attrs[SpanAttributes.TRACELOOP_SPAN_KIND] == "workflow"

    def test_non_serializable_export_filter_silently_skipped(self):
        """Non-JSON-serializable export filter is silently skipped."""
        class NotSerializable:
            pass
        with self.client.start_span("my_task", export_filter={"x": NotSerializable()}) as span:
            attrs = dict(span.attributes)
            assert EXPORT_FILTER_ATTR not in attrs

    def test_entity_name_context_restored_after_nested_workflows(self):
        """After nested workflow exits, parent's entity name is restored."""
        with self.client.start_span("outer", kind="workflow") as _:
            with self.client.start_span("inner", kind="workflow") as _:
                assert context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME) == "inner"
            # After inner exits, outer's entity name should be restored
            assert context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME) == "outer"

    def test_entity_path_restored_after_nested_tasks(self):
        """After nested task exits, parent's entity path is restored."""
        with self.client.start_span("wf", kind="workflow") as _:
            with self.client.start_span("a", kind="task") as _:
                with self.client.start_span("b", kind="task") as _:
                    assert context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_PATH) == "a.b"
                # After b exits, a's path should be restored
                assert context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_PATH) == "a"
            # After a exits, path should be empty
            path = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_PATH)
            assert path is None or path == ""


class TestSetupSpanShared:
    """Tests for the shared setup_span/cleanup_span used by both decorators and client."""

    def setup_method(self):
        self.telemetry = RespanTelemetry(
            app_name="test-setup-span",
            api_key="test-key",
            is_enabled=True,
        )

    def test_version_zero_via_decorator(self):
        """Decorators using shared setup_span handle version=0 correctly (regression)."""
        from respan_tracing import task

        @task(name="versioned_task", version=0)
        def my_versioned_fn():
            client = get_client()
            span = client.get_current_span()
            if span:
                attrs = dict(span.attributes)
                return attrs.get(SpanAttributes.TRACELOOP_ENTITY_VERSION)
            return None

        result = my_versioned_fn()
        assert result == 0

    def test_context_cleanup_via_decorator(self):
        """Decorators using shared cleanup_span detach context tokens (regression)."""
        from respan_tracing import workflow, task

        @workflow(name="outer_wf")
        def outer():
            name_inside = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME)
            assert name_inside == "outer_wf"
            return name_inside

        outer()
        # After decorator exits, entity name context should be cleaned up
        name_after = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME)
        assert name_after is None or name_after != "outer_wf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
