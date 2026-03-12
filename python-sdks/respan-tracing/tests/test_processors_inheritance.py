"""Tests for processors attribute inheritance from parent spans.

When a parent span has processors="dogfood,production" and a child span
doesn't explicitly set processors, the child should inherit the parent's
processors attribute. This enables a single @workflow(processors=...) to
route all child @task spans to the same processors automatically.
"""
import pytest
from unittest.mock import MagicMock, patch

from respan_tracing import RespanTelemetry, get_client
from respan_tracing.constants.tracing import PROCESSORS_ATTR
from respan_tracing.decorators import workflow, task


class TestProcessorsInheritance:
    """Tests for processors inheritance from parent to child spans."""

    def setup_method(self):
        self.telemetry = RespanTelemetry(
            app_name="test-processors-inheritance",
            api_key="test-key",
            is_enabled=True,
        )
        self.client = get_client()

    def test_child_inherits_processors_from_parent(self):
        """Child span without explicit processors inherits parent's processors."""
        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
            processors="dogfood,production",
        ) as parent_span:
            with self.client.start_span("child_task", kind="task") as child_span:
                assert child_span.attributes.get(PROCESSORS_ATTR) == "dogfood,production"

    def test_child_with_explicit_processors_not_overridden(self):
        """Child span with its own processors keeps them (no override)."""
        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
            processors="dogfood,production",
        ) as parent_span:
            with self.client.start_span(
                "child_task",
                kind="task",
                processors="debug",
            ) as child_span:
                assert child_span.attributes.get(PROCESSORS_ATTR) == "debug"

    def test_no_inheritance_when_parent_has_no_processors(self):
        """Child span stays without processors when parent has none."""
        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
        ) as parent_span:
            with self.client.start_span("child_task", kind="task") as child_span:
                assert child_span.attributes.get(PROCESSORS_ATTR) is None

    def test_deep_nesting_inherits_through_chain(self):
        """Processors propagate through multiple nesting levels."""
        with self.client.start_span(
            "root",
            kind="workflow",
            processors="dogfood",
        ) as root:
            with self.client.start_span("mid", kind="task") as mid:
                # mid inherited "dogfood" from root
                assert mid.attributes.get(PROCESSORS_ATTR) == "dogfood"
                with self.client.start_span("leaf", kind="task") as leaf:
                    # leaf inherits "dogfood" from mid (which inherited from root)
                    assert leaf.attributes.get(PROCESSORS_ATTR) == "dogfood"

    def test_span_buffer_children_inherit_processors(self):
        """Spans created inside SpanBuffer inherit processors from parent context."""
        # When SpanBuffer is used with parent context, child spans should
        # inherit processors from whatever span is in context
        with self.client.start_span(
            "outer_workflow",
            kind="workflow",
            processors="dogfood,production",
        ) as outer:
            with self.client.get_span_buffer("trace-123") as buffer:
                # Spans created in buffer should inherit from outer_workflow
                span_id = buffer.create_span("buffered_step", {"status": "ok"})

            # Check the buffered span got processors inherited
            spans = buffer.get_all_spans()
            assert len(spans) == 1
            assert spans[0].attributes.get(PROCESSORS_ATTR) == "dogfood,production"

    def test_decorator_processors_inheritance(self):
        """@task inside @workflow inherits processors via decorator."""
        from opentelemetry import trace as trace_api
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        exporter = InMemorySpanExporter()
        provider = trace_api.get_tracer_provider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        @workflow(name="test_wf", processors="dogfood")
        def my_workflow():
            @task(name="test_task")
            def my_task():
                return "done"
            return my_task()

        my_workflow()
        provider.force_flush()

        spans = exporter.get_finished_spans()
        task_spans = [s for s in spans if "test_task" in s.name]
        assert len(task_spans) >= 1, f"Expected task span, got: {[s.name for s in spans]}"
        assert task_spans[0].attributes.get(PROCESSORS_ATTR) == "dogfood"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
