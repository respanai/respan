from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.trace import SpanKind

from respan_tracing import RespanTelemetry, task, workflow
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.testing import InMemorySpanExporter
from respan_tracing.utils import auto_flush
from respan_tracing.utils.span_factory import build_readable_span, inject_span


@pytest.fixture(autouse=True)
def reset_tracer_and_policy():
    RespanTracer.reset_instance()
    auto_flush.configure_auto_flush("root")
    yield
    RespanTracer.reset_instance()
    auto_flush.configure_auto_flush("root")


def test_flush_remains_non_terminal():
    telemetry = RespanTelemetry(app_name="flush-test", auto_flush="off")

    with patch.object(telemetry.tracer.tracer_provider, "force_flush") as force_flush, patch.object(
        telemetry.tracer.tracer_provider, "shutdown"
    ) as shutdown:
        telemetry.flush()

    force_flush.assert_called_once()
    shutdown.assert_not_called()


def test_auto_flush_root_drains_once_after_root_workflow():
    RespanTelemetry(app_name="root-test", auto_flush="root")

    @task(name="child_task")
    def child_task():
        return "child"

    @workflow(name="root_workflow")
    def root_workflow():
        return child_task()

    with patch("respan_tracing.utils.auto_flush.flush_now", return_value=True) as flush_now:
        assert root_workflow() == "child"

    flush_now.assert_called_once()


def test_auto_flush_off_does_not_drain_after_root_workflow():
    RespanTelemetry(app_name="off-test", auto_flush="off")

    @workflow(name="root_workflow")
    def root_workflow():
        return "done"

    with patch("respan_tracing.utils.auto_flush.flush_now", return_value=True) as flush_now:
        assert root_workflow() == "done"

    flush_now.assert_not_called()


def test_auto_flush_always_drains_nested_spans():
    RespanTelemetry(app_name="always-test", auto_flush="always")

    @task(name="child_task")
    def child_task():
        return "child"

    @workflow(name="root_workflow")
    def root_workflow():
        return child_task()

    with patch("respan_tracing.utils.auto_flush.flush_now", return_value=True) as flush_now:
        assert root_workflow() == "child"

    assert flush_now.call_count == 2


def test_auto_flush_root_exports_batched_spans_without_explicit_flush():
    exporter = InMemorySpanExporter()
    telemetry = RespanTelemetry(app_name="export-root-test", auto_flush="root")
    telemetry.add_processor(exporter=exporter)

    @task(name="child_task")
    def child_task():
        return "child"

    @workflow(name="root_workflow")
    def root_workflow():
        return child_task()

    assert root_workflow() == "child"

    exported_names = {span.name for span in exporter.get_finished_spans()}
    assert {"root_workflow.workflow", "child_task.task"}.issubset(exported_names)


def test_auto_flush_off_keeps_batched_spans_pending_until_flush():
    exporter = InMemorySpanExporter()
    telemetry = RespanTelemetry(app_name="export-off-test", auto_flush="off")
    telemetry.add_processor(exporter=exporter)

    @workflow(name="root_workflow")
    def root_workflow():
        return "done"

    assert root_workflow() == "done"
    assert exporter.get_finished_spans() == ()

    telemetry.flush()
    exported_names = {span.name for span in exporter.get_finished_spans()}
    assert "root_workflow.workflow" in exported_names


def test_inject_span_runs_auto_flush_hook_after_processor_on_end():
    span = build_readable_span(
        "synthetic.task",
        attributes={
            "traceloop.span.kind": "task",
            "traceloop.entity.path": "",
        },
        kind=SpanKind.INTERNAL,
    )
    processor = MagicMock()
    provider = MagicMock(_active_span_processor=processor)

    with patch(
        "respan_tracing.utils.span_factory.trace.get_tracer_provider",
        return_value=provider,
    ), patch("respan_tracing.utils.span_factory.flush_after_injected_span") as flush_hook:
        assert inject_span(span) is True

    processor.on_end.assert_called_once_with(span)
    flush_hook.assert_called_once()


def test_injected_span_policy_uses_debounce_for_root_mode():
    auto_flush.configure_auto_flush("root")

    with patch("respan_tracing.utils.auto_flush._schedule_debounced_flush") as schedule, patch(
        "respan_tracing.utils.auto_flush.flush_now"
    ) as flush_now:
        auto_flush.flush_after_injected_span()

    schedule.assert_called_once()
    flush_now.assert_not_called()


def test_injected_span_policy_flushes_immediately_for_always_mode():
    auto_flush.configure_auto_flush("always")

    with patch("respan_tracing.utils.auto_flush.flush_now", return_value=True) as flush_now:
        auto_flush.flush_after_injected_span()

    flush_now.assert_called_once()
