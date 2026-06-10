from concurrent.futures import ThreadPoolExecutor

from respan_tracing import (
    ContextPropagatingThreadPoolExecutor,
    RespanTelemetry,
    get_client,
    submit_with_current_context,
    task,
)
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.testing.exporters import InMemorySpanExporter


class TestThreadPoolContextPropagation:
    @classmethod
    def setup_class(cls):
        RespanTracer.reset_instance()
        cls.exporter = InMemorySpanExporter()
        cls.telemetry = RespanTelemetry(
            app_name="test-threadpool-context",
            is_batching_enabled=False,
            is_auto_instrument=False,
            is_enabled=True,
        )
        cls.telemetry.add_processor(cls.exporter, name="dogfood")
        cls.client = get_client()

    @classmethod
    def teardown_class(cls):
        RespanTracer.reset_instance()

    def setup_method(self):
        self.exporter.clear()

    def exported_names(self) -> list[str]:
        self.telemetry.flush()
        return [span.name for span in self.exporter.get_finished_spans()]

    def assert_exported(self, names: list[str], span_name: str):
        assert any(name.startswith(span_name) for name in names)

    def assert_not_exported(self, names: list[str], span_name: str):
        assert not any(name.startswith(span_name) for name in names)

    def test_regular_threadpool_loses_processor_routing(self):
        @task(name="threadpool_worker")
        def worker():
            return "done"

        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
            processors="dogfood",
        ):
            with ThreadPoolExecutor(max_workers=1) as executor:
                assert executor.submit(worker).result() == "done"

        names = self.exported_names()

        self.assert_exported(names, "parent_workflow")
        self.assert_not_exported(names, "threadpool_worker")

    def test_context_propagating_threadpool_preserves_routing(self):
        @task(name="threadpool_worker")
        def worker():
            return "done"

        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
            processors="dogfood",
        ):
            with ContextPropagatingThreadPoolExecutor(max_workers=1) as executor:
                assert executor.submit(worker).result() == "done"

        names = self.exported_names()

        self.assert_exported(names, "parent_workflow")
        self.assert_exported(names, "threadpool_worker")

    def test_submit_with_current_context_preserves_routing(self):
        @task(name="submitted_worker")
        def worker():
            return "done"

        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
            processors="dogfood",
        ):
            with ThreadPoolExecutor(max_workers=1) as executor:
                assert submit_with_current_context(executor, worker).result() == "done"

        names = self.exported_names()

        self.assert_exported(names, "parent_workflow")
        self.assert_exported(names, "submitted_worker")

    def test_threadpool_spans_inside_span_buffer_flush_with_parent_context(self):
        def worker(index: int):
            with self.client.start_span(f"buffered_worker_{index}", kind="task"):
                return index

        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
            processors="dogfood",
        ):
            trace_id = self.client.get_current_trace_id()
            with self.client.get_span_buffer(trace_id) as buffer:
                with ContextPropagatingThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(worker, i) for i in range(12)]
                    assert [future.result() for future in futures] == list(range(12))

            assert buffer.get_span_count() == 12

        names = self.exported_names()

        self.assert_exported(names, "parent_workflow")
        for index in range(12):
            self.assert_exported(names, f"buffered_worker_{index}")
