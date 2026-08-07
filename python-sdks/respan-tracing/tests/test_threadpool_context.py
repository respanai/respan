import asyncio

import pytest

from concurrent.futures import ThreadPoolExecutor

from respan_tracing import (
    ContextPropagatingThread,
    ContextPropagatingThreadPoolExecutor,
    RespanTelemetry,
    add_done_callback_with_current_context,
    get_client,
    run_in_executor_with_current_context,
    submit_with_current_context,
    task,
    to_thread_with_current_context,
    wrap_with_current_context,
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

    def exported_spans(self):
        self.telemetry.flush()
        return self.exporter.get_finished_spans()

    def assert_exported(self, names: list[str], span_name: str):
        assert any(name.startswith(span_name) for name in names)

    def assert_not_exported(self, names: list[str], span_name: str):
        assert not any(name.startswith(span_name) for name in names)

    def span_by_prefix(self, prefix: str):
        matches = [
            span for span in self.exported_spans() if span.name.startswith(f"{prefix}.")
        ]
        assert len(matches) == 1, [span.name for span in self.exported_spans()]
        return matches[0]

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

        parent = self.span_by_prefix("parent_workflow")
        worker_span = self.span_by_prefix("threadpool_worker")
        assert worker_span.context.trace_id == parent.context.trace_id
        assert worker_span.parent.span_id == parent.context.span_id

    def test_context_propagating_threadpool_map_preserves_routing(self):
        @task(name="mapped_worker")
        def worker(index: int):
            return index * 2

        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
            processors="dogfood",
        ):
            with ContextPropagatingThreadPoolExecutor(max_workers=3) as executor:
                assert list(executor.map(worker, range(6))) == [0, 2, 4, 6, 8, 10]

        names = self.exported_names()

        self.assert_exported(names, "parent_workflow")
        assert sum(name.startswith("mapped_worker") for name in names) == 6

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

    def test_context_propagating_thread_preserves_routing(self):
        @task(name="thread_worker")
        def worker():
            return "done"

        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
            processors="dogfood",
        ):
            thread = ContextPropagatingThread(target=worker)
            thread.start()
            thread.join()

        names = self.exported_names()

        self.assert_exported(names, "parent_workflow")
        self.assert_exported(names, "thread_worker")

        parent = self.span_by_prefix("parent_workflow")
        worker_span = self.span_by_prefix("thread_worker")
        assert worker_span.context.trace_id == parent.context.trace_id
        assert worker_span.parent.span_id == parent.context.span_id

    def test_future_done_callback_preserves_routing(self):
        def complete_work():
            return "done"

        def callback(_future):
            with self.client.start_span("future_callback", kind="task"):
                pass

        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
            processors="dogfood",
        ):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(complete_work)
                add_done_callback_with_current_context(future, callback)
                assert future.result() == "done"

        names = self.exported_names()

        self.assert_exported(names, "parent_workflow")
        self.assert_exported(names, "future_callback")

        parent = self.span_by_prefix("parent_workflow")
        callback_span = self.span_by_prefix("future_callback")
        assert callback_span.context.trace_id == parent.context.trace_id
        assert callback_span.parent.span_id == parent.context.span_id

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

        parent = self.span_by_prefix("parent_workflow")
        for index in range(12):
            worker_span = self.span_by_prefix(f"buffered_worker_{index}")
            assert worker_span.context.trace_id == parent.context.trace_id
            assert worker_span.parent.span_id == parent.context.span_id

    @pytest.mark.asyncio
    async def test_plain_asyncio_run_in_executor_loses_processor_routing(self):
        @task(name="async_executor_worker")
        def worker():
            return "done"

        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
            processors="dogfood",
        ):
            with ThreadPoolExecutor(max_workers=1) as executor:
                loop = asyncio.get_running_loop()
                assert await loop.run_in_executor(executor, worker) == "done"

        names = self.exported_names()

        self.assert_exported(names, "parent_workflow")
        self.assert_not_exported(names, "async_executor_worker")

    @pytest.mark.asyncio
    async def test_run_in_executor_with_current_context_preserves_routing(self):
        @task(name="async_executor_worker")
        def worker(value: int, *, scale: int):
            return value * scale

        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
            processors="dogfood",
        ):
            with ThreadPoolExecutor(max_workers=1) as executor:
                assert (
                    await run_in_executor_with_current_context(
                        executor,
                        worker,
                        7,
                        scale=6,
                    )
                    == 42
                )

        names = self.exported_names()

        self.assert_exported(names, "parent_workflow")
        self.assert_exported(names, "async_executor_worker")

        parent = self.span_by_prefix("parent_workflow")
        worker_span = self.span_by_prefix("async_executor_worker")
        assert worker_span.context.trace_id == parent.context.trace_id
        assert worker_span.parent.span_id == parent.context.span_id

    @pytest.mark.asyncio
    async def test_to_thread_with_current_context_preserves_routing(self):
        @task(name="to_thread_worker")
        def worker():
            return "done"

        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
            processors="dogfood",
        ):
            assert await to_thread_with_current_context(worker) == "done"

        names = self.exported_names()

        self.assert_exported(names, "parent_workflow")
        self.assert_exported(names, "to_thread_worker")

    def test_run_preserves_fn_and_self_kwargs(self):
        # Regression: RespanContextSnapshot.run(fn, /, ...) is positional-only, so a
        # wrapped callable that itself takes `fn` or `self` keyword arguments does
        # not collide with run()'s own parameters (previously raised TypeError).
        @task(name="kwarg_worker")
        def worker(*, fn, self):
            return (fn, self)

        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
            processors="dogfood",
        ):
            with ContextPropagatingThreadPoolExecutor(max_workers=1) as executor:
                assert executor.submit(worker, fn="cb", self="obj").result() == (
                    "cb",
                    "obj",
                )
            with ThreadPoolExecutor(max_workers=1) as executor:
                assert submit_with_current_context(
                    executor, worker, fn="x", self="y"
                ).result() == ("x", "y")

        names = self.exported_names()
        self.assert_exported(names, "parent_workflow")
        self.assert_exported(names, "kwarg_worker")

    def test_wrapped_callable_reusable_across_concurrent_threads(self):
        # Regression: a single captured snapshot / wrapped callable must be safe to
        # invoke concurrently. A live contextvars.Context can only be entered once,
        # so reusing the stored one directly raised RuntimeError under parallel load.
        @task(name="wrapped_worker")
        def worker(index: int):
            return index * 2

        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
            processors="dogfood",
        ):
            # One wrapped callable fanned across worker threads via ex.map.
            wrapped = wrap_with_current_context(worker)
            with ThreadPoolExecutor(max_workers=4) as executor:
                assert sorted(executor.map(wrapped, range(12))) == [
                    i * 2 for i in range(12)
                ]

        names = self.exported_names()
        self.assert_exported(names, "parent_workflow")
        assert sum(name.startswith("wrapped_worker") for name in names) == 12

    def test_context_propagating_thread_subclass_run_preserves_routing(self):
        # Regression: subclasses overriding run() and calling super().run() must
        # still propagate the captured context (not only the target= form).
        class Worker(ContextPropagatingThread):
            def run(self):
                super().run()

        def work():
            with self.client.start_span("subclass_worker", kind="task"):
                pass

        with self.client.start_span(
            "parent_workflow",
            kind="workflow",
            processors="dogfood",
        ):
            thread = Worker(target=work)
            thread.start()
            thread.join()

        names = self.exported_names()
        self.assert_exported(names, "parent_workflow")
        self.assert_exported(names, "subclass_worker")

        parent = self.span_by_prefix("parent_workflow")
        worker_span = self.span_by_prefix("subclass_worker")
        assert worker_span.context.trace_id == parent.context.trace_id
        assert worker_span.parent.span_id == parent.context.span_id
