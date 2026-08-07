import contextvars
import asyncio
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from threading import Thread
from typing import Awaitable, Callable, Iterator, Optional, ParamSpec, TypeVar

from opentelemetry import context as context_api
from opentelemetry.context import Context
from opentelemetry.semconv_ai import SpanAttributes

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class RespanContextSnapshot:
    """Captured Python and OpenTelemetry context for work crossing threads."""

    python_context: contextvars.Context
    otel_context: Context

    def run(self, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> R:
        """Run ``fn`` with the context that was active when this snapshot was made.

        ``fn`` is positional-only so a wrapped callable that itself takes ``fn`` or
        ``self`` keyword arguments does not collide with this method's parameters.

        Each call runs inside a *fresh* copy of the captured context, so a single
        snapshot can be reused as a wrapper and invoked repeatedly or concurrently
        from multiple threads — a live ``contextvars.Context`` can only be entered
        once, so re-entering the stored one directly would raise ``RuntimeError``
        under concurrency and would leak contextvar mutations between calls.
        """

        captured = list(self.python_context.items())
        otel_context = self.otel_context

        def invoke() -> R:
            for var, value in captured:
                var.set(value)
            token = context_api.attach(otel_context)
            try:
                return fn(*args, **kwargs)
            finally:
                context_api.detach(token)

        return contextvars.copy_context().run(invoke)

    def wrap(self, fn: Callable[P, R]) -> Callable[P, R]:
        """Return a callable that always runs in this captured context."""

        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            return self.run(fn, *args, **kwargs)

        return wrapped


def capture_context() -> RespanContextSnapshot:
    """Capture the active Respan, Python contextvars, and OpenTelemetry context."""

    return RespanContextSnapshot(
        python_context=contextvars.copy_context(),
        otel_context=context_api.get_current(),
    )


def wrap_with_current_context(fn: Callable[P, R]) -> Callable[P, R]:
    """Wrap ``fn`` so it runs with the context active at wrap time."""

    return capture_context().wrap(fn)


def add_done_callback_with_current_context(
    future: Future[R],
    callback: Callable[[Future[R]], object],
) -> Future[R]:
    """Register a Future callback that runs with the current tracing context."""

    snapshot = capture_context()

    def wrapped(done: Future[R]) -> None:
        snapshot.run(callback, done)

    future.add_done_callback(wrapped)
    return future


def submit_with_current_context(
    executor: Executor,
    fn: Callable[P, R],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> Future[R]:
    """Submit work to an executor while preserving the current tracing context."""

    snapshot = capture_context()
    return executor.submit(snapshot.run, fn, *args, **kwargs)


def run_in_executor_with_current_context(
    executor: Executor | None,
    fn: Callable[P, R],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> Awaitable[R]:
    """Run blocking work through asyncio while preserving tracing context.

    ``asyncio`` tasks preserve contextvars, but many agent workflows still cross
    into blocking SDK calls through ``loop.run_in_executor``. This helper
    captures the Respan/OpenTelemetry context before that handoff so spans
    created inside the executor keep the active parent, entity path, SpanBuffer,
    and processor routing.
    """

    loop = asyncio.get_running_loop()
    snapshot = capture_context()
    work = partial(snapshot.run, fn, *args, **kwargs)
    return loop.run_in_executor(executor, work)


async def to_thread_with_current_context(
    fn: Callable[P, R],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    """Async ``to_thread`` helper that preserves the active tracing context."""

    snapshot = capture_context()
    return await asyncio.to_thread(snapshot.run, fn, *args, **kwargs)


class ContextPropagatingThread(Thread):
    """Thread that preserves the Respan and OpenTelemetry context from creation.

    Both the ``target=`` form and subclasses that call ``super().run()`` from an
    overridden ``run()`` propagate the captured context. A subclass that overrides
    ``run()`` without calling ``super().run()`` must apply the context itself
    (e.g. wrap its body with ``capture_context().run(...)``), since a base class
    cannot wrap a method the subclass replaces.

    A ``context_snapshot`` may be shared across several threads — each thread runs
    inside its own fresh copy (see :meth:`RespanContextSnapshot.run`).
    """

    def __init__(
        self,
        group=None,
        target: Optional[Callable[..., object]] = None,
        name: Optional[str] = None,
        args=(),
        kwargs=None,
        *,
        daemon: Optional[bool] = None,
        context_snapshot: Optional[RespanContextSnapshot] = None,
    ) -> None:
        self._respan_snapshot = context_snapshot or capture_context()
        super().__init__(
            group=group,
            target=target,
            name=name,
            args=args,
            kwargs=kwargs or {},
            daemon=daemon,
        )

    def run(self) -> None:
        # Apply the captured context around the whole thread body so both the
        # target= form and subclasses that call super().run() propagate correctly.
        self._respan_snapshot.run(super().run)


class ContextPropagatingThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that preserves Respan and OpenTelemetry context.

    Use this when parallel agent or workflow steps are launched from inside a
    Respan span or SpanBuffer. Each submitted task receives the context that was
    active at submit time, so child spans stay attached to the right trace and
    buffered spans are flushed with the parent buffer.
    """

    def submit(
        self,
        fn: Callable[P, R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[R]:
        snapshot = capture_context()
        return super().submit(snapshot.run, fn, *args, **kwargs)


@contextmanager
def suppressed_parent_context() -> Iterator[None]:
    """Suppress the active OTel parent context for spans created in this block.

    Spans created (via @workflow / @task / @agent / @tool decorators or
    client.start_span) while this block is active see no active parent —
    they start fresh root traces. The OUTER span itself is untouched; only
    spans created INSIDE the with-block get the empty context.

    Use at execution boundaries where the inner work is conceptually
    independent of the outer span — most commonly, a Pulsar / Kafka /
    Celery batch consumer dispatching independent per-message tasks from
    inside a batch-level @workflow span:

        @workflow(name="my_consumer_handle_batch")
        async def _handle_batch(consumer, messages):
            for message in messages:
                with suppressed_parent_context():
                    await asyncio.to_thread(task.run, **message.payload)

    Without this, every per-message dispatch inherits _handle_batch's
    trace_id and downstream `count(distinct trace_unique_id)` collapses N
    messages into one trace per batch.

    Sub-workflow composition (workflow → workflow → workflow as one trace)
    is unaffected — the @workflow decorator's standard child-of-context
    behavior continues to work everywhere except inside this block.
    """
    token = context_api.attach(Context())
    try:
        yield
    finally:
        context_api.detach(token)


def get_entity_path(ctx: Optional[Context] = None) -> Optional[str]:
    """
    Retrieves the current entity path from the active context.
    This builds the hierarchical path like "workflow.task.subtask".

    Args:
        ctx: The context to read from (defaults to current active context)

    Returns:
        The entity path string or None if not set
    """
    if ctx is None:
        ctx = context_api.get_current()

    # First check for full entity path (set by TOOL/TASK spans)
    entity_path = context_api.get_value(
        SpanAttributes.TRACELOOP_ENTITY_PATH, context=ctx
    )
    if entity_path:
        return entity_path

    # Fall back to workflow name (set by WORKFLOW/AGENT spans)
    workflow_name = context_api.get_value(
        SpanAttributes.TRACELOOP_ENTITY_NAME, context=ctx
    )
    return workflow_name
