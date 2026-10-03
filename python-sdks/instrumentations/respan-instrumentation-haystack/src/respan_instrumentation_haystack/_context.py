"""Execution contexts shared by Haystack pipelines and component spans."""

import contextvars
from dataclasses import dataclass, field
from typing import Any

from ._constants import (
    HAYSTACK_COMPONENT_NAME_PARAMETER,
    RESPAN_HAYSTACK_COMPONENT_CONTEXT_VAR_NAME,
    RESPAN_HAYSTACK_PIPELINE_CONTEXT_VAR_NAME,
)


@dataclass
class _HaystackPipelineRunContext:
    graph: Any
    completed_span_id_by_component: dict[str, str] = field(default_factory=dict)
    completion_order_by_component: dict[str, int] = field(default_factory=dict)
    completion_counter: int = 0
    pipeline_span_id: str | None = None

    def record_completion(self, component_name: str, span_id: str) -> None:
        self.completion_counter += 1
        self.completed_span_id_by_component[component_name] = span_id
        self.completion_order_by_component[component_name] = self.completion_counter


@dataclass(frozen=True)
class _HaystackComponentRunContext:
    component_name: str
    pipeline_context: _HaystackPipelineRunContext | None


_CURRENT_PIPELINE_RUN_CONTEXT: contextvars.ContextVar[
    _HaystackPipelineRunContext | None
] = contextvars.ContextVar(
    RESPAN_HAYSTACK_PIPELINE_CONTEXT_VAR_NAME,
    default=None,
)
_CURRENT_COMPONENT_RUN_CONTEXT: contextvars.ContextVar[
    _HaystackComponentRunContext | None
] = contextvars.ContextVar(
    RESPAN_HAYSTACK_COMPONENT_CONTEXT_VAR_NAME,
    default=None,
)


def _pipeline_run_context_wrapper(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    token = _CURRENT_PIPELINE_RUN_CONTEXT.set(
        _HaystackPipelineRunContext(graph=getattr(instance, "graph", None))
    )
    try:
        return wrapped(*args, **kwargs)
    finally:
        _CURRENT_PIPELINE_RUN_CONTEXT.reset(token)


async def _async_pipeline_run_context_wrapper(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    token = _CURRENT_PIPELINE_RUN_CONTEXT.set(
        _HaystackPipelineRunContext(graph=getattr(instance, "graph", None))
    )
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _CURRENT_PIPELINE_RUN_CONTEXT.reset(token)


def _async_pipeline_run_async_generator_context_wrapper(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    async def run_with_context():
        pipeline_context = _HaystackPipelineRunContext(
            graph=getattr(instance, "graph", None)
        )
        iterator = wrapped(*args, **kwargs)
        try:
            while True:
                token = _CURRENT_PIPELINE_RUN_CONTEXT.set(pipeline_context)
                try:
                    output = await anext(iterator)
                except StopAsyncIteration:
                    return
                finally:
                    _CURRENT_PIPELINE_RUN_CONTEXT.reset(token)
                # The caller must never inherit a token owned by this iterator.
                yield output
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                token = _CURRENT_PIPELINE_RUN_CONTEXT.set(pipeline_context)
                try:
                    await close()
                finally:
                    _CURRENT_PIPELINE_RUN_CONTEXT.reset(token)

    return run_with_context()


def _component_run_context_wrapper(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    component_name = _get_component_name(args=args, kwargs=kwargs)
    token = _CURRENT_COMPONENT_RUN_CONTEXT.set(
        _HaystackComponentRunContext(
            component_name=component_name,
            pipeline_context=_CURRENT_PIPELINE_RUN_CONTEXT.get(),
        )
    )
    try:
        return wrapped(*args, **kwargs)
    finally:
        _CURRENT_COMPONENT_RUN_CONTEXT.reset(token)


async def _async_component_run_context_wrapper(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    component_name = _get_component_name(args=args, kwargs=kwargs)
    token = _CURRENT_COMPONENT_RUN_CONTEXT.set(
        _HaystackComponentRunContext(
            component_name=component_name,
            pipeline_context=_CURRENT_PIPELINE_RUN_CONTEXT.get(),
        )
    )
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _CURRENT_COMPONENT_RUN_CONTEXT.reset(token)


def _get_component_name(*, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    if args:
        return str(args[0])
    return str(kwargs.get(HAYSTACK_COMPONENT_NAME_PARAMETER, ""))
