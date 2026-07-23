"""Replicate SDK instrumentation plugin for Respan."""

from __future__ import annotations

import contextlib
import contextvars
import importlib
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable

from opentelemetry import trace

from respan_instrumentation_replicate._constants import (
    ASYNC_PREFIX,
    INPUT_KEY,
    MAX_STREAM_CHUNKS,
    PREDICTION_RESPAN_MODEL_ATTR,
    REPLICATE_INSTRUMENTATION_NAME,
    REPLICATE_PREDICTION_CREATE_SPAN_NAME,
    REPLICATE_PREDICTION_WAIT_SPAN_NAME,
    REPLICATE_RUN_SPAN_NAME,
    REPLICATE_STREAM_SPAN_NAME,
    RESPAN_PARAMS_KEY,
    RESPAN_PARAMS_MODEL_KEY,
)
from respan_instrumentation_replicate._translator import (
    build_model_call_span_data,
    build_operation_span_data,
    output_to_text,
)
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
)
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import build_readable_span, inject_span

logger = logging.getLogger(__name__)

_SUPPRESSED_SPAN_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "respan_replicate_suppressed_span_depth",
    default=0,
)


def _spans_suppressed() -> bool:
    return _SUPPRESSED_SPAN_DEPTH.get() > 0


@contextlib.contextmanager
def _suppress_nested_spans():
    token = _SUPPRESSED_SPAN_DEPTH.set(_SUPPRESSED_SPAN_DEPTH.get() + 1)
    try:
        yield
    finally:
        _SUPPRESSED_SPAN_DEPTH.reset(token)


def _current_otel_parent() -> tuple[str | None, str | None]:
    current_span = trace.get_current_span()
    try:
        span_context = current_span.get_span_context()
    except Exception:
        return None, None

    trace_id = getattr(span_context, "trace_id", 0)
    span_id = getattr(span_context, "span_id", 0)
    if not isinstance(trace_id, int) or not isinstance(span_id, int):
        return None, None
    if trace_id == 0 or span_id == 0:
        return None, None
    return format_trace_id(trace_id=trace_id), format_span_id(span_id=span_id)


def _emit_span(
    *,
    span_name: str,
    attributes: dict[str, Any],
    start_time_ns: int,
    end_time_ns: int | None = None,
    error: Exception | None = None,
    parent_context: tuple[str | None, str | None] | None = None,
) -> None:
    if _spans_suppressed():
        return

    trace_id, parent_id = parent_context or _current_otel_parent()
    span = build_readable_span(
        name=span_name,
        trace_id=trace_id,
        parent_id=parent_id,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns or time.time_ns(),
        attributes=attributes,
        status_code=500 if error is not None else 200,
        error_message=str(error) if error is not None else None,
    )
    inject_span(span=span)


def _pop_respan_params(kwargs: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    call_kwargs = dict(kwargs)
    respan_params = call_kwargs.pop(RESPAN_PARAMS_KEY, None)
    return call_kwargs, respan_params


def _reported_model_from_respan_params(respan_params: Any) -> str | None:
    if not isinstance(respan_params, dict):
        return None
    model = respan_params.get(RESPAN_PARAMS_MODEL_KEY)
    return str(model) if model else None


def _set_prediction_reported_model(prediction: Any, respan_params: Any) -> None:
    reported_model = _reported_model_from_respan_params(respan_params)
    if not reported_model:
        return
    try:
        object.__setattr__(prediction, PREDICTION_RESPAN_MODEL_ATTR, reported_model)
    except Exception:
        try:
            setattr(prediction, PREDICTION_RESPAN_MODEL_ATTR, reported_model)
        except Exception:
            return


def _is_file_output(value: Any) -> bool:
    return value.__class__.__name__ == "FileOutput"


def _is_sync_iterator(value: Any) -> bool:
    return not _is_file_output(value) and isinstance(value, Iterator)


def _is_async_iterator(value: Any) -> bool:
    return isinstance(value, AsyncIterator)


class _SyncIteratorProxy:
    def __init__(
        self,
        *,
        iterator: Iterator[Any],
        emit_once: Callable[[list[Any], Exception | None], None],
    ) -> None:
        self._iterator = iterator
        self._emit_once = emit_once
        self._chunks: list[Any] = []
        self._emitted = False

    def __iter__(self) -> "_SyncIteratorProxy":
        return self

    def __next__(self) -> Any:
        try:
            with _suppress_nested_spans():
                chunk = next(self._iterator)
        except StopIteration:
            self._emit(error=None)
            raise
        except Exception as exc:
            self._emit(error=exc)
            raise
        if len(self._chunks) < MAX_STREAM_CHUNKS:
            self._chunks.append(chunk)
        return chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._iterator, name)

    def _emit(self, *, error: Exception | None) -> None:
        if self._emitted:
            return
        self._emitted = True
        self._emit_once(self._chunks, error)


class _AsyncIteratorProxy:
    def __init__(
        self,
        *,
        iterator: AsyncIterator[Any],
        emit_once: Callable[[list[Any], Exception | None], None],
    ) -> None:
        self._iterator = iterator
        self._chunks: list[Any] = []
        self._emit_once = emit_once
        self._emitted = False

    def __aiter__(self) -> "_AsyncIteratorProxy":
        return self

    async def __anext__(self) -> Any:
        try:
            with _suppress_nested_spans():
                chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._emit(error=None)
            raise
        except Exception as exc:
            self._emit(error=exc)
            raise
        if len(self._chunks) < MAX_STREAM_CHUNKS:
            self._chunks.append(chunk)
        return chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._iterator, name)

    def _emit(self, *, error: Exception | None) -> None:
        if self._emitted:
            return
        self._emitted = True
        self._emit_once(self._chunks, error)


def _wrap_sync_run(original: Any, *, span_name: str, stream: bool = False) -> Any:
    def wrapper(self: Any, ref: Any, input: Any = None, *args: Any, **kwargs: Any) -> Any:
        call_kwargs, respan_params = _pop_respan_params(kwargs)
        event_kwargs = {**call_kwargs, RESPAN_PARAMS_KEY: respan_params}
        parent_context = _current_otel_parent()
        start_ns = time.time_ns()
        try:
            with _suppress_nested_spans():
                if stream:
                    output = original(self, ref, *args, input=input, **call_kwargs)
                else:
                    output = original(self, ref, input, *args, **call_kwargs)
        except Exception as exc:
            resolved_span_name, attrs = build_model_call_span_data(
                span_name=span_name,
                ref=ref,
                input_value=input,
                kwargs=event_kwargs,
                error=exc,
                stream=stream,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                error=exc,
                parent_context=parent_context,
            )
            raise

        if _is_sync_iterator(output):

            def emit_once(chunks: list[Any], error: Exception | None) -> None:
                resolved_span_name, attrs = build_model_call_span_data(
                    span_name=span_name,
                    ref=ref,
                    input_value=input,
                    kwargs=event_kwargs,
                    output=chunks,
                    error=error,
                    stream=True,
                )
                _emit_span(
                    span_name=resolved_span_name,
                    attributes=attrs,
                    start_time_ns=start_ns,
                    error=error,
                    parent_context=parent_context,
                )

            return _SyncIteratorProxy(iterator=output, emit_once=emit_once)

        resolved_span_name, attrs = build_model_call_span_data(
            span_name=span_name,
            ref=ref,
            input_value=input,
            kwargs=event_kwargs,
            output=output,
            stream=stream,
        )
        _emit_span(
            span_name=resolved_span_name,
            attributes=attrs,
            start_time_ns=start_ns,
            parent_context=parent_context,
        )
        return output

    return wrapper


def _wrap_async_run(original: Any, *, span_name: str, stream: bool = False) -> Any:
    async def wrapper(
        self: Any, ref: Any, input: Any = None, *args: Any, **kwargs: Any
    ) -> Any:
        call_kwargs, respan_params = _pop_respan_params(kwargs)
        event_kwargs = {**call_kwargs, RESPAN_PARAMS_KEY: respan_params}
        parent_context = _current_otel_parent()
        start_ns = time.time_ns()
        try:
            with _suppress_nested_spans():
                if stream:
                    output = await original(self, ref, input=input, **call_kwargs)
                else:
                    output = await original(self, ref, input, *args, **call_kwargs)
        except Exception as exc:
            resolved_span_name, attrs = build_model_call_span_data(
                span_name=span_name,
                ref=ref,
                input_value=input,
                kwargs=event_kwargs,
                error=exc,
                stream=stream,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                error=exc,
                parent_context=parent_context,
            )
            raise

        if _is_async_iterator(output):

            def emit_once(chunks: list[Any], error: Exception | None) -> None:
                resolved_span_name, attrs = build_model_call_span_data(
                    span_name=span_name,
                    ref=ref,
                    input_value=input,
                    kwargs=event_kwargs,
                    output=chunks,
                    error=error,
                    stream=True,
                )
                _emit_span(
                    span_name=resolved_span_name,
                    attributes=attrs,
                    start_time_ns=start_ns,
                    error=error,
                    parent_context=parent_context,
                )

            return _AsyncIteratorProxy(iterator=output, emit_once=emit_once)

        resolved_span_name, attrs = build_model_call_span_data(
            span_name=span_name,
            ref=ref,
            input_value=input,
            kwargs=event_kwargs,
            output=output,
            stream=stream,
        )
        _emit_span(
            span_name=resolved_span_name,
            attributes=attrs,
            start_time_ns=start_ns,
            parent_context=parent_context,
        )
        return output

    return wrapper


def _wrap_prediction_create(original: Any, *, is_async: bool = False) -> Any:
    if is_async:

        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            call_kwargs, respan_params = _pop_respan_params(kwargs)
            event_kwargs = {**call_kwargs, RESPAN_PARAMS_KEY: respan_params}
            parent_context = _current_otel_parent()
            start_ns = time.time_ns()
            try:
                with _suppress_nested_spans():
                    prediction = await original(self, *args, **call_kwargs)
            except Exception as exc:
                resolved_span_name, attrs = build_model_call_span_data(
                    span_name=REPLICATE_PREDICTION_CREATE_SPAN_NAME,
                    ref=args[0] if args else None,
                    input_value=call_kwargs.get(INPUT_KEY),
                    kwargs=event_kwargs,
                    error=exc,
                )
                _emit_span(
                    span_name=resolved_span_name,
                    attributes=attrs,
                    start_time_ns=start_ns,
                    error=exc,
                    parent_context=parent_context,
                )
                raise

            resolved_span_name, attrs = build_model_call_span_data(
                span_name=REPLICATE_PREDICTION_CREATE_SPAN_NAME,
                ref=args[0] if args else None,
                input_value=call_kwargs.get(INPUT_KEY),
                kwargs=event_kwargs,
                prediction=prediction,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                parent_context=parent_context,
            )
            _set_prediction_reported_model(prediction, respan_params)
            return prediction

        return async_wrapper

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        call_kwargs, respan_params = _pop_respan_params(kwargs)
        event_kwargs = {**call_kwargs, RESPAN_PARAMS_KEY: respan_params}
        parent_context = _current_otel_parent()
        start_ns = time.time_ns()
        try:
            with _suppress_nested_spans():
                prediction = original(self, *args, **call_kwargs)
        except Exception as exc:
            resolved_span_name, attrs = build_model_call_span_data(
                span_name=REPLICATE_PREDICTION_CREATE_SPAN_NAME,
                ref=args[0] if args else None,
                input_value=call_kwargs.get(INPUT_KEY),
                kwargs=event_kwargs,
                error=exc,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                error=exc,
                parent_context=parent_context,
            )
            raise

        resolved_span_name, attrs = build_model_call_span_data(
            span_name=REPLICATE_PREDICTION_CREATE_SPAN_NAME,
            ref=args[0] if args else None,
            input_value=call_kwargs.get(INPUT_KEY),
            kwargs=event_kwargs,
            prediction=prediction,
        )
        _emit_span(
            span_name=resolved_span_name,
            attributes=attrs,
            start_time_ns=start_ns,
            parent_context=parent_context,
        )
        _set_prediction_reported_model(prediction, respan_params)
        return prediction

    return wrapper


def _wrap_prediction_wait(original: Any, *, is_async: bool = False) -> Any:
    if is_async:

        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            parent_context = _current_otel_parent()
            start_ns = time.time_ns()
            try:
                with _suppress_nested_spans():
                    result = await original(self, *args, **kwargs)
            except Exception as exc:
                resolved_span_name, attrs = build_model_call_span_data(
                    span_name=REPLICATE_PREDICTION_WAIT_SPAN_NAME,
                    prediction=self,
                    error=exc,
                )
                _emit_span(
                    span_name=resolved_span_name,
                    attributes=attrs,
                    start_time_ns=start_ns,
                    error=exc,
                    parent_context=parent_context,
                )
                raise

            resolved_span_name, attrs = build_model_call_span_data(
                span_name=REPLICATE_PREDICTION_WAIT_SPAN_NAME,
                prediction=self,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                parent_context=parent_context,
            )
            return result

        return async_wrapper

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        parent_context = _current_otel_parent()
        start_ns = time.time_ns()
        try:
            with _suppress_nested_spans():
                result = original(self, *args, **kwargs)
        except Exception as exc:
            resolved_span_name, attrs = build_model_call_span_data(
                span_name=REPLICATE_PREDICTION_WAIT_SPAN_NAME,
                prediction=self,
                error=exc,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                error=exc,
                parent_context=parent_context,
            )
            raise

        resolved_span_name, attrs = build_model_call_span_data(
            span_name=REPLICATE_PREDICTION_WAIT_SPAN_NAME,
            prediction=self,
        )
        _emit_span(
            span_name=resolved_span_name,
            attributes=attrs,
            start_time_ns=start_ns,
            parent_context=parent_context,
        )
        return result

    return wrapper


def _wrap_prediction_stream(original: Any, *, is_async: bool = False) -> Any:
    if is_async:

        def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            parent_context = _current_otel_parent()
            start_ns = time.time_ns()
            try:
                with _suppress_nested_spans():
                    iterator = original(self, *args, **kwargs)
            except Exception as exc:
                resolved_span_name, attrs = build_model_call_span_data(
                    span_name=REPLICATE_STREAM_SPAN_NAME,
                    prediction=self,
                    error=exc,
                    stream=True,
                )
                _emit_span(
                    span_name=resolved_span_name,
                    attributes=attrs,
                    start_time_ns=start_ns,
                    error=exc,
                    parent_context=parent_context,
                )
                raise

            def emit_once(chunks: list[Any], error: Exception | None) -> None:
                resolved_span_name, attrs = build_model_call_span_data(
                    span_name=REPLICATE_STREAM_SPAN_NAME,
                    prediction=self,
                    output=chunks,
                    error=error,
                    stream=True,
                )
                _emit_span(
                    span_name=resolved_span_name,
                    attributes=attrs,
                    start_time_ns=start_ns,
                    error=error,
                    parent_context=parent_context,
                )

            return _AsyncIteratorProxy(iterator=iterator, emit_once=emit_once)

        return async_wrapper

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        parent_context = _current_otel_parent()
        start_ns = time.time_ns()
        try:
            with _suppress_nested_spans():
                iterator = original(self, *args, **kwargs)
        except Exception as exc:
            resolved_span_name, attrs = build_model_call_span_data(
                span_name=REPLICATE_STREAM_SPAN_NAME,
                prediction=self,
                error=exc,
                stream=True,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                error=exc,
                parent_context=parent_context,
            )
            raise

        def emit_once(chunks: list[Any], error: Exception | None) -> None:
            resolved_span_name, attrs = build_model_call_span_data(
                span_name=REPLICATE_STREAM_SPAN_NAME,
                prediction=self,
                output=chunks,
                error=error,
                stream=True,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                error=error,
                parent_context=parent_context,
            )

        return _SyncIteratorProxy(iterator=iterator, emit_once=emit_once)

    return wrapper


def _wrap_operation(original: Any, *, span_name: str, is_async: bool = False) -> Any:
    if is_async:

        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            parent_context = _current_otel_parent()
            start_ns = time.time_ns()
            input_value = {"args": args, "kwargs": kwargs}
            try:
                with _suppress_nested_spans():
                    result = await original(self, *args, **kwargs)
            except Exception as exc:
                resolved_span_name, attrs = build_operation_span_data(
                    span_name=span_name,
                    input_value=input_value,
                    error=exc,
                )
                _emit_span(
                    span_name=resolved_span_name,
                    attributes=attrs,
                    start_time_ns=start_ns,
                    error=exc,
                    parent_context=parent_context,
                )
                raise

            resolved_span_name, attrs = build_operation_span_data(
                span_name=span_name,
                input_value=input_value,
                output=output_to_text(result),
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                parent_context=parent_context,
            )
            return result

        return async_wrapper

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        parent_context = _current_otel_parent()
        start_ns = time.time_ns()
        input_value = {"args": args, "kwargs": kwargs}
        try:
            with _suppress_nested_spans():
                result = original(self, *args, **kwargs)
        except Exception as exc:
            resolved_span_name, attrs = build_operation_span_data(
                span_name=span_name,
                input_value=input_value,
                error=exc,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                error=exc,
                parent_context=parent_context,
            )
            raise

        resolved_span_name, attrs = build_operation_span_data(
            span_name=span_name,
            input_value=input_value,
            output=output_to_text(result),
        )
        _emit_span(
            span_name=resolved_span_name,
            attributes=attrs,
            start_time_ns=start_ns,
            parent_context=parent_context,
        )
        return result

    return wrapper


def _module_run_wrapper(module: Any, *, method_name: str) -> Callable[..., Any]:
    def wrapper(ref: Any, input: Any = None, *args: Any, **kwargs: Any) -> Any:
        client = getattr(module, "default_client")
        method = getattr(client, method_name)
        return method(ref, input, *args, **kwargs)

    return wrapper


def _module_async_run_wrapper(module: Any, *, method_name: str) -> Callable[..., Any]:
    async def wrapper(ref: Any, input: Any = None, *args: Any, **kwargs: Any) -> Any:
        client = getattr(module, "default_client")
        method = getattr(client, method_name)
        return await method(ref, input, *args, **kwargs)

    return wrapper


def _module_stream_wrapper(module: Any, *, method_name: str) -> Callable[..., Any]:
    def wrapper(ref: Any, *, input: Any = None, **kwargs: Any) -> Any:
        client = getattr(module, "default_client")
        method = getattr(client, method_name)
        return method(ref, input=input, **kwargs)

    return wrapper


def _module_async_stream_wrapper(module: Any, *, method_name: str) -> Callable[..., Any]:
    async def wrapper(ref: Any, input: Any = None, **kwargs: Any) -> Any:
        client = getattr(module, "default_client")
        method = getattr(client, method_name)
        return await method(ref, input=input, **kwargs)

    return wrapper


class ReplicateInstrumentor:
    """Respan instrumentor for the Replicate Python SDK."""

    name = REPLICATE_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._originals: dict[tuple[Any, str], Any] = {}
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def _patch_attr(self, owner: Any, attr_name: str, replacement: Any) -> None:
        key = (owner, attr_name)
        if key not in self._originals:
            self._originals[key] = getattr(owner, attr_name)
        setattr(owner, attr_name, replacement)

    def activate(self) -> None:
        """Monkey-patch the Replicate SDK."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Replicate instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            replicate_module = importlib.import_module("replicate")
            client_module = importlib.import_module("replicate.client")
            prediction_module = importlib.import_module("replicate.prediction")
        except ImportError as exc:
            logger.warning(
                "Failed to activate Replicate instrumentation - missing dependency: %s",
                exc,
            )
            return

        Client = getattr(client_module, "Client")
        Predictions = getattr(prediction_module, "Predictions")
        Prediction = getattr(prediction_module, "Prediction")

        self._patch_attr(
            Client,
            "run",
            _wrap_sync_run(
                getattr(Client, "run"),
                span_name=REPLICATE_RUN_SPAN_NAME,
            ),
        )
        self._patch_attr(
            Client,
            "async_run",
            _wrap_async_run(
                getattr(Client, "async_run"),
                span_name=f"{ASYNC_PREFIX}{REPLICATE_RUN_SPAN_NAME}",
            ),
        )
        self._patch_attr(
            Client,
            "stream",
            _wrap_sync_run(
                getattr(Client, "stream"),
                span_name=REPLICATE_STREAM_SPAN_NAME,
                stream=True,
            ),
        )
        self._patch_attr(
            Client,
            "async_stream",
            _wrap_async_run(
                getattr(Client, "async_stream"),
                span_name=f"{ASYNC_PREFIX}{REPLICATE_STREAM_SPAN_NAME}",
                stream=True,
            ),
        )

        self._patch_attr(
            Predictions,
            "create",
            _wrap_prediction_create(getattr(Predictions, "create")),
        )
        self._patch_attr(
            Predictions,
            "async_create",
            _wrap_prediction_create(getattr(Predictions, "async_create"), is_async=True),
        )
        for method_name in ("list", "get", "cancel"):
            self._patch_attr(
                Predictions,
                method_name,
                _wrap_operation(
                    getattr(Predictions, method_name),
                    span_name=f"replicate.predictions.{method_name}",
                ),
            )
        for method_name in ("async_list", "async_get", "async_cancel"):
            self._patch_attr(
                Predictions,
                method_name,
                _wrap_operation(
                    getattr(Predictions, method_name),
                    span_name=f"replicate.predictions.{method_name}",
                    is_async=True,
                ),
            )

        self._patch_attr(
            Prediction,
            "wait",
            _wrap_prediction_wait(getattr(Prediction, "wait")),
        )
        self._patch_attr(
            Prediction,
            "async_wait",
            _wrap_prediction_wait(getattr(Prediction, "async_wait"), is_async=True),
        )
        self._patch_attr(
            Prediction,
            "stream",
            _wrap_prediction_stream(getattr(Prediction, "stream")),
        )
        self._patch_attr(
            Prediction,
            "async_stream",
            _wrap_prediction_stream(getattr(Prediction, "async_stream"), is_async=True),
        )

        self._patch_attr(
            replicate_module,
            "run",
            _module_run_wrapper(replicate_module, method_name="run"),
        )
        self._patch_attr(
            replicate_module,
            "async_run",
            _module_async_run_wrapper(replicate_module, method_name="async_run"),
        )
        self._patch_attr(
            replicate_module,
            "stream",
            _module_stream_wrapper(replicate_module, method_name="stream"),
        )
        self._patch_attr(
            replicate_module,
            "async_stream",
            _module_async_stream_wrapper(replicate_module, method_name="async_stream"),
        )

        self._is_instrumented = True
        logger.info("Replicate instrumentation activated")

    def deactivate(self) -> None:
        """Restore patched Replicate SDK methods."""
        for (owner, attr_name), original in reversed(self._originals.items()):
            try:
                setattr(owner, attr_name, original)
            except Exception:
                logger.debug("Failed to restore Replicate SDK attr %s", attr_name)
        self._originals.clear()
        self._is_instrumented = False
        logger.info("Replicate instrumentation deactivated")
