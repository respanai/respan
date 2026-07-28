"""Burr lifecycle adapter that emits canonical Respan spans."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from typing import Any

from burr.lifecycle.base import (
    DoLogAttributeHook,
    PostApplicationExecuteCallHook,
    PostEndSpanHook,
    PostEndStreamHook,
    PostRunStepHook,
    PostStreamItemHook,
    PreApplicationExecuteCallHook,
    PreRunStepHook,
    PreStartSpanHook,
    PreStartStreamHook,
)
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_burr._constants import BURR_ADAPTER_MARKER
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_sdk.utils.serialization import serialize_value

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _ActiveSpan:
    scope: str
    span: Any
    token: Any
    metadata: dict[str, Any]


_ACTIVE_SPANS: ContextVar[tuple[_ActiveSpan, ...]] = ContextVar(
    "respan_burr_active_spans",
    default=(),
)


def _jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"type": "bytes", "length": len(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, depth=depth + 1) for key, item in value.items()
        }
    if isinstance(value, Sequence):
        return [_jsonable(item, depth=depth + 1) for item in value]
    serializer = getattr(value, "serialize", None)
    if callable(serializer):
        try:
            return _jsonable(serializer(), depth=depth + 1)
        except Exception:
            pass
    get_all = getattr(value, "get_all", None)
    if callable(get_all):
        try:
            return _jsonable(get_all(), depth=depth + 1)
        except Exception:
            pass
    try:
        return serialize_value(value=value)
    except Exception:
        return repr(value)


def _json_string(value: Any) -> str:
    return json.dumps(_jsonable(value), default=str, ensure_ascii=False)


def _method_value(method: Any) -> str:
    return str(getattr(method, "value", method))


def _action_metadata(action: Any) -> dict[str, Any]:
    metadata = {
        "name": str(getattr(action, "name", type(action).__name__)),
        "reads": list(getattr(action, "reads", ()) or ()),
        "writes": list(getattr(action, "writes", ()) or ()),
        "tags": list(getattr(action, "tags", ()) or ()),
        "streaming": bool(getattr(action, "streaming", False)),
    }
    inputs = getattr(action, "inputs", None)
    if inputs:
        metadata["declared_inputs"] = _jsonable(inputs)
    return metadata


class BurrLifecycleAdapter(
    PreApplicationExecuteCallHook,
    PostApplicationExecuteCallHook,
    PreRunStepHook,
    PostRunStepHook,
    PreStartSpanHook,
    PostEndSpanHook,
    DoLogAttributeHook,
    PreStartStreamHook,
    PostStreamItemHook,
    PostEndStreamHook,
):
    """Map Burr applications, actions, custom spans, and streams to Respan."""

    def __init__(
        self,
        *,
        capture_content: bool = True,
        tracer: Any | None = None,
    ) -> None:
        self.capture_content = capture_content
        self.tracer = tracer or trace.get_tracer(__name__)
        self.enabled = True
        setattr(self, BURR_ADAPTER_MARKER, True)

    def _attributes(
        self,
        *,
        log_type: str,
        entity_name: str,
        app_id: str,
        partition_key: str | None,
        metadata: dict[str, Any],
        input_value: Any,
    ) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
            RESPAN_LOG_TYPE: log_type,
            RESPAN_TRACE_GROUP_ID: str(app_id),
            RESPAN_METADATA: _json_string({"burr": metadata}),
            SpanAttributes.TRACELOOP_ENTITY_NAME: entity_name,
            SpanAttributes.TRACELOOP_ENTITY_PATH: entity_name,
        }
        if partition_key:
            attrs[RESPAN_THREADS_ID] = str(partition_key)
        if self.capture_content:
            attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_string(input_value)
        return attrs

    def _start(
        self,
        *,
        scope: str,
        name: str,
        attributes: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        span = self.tracer.start_span(name, attributes=attributes)
        token = context_api.attach(trace.set_span_in_context(span))
        _ACTIVE_SPANS.set(
            (*_ACTIVE_SPANS.get(), _ActiveSpan(scope, span, token, metadata))
        )

    def _end(
        self,
        *,
        scope: str,
        exception: BaseException | None,
        output_value: Any,
    ) -> None:
        stack = list(_ACTIVE_SPANS.get())
        if not stack:
            return
        active = stack[-1]
        if active.scope != scope:
            logger.warning(
                "Burr span lifecycle mismatch: expected %s, found %s",
                scope,
                active.scope,
            )
            return
        stack.pop()
        _ACTIVE_SPANS.set(tuple(stack))
        context_api.detach(active.token)
        if exception is None:
            active.span.set_status(Status(StatusCode.OK))
            active.span.set_attribute("status_code", 200)
        else:
            message = str(exception) or type(exception).__name__
            active.span.record_exception(exception)
            active.span.set_status(Status(StatusCode.ERROR, message))
            active.span.set_attribute("status_code", 500)
            active.span.set_attribute(ERROR_MESSAGE_ATTR, message)
        if self.capture_content:
            active.span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                _json_string(output_value),
            )
        active.span.end()

    def _current(self) -> _ActiveSpan | None:
        stack = _ACTIVE_SPANS.get()
        return stack[-1] if stack else None

    def pre_run_execute_call(
        self,
        *,
        app_id: str,
        partition_key: str,
        state: Any,
        method: Any,
        **future_kwargs: Any,
    ) -> None:
        del future_kwargs
        method_name = _method_value(method)
        metadata = {
            "scope": "application",
            "app_id": app_id,
            "partition_key": partition_key,
            "method": method_name,
        }
        self._start(
            scope="application",
            name=f"burr.application.{method_name}",
            attributes=self._attributes(
                log_type="workflow",
                entity_name=f"burr.application.{method_name}",
                app_id=app_id,
                partition_key=partition_key,
                metadata=metadata,
                input_value={
                    **metadata,
                    "state": _jsonable(state),
                },
            ),
            metadata=metadata,
        )

    def post_run_execute_call(
        self,
        *,
        state: Any,
        exception: BaseException | None,
        **future_kwargs: Any,
    ) -> None:
        del future_kwargs
        output = (
            {"status": "completed", "state": _jsonable(state)}
            if exception is None
            else {
                "status": "error",
                "error": type(exception).__name__,
                "message": str(exception),
                "state": _jsonable(state),
            }
        )
        self._end(scope="application", exception=exception, output_value=output)

    def pre_run_step(
        self,
        *,
        app_id: str,
        partition_key: str,
        sequence_id: int,
        state: Any,
        action: Any,
        inputs: dict[str, Any],
        **future_kwargs: Any,
    ) -> None:
        del future_kwargs
        action_details = _action_metadata(action)
        metadata = {
            "scope": "action",
            "app_id": app_id,
            "partition_key": partition_key,
            "sequence_id": sequence_id,
            "action": action_details,
        }
        action_name = action_details["name"]
        self._start(
            scope="action",
            name=f"burr.action.{action_name}",
            attributes=self._attributes(
                log_type="task",
                entity_name=action_name,
                app_id=app_id,
                partition_key=partition_key,
                metadata=metadata,
                input_value={
                    **metadata,
                    "inputs": _jsonable(inputs),
                    "state": _jsonable(state),
                },
            ),
            metadata=metadata,
        )

    def post_run_step(
        self,
        *,
        state: Any,
        result: dict[str, Any] | None,
        exception: BaseException | None,
        **future_kwargs: Any,
    ) -> None:
        del future_kwargs
        output = {
            "status": "completed" if exception is None else "error",
            "result": _jsonable(result),
            "state": _jsonable(state),
        }
        if exception is not None:
            output.update(
                {
                    "error": type(exception).__name__,
                    "message": str(exception),
                }
            )
        self._end(scope="action", exception=exception, output_value=output)

    def pre_start_span(
        self,
        *,
        action: str,
        action_sequence_id: int,
        span: Any,
        span_dependencies: list[str],
        app_id: str,
        partition_key: str | None,
        **future_kwargs: Any,
    ) -> None:
        del future_kwargs
        span_name = str(getattr(span, "name", None) or "custom")
        metadata = {
            "scope": "custom_span",
            "app_id": app_id,
            "partition_key": partition_key,
            "action": action,
            "action_sequence_id": action_sequence_id,
            "span_name": span_name,
            "span_dependencies": list(span_dependencies),
        }
        self._start(
            scope="custom_span",
            name=f"burr.span.{span_name}",
            attributes=self._attributes(
                log_type="task",
                entity_name=span_name,
                app_id=app_id,
                partition_key=partition_key,
                metadata=metadata,
                input_value=metadata,
            ),
            metadata=metadata,
        )

    def post_end_span(self, **future_kwargs: Any) -> None:
        del future_kwargs
        self._end(
            scope="custom_span",
            exception=None,
            output_value={"status": "completed"},
        )

    def do_log_attributes(
        self,
        *,
        attributes: dict[str, Any],
        tags: dict,
        **future_kwargs: Any,
    ) -> None:
        del future_kwargs
        active = self._current()
        if not self.enabled or active is None:
            return
        if self.capture_content:
            active.metadata["logged_attributes"] = _jsonable(attributes)
        if tags:
            active.metadata["tags"] = _jsonable(tags)
        active.span.set_attribute(
            RESPAN_METADATA,
            _json_string({"burr": active.metadata}),
        )

    def pre_start_stream(
        self,
        *,
        action: str,
        sequence_id: int,
        app_id: str,
        partition_key: str | None,
        **future_kwargs: Any,
    ) -> None:
        del future_kwargs
        active = self._current()
        if not self.enabled or active is None:
            return
        active.span.add_event(
            "burr.stream.start",
            {
                "burr.action": action,
                "burr.sequence_id": sequence_id,
                "burr.app_id": app_id,
                "burr.partition_key": partition_key or "",
            },
        )

    def post_stream_item(
        self,
        *,
        item: Any,
        item_index: int,
        action: str,
        sequence_id: int,
        **future_kwargs: Any,
    ) -> None:
        del future_kwargs
        active = self._current()
        if not self.enabled or active is None:
            return
        attributes = {
            "burr.action": action,
            "burr.sequence_id": sequence_id,
            "burr.item_index": item_index,
        }
        if self.capture_content:
            attributes["burr.item"] = _json_string(item)
        active.span.add_event("burr.stream.item", attributes)

    def post_end_stream(
        self,
        *,
        action: str,
        sequence_id: int,
        **future_kwargs: Any,
    ) -> None:
        del future_kwargs
        active = self._current()
        if not self.enabled or active is None:
            return
        active.span.add_event(
            "burr.stream.end",
            {
                "burr.action": action,
                "burr.sequence_id": sequence_id,
            },
        )
