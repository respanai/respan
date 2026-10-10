"""Burr lifecycle adapter that emits canonical Respan spans."""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from burr.core import State
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
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)

from respan_instrumentation_burr._constants import BURR_ADAPTER_MARKER
from respan_instrumentation_burr._policy import CapturePolicy, permitted, suppressed

logger = logging.getLogger(__name__)
_BURR_METADATA_ATTRIBUTE = f"{RESPAN_METADATA}.burr"
_MAX_CAPTURED_STREAM_ITEMS = 32


@dataclasses.dataclass
class _ActiveSpan:
    scope: str
    span: Any
    token: Any
    metadata: dict[str, Any]
    capture: bool = False
    application_exception: BaseException | None = None
    pending_end: tuple[BaseException | None, Callable[[], Any] | None] | None = None


_CUSTOM_EXCEPTION: ContextVar[BaseException | None] = ContextVar(
    "respan_burr_custom_exception", default=None
)


_ACTIVE_SPANS: ContextVar[tuple[_ActiveSpan, ...]] = ContextVar(
    "respan_burr_active_spans", default=()
)


def _safe_hook(method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        try:
            return method(*args, **kwargs)
        except Exception:  # noqa: BLE001 - telemetry cannot change a native Burr result.
            logger.debug("Burr telemetry hook failed")
            args[0]._abort_current()
            return None

    return guarded


_CREDENTIAL_FIELDS = {
    "authorization",
    "proxy_authorization",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "password",
    "client_secret",
    "secret",
    "credentials",
    "credential",
}
_SCHEMA_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def _schema(value: Any) -> bool:
    return type(value) is dict and (
        type(value.get("properties")) is dict
        or (type(value.get("type")) is str and value["type"] in _SCHEMA_TYPES)
        or any(name in value for name in ("$schema", "$ref", "anyOf", "oneOf", "allOf"))
    )


def _jsonable(value: Any, seen: set[int] | None = None, *, schema: bool = False) -> Any:
    if value is None or type(value) in (str, int, float, bool):
        return value
    if type(value) is bytes:
        return value.decode("utf-8", errors="replace")
    seen = set() if seen is None else seen
    if id(value) in seen:
        return {"type": "cyclic"}
    seen = {*seen, id(value)}
    if type(value) is dict:
        result = {}
        for k, v in value.items():
            name = str(k) if type(k) in (str, int, float, bool) else type(k).__name__
            sensitive = name.lower().replace("-", "_") in _CREDENTIAL_FIELDS
            if sensitive and not schema and not _schema(v):
                result[name] = "[REDACTED]"
            else:
                result[name] = _jsonable(v, seen, schema=schema or _schema(v))
        return result
    if type(value) in (list, tuple):
        return [_jsonable(v, seen, schema=schema) for v in value]
    if type(value) is State:
        return _jsonable(value.get_all(), seen)
    # Unknown containers, dataclasses, serializers, and iterators can execute
    # application code. Retain type identity without invoking those protocols.
    return {"type": type(value).__name__}


def _json_string(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False)


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
    """Map Burr's native hooks without wrapping its execution or iterators."""

    def __init__(
        self, *, capture_content: bool = True, tracer: Any | None = None
    ) -> None:
        self.capture_content = capture_content
        self.tracer = tracer
        self.enabled = True
        self.policy = CapturePolicy(capture_content)
        self.policy.scrub = self._scrub
        self.owned: set[Any] = set()
        setattr(self, BURR_ADAPTER_MARKER, True)

    def _scrub(self, span: Any) -> None:
        if span not in self.owned:
            return
        attrs = getattr(span, "_attributes", None)
        if attrs is not None:
            for name in (
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                RESPAN_METADATA,
                _BURR_METADATA_ATTRIBUTE,
                ERROR_MESSAGE_ATTR,
            ):
                attrs.pop(name, None)
        if hasattr(span, "_events"):
            span._events = BoundedList(0)
        if (
            getattr(getattr(span, "status", None), "status_code", None)
            is StatusCode.ERROR
        ):
            span._status = Status(StatusCode.ERROR)

    def close(self) -> None:
        self.enabled = False
        self.policy.close()

    def _start(
        self,
        *,
        scope: str,
        name: str,
        app_id: str,
        partition_key: str | None,
        metadata: dict[str, Any],
        input_value: Callable[[], Any],
    ) -> None:
        active = _ActiveSpan(scope, None, None, metadata)
        _ACTIVE_SPANS.set((*_ACTIVE_SPANS.get(), active))
        if not self.enabled or suppressed():
            return
        span = None
        token = None
        try:
            initial = permitted(self.capture_content)
            self.policy.ensure_provider()
            tracer = self.tracer or trace.get_tracer(__name__)
            attrs = {
                RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
                RESPAN_LOG_TYPE: "workflow" if scope == "application" else "task",
                RESPAN_TRACE_GROUP_ID: str(app_id),
                SpanAttributes.TRACELOOP_ENTITY_NAME: name,
                SpanAttributes.TRACELOOP_ENTITY_PATH: "",
            }
            if partition_key:
                attrs[RESPAN_THREADS_ID] = str(partition_key)
            parent_context = context_api.get_current()
            span = tracer.start_span(name, attributes=attrs)
            self.owned.add(span)
            self.policy.enroll(span, parent_context)
            active.span = span
            active.capture = initial and self.policy.allowed(span)
            if active.capture:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_INPUT, _json_string(input_value())
                )
                self._sync_metadata(active)
            token = context_api.attach(trace.set_span_in_context(span))
            active.token = token
        except Exception:  # noqa: BLE001 - roll back partial telemetry, preserve native call.
            active.span = None
            active.token = None
            if token is not None:
                context_api.detach(token)
            if span is not None:
                try:
                    self._scrub(span)
                    span.end()
                except Exception:  # noqa: BLE001 - one broken span must not stop cleanup.
                    logger.debug("Burr telemetry cleanup failed")
                self.policy.on_end(span)
                self.owned.discard(span)

    def _abort_current(self) -> None:
        active = self._current()
        if active is None:
            return
        active.capture = False
        try:
            self._end(scope=active.scope, exception=None, output_value=None)
        except Exception:  # noqa: BLE001 - cleanup cannot replace native behavior.
            logger.debug("Burr telemetry cleanup failed")

    def _current(self) -> _ActiveSpan | None:
        stack = _ACTIVE_SPANS.get()
        return stack[-1] if stack else None

    def _capture(self, active: _ActiveSpan) -> bool:
        if active.span is None:
            return False
        allowed = active.capture and self.policy.allowed(active.span)
        if not allowed:
            active.capture = False
            self._scrub(active.span)
        return allowed

    def _sync_metadata(self, active: _ActiveSpan) -> None:
        if self._capture(active):
            raw = getattr(active.span, "attributes", {}).get(RESPAN_METADATA, "{}")
            try:
                metadata = json.loads(raw)
            except (TypeError, ValueError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["burr"] = active.metadata
            active.span.set_attribute(RESPAN_METADATA, _json_string(metadata))
            active.span.set_attribute(
                _BURR_METADATA_ATTRIBUTE, _json_string(active.metadata)
            )

    def _end(
        self,
        *,
        scope: str,
        exception: BaseException | None,
        output_value: Callable[[], Any] | None,
    ) -> None:
        stack = list(_ACTIVE_SPANS.get())
        if not stack or stack[-1].scope != scope:
            return
        active = stack.pop()
        _ACTIVE_SPANS.set(tuple(stack))
        if active.span is None:
            return
        try:
            capture = self._capture(active)
            if exception is None:
                active.span.set_status(Status(StatusCode.OK))
            else:
                active.span.set_status(
                    Status(StatusCode.ERROR, str(exception) if capture else None)
                )
                if capture:
                    active.span.record_exception(exception)
                    active.span.set_attribute(
                        ERROR_MESSAGE_ATTR, str(exception) or type(exception).__name__
                    )
            if capture and output_value is not None:
                active.span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT, _json_string(output_value())
                )
            self._capture(active)
        except Exception:  # noqa: BLE001 - telemetry cannot mask the action's outcome.
            self._scrub(active.span)
        finally:
            try:
                # Check the active context before detachment can widen privacy.
                try:
                    self._capture(active)
                except Exception:  # noqa: BLE001 - a policy fault cannot skip detachment.
                    self.policy.fail_closed()
                if active.token is not None:
                    context_api.detach(active.token)
            finally:
                try:
                    active.span.end()
                finally:
                    self.policy.on_end(active.span)
                    self.owned.discard(active.span)

    def _remember_application_exception(self, exception: BaseException) -> None:
        for active in reversed(_ACTIVE_SPANS.get()):
            if active.scope == "application":
                active.application_exception = exception
                return

    @_safe_hook
    def pre_run_execute_call(
        self,
        *,
        app_id: str,
        partition_key: str | None,
        state: Any,
        method: Any,
        **future_kwargs: Any,
    ) -> None:
        method_name = str(getattr(method, "value", method))
        metadata = {
            "scope": "application",
            "app_id": app_id,
            "partition_key": partition_key,
            "method": method_name,
        }
        self._start(
            scope="application",
            name=f"burr.application.{method_name}",
            app_id=app_id,
            partition_key=partition_key,
            metadata=metadata,
            input_value=lambda: {**metadata, "state": state},
        )

    @_safe_hook
    def post_run_execute_call(
        self, *, state: Any, exception: BaseException | None, **future_kwargs: Any
    ) -> None:
        active = next(
            (
                item
                for item in reversed(_ACTIVE_SPANS.get())
                if item.scope == "application"
            ),
            None,
        )
        exception = exception or (active.application_exception if active else None)
        output = lambda: {"state": state}
        if active is not None and active is not self._current():
            active.pending_end = (exception, output)
            return
        self._end(scope="application", exception=exception, output_value=output)

    @_safe_hook
    def pre_run_step(
        self,
        *,
        app_id: str,
        partition_key: str | None,
        sequence_id: int,
        state: Any,
        action: Any,
        inputs: dict[str, Any],
        **future_kwargs: Any,
    ) -> None:
        name = str(action.name)
        metadata = {
            "scope": "action",
            "app_id": app_id,
            "partition_key": partition_key,
            "sequence_id": sequence_id,
            "action": {"name": name},
        }

        def payload() -> Any:
            metadata["action"].update(
                {
                    "reads": action.reads,
                    "writes": action.writes,
                    "tags": action.tags,
                    "streaming": bool(getattr(action, "streaming", False)),
                    "declared_inputs": action.inputs,
                }
            )
            return {**metadata, "inputs": inputs, "state": state}

        self._start(
            scope="action",
            name=name,
            app_id=app_id,
            partition_key=partition_key,
            metadata=metadata,
            input_value=payload,
        )

    @_safe_hook
    def post_run_step(
        self,
        *,
        state: Any,
        result: dict[str, Any] | None,
        exception: BaseException | None,
        **future_kwargs: Any,
    ) -> None:
        if exception is not None:
            self._remember_application_exception(exception)
        active = self._current()
        if active is not None and "stream" in active.metadata:
            active.metadata["stream"]["completed"] = exception is None
            self._sync_metadata(active)
        output = lambda: {"result": result, "state": state}
        if active is not None and "stream" in active.metadata:
            active.pending_end = (exception, output)
            return
        self._end(scope="action", exception=exception, output_value=output)

    @_safe_hook
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
        name = str(span.name or "custom")
        metadata = {
            "scope": "custom_span",
            "app_id": app_id,
            "partition_key": partition_key,
            "action": action,
            "action_sequence_id": action_sequence_id,
            "span_name": name,
            "span_dependencies": span_dependencies,
            "span_id": span.uid,
        }
        self._start(
            scope="custom_span",
            name=name,
            app_id=app_id,
            partition_key=partition_key,
            metadata=metadata,
            input_value=lambda: metadata,
        )

    @_safe_hook
    def post_end_span(self, **future_kwargs: Any) -> None:
        # The owned native __exit__ wrapper supplies its exact exception value;
        # lifecycle callback arguments alone do not expose this in Burr 0.42.
        self._end(
            scope="custom_span", exception=_CUSTOM_EXCEPTION.get(), output_value=None
        )

    @_safe_hook
    def do_log_attributes(
        self, *, attributes: dict[str, Any], tags: dict, **future_kwargs: Any
    ) -> None:
        active = self._current()
        if active is not None and self._capture(active):
            active.metadata.setdefault("logged_attributes", {}).update(
                _jsonable(attributes)
            )
            if tags:
                active.metadata.setdefault("tags", {}).update(_jsonable(tags))
            self._sync_metadata(active)

    @_safe_hook
    def pre_start_stream(
        self,
        *,
        action: str,
        sequence_id: int,
        app_id: str,
        partition_key: str | None,
        **future_kwargs: Any,
    ) -> None:
        active = self._current()
        if active is not None and self._capture(active):
            active.metadata["stream"] = {
                "action": action,
                "sequence_id": sequence_id,
                "started": True,
                "completed": False,
                "item_count": 0,
                "items": [],
            }
            self._sync_metadata(active)
            active.span.add_event(
                "burr.stream.start",
                {"burr.action": action, "burr.sequence_id": sequence_id},
            )

    @_safe_hook
    def post_stream_item(
        self,
        *,
        item: Any,
        item_index: int,
        action: str,
        sequence_id: int,
        **future_kwargs: Any,
    ) -> None:
        active = self._current()
        if active is None or not self._capture(active):
            return
        stream = active.metadata.get("stream", {})
        if stream.get("action") != action:
            return
        stream["item_count"] += 1
        if len(stream["items"]) < _MAX_CAPTURED_STREAM_ITEMS:
            stream["items"].append({"index": item_index, "value": _jsonable(item)})
        self._sync_metadata(active)
        active.span.add_event(
            "burr.stream.item",
            {
                "burr.action": action,
                "burr.sequence_id": sequence_id,
                "burr.item_index": item_index,
                "burr.item": _json_string(item),
            },
        )

    @_safe_hook
    def post_end_stream(
        self, *, action: str, sequence_id: int, **future_kwargs: Any
    ) -> None:
        active = self._current()
        if active is not None and self._capture(active):
            stream = active.metadata.get("stream", {})
            if stream.get("action") == action:
                stream["completed"] = True
                self._sync_metadata(active)
                active.span.add_event(
                    "burr.stream.end",
                    {"burr.action": action, "burr.sequence_id": sequence_id},
                )
        if active is not None and active.pending_end is not None:
            exception, output = active.pending_end
            self._end(scope=active.scope, exception=exception, output_value=output)
            parent = self._current()
            if parent is not None and parent.pending_end is not None:
                exception, output = parent.pending_end
                self._end(scope=parent.scope, exception=exception, output_value=output)
