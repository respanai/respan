"""Create real spans at native operation start and finish actual SDK outcomes."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from opentelemetry.semconv.attributes.exception_attributes import (
    EXCEPTION_MESSAGE,
    EXCEPTION_TYPE,
)
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from requests import Response
from respan_sdk.constants.llm_logging import LOG_TYPE_TASK, LogMethodChoices
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE

from respan_instrumentation_arize._constants import (
    ARIZE_INSTRUMENTATION_NAME,
    ARIZE_METADATA_INTEGRATION,
    ARIZE_METADATA_OPERATION,
    ARIZE_METADATA_RESOURCE,
)
from respan_instrumentation_arize._policy import clear_content, content_allowed
from respan_instrumentation_arize._serialization import safe_json_dumps, safe_text


def build_arize_span_attributes(*, resource, method_name, **unused):
    name = f"arize.{resource}.{method_name}"
    return {
        RESPAN_LOG_TYPE: LOG_TYPE_TASK,
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        SpanAttributes.TRACELOOP_ENTITY_NAME: name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: "",
        ARIZE_METADATA_INTEGRATION: ARIZE_INSTRUMENTATION_NAME,
        ARIZE_METADATA_RESOURCE: resource,
        ARIZE_METADATA_OPERATION: method_name,
    }


def _http_status(value):
    if isinstance(value, Response):
        status = value.__dict__.get("status_code")
    elif type(value).__module__.startswith("arize.") and isinstance(
        value, BaseException
    ):
        status = value.__dict__.get("status")
    else:
        return None
    return status if type(status) is int and 100 <= status <= 599 else None


def _error_message(error):
    fields = object.__getattribute__(error, "__dict__")
    body = fields.get("body")
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
            if type(parsed) is dict and isinstance(parsed.get("message"), str):
                return safe_text(parsed["message"])
        except (ValueError, TypeError):
            pass
    reason = fields.get("reason")
    if isinstance(reason, str):
        return safe_text(reason)
    args = object.__getattribute__(error, "args")
    return safe_text(args[0]) if args and isinstance(args[0], str) else None


@dataclass(eq=False)
class Operation:
    span: Any
    capture: bool
    setting: bool
    parent: Any = None
    finished: bool = False
    lock: Any = field(default_factory=RLock)
    on_finished: Any = None

    def veto(self):
        current = self
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            with current.lock:
                current.capture = False
                clear_content(current.span)
            current = current.parent

    def finish(self, result=None, error=None):
        with self.lock:
            if self.finished:
                return
            self.finished = True
            try:
                if not self.span.is_recording():
                    return
                if (
                    not content_allowed(self.setting)
                    or self.parent is not None
                    and not self.parent.capture
                ):
                    self.veto()
                status = _http_status(error if error is not None else result)
                if status is not None:
                    self.span.set_attribute(HTTP_RESPONSE_STATUS_CODE, status)
                if error is not None and not isinstance(
                    error,
                    (
                        asyncio.CancelledError,
                        concurrent.futures.CancelledError,
                        GeneratorExit,
                    ),
                ):
                    self.span.set_status(Status(StatusCode.ERROR))
                    attributes = {EXCEPTION_TYPE: type(error).__name__}
                    if self.capture:
                        message = _error_message(error)
                        if message is not None:
                            attributes[EXCEPTION_MESSAGE] = message
                    self.span.add_event("exception", attributes)
                elif error is None:
                    if status is not None and status >= 400:
                        self.span.set_status(Status(StatusCode.ERROR))
                    if self.capture:
                        self.span.set_attribute(
                            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                            safe_json_dumps(result),
                        )
                if not self.capture:
                    clear_content(self.span)
            except Exception:  # noqa: BLE001 - telemetry must not alter native SDK behavior
                # Observing a result can never replace the native SDK result/error.
                clear_content(self.span)
            finally:
                self.span.end()
                if self.on_finished is not None:
                    self.on_finished(self)
                self.span = None
                self.parent = None
                self.on_finished = None

    def close(self):
        with self.lock:
            if self.finished:
                return
            self.finished = True
            if (
                not self.capture
                or not content_allowed(self.setting)
                or self.parent is not None
                and not self.parent.capture
            ):
                self.veto()
            self.span.end()
            if self.on_finished is not None:
                self.on_finished(self)
            self.span = None
            self.parent = None
            self.on_finished = None

    def observe_future(self, future):
        if not self.span.is_recording():
            self.finish()
            return

        def completed(native):
            if self.finished:
                return
            try:
                if native.cancelled():
                    self.finish(error=concurrent.futures.CancelledError())
                else:
                    self.finish(result=native.result())
            except BaseException as error:  # noqa: BLE001 - preserve native errors including cancellation
                self.finish(error=error)

        try:
            future.add_done_callback(completed)
        except Exception:  # noqa: BLE001 - telemetry must not alter native SDK behavior
            self.close()
