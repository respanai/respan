"""Temporal's official tracing interceptor adapted to the Respan contract."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import re
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from threading import RLock
from typing import Any
from weakref import WeakKeyDictionary

from opentelemetry import baggage, trace
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_TRACE_GROUP_ID
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import read_propagated_attributes

from respan_instrumentation_temporal._constants import (
    TASK_LOG_TYPE,
    TEMPORAL_CAPTURED_INPUT,
    TEMPORAL_CLIENT_CONNECT_TARGET,
    TEMPORAL_CLIENT_MODULE,
    TEMPORAL_INSTRUMENTATION_NAME,
    TEMPORAL_OTEL_MODULE,
    TEMPORAL_RAW_ATTRIBUTE_KEYS,
    WORKFLOW_LOG_TYPE,
    WORKFLOW_OPERATION_PREFIXES,
)
from respan_instrumentation_temporal._serialization import (
    json_dumps,
    safe_baggage_value,
    safe_error_message,
)

logger = logging.getLogger(__name__)

_CAMEL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")
_SAFE_DETAIL = re.compile(r"[^a-zA-Z0-9_.-]+")
_MISSING = object()
_INTERNAL_WORKFLOW_ID = "__respan_temporal_workflow_id__"
_CLIENT_CONNECT_ATTRIBUTE = TEMPORAL_CLIENT_CONNECT_TARGET.rsplit(".", 1)[-1]
_RESPAN_BAGGAGE_PREFIX = "respan."


def _instrumentation_version() -> str | None:
    try:
        return importlib.metadata.version("respan-instrumentation-temporal")
    except importlib.metadata.PackageNotFoundError:
        return None


def _context_with_respan_baggage(context: Any) -> Any:
    propagated = read_propagated_attributes()
    if not propagated:
        return context
    result = context if context is not None else otel_context.get_current()
    for key, value in propagated.items():
        if isinstance(key, str) and key.startswith(_RESPAN_BAGGAGE_PREFIX):
            result = baggage.set_baggage(
                key,
                safe_baggage_value(key, value),
                context=result,
            )
    return result


def _apply_respan_baggage(attrs: dict[str, Any], context: Any) -> None:
    try:
        values = baggage.get_all(context=context)
    except BaseException:  # noqa: BLE001 - propagation must remain fail-open
        return
    for key, value in values.items():
        if isinstance(key, str) and key.startswith(_RESPAN_BAGGAGE_PREFIX):
            attrs.setdefault(key, safe_baggage_value(key, value))


def _json_dumps(value: Any, *, max_chars: int | None = None) -> str:
    return json_dumps(value, max_bytes=max_chars)


def _snake_case(value: str) -> str:
    value = _CAMEL_BOUNDARY_1.sub(r"\1_\2", value)
    value = _CAMEL_BOUNDARY_2.sub(r"\1_\2", value)
    return value.replace("-", "_").lower()


def _span_parts(name: str) -> tuple[str, str | None]:
    operation, separator, detail = name.partition(":")
    return operation, detail if separator and detail else None


def _safe_detail(detail: str | None) -> str | None:
    if not detail:
        return None
    cleaned = _SAFE_DETAIL.sub("_", detail).strip("_.-")
    return cleaned[:120] or None


def _extract_temporal_input(value: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in (
        "args",
        "arg",
        "id",
        "workflow",
        "workflow_type",
        "activity",
        "activity_type",
        "query",
        "signal",
        "update",
        "update_id",
        "task_queue",
    ):
        item = getattr(value, field_name, _MISSING)
        if item is not _MISSING and item is not None and not callable(item):
            payload[field_name] = item
    return payload


def _canonical_attributes(
    name: str,
    attributes: Mapping[str, Any] | None,
    *,
    capture_content: bool,
    max_attribute_chars: int | None,
) -> dict[str, Any]:
    source = dict(attributes or {})
    captured_input = source.pop(TEMPORAL_CAPTURED_INPUT, None)
    captured_output = source.pop(_CAPTURED_OUTPUT, _MISSING)
    if callable(captured_input) and capture_content:
        captured_input = captured_input()
    temporal_attributes = {
        key: source.pop(key)
        for key in tuple(source)
        if key in TEMPORAL_RAW_ATTRIBUTE_KEYS
    }
    workflow_id = temporal_attributes.get("temporalWorkflowID")
    operation, detail = _span_parts(name)
    safe_detail = _safe_detail(detail)
    operation_name = _snake_case(operation)
    entity_name = f"temporal.{operation_name}"
    if safe_detail:
        entity_name = f"{entity_name}.{safe_detail}"
    log_type = (
        WORKFLOW_LOG_TYPE if operation in WORKFLOW_OPERATION_PREFIXES else TASK_LOG_TYPE
    )

    input_payload: dict[str, Any] = {
        "operation": operation_name,
        "detail": detail,
        "content_captured": capture_content,
    }
    if capture_content:
        if temporal_attributes:
            input_payload["temporal"] = temporal_attributes
        if captured_input:
            input_payload["input"] = captured_input

    source[RESPAN_LOG_TYPE] = log_type
    if isinstance(workflow_id, str) and workflow_id:
        source[_INTERNAL_WORKFLOW_ID] = workflow_id
    source[SpanAttributes.TRACELOOP_ENTITY_NAME] = entity_name
    source[SpanAttributes.TRACELOOP_ENTITY_PATH] = entity_name
    if log_type == WORKFLOW_LOG_TYPE and safe_detail:
        source[RESPAN_TRACE_GROUP_ID] = safe_detail
    source[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_dumps(
        input_payload, max_chars=max_attribute_chars
    )
    if capture_content and captured_output is not _MISSING:
        source[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_dumps(
            captured_output, max_chars=max_attribute_chars
        )
    return source


def _attempt(function: Any, default: Any = None) -> Any:
    try:
        return function()
    except BaseException:  # noqa: BLE001 - telemetry cannot change native behavior
        return default


_CAPTURED_OUTPUT = "__respan_temporal_captured_output__"
_OBSERVED: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()
_CONTEXT_DECISIONS: OrderedDict[tuple[int, int], Any] = OrderedDict()
_PROVIDERS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()
_POLICY_LOCK = RLock()


def _span_key(span: Any) -> Any:
    value = _attempt(span.get_span_context, trace.INVALID_SPAN_CONTEXT)
    return (value.trace_id, value.span_id) if value.is_valid else None


def _remember(span: Any, decision: Any) -> None:
    key = _span_key(span)
    if key is not None:
        with _POLICY_LOCK:
            _OBSERVED[span] = decision
            _CONTEXT_DECISIONS[key] = decision
            while len(_CONTEXT_DECISIONS) > 4096:
                _CONTEXT_DECISIONS.popitem(last=False)


def _decision(span: Any) -> Any:
    with _POLICY_LOCK:
        return _OBSERVED.get(span) or _CONTEXT_DECISIONS.get(_span_key(span))


class _PrivacyObserver(SpanProcessor):
    def on_start(self, span: Any, parent_context: Any = None) -> None:
        contexts = (otel_context.get_current(), parent_context)
        allowed, parents = _parent_decisions(contexts)
        _attempt(lambda: _remember(span, _PrivacyDecision(allowed, contexts, parents)))

    def on_end(self, span: Any) -> None:
        key = (span.context.trace_id, span.context.span_id)
        with _POLICY_LOCK:
            state = _CONTEXT_DECISIONS.get(key)
        if state is not None:
            _attempt(state.allowed, False)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _observe_provider() -> None:
    provider = trace.get_tracer_provider()
    with _POLICY_LOCK:
        if provider not in _PROVIDERS and callable(
            getattr(provider, "add_span_processor", None)
        ):
            observer = _PrivacyObserver()
            provider.add_span_processor(observer)
            _PROVIDERS[provider] = observer


def _context_allows(context: Any) -> bool:
    return (
        otel_context.get_value(ENABLE_CONTENT_TRACING_KEY, context=context) is not False
        and baggage.get_baggage(ENABLE_CONTENT_TRACING_KEY, context=context) != "false"
        and not _suppressed(context)
        and os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower()
        not in {"false", "0", "off", "no"}
    )


def _suppressed(context: Any) -> bool:
    return bool(
        otel_context.get_value(
            otel_context._SUPPRESS_INSTRUMENTATION_KEY, context=context
        )
        or otel_context.get_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, context=context
        )
    )


def _parent_decisions(contexts: tuple[Any, ...]) -> tuple[bool, list[Any]]:
    allowed = True
    parents = []
    for context in contexts:
        allowed = allowed and bool(
            _attempt(lambda context=context: _context_allows(context), False)
        )
        parent = _attempt(
            lambda context=context: trace.get_current_span(context), trace.INVALID_SPAN
        )
        if not _attempt(
            lambda parent=parent: parent.get_span_context().is_valid, False
        ):
            continue
        state = _attempt(lambda parent=parent: _decision(parent))
        if state is not None:
            parents.append(state)
            allowed = allowed and bool(_attempt(state.allowed, False))
        elif _attempt(parent.is_recording, False):
            # An application-owned recording ancestor has no observed privacy
            # decision. Do not assume that a late provider permitted content.
            allowed = False
    return allowed, parents


class _PrivacyDecision:
    def __init__(
        self, allowed: bool, contexts: tuple[Any, ...], parents: list[Any]
    ) -> None:
        self.denied = not allowed
        self.contexts = contexts
        self.parents = parents
        if self.denied:
            self.deny_chain()

    def deny_chain(self) -> None:
        pending = [self]
        seen: set[int] = set()
        while pending:
            decision = pending.pop()
            if id(decision) in seen:
                continue
            seen.add(id(decision))
            decision.denied = True
            pending.extend(decision.parents)

    def allowed(self) -> bool:
        pending = [self]
        seen: set[int] = set()
        current = otel_context.get_current()
        while pending:
            decision = pending.pop()
            if id(decision) in seen:
                continue
            seen.add(id(decision))
            if decision.denied or not all(
                bool(_attempt(lambda c=c: _context_allows(c), False))
                for c in (*decision.contexts, current)
            ):
                self.deny_chain()
                return False
            pending.extend(decision.parents)
        return True


class _CanonicalSpanProxy(trace.Span):
    def __init__(
        self,
        span: Any,
        *,
        capture_content: bool,
        max_attribute_chars: int | None,
        attributes: Any = None,
        name: str = "",
        contexts: tuple[Any, ...] = (),
        parents: list[Any] | None = None,
        on_end: Any = None,
    ) -> None:
        self._span = span
        self._capture_content = capture_content
        self._max_attribute_chars = max_attribute_chars
        self._source = attributes
        self._name = name
        self._contexts = contexts
        self._parents = parents or []
        self._on_end = on_end
        self._ended = False
        self._has_error = False
        self._scrubbed = False
        self._owned_content: set[str] = set()
        self._error_type = None
        self._error_message: str | None = None
        self._decision = _attempt(lambda: _decision(span)) or _PrivacyDecision(
            capture_content, contexts, self._parents
        )
        if not capture_content:
            self._decision.deny_chain()
        if self.get_span_context().is_valid:
            _attempt(lambda: _remember(self, self._decision))
            _attempt(lambda: _remember(span, self._decision))

    @property
    def _denied(self) -> bool:
        return self._decision.denied

    def allowed(self) -> bool:
        allowed = self._decision.allowed()
        if not allowed and not self._scrubbed:
            self._scrub()
        return allowed

    def _scrub(self) -> None:
        self._scrubbed = True
        attributes = getattr(self._span, "_attributes", None)
        for key in self._owned_content:
            if attributes is not None:
                _attempt(lambda key=key: attributes.pop(key, None))
            else:
                _attempt(lambda key=key: self._span.set_attribute(key, "[REDACTED]"))
        # OTel spans have no public attribute deletion API: replace every
        # content-bearing field already owned by this adapter before end.
        safe = _attempt(
            lambda: _canonical_attributes(
                self._name,
                self._source,
                capture_content=False,
                max_attribute_chars=self._max_attribute_chars,
            ),
            {},
        )
        for key, value in safe.items():
            if key != _INTERNAL_WORKFLOW_ID:
                _attempt(
                    lambda key=key, value=value: self._span.set_attribute(key, value)
                )
        if self._has_error:
            self._error(None)

    def capture_baggage(self, attrs: dict[str, Any]) -> None:
        if self.is_recording() and self.allowed():
            for key, value in attrs.items():
                self._owned_content.add(key)
                _attempt(
                    lambda key=key, value=value: self._span.set_attribute(key, value)
                )

    def capture_input(self, value: Any) -> None:
        if self.is_recording() and self.allowed():
            _attempt(
                lambda: self._span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_INPUT,
                    _json_dumps(value, max_chars=self._max_attribute_chars),
                )
            )

    def capture_output(self, value: Any) -> None:
        if self.is_recording() and self.allowed():
            self._owned_content.add(SpanAttributes.TRACELOOP_ENTITY_OUTPUT)
            _attempt(
                lambda: self._span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    _json_dumps(value, max_chars=self._max_attribute_chars),
                )
            )

    def _error(self, message: str | None) -> None:
        if self._denied:
            self._error_message = None
            _attempt(lambda: self._span.set_status(Status(StatusCode.ERROR)))
            attributes = getattr(self._span, "_attributes", None)
            if attributes is not None:
                _attempt(lambda: attributes.pop(ERROR_MESSAGE, None))
        else:
            if message is not None:
                self._error_message = message
            _attempt(
                lambda: self._span.set_status(
                    Status(StatusCode.ERROR, self._error_message)
                )
            )
            if self._error_message is not None:
                self._owned_content.add(ERROR_MESSAGE)
                _attempt(
                    lambda: self._span.set_attribute(ERROR_MESSAGE, self._error_message)
                )
        if self._error_type:
            _attempt(lambda: self._span.set_attribute(ERROR_TYPE, self._error_type))
        # Exceptions are diagnostics, never a successful native operation result.
        # Preserve a previously captured actual result only while privacy permits.

    def record_exception(
        self, exception: BaseException, *args: Any, **kwargs: Any
    ) -> None:
        self._has_error = True
        self._error_type = type(exception).__name__
        if self.is_recording():
            self._error(safe_error_message(exception, capture_content=self.allowed()))
        # Never delegate raw exception events: native OTel can include traceback,
        # repr and exception text after privacy has been disabled.

    def set_status(self, status: Any, description: str | None = None) -> None:
        code = getattr(status, "status_code", status)
        if code == StatusCode.ERROR:
            self._has_error = True
            native_description = getattr(status, "description", description)
            message = None
            if self.allowed() and type(native_description) is str:
                message = (
                    safe_error_message(
                        RuntimeError(native_description), capture_content=True
                    )
                    if native_description
                    else ""
                )
            self._error(message)
        else:
            _attempt(lambda: self._span.set_status(status, description))

    def set_attribute(self, key: str, value: Any) -> None:
        if key in (
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        ):
            self._owned_content.add(key)
            if self.is_recording() and self.allowed():
                _attempt(
                    lambda key=key, value=value: self._span.set_attribute(key, value)
                )
        elif key == ERROR_MESSAGE:
            self._owned_content.add(ERROR_MESSAGE)
            if self.allowed():
                if type(value) is str:
                    self._error_message = value
                _attempt(
                    lambda key=key, value=value: self._span.set_attribute(key, value)
                )
        else:
            _attempt(lambda key=key, value=value: self._span.set_attribute(key, value))

    def set_attributes(self, attributes: Any) -> None:
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def add_event(
        self, name: str, attributes: Any = None, timestamp: Any = None
    ) -> None:
        # Native exception events contain raw traceback/text. This adapter emits
        # sanitized status and error fields instead, which can be scrubbed before
        # end when a later privacy decision denies content.
        self.allowed()

    def update_name(self, name: str) -> None:
        _attempt(lambda: self._span.update_name(name))

    def get_span_context(self) -> Any:
        return _attempt(self._span.get_span_context, trace.INVALID_SPAN_CONTEXT)

    def is_recording(self) -> bool:
        return bool(_attempt(self._span.is_recording, False)) and not self._ended

    def end(self, *args: Any, **kwargs: Any) -> None:
        if self._ended:
            return
        self.allowed()
        self._ended = True
        _attempt(lambda: self._span.end(*args, **kwargs))
        if self._on_end is not None:
            _attempt(self._on_end)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._span, name)


class _CanonicalTracer:
    def __init__(
        self, tracer: Any, *, capture_content: bool, max_attribute_chars: int | None
    ) -> None:
        self._tracer = tracer
        self._capture_content = capture_content
        self._max_attribute_chars = max_attribute_chars
        self._workflow_groups: dict[int, str] = {}
        self._workflow_contexts: dict[str, Any] = {}

    def start_span(self, name: str, *args: Any, **kwargs: Any) -> _CanonicalSpanProxy:
        _attempt(_observe_provider)
        # Temporal passes context positionally for completed workflow spans.
        supplied = kwargs.get("context", args[0] if args else None)
        contexts = (otel_context.get_current(), supplied)
        allowed, parents = _parent_decisions(contexts)
        allowed = (
            allowed
            and self._capture_content
            and os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower()
            not in {"false", "0", "off", "no"}
        )
        source = kwargs.get("attributes")
        attrs = _attempt(
            lambda: _canonical_attributes(
                name,
                source,
                capture_content=False,
                max_attribute_chars=self._max_attribute_chars,
            ),
            {},
        )
        workflow_id = attrs.pop(_INTERNAL_WORKFLOW_ID, None)
        parent_context = _attempt(
            lambda: trace.get_current_span(supplied).get_span_context(),
            trace.INVALID_SPAN_CONTEXT,
        )
        if not parent_context.is_valid:
            attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] = ""
        elif parent_context.trace_id in self._workflow_groups:
            attrs.setdefault(
                RESPAN_TRACE_GROUP_ID, self._workflow_groups[parent_context.trace_id]
            )
        kwargs["attributes"] = attrs
        # Suppression must precede sampler callbacks and all vendor payload access.
        suppressed = any(
            bool(_attempt(lambda c=c: _suppressed(c), True)) for c in contexts
        )
        span = (
            trace.INVALID_SPAN
            if suppressed
            else _attempt(
                lambda: self._tracer.start_span(name, *args, **kwargs),
                trace.INVALID_SPAN,
            )
        )
        group = attrs.get(RESPAN_TRACE_GROUP_ID)
        span_context = _attempt(span.get_span_context, trace.INVALID_SPAN_CONTEXT)
        if group and span_context.is_valid:
            self._workflow_groups[span_context.trace_id] = group
        on_end = None
        if workflow_id and name.startswith("CompleteWorkflow:"):
            on_end = lambda: self._workflow_contexts.pop(workflow_id, None)
        proxy = _CanonicalSpanProxy(
            span,
            capture_content=allowed,
            max_attribute_chars=self._max_attribute_chars,
            attributes=source,
            name=name,
            contexts=contexts,
            parents=parents,
            on_end=on_end,
        )
        if proxy.is_recording() and proxy.allowed():
            baggage_attrs: dict[str, Any] = {}
            _attempt(lambda: _apply_respan_baggage(baggage_attrs, supplied))
            proxy.capture_baggage(baggage_attrs)
            full = _attempt(
                lambda: _canonical_attributes(
                    name,
                    source,
                    capture_content=True,
                    max_attribute_chars=self._max_attribute_chars,
                ),
                {},
            )
            for key in (
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            ):
                if key in full:
                    proxy.set_attribute(key, full[key])
        return proxy

    @contextmanager
    def start_as_current_span(self, name: str, *args: Any, **kwargs: Any):
        kwargs.pop("end_on_exit", None)
        kwargs.pop("record_exception", None)
        kwargs.pop("set_status_on_exception", None)
        span = self.start_span(name, *args, **kwargs)
        token = None
        if span.get_span_context().is_valid:
            token = _attempt(
                lambda: otel_context.attach(trace.set_span_in_context(span))
            )
        try:
            yield span
        except BaseException as exc:
            span.record_exception(exc)
            raise
        finally:
            span.end()
            if token is not None:
                _attempt(lambda: otel_context.detach(token))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tracer, name)


def _build_interceptor(
    base_class: type,
    *,
    tracer: Any,
    capture_content: bool,
    max_attribute_chars: int | None,
    always_create_workflow_spans: bool,
) -> Any:
    canonical_tracer = _CanonicalTracer(
        tracer,
        capture_content=capture_content,
        max_attribute_chars=max_attribute_chars,
    )

    class RespanTemporalTracingInterceptor(base_class):
        @contextmanager
        def _start_as_current_span(
            self,
            name: str,
            *,
            attributes: Mapping[str, Any] | None,
            input_with_headers: Any = None,
            input_with_ctx: Any = None,
            kind: Any,
            context: Any = None,
        ):
            enriched = _attempt(lambda: dict(attributes or {}), {})
            # Defer payload access until after the real sampler and privacy gate.
            enriched[TEMPORAL_CAPTURED_INPUT] = lambda: {
                **(
                    _extract_temporal_input(input_with_headers)
                    if input_with_headers is not None
                    else {}
                ),
                **(
                    _extract_temporal_input(input_with_ctx)
                    if input_with_ctx is not None
                    else {}
                ),
            }
            allowed, _ = _parent_decisions((otel_context.get_current(), context))
            if not capture_content or not allowed:
                context = _attempt(
                    lambda: baggage.set_baggage(
                        ENABLE_CONTENT_TRACING_KEY, "false", context=context
                    ),
                    context,
                )
            token = (
                _attempt(lambda: otel_context.attach(context))
                if context is not None
                else None
            )
            try:
                with canonical_tracer.start_as_current_span(
                    name, attributes=enriched, kind=kind, context=context
                ) as span:
                    propagated_token = None
                    if span.is_recording() and span.allowed():
                        propagated = _attempt(
                            lambda: _context_with_respan_baggage(
                                otel_context.get_current()
                            )
                        )
                        if propagated is not None:
                            propagated_token = _attempt(
                                lambda: otel_context.attach(propagated)
                            )
                            attrs: dict[str, Any] = {}
                            _attempt(lambda: _apply_respan_baggage(attrs, propagated))
                            span.capture_baggage(attrs)
                    if input_with_headers is not None:
                        updated = _attempt(
                            lambda: self._context_to_headers(input_with_headers.headers)
                        )
                        if updated is not None:
                            input_with_headers.headers = updated
                    try:
                        yield None
                    finally:
                        # Observe before the propagated operation context detaches.
                        span.allowed()
                        if propagated_token is not None:
                            _attempt(lambda: otel_context.detach(propagated_token))
            finally:
                if token is not None:
                    _attempt(lambda: otel_context.detach(token))

        def _context_from_headers(self, headers: Any) -> Any:
            return _attempt(
                lambda: super(
                    RespanTemporalTracingInterceptor, self
                )._context_from_headers(headers)
            )

        def _completed_workflow_span(self, params: Any) -> Any:
            return _attempt(
                lambda: super(
                    RespanTemporalTracingInterceptor, self
                )._completed_workflow_span(params)
            )

        def intercept_activity(self, next: Any) -> Any:
            from temporalio.worker import ActivityInboundInterceptor

            class CaptureActivity(ActivityInboundInterceptor):
                async def execute_activity(self, input: Any) -> Any:
                    span = trace.get_current_span()
                    if isinstance(span, _CanonicalSpanProxy):
                        span.capture_input({"args": input.args})
                    result = await super().execute_activity(input)
                    if isinstance(span, _CanonicalSpanProxy):
                        span.capture_output(result)
                    return result

            return super().intercept_activity(CaptureActivity(next))

        def intercept_client(self, next: Any) -> Any:
            from temporalio.client import OutboundInterceptor

            class CaptureClient(OutboundInterceptor):
                async def query_workflow(self, input: Any) -> Any:
                    result = await super().query_workflow(input)
                    span = trace.get_current_span()
                    if isinstance(span, _CanonicalSpanProxy):
                        span.capture_output(result)
                    return result

            return super().intercept_client(CaptureClient(next))

        def workflow_interceptor_class(self, input: Any) -> type:
            native = super().workflow_interceptor_class(input)
            from temporalio import workflow
            from temporalio.worker import WorkflowInboundInterceptor

            class CaptureWorkflow(native):
                async def execute_workflow(self, value: Any) -> Any:
                    self._respan_input = {"args": value.args}
                    self._respan_result = _MISSING
                    # Keep the native replay-aware context and completed spans;
                    # the only addition is data passed through its extern hook.
                    with self._top_level_workflow_context(success_is_complete=True):
                        self._completed_span(
                            f"RunWorkflow:{workflow.info().workflow_type}",
                            kind=trace.SpanKind.SERVER,
                        )
                        result = await WorkflowInboundInterceptor.execute_workflow(
                            self, value
                        )
                        self._respan_result = result
                        return result

                def _completed_span(self, span_name: str, **kwargs: Any) -> None:
                    attributes = dict(kwargs.pop("additional_attributes", None) or {})
                    outbound = kwargs.get("add_to_outbound")
                    if outbound is not None:
                        attributes[TEMPORAL_CAPTURED_INPUT] = {
                            "args": getattr(outbound, "args", ())
                        }
                    if span_name.startswith(("RunWorkflow:", "CompleteWorkflow:")):
                        attributes[TEMPORAL_CAPTURED_INPUT] = getattr(
                            self, "_respan_input", {}
                        )
                    if span_name.startswith("CompleteWorkflow:"):
                        result = getattr(self, "_respan_result", _MISSING)
                        if result is not _MISSING:
                            attributes[_CAPTURED_OUTPUT] = result
                    return _attempt(
                        lambda: super(CaptureWorkflow, self)._completed_span(
                            span_name, additional_attributes=attributes, **kwargs
                        )
                    )

            return CaptureWorkflow

    RespanTemporalTracingInterceptor.__name__ = "RespanTemporalTracingInterceptor"
    return RespanTemporalTracingInterceptor(
        tracer=canonical_tracer,
        always_create_workflow_spans=always_create_workflow_spans,
    )


class TemporalInstrumentor:
    """Inject a canonicalized official Temporal tracing interceptor."""

    name = TEMPORAL_INSTRUMENTATION_NAME
    _patches_applied = False
    _activation_count = 0
    _lock = RLock()
    _shared_config: tuple[bool, bool, int | None] | None = None
    _client_class: type[Any] | None = None
    _original_connect_descriptor_holder: tuple[Any, ...] = ()
    _installed_connect_function: Any = None
    _patch_generation = 0

    def __init__(
        self,
        *,
        capture_content: bool = True,
        always_create_workflow_spans: bool = False,
        max_attribute_chars: int | None = None,
    ) -> None:
        self._capture_content = capture_content
        self._always_create_workflow_spans = always_create_workflow_spans
        self._max_attribute_chars = (
            max(512, int(max_attribute_chars))
            if max_attribute_chars is not None
            else None
        )
        self._is_instrumented = False
        self._interceptor: Any = None
        self._base_interceptor_class: type | None = None

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def _ensure_interceptor(self) -> Any:
        if self._interceptor is not None:
            return self._interceptor
        _attempt(_observe_provider)
        otel_module = importlib.import_module(TEMPORAL_OTEL_MODULE)
        self._base_interceptor_class = otel_module.TracingInterceptor
        self._interceptor = _build_interceptor(
            self._base_interceptor_class,
            tracer=trace.get_tracer(
                TEMPORAL_INSTRUMENTATION_NAME,
                _instrumentation_version(),
            ),
            capture_content=self._capture_content,
            max_attribute_chars=self._max_attribute_chars,
            always_create_workflow_spans=self._always_create_workflow_spans,
        )
        return self._interceptor

    @property
    def interceptor(self) -> Any:
        """The interceptor for explicit Temporal client/test-environment wiring."""
        return self._ensure_interceptor()

    async def _connect(
        self,
        wrapped: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        interceptor = _attempt(self._ensure_interceptor)
        if interceptor is None:
            return await wrapped(*args, **kwargs)
        connect_kwargs = dict(kwargs)
        interceptors = list(connect_kwargs.get("interceptors") or ())
        base_class = self._base_interceptor_class
        has_temporal_tracing = bool(
            base_class is not None
            and any(isinstance(candidate, base_class) for candidate in interceptors)
        )
        if not has_temporal_tracing:
            interceptors.append(interceptor)
            connect_kwargs["interceptors"] = interceptors
        return await wrapped(*args, **connect_kwargs)

    def activate(self) -> None:
        """Patch `Client.connect` to inject the Respan Temporal interceptor."""
        cls = type(self)
        if not self._is_respan_tracing_enabled():
            logger.info(
                "Temporal instrumentation skipped because Respan tracing is disabled"
            )
            return
        with cls._lock:
            if self._is_instrumented:
                return
            config = (
                self._capture_content,
                self._always_create_workflow_spans,
                self._max_attribute_chars,
            )
            if cls._patches_applied:
                if cls._shared_config != config:
                    raise ValueError(
                        "Temporal instrumentation is already active with different settings"
                    )
                cls._activation_count += 1
                self._is_instrumented = True
                return
            try:
                client_module = importlib.import_module(TEMPORAL_CLIENT_MODULE)
                client_class = getattr(client_module, "Client", None)
                if client_class is None or not hasattr(client_class, "connect"):
                    logger.warning("Temporal Client.connect is unavailable")
                    return
                self._ensure_interceptor()

                original_descriptor = client_class.__dict__.get(
                    _CLIENT_CONNECT_ATTRIBUTE
                )
                original_connect = getattr(client_class, _CLIENT_CONNECT_ATTRIBUTE)
                cls._patch_generation += 1
                generation = cls._patch_generation

                async def traced_connect(
                    client_cls: type[Any], *args: Any, **kwargs: Any
                ) -> Any:
                    if (
                        type(self)._activation_count == 0
                        or type(self)._patch_generation != generation
                    ):
                        return await original_connect(*args, **kwargs)
                    return await self._connect(original_connect, args, kwargs)

                installed_descriptor = classmethod(traced_connect)
                setattr(
                    client_class,
                    _CLIENT_CONNECT_ATTRIBUTE,
                    installed_descriptor,
                )
                cls._client_class = client_class
                cls._original_connect_descriptor_holder = (original_descriptor,)
                cls._installed_connect_function = traced_connect
            except ImportError as exc:
                logger.warning(
                    "Failed to activate Temporal instrumentation - missing dependency: %s",
                    exc,
                )
                return
            except Exception:
                logger.exception("Failed to activate Temporal instrumentation")
                return
            cls._patches_applied = True
            cls._activation_count = 1
            cls._shared_config = config
            self._is_instrumented = True
            logger.info("Temporal instrumentation activated")

    def deactivate(self) -> None:
        """Restore `Client.connect`; existing clients retain their interceptor."""
        cls = type(self)
        with cls._lock:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            cls._activation_count = max(cls._activation_count - 1, 0)
            if cls._activation_count:
                return
            if (
                cls._client_class is not None
                and getattr(
                    cls._client_class.__dict__.get(_CLIENT_CONNECT_ATTRIBUTE),
                    "__func__",
                    None,
                )
                is cls._installed_connect_function
            ):
                setattr(
                    cls._client_class,
                    _CLIENT_CONNECT_ATTRIBUTE,
                    cls._original_connect_descriptor_holder[0],
                )
            cls._client_class = None
            cls._original_connect_descriptor_holder = ()
            cls._installed_connect_function = None
            cls._patches_applied = False
            cls._shared_config = None
            logger.info("Temporal instrumentation deactivated")
