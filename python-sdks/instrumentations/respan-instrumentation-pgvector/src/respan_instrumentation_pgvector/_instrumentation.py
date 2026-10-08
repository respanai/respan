"""Observe real pgvector/psycopg APIs without altering native objects or resources."""

# ruff: noqa: BLE001 -- telemetry faults fail closed, never change native outcomes.
from __future__ import annotations

import contextvars
import functools
import importlib
import json
import logging
import re
import threading
import types
import weakref

from opentelemetry import context, trace
from opentelemetry.sdk.trace import _Span
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes.db_attributes import (
    DB_NAMESPACE,
    DB_OPERATION_NAME,
    DB_QUERY_TEXT,
    DB_RESPONSE_STATUS_CODE,
    DB_SYSTEM_NAME,
)
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_MESSAGE,
    ERROR_TYPE,
)
from opentelemetry.semconv._incubating.attributes.server_attributes import (
    SERVER_ADDRESS,
    SERVER_PORT,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from respan_sdk.constants.llm_logging import LOG_TYPE_TASK
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA
from respan_tracing.core.tracer import RespanTracer

from ._constants import CURSOR_METHODS
from ._native import (
    CURSORS,
    compiled_query,
    cursor_metadata,
    database_metadata,
    member,
    namespace,
    native_json,
    storage,
)
from ._policy import (
    CREATING_CALL,
    AncestorPolicy,
    content_allowed,
    span_key,
    suppressed,
)
from ._serialization import (
    REDACTED,
    json_dumps,
    safe_exception_message,
    safe_text,
    safe_type_name,
    sensitive_key,
)

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_ACTIVE = contextvars.ContextVar("respan_pgvector_call", default=None)
_POLICIES = weakref.WeakKeyDictionary()
_PENDING = weakref.WeakSet()
_PATCHES = []
_CONFIG = None
_OWNERS = 0
_NATIVE_END = _Span.end
_SQL_OPERATION = re.compile(
    r"^\s*(?:/\*.*?\*/\s*)*(?:--[^\n]*\n\s*)*([A-Za-z]+)", re.DOTALL
)


def _attempt(fn, default=None):
    try:
        return fn()
    except BaseException:
        logger.debug("pgvector telemetry operation failed")
        return default


def _restore_context(ambient):
    if context.get_current() is ambient:
        return
    _attempt(lambda: context._RUNTIME_CONTEXT.attach(ambient))
    if context.get_current() is not ambient:
        for item in storage(context._RUNTIME_CONTEXT).values():
            if type(item) is contextvars.ContextVar:
                item.set(ambient)
                break


def _enabled():
    current = RespanTracer._instance
    return current is None or storage(current).get("is_enabled", True) is not False


def _provider():
    provider = _CONFIG[0] if _CONFIG is not None else None
    provider = provider if provider is not None else trace.get_tracer_provider()
    if type(provider) is trace.ProxyTracerProvider:
        provider = trace._TRACER_PROVIDER
    return provider


def _policy():
    provider = _provider()
    if provider is None or not hasattr(provider, "add_span_processor"):
        return None
    policy = _POLICIES.get(provider)
    if policy is None:
        policy = AncestorPolicy(_CONFIG[1])
        _POLICIES[provider] = policy
        provider.add_span_processor(policy)
        processor = provider._active_span_processor
        processor._span_processors = (policy,) + tuple(
            p for p in processor._span_processors if p is not policy
        )
    return policy


def _remove_policies():
    for provider, policy in list(_POLICIES.items()):
        policy.enabled = False
        processor = provider._active_span_processor
        _attempt(
            lambda processor=processor, policy=policy: setattr(
                processor,
                "_span_processors",
                tuple(p for p in processor._span_processors if p is not policy),
            )
        )
        policy.clear()
    _POLICIES.clear()


def _observe(call, fn):
    ambient = context.get_current()
    try:
        return fn()
    except BaseException:
        call.failed = True
        _attempt(call.scrub)
        return None
    finally:
        if not call.finished and call.span is not None:
            _attempt(lambda: call.allowed(honor_suppression=False))
        _restore_context(ambient)


class _Call:
    def __init__(self, operation, method):
        self.span = None
        self.policy = None
        self.finished = False
        self.failed = False
        self.creation_name = f"pgvector.{operation}"
        self.operation = operation
        self.method = method
        self.payload = None
        self.output = None
        self.has_output = False
        self.native_meta = None
        self.owner = None
        self.cleanups = []
        self.initially_allowed = content_allowed(_CONFIG[1])
        self.base = {
            RESPAN_LOG_TYPE: LOG_TYPE_TASK,
            DB_SYSTEM_NAME: "postgresql",
            DB_OPERATION_NAME: method or operation,
            SpanAttributes.TRACELOOP_ENTITY_NAME: self.creation_name,
            SpanAttributes.TRACELOOP_ENTITY_PATH: self.creation_name
            if trace.get_current_span().get_span_context().is_valid
            else "",
        }
        _PENDING.add(self)

    def recording(self):
        return self.span is not None and _attempt(self.span.is_recording, False)

    def allowed(self, *, honor_suppression=True):
        allowed = (
            self.initially_allowed
            and not self.failed
            and self.recording()
            and self.policy is not None
            and self.policy.observe(self.span, honor_suppression=honor_suppression)
        )
        if not allowed:
            if self.policy is not None:
                self.policy._deny_chain(span_key(self.span))
            self.scrub()
        return bool(allowed)

    def scrub(self, readable=None):
        self.payload = None
        self.output = None
        self.has_output = False
        self.native_meta = None
        self.owner = None
        structural = set(self.base) | {ERROR_TYPE, DB_RESPONSE_STATUS_CODE}
        if self.span is not None:
            attributes = member(self.span, "_attributes")
            raw = storage(attributes).get("_dict")
            if type(raw) is dict:
                keep = {
                    k: v for k, v in raw.items() if type(k) is str and k in structural
                }
                raw.clear()
                raw.update(keep)
            if member(self.span, "_events") is not None:
                self.span._events = BoundedList(0)
            status = member(self.span, "_status")
            if status is not None:
                self.span._status = Status(status.status_code)
        if readable is not None:
            readable._attributes = types.MappingProxyType(
                {
                    k: v
                    for k, v in (readable.attributes or {}).items()
                    if type(k) is str and k in structural
                }
            )
            readable._events = ()
            readable._status = Status(readable.status.status_code)

    def set(self, key, item):
        self.span.set_attribute(key, item)

    def request(self, instance, args, kwargs):
        if not self.allowed():
            return
        if any(type(instance) is cls for cls in CURSORS):
            self.owner = instance
        self.payload = {
            "operation": self.operation,
            "arguments": native_json(args),
            "parameters": native_json(kwargs),
        }
        metadata = database_metadata(
            instance if instance is not None else (args[0] if args else None)
        )
        for key, target in (
            ("database", DB_NAMESPACE),
            ("host", SERVER_ADDRESS),
            ("port", SERVER_PORT),
        ):
            if key in metadata:
                value = metadata[key]
                if key == "port" and value.isdecimal():
                    value = int(value)
                self.base[target] = value
                self.set(target, value)
        query = kwargs.get("query", args[0] if args else None)
        if self.method in ("execute", "executemany") and type(query) is str:
            self.set(DB_QUERY_TEXT, safe_text(query))
            operation = _SQL_OPERATION.match(query)
            if operation:
                self.base[DB_OPERATION_NAME] = operation.group(1).upper()
                self.set(DB_OPERATION_NAME, self.base[DB_OPERATION_NAME])

    def response(self, result):
        if not self.allowed():
            return
        query = compiled_query(result)
        if query is not None and self.payload is not None:
            self.payload["compiled_query"] = safe_text(query)
            self.set(DB_QUERY_TEXT, safe_text(query))
        self.output = native_json(result)
        self.has_output = True
        metadata = cursor_metadata(result)
        if not metadata and self.owner is not None:
            metadata = cursor_metadata(self.owner)
        columns = metadata.get("columns", []) if type(metadata) is dict else []
        secret_positions = [
            i for i, column in enumerate(columns) if sensitive_key(column.get("name"))
        ]
        if self.method.startswith("fetch") and secret_positions:
            rows = [self.output] if self.method == "fetchone" else self.output
            if type(rows) is list:
                for row in rows:
                    if type(row) is list:
                        for position in secret_positions:
                            if position < len(row):
                                row[position] = REDACTED
        if metadata:
            self.native_meta = metadata

    def error(self, error):
        if not self.recording():
            return
        self.output = None
        self.has_output = False
        self.span.set_status(Status(StatusCode.ERROR))
        self.set(ERROR_TYPE, safe_type_name(error))
        cls = type(error)
        module = namespace(cls).get("__module__", "")
        native = type(module) is str and module.startswith(("psycopg", "psycopg2"))
        installed = (
            _attempt(
                lambda: getattr(importlib.import_module(module), safe_type_name(error))
            )
            if native
            else None
        )
        if native and installed is cls:
            code = None
            for base in type.__dict__["__mro__"].__get__(cls):
                candidate = namespace(base).get("sqlstate")
                if type(candidate) is str:
                    code = candidate
                    break
            if code is not None:
                self.set(DB_RESPONSE_STATUS_CODE, code)
        if self.allowed() and (
            (native and installed is cls)
            or any(cls is k for k in (ValueError, TypeError, RuntimeError, OSError))
        ):
            message = safe_exception_message(error)
            if message is not None:
                self.set(ERROR_MESSAGE, message)

    def finish(self):
        if self.finished:
            return
        if self.recording():
            _observe(self, self._finish_attributes)
        self.finished = True
        if self.span is not None:
            try:
                self.span.end()
            except BaseException:
                self.failed = True
                _attempt(self.scrub)
                _attempt(lambda: _NATIVE_END(self.span))
        for undo in reversed(self.cleanups):
            _attempt(undo)
        self.cleanups.clear()
        self.payload = None
        self.output = None
        self.native_meta = None
        self.owner = None
        if self.policy is not None:
            self.policy.calls.pop(span_key(self.span), None)
        _PENDING.discard(self)

    def _finish_attributes(self):
        for key, value in self.base.items():
            self.set(key, value)
        if not self.allowed(honor_suppression=False):
            return
        if self.native_meta is not None:
            self.set(f"{RESPAN_METADATA}.pgvector.result", json_dumps(self.native_meta))
        if self.payload is not None:
            self.set(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_dumps(self.payload))
        if self.has_output:
            self.set(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_dumps(self.output))


def _protect(call):
    span = call.span
    if not call.recording():
        return
    for key in (
        f"{RESPAN_METADATA}.run_id",
        f"{RESPAN_METADATA}.scenario",
        f"{RESPAN_METADATA}.example_set",
    ):
        item = (span.attributes or {}).get(key)
        if type(item) is str:
            call.base[key] = safe_text(item)
    marker = (span.attributes or {}).get(RESPAN_METADATA)
    if type(marker) is str:
        decoded = _attempt(lambda: json.loads(marker), {})
        if type(decoded) is dict:
            keep = {
                k: v
                for k, v in decoded.items()
                if type(k) is str
                and k in {"run_id", "scenario", "example_set"}
                and type(v) is str
            }
            if keep:
                call.base[RESPAN_METADATA] = json_dumps(keep)
    state = storage(span)
    present = "set_attribute" in state
    prior = state.get("set_attribute")
    original = span.set_attribute

    def setter(key, item):
        try:
            return original(key, item)
        except BaseException:
            call.failed = True
            _attempt(call.scrub)
            return None

    def undo():
        if storage(span).get("set_attribute") is setter:
            if present:
                span.set_attribute = prior
            else:
                delattr(span, "set_attribute")

    call.cleanups.append(undo)
    span.set_attribute = setter


def _start(operation, method, instance, args, kwargs):
    ambient = context.get_current()
    call = None
    try:
        if _CONFIG is None or not _enabled() or suppressed():
            return None
        call = _Call(operation, method)

        def startup():
            call.policy = _policy()
            provider = _provider()
            if provider is None:
                return
            token = CREATING_CALL.set(call)
            try:
                call.span = provider.get_tracer("pgvector").start_span(
                    call.creation_name, kind=SpanKind.CLIENT, attributes=call.base
                )
            finally:
                CREATING_CALL.reset(token)
            _protect(call)

        _observe(call, startup)
        if call.failed:
            call.finish()
            return None
        if call.recording():
            _observe(call, lambda: call.request(instance, args, kwargs))
        return call
    except BaseException:
        if call is not None:
            call.failed = True
            _attempt(call.scrub)
            _attempt(call.finish)
        return None
    finally:
        _restore_context(ambient)


def _wrap(original, operation, method, asynchronous, instance_method):
    def setup(args, kwargs):
        instance = args[0] if instance_method and args else None
        actual = args[1:] if instance_method else args
        startup = _ACTIVE.set(True)
        try:
            return _start(operation, method, instance, actual, kwargs)
        finally:
            _ACTIVE.reset(startup)

    async def async_call(*args, **kwargs):
        if _ACTIVE.get() is not None:
            return await original(*args, **kwargs)
        call = setup(args, kwargs)
        if call is None:
            return await original(*args, **kwargs)
        ambient = context.get_current()
        active = _ACTIVE.set(call)
        token = None
        try:
            if call.span is not None:
                token = _attempt(
                    lambda: context.attach(trace.set_span_in_context(call.span))
                )
                if token is None:
                    call.failed = True
                    call.scrub()
                    _restore_context(ambient)
            try:
                result = await original(*args, **kwargs)
            except BaseException as error:
                _observe(call, functools.partial(call.error, error))
                raise
            if call.recording():
                _observe(call, lambda: call.response(result))
            return result
        finally:
            if call.recording():
                _observe(call, lambda: call.allowed(honor_suppression=False))
            if token is not None:
                _attempt(lambda: context.detach(token))
            _restore_context(ambient)
            call.finish()
            _ACTIVE.reset(active)
            _restore_context(ambient)

    def sync_call(*args, **kwargs):
        if _ACTIVE.get() is not None:
            return original(*args, **kwargs)
        call = setup(args, kwargs)
        if call is None:
            return original(*args, **kwargs)
        ambient = context.get_current()
        active = _ACTIVE.set(call)
        token = None
        try:
            if call.span is not None:
                token = _attempt(
                    lambda: context.attach(trace.set_span_in_context(call.span))
                )
                if token is None:
                    call.failed = True
                    call.scrub()
                    _restore_context(ambient)
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                _observe(call, functools.partial(call.error, error))
                raise
            if call.recording():
                _observe(call, lambda: call.response(result))
            return result
        finally:
            if call.recording():
                _observe(call, lambda: call.allowed(honor_suppression=False))
            if token is not None:
                _attempt(lambda: context.detach(token))
            _restore_context(ambient)
            call.finish()
            _ACTIVE.reset(active)
            _restore_context(ambient)

    return functools.wraps(original)(async_call if asynchronous else sync_call)


def _targets():
    import psycopg

    for cls, label, asynchronous in (
        (psycopg.Connection, "connection", False),
        (psycopg.AsyncConnection, "connection", True),
        (psycopg.Cursor, "cursor", False),
        (psycopg.ServerCursor, "server_cursor", False),
        (psycopg.AsyncCursor, "cursor", True),
        (psycopg.AsyncServerCursor, "server_cursor", True),
    ):
        for method in ("execute",) if label == "connection" else CURSOR_METHODS:
            original = None
            for base in type.__dict__["__mro__"].__get__(cls):
                if method in namespace(base):
                    original = namespace(base)[method]
                    break
            if original is not None:
                yield (
                    cls,
                    method,
                    original,
                    f"{label}.{method}",
                    method,
                    asynchronous,
                    True,
                )
    for module, name, label, asynchronous in (
        ("pgvector.psycopg", "register_vector", "register_vector", False),
        ("pgvector.psycopg", "register_vector_async", "register_vector", True),
        ("pgvector.psycopg2", "register_vector", "register_vector_psycopg2", False),
    ):
        try:
            owner = importlib.import_module(module)
        except ImportError:
            continue
        original = vars(owner).get(name)
        if original is not None:
            yield owner, name, original, label, "", asynchronous, False


def _restore():
    for owner, name, original, wrapper, present in reversed(_PATCHES):
        state = namespace(owner) if type(owner) is type else vars(owner)
        if state.get(name) is wrapper:
            if present:
                _attempt(
                    lambda owner=owner, name=name, original=original: setattr(
                        owner, name, original
                    )
                )
            else:
                _attempt(lambda owner=owner, name=name: delattr(owner, name))
    _PATCHES.clear()
    _remove_policies()


class PGVectorInstrumentor:
    """Native registration/execute/fetch observation; no cursor or result proxy."""

    name = "pgvector"

    def __init__(self, *, capture_content=True, tracer_provider=None):
        self.config = (tracer_provider, capture_content is True)
        self._is_instrumented = False

    def activate(self, *, tracer_provider=None):
        global _CONFIG, _OWNERS
        with _LOCK:
            if self._is_instrumented:
                return
            config = (
                self.config
                if tracer_provider is None
                else (tracer_provider, self.config[1])
            )
            if _OWNERS:
                if any(a is not b for a, b in zip(config, _CONFIG)):
                    raise ValueError("pgvector instrumentation configuration conflict")
                _OWNERS += 1
                self._is_instrumented = True
                return
            _CONFIG = config
            try:
                _policy()
                for (
                    owner,
                    name,
                    original,
                    operation,
                    method,
                    asynchronous,
                    instance_method,
                ) in _targets():
                    state = namespace(owner) if type(owner) is type else vars(owner)
                    present = name in state
                    wrapper = _wrap(
                        original, operation, method, asynchronous, instance_method
                    )
                    _PATCHES.append((owner, name, original, wrapper, present))
                    setattr(owner, name, wrapper)
            except BaseException:
                _restore()
                _CONFIG = None
                raise
            _OWNERS = 1
            self._is_instrumented = True

    def deactivate(self):
        global _CONFIG, _OWNERS
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _OWNERS -= 1
            if _OWNERS:
                return
            for call in list(_PENDING):
                call.failed = True
                _attempt(call.scrub)
                _attempt(call.finish)
            _restore()
            _CONFIG = None

    def instrument(self, **kwargs):
        return self.activate(**kwargs)

    def uninstrument(self):
        return self.deactivate()
