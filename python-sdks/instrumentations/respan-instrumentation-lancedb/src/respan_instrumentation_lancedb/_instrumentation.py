"""Observe released native LanceDB operations and Arrow values with real OTel."""

# ruff: noqa: BLE001 -- telemetry faults never alter native behavior.
from __future__ import annotations

import contextvars
import functools
import importlib
import importlib.metadata
import inspect
import threading
import types
import weakref

from opentelemetry import context, trace
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.db_attributes import (
    DB_COLLECTION_NAME,
    DB_OPERATION_NAME,
    DB_SYSTEM_NAME,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_PROMPT,
    RESPAN_SPAN_ATTRIBUTES_MAP,
)
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import _PROPAGATED_ATTRIBUTES

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
    to_jsonable,
)
from ._translator import (
    arguments,
    dumps,
    native_type,
    query_fields,
    schemas,
    storage,
    table_name,
)

_LOCK = threading.RLock()
_ACTIVATION_COUNT = 0
_ENABLED = False
_CAPTURE_CONTENT = True
_PROVIDER = None
_POLICIES = weakref.WeakKeyDictionary()
_PATCHES = []
_PENDING = weakref.WeakSet()
_BUILDERS = weakref.WeakKeyDictionary()
_ACTIVE_CALL = contextvars.ContextVar("respan_lancedb_active_call", default=False)
_HYBRID_SCOPE = contextvars.ContextVar("respan_lancedb_hybrid_scope", default=None)


def _base(operation):
    return {
        RESPAN_LOG_TYPE: "task",
        DB_SYSTEM_NAME: "lancedb",
        DB_OPERATION_NAME: operation.rsplit(".", 1)[-1],
    }


def _tracing_enabled():
    instance = RespanTracer._instance
    if instance is None:
        return True
    return (
        type(instance) is RespanTracer
        and object.__getattribute__(instance, "__dict__").get("is_enabled") is True
    )


def _provider():
    return _PROVIDER if _PROVIDER is not None else trace.get_tracer_provider()


def _policy():
    provider = _provider()
    with _LOCK:
        policy = _POLICIES.get(provider)
        if policy is None:
            if not callable(getattr(provider, "add_span_processor", None)):
                return None
            policy = AncestorPolicy(_CAPTURE_CONTENT)
            _POLICIES[provider] = policy
            try:
                provider.add_span_processor(policy)
                processor = getattr(provider, "_active_span_processor", None)
                if processor is not None:
                    processor._span_processors = (
                        policy,
                        *(
                            item
                            for item in processor._span_processors
                            if item is not policy
                        ),
                    )
            except BaseException:
                _remove_policies()
                raise
        policy.setting = _CAPTURE_CONTENT
        policy.enabled = True
        return policy


def _attempt(fn, default=None):
    try:
        return fn()
    except BaseException:
        return default


def _propagated_attributes():
    # The released bridge calls str() on metadata. Read the same canonical
    # ContextVar without invoking arbitrary customer formatting hooks.
    values = _PROPAGATED_ATTRIBUTES.get()
    result = {}
    if type(values) is not dict:
        return result
    for key, value in values.items():
        if type(key) is not str or key not in RESPAN_SPAN_ATTRIBUTES_MAP:
            continue
        target = RESPAN_SPAN_ATTRIBUTES_MAP[key]
        if target == RESPAN_METADATA and type(value) is dict:
            for name, item in value.items():
                if type(name) is str:
                    cleaned = REDACTED if sensitive_key(name) else to_jsonable(item)
                    result[f"{target}.{safe_text(name)}"] = (
                        cleaned if type(cleaned) is str else json_dumps(cleaned)
                    )
        elif target == RESPAN_PROMPT:
            result[target] = json_dumps(value)
        elif any(type(value) is kind for kind in (str, bool, int, float)):
            result[target] = to_jsonable(value)
    return result


def _restore_context(ambient):
    if context.get_current() is ambient:
        return
    _attempt(lambda: context._RUNTIME_CONTEXT.attach(ambient))
    if context.get_current() is not ambient:
        state = object.__getattribute__(context._RUNTIME_CONTEXT, "__dict__")
        for candidate in state.values():
            if type(candidate) is contextvars.ContextVar:
                candidate.set(ambient)
                break


class _Call:
    def __init__(self, kwargs, *, name, operation):
        self.span = None
        self.creation_name = name
        self.policy = None
        self.finished = False
        self.failed = False
        self.operation = operation
        self.chunks = []
        self.content = set()
        self.cleanups = []
        self.propagated = {}
        self.priority = {}
        self.base = _base(operation)
        self.base[SpanAttributes.TRACELOOP_ENTITY_NAME] = name
        self.base[SpanAttributes.TRACELOOP_ENTITY_PATH] = (
            "" if not trace.get_current_span().get_span_context().is_valid else name
        )
        _PENDING.add(self)
        self.policy = _policy()
        creation_token = CREATING_CALL.set(self)
        try:
            self.span = (
                _provider()
                .get_tracer(
                    "lancedb",
                    importlib.metadata.version("respan-instrumentation-lancedb"),
                )
                .start_span(name, kind=trace.SpanKind.CLIENT, attributes=self.base)
            )
        finally:
            CREATING_CALL.reset(creation_token)
        if self.recording() and self.allowed():
            self.propagated = _propagated_attributes()
            self.set_attributes({SpanAttributes.TRACELOOP_ENTITY_INPUT: dumps(kwargs)})

    def __del__(self):
        try:
            if not getattr(self, "finished", True):
                if self.recording() and self.chunks:
                    _observe(self, lambda: self.output(self.chunks))
                self.finish(completed=False)
        except BaseException:  # noqa: S110 - GC telemetry must not affect native resources.
            pass

    def recording(self):
        return self.span is not None and bool(_attempt(self.span.is_recording, False))

    def scrub(self, readable=None):
        builder = getattr(self, "builder", None)
        if builder is not None:
            value = builder()
            if value is not None and value in _BUILDERS:
                state = _BUILDERS[value]
                _BUILDERS[value] = (False, [], state[2], state[3], state[4])
        self.chunks.clear()
        self.propagated.clear()
        self.priority.clear()
        attributes = getattr(self.span, "_attributes", None)
        structural = set(self.base) | {ERROR_TYPE}
        if attributes is not None:
            for key in list(attributes):
                if key not in structural:
                    _attempt(lambda key=key: attributes.pop(key, None))
        if getattr(self.span, "_events", None) is not None:
            self.span._events = BoundedList(0)
        if readable is not None:
            readable._attributes = types.MappingProxyType(
                {
                    key: value
                    for key, value in (readable.attributes or {}).items()
                    if key in structural
                }
            )
            readable._events = ()
            readable._status = Status(readable.status.status_code)
        status = getattr(self.span, "status", None)
        if status is not None:
            self.span._status = Status(status.status_code)

    def allowed(self, *, honor_suppression=True):
        if self.failed and self.policy is not None:
            self.policy._deny_chain(span_key(self.span))
        value = (
            not self.failed
            and self.policy is not None
            and self.policy.observe(self.span, honor_suppression=honor_suppression)
        )
        if not value:
            self.scrub()
        return value

    def set_attributes(self, values):
        priority = {
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            RESPAN_LOG_TYPE,
            f"{RESPAN_METADATA}.lancedb.request",
            f"{RESPAN_METADATA}.lancedb.result",
        }
        for key, value in values.items():
            if key not in priority:
                self.content.add(key)
                self.span.set_attribute(key, value)
        # Reassert structure/propagation after indexed convenience fields. Full
        # native JSON is last under the SDK's own attribute-count bounds.
        for key, value in {**self.base, **self.propagated}.items():
            self.content.add(key)
            self.span.set_attribute(key, value)
        self.priority.update(
            {key: value for key, value in values.items() if key in priority}
        )
        for key, value in self.priority.items():
            self.content.add(key)
            self.span.set_attribute(key, value)

    def capture(self, response):
        if self.recording() and self.allowed():
            self.chunks.append(response)

    def output(self, response):
        if self.recording() and self.allowed():
            values = {SpanAttributes.TRACELOOP_ENTITY_OUTPUT: dumps(response)}
            actual_schemas = schemas(response)
            if actual_schemas:
                values[f"{RESPAN_METADATA}.lancedb.result"] = dumps(
                    {"schemas": actual_schemas}
                )
            self.set_attributes(values)

    def error(self, exc):
        if not self.recording():
            return
        allowed = self.allowed()
        message = (
            safe_exception_message(exc)
            if allowed
            and any(
                type(exc) is kind
                for kind in (
                    ValueError,
                    RuntimeError,
                    TypeError,
                    OSError,
                    IOError,
                    KeyError,
                )
            )
            else None
        )
        self.span.set_status(Status(StatusCode.ERROR, message))
        self.span.set_attribute(ERROR_TYPE, safe_type_name(exc))
        if message is not None:
            self.content.add(ERROR_MESSAGE)
            self.span.set_attribute(ERROR_MESSAGE, message)

    def attach(self):
        ambient = context.get_current()
        try:
            token = (
                context.attach(trace.set_span_in_context(self.span))
                if self.recording()
                else None
            )
        except BaseException:
            _attempt(lambda: _restore_context(ambient))
            raise
        return token, ambient

    def detach(self, state):
        if state is None:
            return
        token, ambient = state
        if not self.finished:
            _attempt(lambda: self.allowed(honor_suppression=False))
        if token is not None:
            try:
                context.detach(token)
            except BaseException:
                _attempt(lambda: context._RUNTIME_CONTEXT.detach(token))
        _attempt(lambda: _restore_context(ambient))

    def finish(self, error=None, *, completed=True):
        if self.finished:
            return
        self.finished = True
        if self.recording():
            if error is None and completed:
                _attempt(lambda: self.span.set_status(Status(StatusCode.OK)))
            elif error is not None:
                _attempt(lambda: self.error(error))
            if not _attempt(lambda: self.allowed(honor_suppression=False), False):
                _attempt(self.scrub)
            ambient = context.get_current()
            try:
                _attempt(self.span.end)
            finally:
                _attempt(lambda: _restore_context(ambient))
        for cleanup in reversed(self.cleanups):
            _attempt(cleanup)
        self.cleanups.clear()
        self.chunks.clear()
        self.content.clear()
        self.propagated.clear()
        self.priority.clear()
        if self.policy is not None and self.span is not None:
            self.policy.calls.pop(span_key(self.span), None)
        self.policy = None
        _PENDING.discard(self)


def _observe(call, fn, *, keep_context=False):
    ambient = context.get_current()
    try:
        return fn()
    except BaseException:
        call.failed = True
        _attempt(call.scrub)
        _attempt(lambda: _restore_context(ambient))
        return None
    finally:
        if not keep_context:
            if not call.finished and call.policy is not None and call.span is not None:
                _attempt(
                    lambda: call.policy.observe(call.span, honor_suppression=False)
                )
            _attempt(lambda: _restore_context(ambient))


def _remove_policies():
    for provider, policy in list(_POLICIES.items()):
        policy.enabled = False
        processor = getattr(provider, "_active_span_processor", None)
        if processor is not None:
            _attempt(
                lambda processor=processor, policy=policy: setattr(
                    processor,
                    "_span_processors",
                    tuple(p for p in processor._span_processors if p is not policy),
                )
            )
        policy.clear()
    _POLICIES.clear()


def _start(original, instance, args, kwargs, operation):
    ambient = context.get_current()
    call = None
    try:
        builder_state = _BUILDERS.get(instance) if native_type(instance) else None
        if (
            builder_state is not None
            and builder_state[4] is not None
            and builder_state[4]["active"]
        ):
            # Native hybrid legs run on ThreadPoolExecutor workers without
            # ContextVars. Their builders carry this bodyless owned call guard;
            # the actual aggregate operation owns the single result/error span.
            return None
        if not _tracing_enabled() or suppressed() or _ACTIVE_CALL.get():
            return None
        # Allocate before __init__ so an on_start failure can still end the
        # actual span captured by the first owned observer.
        call = _Call.__new__(_Call)
        _Call.__init__(call, {}, name=f"lancedb.{operation}", operation=operation)
        if call.recording():
            builder_owner = (
                args[0]
                if operation == "merge.execute" and args and native_type(args[0])
                else instance
            )
            if native_type(builder_owner):
                call.builder = weakref.ref(builder_owner)
            state = _BUILDERS.get(builder_owner) if native_type(builder_owner) else None
            if state is not None and (not state[0] or not state[3]._allowed(state[2])):
                call.failed = True
            if call.allowed():
                payload = {
                    "operation": operation,
                    "arguments": arguments(original, instance, args, kwargs),
                }
                if state is not None:
                    payload["builder"] = state[1]
                if operation.startswith(("query.", "merge.")):
                    payload["native_configuration"] = query_fields(builder_owner)
                collection = table_name(instance)
                if collection is None and operation in (
                    "connection.create_table",
                    "connection.open_table",
                    "connection.drop_table",
                ):
                    candidate = payload["arguments"].get("name")
                    collection = (
                        safe_text(candidate) if type(candidate) is str else None
                    )
                if collection is not None:
                    call.base[DB_COLLECTION_NAME] = collection
                call.set_attributes(
                    {SpanAttributes.TRACELOOP_ENTITY_INPUT: dumps(payload)}
                )
                native_schemas = [
                    schema
                    for value in payload["arguments"].values()
                    for schema in schemas(value)
                ]
                if native_schemas:
                    call.set_attributes(
                        {
                            f"{RESPAN_METADATA}.lancedb.request": dumps(
                                {"schemas": native_schemas}
                            )
                        }
                    )
        return call
    except BaseException:
        if call is not None and getattr(call, "span", None) is not None:
            call.failed = True
            _attempt(call.scrub)
            _attempt(lambda: call.finish(completed=False))
        return None
    finally:
        _attempt(lambda: _restore_context(ambient))


def _remember(original, instance, args, kwargs, result, name):
    if not native_type(result):
        return
    ambient = context.get_current()
    try:
        policy = _policy()
        current = trace.get_current_span()
        remote = (
            type(current) is trace.NonRecordingSpan
            and current.get_span_context().is_remote
        )
        allowed = (
            policy is not None
            and (remote or policy.observe(current))
            and content_allowed(_CAPTURE_CONTENT)
        )
        previous = _BUILDERS.get(instance)
        if previous is not None:
            allowed = allowed and previous[0]
        # Retain original argument references without conversion or iteration;
        # payload inspection happens only on a recording execution span. The
        # SDK itself owns these values throughout the builder's lifetime.
        history = list(previous[1]) if allowed and previous is not None else []
        if allowed:
            history.append({"method": name, "args": args, "kwargs": kwargs})
        current = trace.get_current_span()
        parent_key = (
            None
            if type(current) is trace.NonRecordingSpan
            and current.get_span_context().is_remote
            else span_key(current)
        )
        _BUILDERS[result] = (allowed, history, parent_key, policy, _HYBRID_SCOPE.get())
    except BaseException:
        _BUILDERS[result] = (False, [], None, policy, _HYBRID_SCOPE.get())
    finally:
        _restore_context(ambient)


class _AsyncTap:
    def __init__(self, source, call):
        self.source = source
        self.call = call

    def __aiter__(self):
        return self

    async def __anext__(self):
        state = _observe(self.call, self.call.attach, keep_context=True)
        try:
            value = await self.source.__anext__()
        except StopAsyncIteration:
            _observe(self.call, lambda: self.call.output(self.call.chunks))
            self.call.finish()
            raise
        except BaseException as error:
            if self.call.chunks:
                _observe(self.call, lambda: self.call.output(self.call.chunks))
            self.call.finish(error)
            raise
        else:
            _observe(self.call, lambda: self.call.capture(value))
            return value
        finally:
            self.call.detach(state)


def _tap_reader(result, call):
    from lancedb._lancedb import RecordBatchStream
    from lancedb.arrow import AsyncRecordBatchReader

    if type(result) is not AsyncRecordBatchReader:
        return False
    fields = storage(result)
    source = fields.get("_inner")
    if (
        type(source) is not RecordBatchStream
        and type(source) is not types.AsyncGeneratorType
    ):
        return False
    tap = _AsyncTap(source, call)

    def cleanup():
        if fields.get("_inner") is tap:
            fields["_inner"] = source

    call.cleanups.append(cleanup)
    fields["_inner"] = tap
    return True


def _wrap(original, operation, *, builder=False):
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def async_wrapper(instance, *args, **kwargs):
            if builder:
                result = await original(instance, *args, **kwargs)
                if _ENABLED and _tracing_enabled():
                    _attempt(
                        lambda: _remember(
                            original, instance, args, kwargs, result, operation
                        )
                    )
                return result
            call = (
                _start(original, instance, args, kwargs, operation)
                if _ENABLED
                else None
            )
            if call is None:
                token = _ACTIVE_CALL.set(True)
                try:
                    return await original(instance, *args, **kwargs)
                finally:
                    _ACTIVE_CALL.reset(token)
            token = _ACTIVE_CALL.set(True)
            state = _observe(call, call.attach, keep_context=True)
            try:
                result = await original(instance, *args, **kwargs)
            except BaseException as error:
                _observe(call, lambda error=error: call.finish(error))
                raise
            finally:
                call.detach(state)
                _ACTIVE_CALL.reset(token)
            if not _observe(call, lambda: _tap_reader(result, call)):
                _observe(call, lambda: call.output(result))
                _observe(call, call.finish)
            return result

        return async_wrapper

    @functools.wraps(original)
    def sync_wrapper(instance, *args, **kwargs):
        if builder:
            result = original(instance, *args, **kwargs)
            if _ENABLED and _tracing_enabled():
                _attempt(
                    lambda: _remember(
                        original, instance, args, kwargs, result, operation
                    )
                )
            return result
        if operation == "merge.execute":
            from lancedb.table import AsyncTable

            if type(storage(instance).get("_table")) is AsyncTable:
                # Native execute is synchronous builder dispatch returning the
                # actual async _do_merge coroutine. Observe that async boundary.
                return original(instance, *args, **kwargs)
        call = _start(original, instance, args, kwargs, operation) if _ENABLED else None
        if call is None:
            token = _ACTIVE_CALL.set(True)
            try:
                return original(instance, *args, **kwargs)
            finally:
                _ACTIVE_CALL.reset(token)
        token = _ACTIVE_CALL.set(True)
        state = _observe(call, call.attach, keep_context=True)
        try:
            result = original(instance, *args, **kwargs)
        except BaseException as error:
            _observe(call, lambda error=error: call.finish(error))
            raise
        else:
            _observe(call, lambda: call.output(result))
            _observe(call, call.finish)
            return result
        finally:
            call.detach(state)
            _ACTIVE_CALL.reset(token)

    @functools.wraps(original)
    def hybrid_scope(instance, *args, **kwargs):
        from lancedb.query import LanceHybridQueryBuilder

        if (
            type(instance) is not LanceHybridQueryBuilder
            or _HYBRID_SCOPE.get() is not None
        ):
            return sync_wrapper(instance, *args, **kwargs)
        scope = {"active": True}
        token = _HYBRID_SCOPE.set(scope)
        try:
            return sync_wrapper(instance, *args, **kwargs)
        finally:
            scope["active"] = False
            _HYBRID_SCOPE.reset(token)

    return hybrid_scope


def _targets():
    modules = {
        name: importlib.import_module(f"lancedb.{name}")
        for name in ("db", "table", "query", "merge")
    }
    groups = (
        (
            "db",
            ("LanceDBConnection", "AsyncConnection"),
            ("create_table", "open_table", "drop_table", "table_names", "list_tables"),
            "connection",
            False,
        ),
        (
            "table",
            ("LanceTable", "AsyncTable"),
            (
                "add",
                "delete",
                "update",
                "create_index",
                "create_scalar_index",
                "create_fts_index",
                "optimize",
            ),
            "table",
            False,
        ),
        (
            "query",
            tuple(
                name
                for name, cls in vars(modules["query"]).items()
                if isinstance(cls, type)
                and vars(cls).get("__module__") == "lancedb.query"
                and ("Query" in name)
            ),
            ("to_list", "to_arrow", "to_pandas", "to_batches", "explain_plan"),
            "query",
            False,
        ),
        ("merge", ("LanceMergeInsertBuilder",), ("execute",), "merge", False),
        ("table", ("AsyncTable",), ("_do_merge",), "merge", False),
        (
            "table",
            ("LanceTable", "AsyncTable"),
            ("search", "query", "merge_insert"),
            "table",
            True,
        ),
        (
            "query",
            tuple(
                name
                for name, cls in vars(modules["query"]).items()
                if isinstance(cls, type)
                and vars(cls).get("__module__") == "lancedb.query"
                and ("Query" in name)
            ),
            (
                "where",
                "limit",
                "select",
                "offset",
                "nearest_to",
                "nearest_to_text",
                "distance_type",
                "nprobes",
                "refine_factor",
                "rerank",
                "bypass_vector_index",
                "with_row_id",
                "with_row_address",
                "prefilter",
                "postfilter",
                "minimum_nprobes",
                "maximum_nprobes",
                "ef",
                "fast_search",
                "distance_range",
            ),
            "query",
            True,
        ),
        (
            "merge",
            ("LanceMergeInsertBuilder",),
            (
                "when_matched_update_all",
                "when_not_matched_insert_all",
                "when_not_matched_by_source_delete",
            ),
            "merge",
            True,
        ),
    )
    seen = set()
    for module, names, methods, label, builder in groups:
        for name in names:
            cls = vars(modules[module]).get(name)
            if not isinstance(cls, type):
                continue
            for method in methods:
                for owner in type.__dict__["__mro__"].__get__(cls):
                    original = vars(owner).get(method)
                    if original is not None:
                        key = (owner, method)
                        if key not in seen and callable(original):
                            seen.add(key)
                            operation = (
                                "merge.execute"
                                if method == "_do_merge"
                                else f"{label}.{method}"
                            )
                            yield owner, method, original, operation, builder
                        break


def _restore():
    for owner, name, original, wrapper in reversed(_PATCHES):
        if vars(owner).get(name) is wrapper:
            _attempt(
                lambda owner=owner, name=name, original=original: setattr(
                    owner, name, original
                )
            )
    _PATCHES.clear()
    _BUILDERS.clear()
    _remove_policies()


class LanceDBInstrumentor:
    """Instrument native embedded DB actions; builders retain native behavior."""

    name = "lancedb"

    def __init__(self):
        self._is_instrumented = False

    def activate(self, *, tracer_provider=None, capture_content=True):
        global _ENABLED, _CAPTURE_CONTENT, _PROVIDER, _ACTIVATION_COUNT
        with _LOCK:
            if self._is_instrumented:
                return
            if _ACTIVATION_COUNT:
                if (
                    tracer_provider is not _PROVIDER
                    or capture_content is not _CAPTURE_CONTENT
                ):
                    raise ValueError("LanceDB instrumentation configuration conflict")
                _ACTIVATION_COUNT += 1
                self._is_instrumented = True
                return
            _PROVIDER = tracer_provider
            _CAPTURE_CONTENT = capture_content is True
            try:
                _policy()
                for owner, name, original, operation, builder in _targets():
                    wrapper = _wrap(original, operation, builder=builder)
                    # Record ownership before a mutating setter can raise.
                    _PATCHES.append((owner, name, original, wrapper))
                    setattr(owner, name, wrapper)
            except BaseException:
                _restore()
                _PROVIDER = None
                raise
            _ENABLED = True
            _ACTIVATION_COUNT = 1
            self._is_instrumented = True

    def deactivate(self):
        global _ENABLED, _ACTIVATION_COUNT, _PROVIDER
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _ACTIVATION_COUNT -= 1
            if _ACTIVATION_COUNT:
                return
            _ENABLED = False
            for call in list(_PENDING):
                if call.chunks:
                    _observe(call, lambda call=call: call.output(call.chunks))
                _observe(call, lambda call=call: call.finish(completed=False))
            _restore()
            _PROVIDER = None
