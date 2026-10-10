"""Observe released Cursor SDK runs without replacing native run handles."""

from __future__ import annotations

import contextvars
import functools
import importlib
import inspect
import logging
import threading
import time
import weakref

from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_TOOL_CALL_ID,
)
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA
from respan_tracing.utils.span_factory import build_readable_span, inject_span

from ._policy import Policy, permitted, span_key, suppressed
from ._serialization import dumps, safe

_LOCK = threading.RLock()
_RUNTIME = None
_CREATING = contextvars.ContextVar("respan_cursor_creating_run", default=None)
logger = logging.getLogger(__name__)
_SCOPE = "respan.instrumentation.cursor-sdk"


def current_policy():
    return _RUNTIME.policy if _RUNTIME is not None and _RUNTIME.active else None


def attributes(name, kind):
    return {
        RESPAN_LOG_TYPE: kind,
        SpanAttributes.TRACELOOP_ENTITY_NAME: name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: "",
        SpanAttributes.TRACELOOP_WORKFLOW_NAME: "cursor-sdk",
    }


class Call:
    def __init__(self, runtime, name, kind, payload):
        self.runtime = runtime
        self.parent = span_key(trace.get_current_span())
        self.allowed = (
            runtime.capture and permitted() and runtime.policy.ancestors(self.parent)
        )
        self.span = runtime.tracer.start_span(name, attributes=attributes(name, kind))
        self.done = not self.span.is_recording()
        self.output = None
        self.events = []
        self.tools = {}
        self.seen_tools = set()
        self.completed_tools = []
        self.terminal = False
        self.start = time.time_ns()
        if not self.done:
            runtime.pending.add(self)
            try:
                self.allowed = (
                    self.allowed
                    and runtime.active
                    and permitted()
                    and runtime.policy.ancestors(span_key(self.span))
                )
                if self.allowed:
                    self.span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_INPUT, dumps(payload)
                    )
            except Exception:  # noqa: BLE001 - Clean up partially initialized telemetry.
                runtime.finish(self, empty=True)

    def checkpoint(self):
        self.allowed = (
            self.allowed
            and self.runtime.active
            and permitted()
            and self.runtime.policy.ancestors(self.parent)
            and self.runtime.policy.ancestors(span_key(self.span))
        )
        if not self.allowed:
            self.output = None
            self.events.clear()
            self.tools.clear()
            self.span._status = Status(self.span.status.status_code)
            for span in self.completed_tools:
                span._status = Status(span.status.status_code)
                span._attributes = {
                    k: v
                    for k, v in span.attributes.items()
                    if k
                    not in (
                        SpanAttributes.TRACELOOP_ENTITY_INPUT,
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        ERROR_MESSAGE,
                    )
                }
            attrs = self.span._attributes if self.span.is_recording() else None
            if attrs is not None:
                for key in (
                    SpanAttributes.TRACELOOP_ENTITY_INPUT,
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    ERROR_MESSAGE,
                ):
                    attrs.pop(key, None)
        return self.allowed and not self.done

    def observe(self, raw):
        if self.done or type(raw) is not dict:
            return
        capture = self.checkpoint()
        if capture:
            self.events.append(safe(raw))
        message = raw.get("sdkMessage")
        if (
            type(message) is dict
            and type(message.get("message")) is dict
            and "type" in message["message"]
        ):
            message = message["message"]
        step = raw.get("step")
        if type(step) is dict:
            step = step.get("step", step)
            if (
                type(step) is dict
                and step.get("type") == "toolCall"
                and type(step.get("message")) is dict
            ):
                data = step["message"]
                if data.get("status") in ("completed", "error"):
                    message = {
                        **data,
                        "type": "tool_call",
                        "callId": data.get("callId", data.get("id")),
                    }
        if type(message) is dict:
            call_id = message.get("callId", message.get("call_id"))
            if (
                message.get("type") == "tool_call"
                and type(call_id) is str
                and call_id not in self.seen_tools
            ):
                tool = self.tools.setdefault(call_id, {"start": time.time_ns()})
                tool["data"] = (
                    safe(message)
                    if capture
                    else {"name": message.get("name"), "status": message.get("status")}
                )
                if message.get("status") in ("completed", "error"):
                    tool = self.tools.pop(call_id, None)
                    self.seen_tools.add(call_id)
                    if tool:
                        data = tool["data"]
                        attrs = attributes(data.get("name") or "tool", LOG_TYPE_TOOL)
                        attrs[GEN_AI_TOOL_CALL_ID] = call_id
                        if "args" in data:
                            attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = dumps(
                                {"name": data.get("name"), "arguments": data["args"]}
                            )
                        if "result" in data and message.get("status") != "error":
                            attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = dumps(
                                data["result"]
                            )
                        attrs[RESPAN_METADATA] = dumps(
                            {
                                "cursor.tool_status": message.get("status"),
                                "cursor.truncated": data.get("truncated"),
                            }
                        )
                        sc = self.span.get_span_context()
                        span = build_readable_span(
                            name="tool." + (data.get("name") or "tool"),
                            trace_id=f"{sc.trace_id:032x}",
                            parent_id=f"{sc.span_id:016x}",
                            start_time_ns=tool["start"],
                            end_time_ns=time.time_ns(),
                            attributes=attrs,
                        )
                        if message.get("status") == "error":
                            span._status = Status(StatusCode.ERROR)
                        self.completed_tools.append(span)
        result = raw.get("result")
        if type(result) is dict:
            result = result.get("result", result)
            if type(result) is dict and result.get("status") in (
                "finished",
                "completed",
                "error",
                "cancelled",
                "canceled",
                "expired",
            ):
                self.terminal = True
                failed = result.get("status") in (
                    "error",
                    "cancelled",
                    "canceled",
                    "expired",
                )
                if failed:
                    error = result.get("error")
                    description = (
                        error.get("message")
                        if type(error) is dict
                        else result["status"]
                    )
                    self.span.set_status(
                        Status(StatusCode.ERROR, safe(description) if capture else None)
                    )
                    if capture and description:
                        self.span.set_attribute(ERROR_MESSAGE, safe(description))
                elif capture:
                    self.output = {**safe(result), "events": self.events.copy()}

    def finish(self, error=None, *, empty=False):
        if self.done:
            return
        if empty:
            self.allowed = False
        capture = self.checkpoint()
        if error is not None:
            self.output = None
            self.span.set_status(Status(StatusCode.ERROR))
            status = (
                getattr(error, "status_code", None)
                if type(error).__module__.startswith("cursor_sdk")
                else None
            )
            if type(status) is int and 100 <= status <= 599:
                self.span.set_attribute(HTTP_RESPONSE_STATUS_CODE, status)
            if capture:
                message = (
                    getattr(error, "message", None)
                    if type(error).__module__.startswith("cursor_sdk")
                    else error.args[0]
                    if error.args and type(error.args[0]) is str
                    else type(error).__name__
                )
                self.span.set_attribute(ERROR_MESSAGE, safe(message))
        elif capture and self.output is not None:
            self.span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT, dumps(self.output)
            )
        self.done = True
        self.output = None
        self.events.clear()
        self.tools.clear()
        self.completed_tools.clear()
        self.runtime.pending.discard(self)
        self.span.end()

    def flush_tools(self):
        self.checkpoint()
        for span in self.completed_tools:
            inject_span(span)
        self.completed_tools.clear()


class Wire:
    def __init__(self, native, call):
        self.native, self.call = native, call

    def __iter__(self):
        return self

    def __next__(self):
        return self._step(self.native.__next__)

    def send(self, value):
        return self._step(self.native.send, value)

    def throw(self, *args):
        return self._step(self.native.throw, *args)

    def close(self):
        try:
            return self.native.close()
        finally:
            self.call.runtime.finish(self.call, empty=True)

    def _step(self, method, *args):
        self.call.runtime.checkpoint(self.call)
        try:
            value = method(*args)
        except StopIteration:
            raise
        except BaseException as error:
            self.call.runtime.finish(self.call, error)
            raise
        self.call.runtime.observe(self.call, value)
        return value


class AsyncWire:
    def __init__(self, native, call):
        self.native, self.call = native, call

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._step(self.native.__anext__)

    async def asend(self, value):
        return await self._step(self.native.asend, value)

    async def athrow(self, *args):
        return await self._step(self.native.athrow, *args)

    async def aclose(self):
        try:
            return await self.native.aclose()
        finally:
            self.call.runtime.finish(self.call, empty=True)

    async def _step(self, method, *args):
        self.call.runtime.checkpoint(self.call)
        try:
            value = await method(*args)
        except StopAsyncIteration:
            raise
        except BaseException as error:
            self.call.runtime.finish(self.call, error)
            raise
        self.call.runtime.observe(self.call, value)
        return value


class Runtime:
    def __init__(self, provider, capture):
        self.provider, self.capture = provider, capture
        self.tracer = provider.get_tracer(_SCOPE)
        self.policy = Policy()
        provider.add_span_processor(self.policy)
        self.active = True
        self.owners = set()
        self.hooks = []
        self.pending = set()
        self.runs = weakref.WeakKeyDictionary()

    def begin(self, name, kind, payload):
        with _LOCK:
            if not self.active or suppressed():
                return None
            try:
                call = Call(self, name, kind, payload)
                if not self.active:
                    self.finish(call, empty=True)
                    return None
                return None if call.done else call
            except Exception:  # noqa: BLE001 - Telemetry faults must not change native execution.
                return None

    def observe(self, call, raw):
        try:
            call.observe(raw)
        except Exception:  # noqa: BLE001 - Telemetry faults must not change native execution.
            self.veto(call)

    @staticmethod
    def veto(call):
        call.allowed = False
        call.output = None
        call.events.clear()
        call.tools.clear()
        call.completed_tools.clear()
        call.span._status = Status(call.span.status.status_code)
        attrs = getattr(call.span, "_attributes", None)
        if attrs is not None:
            for key in (
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                ERROR_MESSAGE,
            ):
                attrs.pop(key, None)

    def checkpoint(self, call):
        try:
            return call.checkpoint()
        except Exception:  # noqa: BLE001 - Optional observation cannot change SDK results.
            self.veto(call)
            return False

    def flush_tools(self, call):
        try:
            call.flush_tools()
        except Exception:  # noqa: BLE001 - Optional export cannot change SDK results.
            self.veto(call)

    def result(self, call, result, *, unary=False):
        try:
            if unary:
                if self.checkpoint(call):
                    call.output = safe(result)
            else:
                self.observe(call, {"result": {"result": safe(result)}})
        except Exception:  # noqa: BLE001 - Native result objects must remain unchanged.
            self.veto(call)

    @staticmethod
    def finish(call, error=None, *, empty=False):
        try:
            call.finish(error, empty=empty)
        except Exception:  # noqa: BLE001 - Telemetry faults must not change native execution.
            Runtime.veto(call)
            call.done = True
            call.runtime.pending.discard(call)
            try:
                call.span.end()
            except Exception:  # noqa: BLE001 - Preserve the original native result or error.
                logger.debug("Cursor telemetry finalization failed")

    def patch(self, cls, name, factory):
        original = inspect.getattr_static(cls, name)
        wrapper = factory(original)
        setattr(cls, name, wrapper)
        self.hooks.append((cls, name, original, wrapper))

    def bind(self, run, call):
        try:
            self.runs[run] = call
            weakref.finalize(run, Runtime.finish, call, empty=True)
        except Exception:  # noqa: BLE001 - Binding telemetry cannot replace a native Run.
            self.finish(call, empty=True)

    def install(self):
        try:
            sdk = importlib.import_module("cursor_sdk")
        except ModuleNotFoundError as error:
            if error.name == "cursor_sdk":
                return
            raise
        for cls in (sdk.Agent, sdk.AsyncAgent):
            self.patch(cls, "send", self.send_wrapper)
        for cls in (sdk.Run, sdk.AsyncRun):
            self.patch(cls, "_handle_event", self.event_wrapper)
            self.patch(cls, "wait", self.wait_wrapper)
        for cls in (sdk.CursorClient, sdk.AsyncClient):
            self.patch(cls, "_agent_stream", self.stream_wrapper)
            self.patch(cls, "_agent_unary", self.unary_wrapper)

    def send_wrapper(self, original):
        def prepare(args, kwargs):
            message = kwargs.get("message", args[1] if len(args) > 1 else None)
            options = kwargs.get("options", args[2] if len(args) > 2 else None)
            return self.begin(
                "agent", LOG_TYPE_AGENT, {"message": message, "options": options}
            )

        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def wrapper(*args, **kwargs):
                call = prepare(args, kwargs)
                if call is None:
                    return await original(*args, **kwargs)
                token = _CREATING.set(call)
                try:
                    with trace.use_span(
                        call.span,
                        end_on_exit=False,
                        record_exception=False,
                        set_status_on_exception=False,
                    ):
                        run = await original(*args, **kwargs)
                        self.bind(run, call)
                        self.checkpoint(call)
                        return run
                except BaseException as error:
                    self.finish(call, error)
                    raise
                finally:
                    _CREATING.reset(token)
        else:

            @functools.wraps(original)
            def wrapper(*args, **kwargs):
                call = prepare(args, kwargs)
                if call is None:
                    return original(*args, **kwargs)
                token = _CREATING.set(call)
                try:
                    with trace.use_span(
                        call.span,
                        end_on_exit=False,
                        record_exception=False,
                        set_status_on_exception=False,
                    ):
                        run = original(*args, **kwargs)
                        self.bind(run, call)
                        self.checkpoint(call)
                        return run
                except BaseException as error:
                    self.finish(call, error)
                    raise
                finally:
                    _CREATING.reset(token)

        return wrapper

    def event_wrapper(self, original):
        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def wrapper(run, *args, **kwargs):
                call = self.runs.get(run) or _CREATING.get()
                if call is None or not self.active:
                    return await original(run, *args, **kwargs)
                self.runs[run] = call
                try:
                    with trace.use_span(
                        call.span,
                        end_on_exit=False,
                        record_exception=False,
                        set_status_on_exception=False,
                    ):
                        value = await original(run, *args, **kwargs)
                        self.checkpoint(call)
                        self.flush_tools(call)
                        if call.terminal:
                            self.finish(call)
                        return value
                except BaseException as error:
                    self.finish(call, error)
                    raise
        else:

            @functools.wraps(original)
            def wrapper(run, *args, **kwargs):
                call = self.runs.get(run) or _CREATING.get()
                if call is None or not self.active:
                    return original(run, *args, **kwargs)
                self.runs[run] = call
                try:
                    with trace.use_span(
                        call.span,
                        end_on_exit=False,
                        record_exception=False,
                        set_status_on_exception=False,
                    ):
                        value = original(run, *args, **kwargs)
                        self.checkpoint(call)
                        self.flush_tools(call)
                        if call.terminal:
                            self.finish(call)
                        return value
                except BaseException as error:
                    self.finish(call, error)
                    raise

        return wrapper

    def wait_wrapper(self, original):
        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def wrapper(run, *args, **kwargs):
                call = self.runs.get(run)
                try:
                    result = await original(run, *args, **kwargs)
                except BaseException as error:
                    if call:
                        self.finish(call, error)
                    raise
                if call and not call.done:
                    self.result(call, result)
                    self.finish(call)
                return result
        else:

            @functools.wraps(original)
            def wrapper(run, *args, **kwargs):
                call = self.runs.get(run)
                try:
                    result = original(run, *args, **kwargs)
                except BaseException as error:
                    if call:
                        self.finish(call, error)
                    raise
                if call and not call.done:
                    self.result(call, result)
                    self.finish(call)
                return result

        return wrapper

    def stream_wrapper(self, original):
        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def wrapper(client, method, *args, **kwargs):
                native = await original(client, method, *args, **kwargs)
                call = _CREATING.get()
                return (
                    AsyncWire(native, call)
                    if self.active and method == "Send" and call is not None
                    else native
                )
        else:

            @functools.wraps(original)
            def wrapper(client, method, *args, **kwargs):
                native = original(client, method, *args, **kwargs)
                call = _CREATING.get()
                return (
                    Wire(native, call)
                    if self.active and method == "Send" and call is not None
                    else native
                )

        return wrapper

    def unary_wrapper(self, original):
        def begin(method, args, kwargs):
            payload = kwargs.get("message", args[0] if args else {})
            return (
                self.begin(
                    "usage",
                    LOG_TYPE_TASK,
                    {k: payload[k] for k in ("agentId", "runId") if k in payload},
                )
                if method == "GetUsage"
                else None
            )

        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def wrapper(client, method, *args, **kwargs):
                call = begin(method, args, kwargs)
                try:
                    result = await original(client, method, *args, **kwargs)
                    if call:
                        self.result(call, result, unary=True)
                        self.finish(call)
                    return result
                except BaseException as error:
                    if call:
                        self.finish(call, error)
                    raise
        else:

            @functools.wraps(original)
            def wrapper(client, method, *args, **kwargs):
                call = begin(method, args, kwargs)
                try:
                    result = original(client, method, *args, **kwargs)
                    if call:
                        self.result(call, result, unary=True)
                        self.finish(call)
                    return result
                except BaseException as error:
                    if call:
                        self.finish(call, error)
                    raise

        return wrapper

    def close(self):
        self.active = False
        for call in list(self.pending):
            self.finish(call, empty=True)
        for cls, name, original, wrapper in reversed(self.hooks):
            if inspect.getattr_static(cls, name) is wrapper:
                setattr(cls, name, original)
        self.hooks.clear()
        self.runs.clear()
        self.policy.close()


def activate(owner, capture):
    global _RUNTIME
    provider = trace.get_tracer_provider()
    with _LOCK:
        if _RUNTIME is not None:
            if not _RUNTIME.active:
                raise RuntimeError("Prior Cursor telemetry is still closing")
            if _RUNTIME.provider is not provider or _RUNTIME.capture != capture:
                raise ValueError(
                    "Active Cursor owners require the same tracer provider and content setting"
                )
            _RUNTIME.owners.add(owner)
            return
        if not hasattr(provider, "add_span_processor"):
            raise RuntimeError(
                "Cursor instrumentation requires an OTel SDK tracer provider"
            )
        runtime = Runtime(provider, capture)
        try:
            runtime.install()
        except BaseException:
            runtime.close()
            raise
        runtime.owners.add(owner)
        _RUNTIME = runtime


def deactivate(owner):
    global _RUNTIME
    with _LOCK:
        if _RUNTIME is None or owner not in _RUNTIME.owners:
            return
        _RUNTIME.owners.remove(owner)
        if not _RUNTIME.owners:
            _RUNTIME.close()
            _RUNTIME = None
