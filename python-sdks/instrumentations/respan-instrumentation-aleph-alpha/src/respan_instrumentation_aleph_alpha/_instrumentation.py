"""Observe native Aleph Alpha calls without replacing its transport or parsers."""

from __future__ import annotations

import inspect
import logging
import threading
import weakref
from contextvars import ContextVar
from functools import wraps

from opentelemetry import context, trace
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv._incubating.attributes.http_attributes import (
    HTTP_RESPONSE_STATUS_CODE,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import SpanAttributes as AI
from opentelemetry.trace import SpanKind, Status, StatusCode
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA
from respan_tracing.core.tracer import RespanTracer

from ._policy import CapturePolicy, PrivacyObserver, explicit_capture, suppressed
from ._serialization import (
    REDACTED,
    initialize_native_types,
    json_string,
    json_value,
    native_dict,
    native_storage,
    redact_text,
)

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_CURRENT = ContextVar("respan_aleph_alpha_call", default=None)
_OWNERS = set()
_MANAGER = None
_METHODS = (
    "chat",
    "complete",
    "embed",
    "embeddings",
    "semantic_embed",
    "batch_semantic_embed",
    "instructable_embed",
    "evaluate",
    "explain",
)
_STREAMS = ("chat_with_streaming", "complete_with_streaming")


def _register(provider, processor):
    active = getattr(provider, "_active_span_processor", None)
    if active is None:
        provider.add_span_processor(processor)
    else:
        with active._lock:
            active._span_processors = (
                processor,
                *(p for p in active._span_processors if p is not processor),
            )


def _remove(provider, processor):
    active = getattr(provider, "_active_span_processor", None)
    if active is not None:
        with active._lock:
            active._span_processors = tuple(
                p for p in active._span_processors if p is not processor
            )


class _Call:
    def __init__(self, manager, operation, args, kwargs):
        self.manager = manager
        self.operation = operation
        self.done = False
        self.finishing = False
        self.keys = set()
        self.request = None
        self.wire = []
        self.output = None
        self.items = []
        self.text = []
        self.model = None
        self.marker = None
        self.span = None
        self.was_recording = False
        self.policy = CapturePolicy(manager.observer, manager.capture, manager.context)
        provider = manager.observe()
        kind = (
            "chat"
            if operation.startswith("chat")
            else "text"
            if operation.startswith("complete")
            else "embedding"
            if "embed" in operation
            else "task"
        )
        self.kind = kind
        self.span = provider.get_tracer(
            "respan.instrumentation.aleph-alpha"
        ).start_span(
            "alephalpha." + operation,
            context=manager.context
            if manager.context is not None
            else self.policy.context,
            kind=SpanKind.CLIENT,
            attributes={
                RESPAN_LOG_TYPE: kind,
                AI.TRACELOOP_ENTITY_NAME: "alephalpha." + operation,
                AI.TRACELOOP_ENTITY_PATH: "",
                **(
                    {
                        AI.LLM_SYSTEM: "alephalpha",
                        AI.LLM_REQUEST_TYPE: "chat"
                        if kind in ("chat", "text")
                        else "embedding",
                    }
                    if kind != "task"
                    else {}
                ),
            },
        )
        self.was_recording = self.span.is_recording()
        self.marker = (getattr(self.span, "attributes", None) or {}).get(
            RESPAN_METADATA
        )
        manager.observer.states.add(self)
        if self.check():
            request = args[0] if args else kwargs.get("request")
            self.request = json_value(request)
        model = args[1] if len(args) > 1 else kwargs.get("model")
        if type(model) is str:
            self.model = redact_text(model)
            self.span.set_attribute(AI.LLM_REQUEST_MODEL, self.model)

    def safe(self, method, *args):
        if self.done:
            return None
        try:
            return getattr(self, method)(*args)
        except Exception:  # noqa: BLE001
            self.abort()
            return None

    def drop(self):
        self.request = self.output = None
        self.wire.clear()
        self.items.clear()
        self.text.clear()

    def scrub(self):
        self.drop()
        attrs = getattr(self.span, "_attributes", None)
        if attrs is not None:
            for key in self.keys:
                attrs.pop(key, None)
            attrs.pop(ERROR_MESSAGE, None)
        if self.span is not None:
            self.span._events = BoundedList(
                maxlen=getattr(getattr(self.span._events, "_dq", None), "maxlen", None)
            )
        if self.span is not None and self.span.status.status_code == StatusCode.ERROR:
            self.span._status = Status(StatusCode.ERROR)

    def clean(self):
        try:
            self.scrub()
        except Exception:  # noqa: BLE001
            self.drop()
            # A failing telemetry setter/storage is discarded, never allowed to
            # export a body whose ownership can no longer be verified.
            if self.span is not None:
                self.span._attributes = {
                    RESPAN_LOG_TYPE: self.kind,
                    AI.TRACELOOP_ENTITY_NAME: "alephalpha." + self.operation,
                }
                self.span._events = BoundedList(maxlen=None)
                if self.span.status.status_code == StatusCode.ERROR:
                    self.span._status = Status(StatusCode.ERROR)

    def release(self):
        self.drop()
        self.keys.clear()
        self.policy.allowed = False
        self.policy.context = self.policy.supplied = None
        self.policy.parent = trace.INVALID_SPAN
        self.policy.supplied_parent = None

    def abort(self):
        if self.done:
            return
        self.policy.allowed = False
        try:
            self.clean()
        except Exception:  # noqa: BLE001
            self.drop()
        self.finishing = True
        try:
            if self.span is not None:
                self.span.end()
        except Exception:  # noqa: BLE001
            logger.debug("Aleph Alpha telemetry discard failed")
        finally:
            self.done = True
            self.finishing = False
            self.manager.observer.states.discard(self)
            self.release()

    def check(self):
        allowed = self.policy.check() and (
            self.span.is_recording() or (self.finishing and self.was_recording)
        )
        if not allowed:
            self.scrub()
        return allowed

    def capture(self, key, value):
        if self.check():
            self.keys.add(key)
            try:
                self.span.set_attribute(key, value)
            finally:
                self.keys.add(key)
            self.check()

    def body(self, body):
        if self.check() and type(body) is dict:
            self.wire.append(json_value(body))
            if self.model is None and type(body.get("model")) is str:
                self.model = redact_text(body["model"])
                self.span.set_attribute(AI.LLM_REQUEST_MODEL, self.model)

    def result(self, result):
        if self.check():
            self.output = json_value(result)

    def consume(self, item):
        if self.check():
            self.items.append(json_value(item))
            data = native_dict(item)
            if data is not None and type(data.get("content")) is str:
                self.text.append((0, data["content"]))
            if data is not None and type(data.get("completion")) is str:
                self.text.append((data.get("index", 0), data["completion"]))

    def failure(self, error):
        self.span.set_status(Status(StatusCode.ERROR))
        self.span.set_attribute(
            ERROR_TYPE, type.__getattribute__(type(error), "__name__")
        )
        if self.check():
            for argument in BaseException.args.__get__(error):
                if type(argument) is str:
                    self.capture(ERROR_MESSAGE, redact_text(argument))
                    break

    def usage(self, data):
        if type(data) is not dict:
            return
        source = data.get("usage") if type(data.get("usage")) is dict else data
        for names, keys in (
            (
                ("num_tokens_prompt_total", "prompt_tokens"),
                (AI.LLM_USAGE_PROMPT_TOKENS, GEN_AI_USAGE_INPUT_TOKENS),
            ),
            (
                ("num_tokens_generated", "completion_tokens"),
                (AI.LLM_USAGE_COMPLETION_TOKENS, GEN_AI_USAGE_OUTPUT_TOKENS),
            ),
            (("total_tokens",), (AI.LLM_USAGE_TOTAL_TOKENS,)),
        ):
            for name in names:
                if type(source.get(name)) is int:
                    for key in keys:
                        self.span.set_attribute(key, source[name])
                    break

    def map(self, exhausted):
        data = self.output
        request = self.request or {}
        body = self.wire[0] if self.wire else request
        if self.operation == "batch_semantic_embed":
            body = request
        if self.kind != "task":
            self.span.set_attribute(AI.LLM_SYSTEM, "alephalpha")
            self.span.set_attribute(
                AI.LLM_REQUEST_TYPE,
                "chat" if self.kind in ("chat", "text") else "embedding",
            )
        for source, key in (
            ("maximum_tokens", AI.LLM_REQUEST_MAX_TOKENS),
            ("temperature", AI.LLM_REQUEST_TEMPERATURE),
            ("top_p", AI.LLM_REQUEST_TOP_P),
        ):
            if type(request.get(source)) in (int, float):
                self.span.set_attribute(key, request[source])
        if not self.check():
            return
        input_value = body.get(
            "messages", body.get("prompt", body.get("prompts", body.get("input", body)))
        )
        if self.kind == "chat":
            for index, message in enumerate(body.get("messages") or []):
                if type(message) is dict:
                    for key in ("role", "content"):
                        if key in message and message[key] is not None:
                            self.capture(
                                f"{AI.LLM_PROMPTS}.{index}.{key}",
                                message[key]
                                if type(message[key]) is str
                                else json_string(message[key]),
                            )
                    if message.get("tool_calls") is not None:
                        self.capture(
                            f"{AI.LLM_PROMPTS}.{index}.tool_calls",
                            json_string(message["tool_calls"]),
                        )
        if self.kind == "text" and input_value is not None:
            self.capture(f"{AI.LLM_PROMPTS}.0.content", json_string(input_value))
        if self.operation in _STREAMS:
            data = self.items if self.items or exhausted else None
            groups = {}
            for index, text in self.text:
                if type(index) is int:
                    groups.setdefault(index, []).append(text)
            for index, parts in groups.items():
                joined = "".join(parts)
                redacted = redact_text(joined)
                if redacted != joined:
                    for item in self.items:
                        if type(item) is dict and item.get("index", 0) == index:
                            for key in (
                                "content",
                                "completion",
                                "raw_completion",
                                "completion_tokens",
                            ):
                                if key in item and item[key] is not None:
                                    item[key] = REDACTED
                self.capture(f"{AI.LLM_COMPLETIONS}.{index}.content", redacted)
            for item in self.items:
                self.usage(item)
                if type(item) is dict and type(item.get("role")) is str:
                    self.capture(f"{AI.LLM_COMPLETIONS}.0.role", item["role"])
            tool_calls = [
                item
                for item in self.items
                if type(item) is dict and "function" in item and "id" in item
            ]
            if tool_calls:
                self.capture(
                    f"{AI.LLM_COMPLETIONS}.0.tool_calls", json_string(tool_calls)
                )
        else:
            self.usage(data)
            if type(data) is dict:
                message = data.get("message")
                if type(message) is dict:
                    for key in ("role", "content"):
                        if type(message.get(key)) is str:
                            self.capture(f"{AI.LLM_COMPLETIONS}.0.{key}", message[key])
                    if message.get("tool_calls") is not None:
                        self.capture(
                            f"{AI.LLM_COMPLETIONS}.0.tool_calls",
                            json_string(message["tool_calls"]),
                        )
                for index, completion in enumerate(data.get("completions") or []):
                    if (
                        type(completion) is dict
                        and type(completion.get("completion")) is str
                    ):
                        self.capture(
                            f"{AI.LLM_COMPLETIONS}.{index}.content",
                            completion["completion"],
                        )
        if body.get("tools") is not None:
            self.capture(AI.LLM_REQUEST_FUNCTIONS, json_string(body["tools"]))
        if self.kind == "embedding" and type(data) is dict:
            envelope = dict(data)
            if "embedding" in envelope:
                output = envelope.pop("embedding")
            elif "embeddings" in envelope:
                output = envelope.pop("embeddings")
            elif type(envelope.get("data")) is list:
                entries = envelope.pop("data")
                output = [
                    entry.get("embedding") for entry in entries if type(entry) is dict
                ]
                envelope["data"] = [
                    {key: value for key, value in entry.items() if key != "embedding"}
                    for entry in entries
                    if type(entry) is dict
                ]
            else:
                output = data
            self.capture(
                RESPAN_METADATA + ".aleph_alpha.response", json_string(envelope)
            )
        else:
            output = data
        # Indexed histories may exceed the native 128-attribute limit. Complete
        # canonical content and request configuration are deliberately written last.
        self.capture(
            RESPAN_METADATA + ".aleph_alpha.request",
            json_string({"native": request, "wire": self.wire}),
        )
        self.capture(AI.TRACELOOP_ENTITY_INPUT, json_string(input_value))
        if output is not None:
            self.capture(AI.TRACELOOP_ENTITY_OUTPUT, json_string(output))

    def finish(self, exhausted=False):
        if self.done or self.finishing:
            return
        self.finishing = True
        try:
            if exhausted and self.span.status.status_code != StatusCode.ERROR:
                self.span.set_status(Status(StatusCode.OK))
            if self.check():
                self.map(exhausted)
            self.span.set_attribute(RESPAN_LOG_TYPE, self.kind)
            if self.kind != "task":
                self.span.set_attribute(AI.LLM_SYSTEM, "alephalpha")
                self.span.set_attribute(
                    AI.LLM_REQUEST_TYPE,
                    "chat" if self.kind in ("chat", "text") else "embedding",
                )
            self.span.set_attribute(
                AI.TRACELOOP_ENTITY_NAME, "alephalpha." + self.operation
            )
            if type(self.marker) is str:
                self.span.set_attribute(RESPAN_METADATA, self.marker)
            if self.model is not None:
                self.span.set_attribute(AI.LLM_REQUEST_MODEL, self.model)
            self.check()
        except Exception:  # noqa: BLE001
            self.policy.allowed = False
            try:
                self.clean()
            except Exception:  # noqa: BLE001
                self.drop()
        finally:
            try:
                self.span.end()
            except Exception:  # noqa: BLE001
                self.policy.allowed = False
                try:
                    self.clean()
                except Exception:  # noqa: BLE001
                    self.drop()
                logger.debug("Aleph Alpha telemetry end failed")
            finally:
                self.done = True
                self.finishing = False
                self.manager.observer.states.discard(self)
                self.release()


class _Scope:
    def __init__(self, state):
        self.state = state
        self.token = None
        self.current = None

    def __enter__(self):
        self.current = _CURRENT.set(self.state)
        ambient = context.get_current()
        try:
            if self.state is not None and not self.state.done:
                self.state.safe("check")
                self.token = context.attach(trace.set_span_in_context(self.state.span))
        except Exception:  # noqa: BLE001
            if self.state is not None:
                self.state.abort()
                try:
                    self.state.manager.original_attach(ambient)
                except Exception:  # noqa: BLE001
                    logger.debug("Aleph Alpha telemetry attach restoration failed")
        return self

    def __exit__(self, kind, error, tb):
        if self.state is not None:
            self.state.safe("check")
        try:
            if self.token is not None:
                context.detach(self.token)
        except Exception:  # noqa: BLE001
            try:
                self.state.manager.original_detach(self.token)
            except Exception:  # noqa: BLE001
                logger.debug("Aleph Alpha telemetry detach failed")
        finally:
            _CURRENT.reset(self.current)


class _Stream:
    """Forward the declared native AsyncGenerator protocol; items are unchanged."""

    def __init__(self, native, state):
        self.native = native
        self.state = state
        self.finalizer = (
            weakref.finalize(self, state.safe, "finish") if state is not None else None
        )

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._pull(self.native.__anext__)

    async def asend(self, value):
        return await self._pull(self.native.asend, value)

    async def athrow(self, *args):
        return await self._pull(self.native.athrow, *args)

    async def _pull(self, callback, *args):
        with _Scope(self.state):
            try:
                item = await callback(*args)
            except StopAsyncIteration:
                if self.state is not None:
                    self.state.safe("finish", True)
                raise
            except BaseException as error:
                if self.state is not None:
                    if isinstance(error, Exception):
                        self.state.safe("failure", error)
                    self.state.safe("finish")
                raise
            if self.state is not None:
                self.state.safe("consume", item)
            return item

    async def aclose(self):
        with _Scope(self.state):
            try:
                return await self.native.aclose()
            finally:
                if self.state is not None:
                    self.state.safe("finish")


class _Manager:
    def __init__(self, capture, provider, supplied):
        self.capture = capture
        self.provider = provider
        self.context = supplied
        self.observer = PrivacyObserver()
        self.providers = []
        self.patches = []
        self.enabled = True

    def observe(self):
        provider = (
            self.provider if self.provider is not None else trace.get_tracer_provider()
        )
        if not any(provider is item for item in self.providers) and hasattr(
            provider, "add_span_processor"
        ):
            _register(provider, self.observer)
            self.providers.append(provider)
        return provider

    def patch(self, obj, name, replacement):
        data = (
            type.__getattribute__(obj, "__dict__")
            if isinstance(obj, type)
            else native_storage(obj, type(obj))
        )
        present = data is not None and name in data
        before = data.get(name) if present else None
        self.patches.append((obj, name, replacement, present, before))
        setattr(obj, name, replacement)

    def state(self, operation, args, kwargs):
        if (
            not self.enabled
            or suppressed()
            or (self.context is not None and suppressed(self.context))
        ):
            return None
        state = _Call.__new__(_Call)
        try:
            state.__init__(self, operation, args, kwargs)
            return state
        except Exception:  # noqa: BLE001
            if hasattr(state, "policy"):
                state.abort()
            return None

    def close(self):
        self.enabled = False
        for state in list(self.observer.states):
            state.policy.allowed = False
            state.safe("finish")
        for obj, name, replacement, present, before in reversed(self.patches):
            if inspect.getattr_static(obj, name, None) is replacement:
                if present:
                    setattr(obj, name, before)
                else:
                    delattr(obj, name)
        for provider in self.providers:
            _remove(provider, self.observer)
        self.providers.clear()


def _wrapper(manager, original, operation, streaming=False):
    if streaming:

        @wraps(original)
        def stream(self, *args, **kwargs):
            state = manager.state(operation, args, kwargs)
            return (
                _Stream(original(self, *args, **kwargs), state)
                if state is not None
                else original(self, *args, **kwargs)
            )

        return stream
    if inspect.iscoroutinefunction(original):

        @wraps(original)
        async def asynchronous(self, *args, **kwargs):
            state = manager.state(operation, args, kwargs)
            with _Scope(state):
                try:
                    result = await original(self, *args, **kwargs)
                except BaseException as error:
                    if state is not None:
                        if isinstance(error, Exception):
                            state.safe("failure", error)
                        state.safe("finish")
                    raise
                if state is not None:
                    state.safe("result", result)
                    state.safe("finish", True)
                return result

        return asynchronous

    @wraps(original)
    def synchronous(self, *args, **kwargs):
        state = manager.state(operation, args, kwargs)
        with _Scope(state):
            try:
                result = original(self, *args, **kwargs)
            except BaseException as error:
                if state is not None:
                    if isinstance(error, Exception):
                        state.safe("failure", error)
                    state.safe("finish")
                raise
            if state is not None:
                state.safe("result", result)
                state.safe("finish", True)
            return result

    return synchronous


class AlephAlphaInstrumentor:
    name = "aleph-alpha"

    def __init__(self, *, capture_content=True, tracer_provider=None, context=None):
        self.capture_content = capture_content is True
        self.provider = tracer_provider
        self.context = context
        self._is_instrumented = False

    def activate(self):
        global _MANAGER
        with _LOCK:
            if self._is_instrumented:
                return
            tracer = getattr(RespanTracer, "_instance", None)
            if tracer is not None and not getattr(tracer, "is_enabled", True):
                return
            if _OWNERS:
                if (
                    self.capture_content is not _MANAGER.capture
                    or self.provider is not _MANAGER.provider
                    or self.context is not _MANAGER.context
                ):
                    raise RuntimeError(
                        "Aleph Alpha instrumentation configuration conflict"
                    )
                _OWNERS.add(self)
                self._is_instrumented = True
                return
            import aleph_alpha_client.aleph_alpha_client as native
            from aleph_alpha_client import AsyncClient, Client

            initialize_native_types()
            manager = _Manager(self.capture_content, self.provider, self.context)
            try:
                manager.observe()
                for cls in (Client, AsyncClient):
                    for method in _METHODS:
                        if hasattr(cls, method):
                            original = getattr(cls, method)
                            manager.patch(
                                cls, method, _wrapper(manager, original, method)
                            )
                    if cls is AsyncClient:
                        for method in _STREAMS:
                            manager.patch(
                                cls,
                                method,
                                _wrapper(manager, getattr(cls, method), method, True),
                            )
                    original_body = cls._build_json_body

                    def make_body(original):
                        @wraps(original)
                        def body(self, *args, **kwargs):
                            result = original(self, *args, **kwargs)
                            state = _CURRENT.get()
                            if state is not None:
                                state.safe("body", result)
                            return result

                        return body

                    manager.patch(cls, "_build_json_body", make_body(original_body))
                original_error = native._raise_for_status

                @wraps(original_error)
                def error(status, text):
                    state = _CURRENT.get()
                    if state is not None and type(status) is int:
                        try:
                            state.span.set_attribute(HTTP_RESPONSE_STATUS_CODE, status)
                        except Exception:  # noqa: BLE001
                            state.abort()
                    return original_error(status, text)

                manager.patch(native, "_raise_for_status", error)
                runtime = context._RUNTIME_CONTEXT
                original_detach = runtime.detach
                manager.original_detach = original_detach
                manager.original_attach = runtime.attach

                def detach(token):
                    # Processor export suppression is transient, and must never
                    # veto a pending sibling merely because its parent exports.
                    if not explicit_capture():
                        manager.observer.notice()
                    return original_detach(token)

                manager.patch(runtime, "detach", detach)
            except Exception:
                manager.close()
                raise
            _MANAGER = manager
            _OWNERS.add(self)
            self._is_instrumented = True

    def deactivate(self):
        global _MANAGER
        with _LOCK:
            if not self._is_instrumented:
                return
            _OWNERS.discard(self)
            self._is_instrumented = False
            if not _OWNERS and _MANAGER is not None:
                _MANAGER.close()
                _MANAGER = None
