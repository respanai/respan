"""Native LiteLLM callbacks with owned request and iterator scope observation."""

from __future__ import annotations

import inspect
import logging
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from opentelemetry import context, trace
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_MESSAGE,
    ERROR_TYPE,
)
from opentelemetry.semconv._incubating.attributes.exception_attributes import (
    EXCEPTION_MESSAGE,
    EXCEPTION_TYPE,
)
from opentelemetry.semconv._incubating.attributes.http_attributes import (
    HTTP_RESPONSE_STATUS_CODE,
)
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.core.tracer import RespanTracer

from ._callback import RespanLiteLLMCallback
from ._serialization import redact_text
from ._translator import build_litellm_span_data, get, raw_usage

logger = logging.getLogger(__name__)
_CALL = ContextVar("respan_litellm_call", default=None)
_BOUND = ContextVar("respan_litellm_content_bound", default=True)
_LOCK = threading.RLock()
_RUNTIME = None
_LISTS = (
    "callbacks",
    "input_callback",
    "success_callback",
    "failure_callback",
    "_async_input_callback",
    "_async_success_callback",
    "_async_failure_callback",
)


def allowed(parent_context=None):
    import litellm

    return (
        _BOUND.get()
        and get(litellm, "turn_off_message_logging") is not True
        and os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower()
        not in {"0", "false", "no", "off"}
        and context.get_value(ENABLE_CONTENT_TRACING_KEY, parent_context) is not False
    )


def source_private(kwargs):
    dynamic = get(
        get(kwargs, "standard_callback_dynamic_params"), "turn_off_message_logging"
    )
    if (
        dynamic is True
        or isinstance(dynamic, str)
        and dynamic.strip().lower() in {"1", "true", "on", "yes"}
    ):
        return True
    params = get(kwargs, "litellm_params", {})
    metadata = get(kwargs, "metadata", get(params, "metadata", {}))
    headers = get(metadata, "headers")
    return bool(
        get(headers, "litellm-enable-message-redaction")
        or get(headers, "x-litellm-enable-message-redaction")
    )


def suppressed():
    return bool(
        context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY)
        or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    )


def enabled():
    instance = get(RespanTracer, "_instance")
    return instance is None or get(instance, "is_enabled", True) is not False


def safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        logger.debug("LiteLLM telemetry observation failed", exc_info=True)


def source_http_status(error):
    """Use retained provider HTTP responses, excluding LiteLLM synthetic defaults."""
    observed = None
    seen = set()
    for _ in range(8):
        if not isinstance(error, BaseException) or id(error) in seen:
            break
        seen.add(id(error))
        if type(error).__module__ in ("openai", "httpx") or type(
            error
        ).__module__.startswith(("openai.", "httpx.")):
            status = get(get(error, "response"), "status_code")
            if type(status) is int and 100 <= status <= 599:
                observed = status
        error = BaseException.__cause__.__get__(
            error
        ) or BaseException.__context__.__get__(error)
    return observed


@contextmanager
def suppress_nested():
    try:
        from respan_instrumentation_openai._instrumentation import (
            suppress_openai_instrumentation,
        )
    except ImportError:
        yield
        return
    with suppress_openai_instrumentation():
        yield


class _Policy(SpanProcessor):
    def __init__(self):
        self.active = True
        self.states = {}
        self.denied = OrderedDict()
        self.lock = threading.RLock()

    def on_start(self, span, parent_context=None):
        if not self.active:
            return
        with self.lock:
            parent = (
                (span.parent.trace_id, span.parent.span_id) if span.parent else None
            )
            capture = (
                safe(allowed) is True
                and safe(allowed, parent_context) is True
                and self.parent_allowed(parent)
            )
            self.states[(span.context.trace_id, span.context.span_id)] = (
                capture,
                parent,
            )

    def parent_allowed(self, key):
        seen = set()
        while key and key not in seen:
            seen.add(key)
            if key in self.denied:
                return False
            state = self.states.get(key)
            if state is None:
                return True
            if not state[0]:
                return False
            key = state[1]
        return True

    def on_end(self, span):
        if not self.active:
            return
        with self.lock:
            capture, parent = self.states.pop(
                (span.context.trace_id, span.context.span_id), (False, None)
            )
            if (
                not capture
                or safe(allowed) is not True
                or not self.parent_allowed(parent)
            ):
                self.denied[(span.context.trace_id, span.context.span_id)] = None
                while len(self.denied) > 4096:
                    self.denied.popitem(last=False)

    def deny(self, span_context):
        with self.lock:
            self.denied[(span_context.trace_id, span_context.span_id)] = None
            while len(self.denied) > 4096:
                self.denied.popitem(last=False)

    def observe_veto(self, span):
        current = span.get_span_context()
        key = (current.trace_id, current.span_id)
        with self.lock:
            state = self.states.get(key)
            if state is not None and (
                safe(allowed) is not True or not self.parent_allowed(state[1])
            ):
                self.states[key] = (False, state[1])
                self.deny(current)

    def shutdown(self):
        self.active = False
        self.states.clear()
        self.denied.clear()

    def force_flush(self, timeout_millis=30000):
        return True


class _Call:
    def __init__(self, runtime, kwargs, owned=False):
        self.runtime = runtime
        self.owned = owned
        self.stream_requested = get(kwargs, "stream") is True
        self.finished = False
        self.ids = set()
        self.kwargs = {}
        self.usage = {}
        self.usage_seen = False
        self.stream_usage_seen = False
        self.chunks = {}
        self.response_items = {}
        self.parent = _CALL.get()
        self.capture = False
        self.span = runtime.tracer.start_span("litellm.request")
        self.parent_id = (
            (self.span.parent.trace_id, self.span.parent.span_id)
            if get(self.span, "parent")
            else None
        )
        self.capture = (
            self.span.is_recording()
            and runtime.include_content
            and safe(allowed) is True
        )
        self.observe(kwargs)

    def veto(self):
        if not self.capture:
            return
        if (
            safe(allowed) is not True
            or not self.runtime.policy.parent_allowed(
                (
                    self.span.get_span_context().trace_id,
                    self.span.get_span_context().span_id,
                )
            )
            or (self.parent is not None and not self.parent.capture)
        ):
            self.capture = False
            self.chunks.clear()
            self.response_items.clear()
            provider = get(get(self.kwargs, "litellm_params"), "custom_llm_provider")
            self.kwargs = {
                k: v
                for k, v in self.kwargs.items()
                if k
                in (
                    "model",
                    "call_type",
                    "stream",
                    "custom_llm_provider",
                    "temperature",
                    "top_p",
                    "max_tokens",
                )
            }
            if isinstance(provider, str):
                self.kwargs["custom_llm_provider"] = provider

    def observe(self, kwargs):
        if safe(source_private, kwargs) is True:
            self.capture = False
            self.chunks.clear()
            self.response_items.clear()
            self.kwargs.clear()
        self.veto()
        keys = (
            "model",
            "call_type",
            "stream",
            "custom_llm_provider",
            "temperature",
            "top_p",
            "max_tokens",
            "litellm_params",
            "optional_params",
            "metadata",
        )
        for key in keys:
            value = get(kwargs, key)
            if value is None:
                continue
            if key == "stream" and self.owned and self.stream_requested:
                value = True
            if (
                key in ("litellm_params", "optional_params", "metadata")
                and not self.capture
            ):
                if key == "litellm_params":
                    provider = get(value, "custom_llm_provider")
                    if isinstance(provider, str):
                        self.kwargs[key] = {"custom_llm_provider": provider}
                continue
            self.kwargs[key] = value
        if self.capture:
            for key in ("messages", "input", "tools", "functions"):
                if (value := get(kwargs, key)) is not None:
                    self.kwargs[key] = value

    @contextmanager
    def scope(self):
        token = _CALL.set(self)
        bound = _BOUND.set(self.capture)
        otel = context.attach(trace.set_span_in_context(self.span))
        try:
            with suppress_nested():
                yield
        finally:
            safe(self.veto)
            context.detach(otel)
            _BOUND.reset(bound)
            _CALL.reset(token)

    def chunk(self, chunk):
        self.veto()
        if get(chunk, "type") == "response.completed":
            response = get(chunk, "response")
            if not self.stream_usage_seen and not self.usage_seen:
                self.usage = raw_usage(response)
            if self.capture:
                self.response_items = {
                    index: item
                    for index, item in enumerate(get(response, "output", ()) or ())
                }
            return
        if not self.capture:
            return
        event_type = get(chunk, "type")
        output_index = get(chunk, "output_index", 0)
        if event_type == "response.output_item.added" and type(output_index) is int:
            from ._translator import plain

            self.response_items[output_index] = plain(get(chunk, "item"))
        elif (
            event_type == "response.function_call_arguments.delta"
            and type(output_index) is int
        ):
            item = self.response_items.get(output_index)
            if isinstance(item, dict) and isinstance(delta := get(chunk, "delta"), str):
                item["arguments"] = item.get("arguments", "") + delta
        elif event_type == "response.output_text.delta" and type(output_index) is int:
            if isinstance(delta := get(chunk, "delta"), str):
                current = self.chunks.setdefault(output_index, {"role": "assistant"})
                current["content"] = current.get("content", "") + delta
        for choice in get(chunk, "choices", ()) or ():
            index = get(choice, "index", 0)
            if type(index) is not int:
                continue
            delta = get(choice, "delta", get(choice, "message"))
            if delta is None:
                continue
            current = self.chunks.setdefault(index, {"role": "assistant"})
            if isinstance(role := get(delta, "role"), str):
                current["role"] = role
            if isinstance(content := get(delta, "content"), str):
                current["content"] = current.get("content", "") + content
            for call in get(delta, "tool_calls", ()) or ():
                call_index = get(call, "index", 0)
                if type(call_index) is not int:
                    continue
                target = current.setdefault("tool_calls", {}).setdefault(
                    call_index,
                    {"type": "function", "function": {"name": "", "arguments": ""}},
                )
                if isinstance(identifier := get(call, "id"), str):
                    target["id"] = identifier
                function = get(call, "function")
                for key in ("name", "arguments"):
                    if isinstance(value := get(function, key), str):
                        target["function"][key] += value

    def response(self):
        if self.response_items:
            return {"output": [item for _, item in sorted(self.response_items.items())]}
        choices = []
        for index, current in sorted(self.chunks.items()):
            result = dict(current)
            if isinstance(result.get("tool_calls"), dict):
                result["tool_calls"] = [
                    v for _, v in sorted(result["tool_calls"].items())
                ]
            choices.append({"index": index, "message": result})
        return {"choices": choices}

    def finish(self, response=None, error=None, failed=False):
        if self.finished:
            return
        self.finished = True
        try:
            self.veto()
            if self.span.is_recording():
                if error is not None or failed:
                    self.span.set_status(trace.StatusCode.ERROR)
                    if error is not None:
                        self.span.set_attribute(ERROR_TYPE, type(error).__name__)
                name, attrs = build_litellm_span_data(
                    kwargs=self.kwargs,
                    response_obj=response,
                    error=error,
                    include_content=self.capture,
                    usage=self.usage,
                )
                self.span.update_name(name)
                self.span.set_attributes(attrs)
                if error is not None or failed:
                    self.span.set_status(trace.StatusCode.ERROR)
                    if error is not None:
                        self.span.set_attribute(ERROR_TYPE, type(error).__name__)
                        event = {EXCEPTION_TYPE: type(error).__name__}
                        status = source_http_status(error)
                        if type(status) is int and 100 <= status <= 599:
                            self.span.set_attribute(HTTP_RESPONSE_STATUS_CODE, status)
                        args = BaseException.args.__get__(error) if self.capture else ()
                        if len(args) == 1 and isinstance(args[0], str):
                            text = redact_text(args[0])
                            self.span.set_attribute(ERROR_MESSAGE, text)
                            event[EXCEPTION_MESSAGE] = text
                        self.span.add_event("exception", event)
        finally:
            self.kwargs.clear()
            self.chunks.clear()
            self.response_items.clear()
            self.usage.clear()
            self.runtime.finish(self)
            if not self.capture and self.span.is_recording():
                self.runtime.policy.deny(self.span.get_span_context())
            self.span.end()


class _Stream:
    def __init__(self, native, call):
        self.native = native
        self.call = call
        self.iterator = iter(native)

    def __iter__(self):
        return self

    def advance(self, method, *args):
        try:
            with self.call.scope():
                result = method(*args)
                safe(self.call.chunk, result)
            return result
        except StopIteration:
            safe(self.call.finish, self.call.response())
            raise
        except BaseException as error:
            safe(self.call.finish, self.call.response(), error)
            raise

    def __next__(self):
        return self.advance(self.iterator.__next__)

    def __getattr__(self, name):
        if name in ("send", "throw"):
            method = getattr(self.iterator, name)
            return lambda *args: self.advance(method, *args)
        if name == "close":
            method = getattr(self.native, name)

            def close(*args):
                try:
                    with self.call.scope():
                        return method(*args)
                except BaseException as error:
                    safe(self.call.finish, self.call.response(), error)
                    raise
                finally:
                    safe(self.call.finish, self.call.response())

            return close
        return getattr(self.native, name)


class _AsyncStream:
    def __init__(self, native, call):
        self.native = native
        self.call = call
        self.iterator = native.__aiter__()

    def __aiter__(self):
        return self

    async def advance(self, method, *args):
        try:
            with self.call.scope():
                result = await method(*args)
                safe(self.call.chunk, result)
            return result
        except StopAsyncIteration:
            safe(self.call.finish, self.call.response())
            raise
        except BaseException as error:
            safe(self.call.finish, self.call.response(), error)
            raise

    async def __anext__(self):
        return await self.advance(self.iterator.__anext__)

    def __getattr__(self, name):
        if name in ("asend", "athrow"):
            method = getattr(self.iterator, name)

            async def advance(*args):
                return await self.advance(method, *args)

            return advance
        if name == "aclose":
            method = getattr(self.native, name)

            async def close(*args):
                try:
                    with self.call.scope():
                        return await method(*args)
                except BaseException as error:
                    safe(self.call.finish, self.call.response(), error)
                    raise
                finally:
                    safe(self.call.finish, self.call.response())

            return close
        return getattr(self.native, name)


class _DualStream(_Stream):
    """Retain native LiteLLM wrappers supporting both iteration protocols."""

    def __init__(self, native, call):
        super().__init__(native, call)
        self.async_iterator = native.__aiter__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await _AsyncStream.advance(self, self.async_iterator.__anext__)

    def __getattr__(self, name):
        if name in ("asend", "athrow"):
            method = getattr(self.async_iterator, name)

            async def advance(*args):
                return await _AsyncStream.advance(self, method, *args)

            return advance
        if name == "aclose":
            return _AsyncStream.__getattr__(self, name)
        return super().__getattr__(name)


def wrap_stream(native, call, *, asynchronous=False):
    if hasattr(type(native), "__iter__") and hasattr(type(native), "__aiter__"):
        return _DualStream(native, call)
    return _AsyncStream(native, call) if asynchronous else _Stream(native, call)


class _Runtime:
    def __init__(self, include_content=True, tracer_provider=None):
        self.provider = tracer_provider or trace.get_tracer_provider()
        self.tracer = self.provider.get_tracer("respan.instrumentation.litellm")
        self.include_content = include_content
        self.active = True
        self.refs = 0
        self.calls = set()
        self.source = {}
        self.finished_ids = OrderedDict()
        self.patches = []
        self.policy = _Policy()
        self.callback = RespanLiteLLMCallback(
            include_content=include_content, _runtime=self
        )

    def begin(self, kwargs, owned=False):
        if not self.active or suppressed() or not enabled():
            return None
        call = _Call(self, kwargs, owned)
        self.calls.add(call)
        return call

    def identify(self, kwargs):
        identifier = get(
            kwargs,
            "litellm_call_id",
            get(get(kwargs, "litellm_params"), "litellm_call_id"),
        )
        current = _CALL.get()
        if current is not None and current.runtime is self and not current.finished:
            if isinstance(identifier, str):
                current.ids.add(identifier)
                self.source[identifier] = current
            return current
        return self.source.get(identifier)

    def pre(self, model, messages, kwargs, include_content=True):
        if not self.active:
            return
        call = self.identify(kwargs)
        identifier = get(
            kwargs,
            "litellm_call_id",
            get(get(kwargs, "litellm_params"), "litellm_call_id"),
        )
        if call is None and identifier not in self.finished_ids:
            call = self.begin({"model": model, "messages": messages, **kwargs})
            if call and isinstance(identifier, str):
                call.ids.add(identifier)
                self.source[identifier] = call
        if call:
            if not include_content:
                call.capture = False
                call.chunks.clear()
                call.kwargs.clear()
            call.observe({"model": model, "messages": messages, **kwargs})

    def post(self, kwargs):
        call = self.identify(kwargs)
        if call and not call.finished and not call.usage_seen:
            call.veto()
            original_response = get(kwargs, "original_response")
            call.usage = raw_usage(original_response)
            if (
                isinstance(original_response, dict)
                or isinstance(original_response, str)
                and original_response.lstrip().startswith("{")
            ):
                call.usage_seen = True

    def event(self, kwargs, response, error, failed=False):
        if not self.active:
            return
        call = self.identify(kwargs)
        if call is None:
            return
        call.veto()
        if not call.owned:
            safe(call.finish, response, error, failed)

    def finish(self, call):
        self.calls.discard(call)
        for identifier in call.ids:
            self.source.pop(identifier, None)
            self.finished_ids[identifier] = None
        while len(self.finished_ids) > 4096:
            self.finished_ids.popitem(last=False)
        call.ids.clear()

    def patch(self, owner, name, replacement):
        original = getattr(owner, name)
        self.patches.append((owner, name, original, replacement))
        setattr(owner, name, replacement)

    def install(self):
        import litellm

        self.module = litellm
        add_processor = get(self.provider, "add_span_processor")
        if callable(add_processor):
            add_processor(self.policy)
        original_detach = context.detach

        @wraps(original_detach)
        def detach(token):
            if self.active:
                safe(self.policy.observe_veto, trace.get_current_span())
            return original_detach(token)

        self.patch(context, "detach", detach)
        callbacks = get(litellm, "callbacks", [])
        self.original_callbacks = callbacks
        if not isinstance(callbacks, list):
            callbacks = list(callbacks or ())
            litellm.callbacks = callbacks
        callbacks.append(self.callback)
        self.installed_callbacks = callbacks
        for name in (
            "completion",
            "acompletion",
            "embedding",
            "aembedding",
            "responses",
            "aresponses",
        ):
            original = get(litellm, name)
            if not callable(original):
                continue
            signature = inspect.signature(original)

            def arguments(args, kwargs, signature=signature, name=name):
                try:
                    bound = dict(signature.bind_partial(*args, **kwargs).arguments)
                    extra = bound.pop("kwargs", {})
                    bound.update(extra)
                    bound.setdefault("call_type", name)
                    return bound
                except (ValueError, TypeError):
                    return {"call_type": name, **kwargs}

            if inspect.iscoroutinefunction(original):

                @wraps(original)
                async def wrapped(
                    *args, __original=original, __arguments=arguments, **kwargs
                ):
                    call = safe(self.begin, __arguments(args, kwargs), True)
                    if call is None:
                        return await __original(*args, **kwargs)
                    try:
                        with call.scope():
                            result = await __original(*args, **kwargs)
                    except BaseException as error:
                        safe(call.finish, None, error)
                        raise
                    if call.stream_requested and hasattr(result, "__aiter__"):
                        return wrap_stream(result, call, asynchronous=True)
                    safe(call.finish, result)
                    return result
            else:

                @wraps(original)
                def wrapped(
                    *args, __original=original, __arguments=arguments, **kwargs
                ):
                    call = safe(self.begin, __arguments(args, kwargs), True)
                    if call is None:
                        return __original(*args, **kwargs)
                    try:
                        with call.scope():
                            result = __original(*args, **kwargs)
                    except BaseException as error:
                        safe(call.finish, None, error)
                        raise
                    if call.stream_requested and hasattr(result, "__iter__"):
                        return wrap_stream(result, call)
                    safe(call.finish, result)
                    return result

            self.patch(litellm, name, wrapped)
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

        original = CustomStreamWrapper.chunk_creator

        @wraps(original)
        def creator(instance, *args, **kwargs):
            result = original(instance, *args, **kwargs)
            call = _CALL.get()
            if self.active and call is not None and call.runtime is self:
                source = kwargs.get("chunk", args[0] if args else None)
                found = safe(raw_usage, source)
                if found and not call.stream_usage_seen:
                    call.usage = found
            return result

        self.patch(CustomStreamWrapper, "chunk_creator", creator)
        # Observe OpenAI's retained raw provider body before model normalization
        # can coerce invalid boolean counters or manufacture missing zero usage.
        from openai._legacy_response import LegacyAPIResponse

        original_parse = LegacyAPIResponse.parse

        @wraps(original_parse)
        def parse(response, *args, **kwargs):
            result = original_parse(response, *args, **kwargs)
            call = _CALL.get()
            if self.active and call is not None and call.runtime is self:
                observed = safe(raw_usage, get(get(response, "http_response"), "text"))
                if observed is not None:
                    call.usage = observed
                    call.usage_seen = True
            return result

        self.patch(LegacyAPIResponse, "parse", parse)

        def observe_stream_usage(source):
            call = _CALL.get()
            if not self.active or call is None or call.runtime is not self:
                return
            if get(source, "type") == "response.completed":
                source = get(source, "response")
            if get(source, "usage") is not None:
                call.usage = raw_usage(source)
                call.stream_usage_seen = True

        from openai._streaming import ServerSentEvent

        original_json = ServerSentEvent.json

        @wraps(original_json)
        def event_json(event, *args, **kwargs):
            result = original_json(event, *args, **kwargs)
            safe(observe_stream_usage, result)
            return result

        self.patch(ServerSentEvent, "json", event_json)

        from litellm.llms.openai.responses.transformation import (
            OpenAIResponsesAPIConfig,
        )

        original_transform = OpenAIResponsesAPIConfig.transform_streaming_response

        @wraps(original_transform)
        def transform(instance, *args, **kwargs):
            result = original_transform(instance, *args, **kwargs)
            source = kwargs.get("parsed_chunk", args[1] if len(args) > 1 else None)
            safe(observe_stream_usage, source)
            return result

        self.patch(OpenAIResponsesAPIConfig, "transform_streaming_response", transform)

    def close(self):
        self.active = False
        for call in tuple(self.calls):
            call.capture = False
            safe(call.finish)
        for owner, name, original, replacement in reversed(self.patches):
            if getattr(owner, name) is replacement:
                setattr(owner, name, original)
        if hasattr(self, "module"):
            for name in _LISTS:
                callbacks = get(self.module, name)
                if isinstance(callbacks, list):
                    callbacks[:] = [c for c in callbacks if c is not self.callback]
            if (
                hasattr(self, "installed_callbacks")
                and get(self.module, "callbacks") is get(self, "installed_callbacks")
                and not isinstance(get(self, "original_callbacks"), list)
            ):
                original = tuple(self.original_callbacks or ())
                if len(original) == len(self.installed_callbacks) and all(
                    x is y for x, y in zip(original, self.installed_callbacks)
                ):
                    self.module.callbacks = self.original_callbacks
        self.policy.shutdown()
        active = get(self.provider, "_active_span_processor")
        if active is not None and hasattr(active, "_span_processors"):
            active._span_processors = tuple(
                p for p in active._span_processors if p is not self.policy
            )
        self.patches.clear()
        self.source.clear()
        self.finished_ids.clear()


class LiteLLMInstrumentor:
    name = "litellm"

    def __init__(self, *, include_content=True, tracer_provider=None):
        self.include_content = include_content
        self.provider = tracer_provider
        self.runtime = None

    def activate(self):
        global _RUNTIME
        with _LOCK:
            if self.runtime is not None:
                return
            if not enabled():
                return
            if _RUNTIME is not None and _RUNTIME.active:
                if (
                    _RUNTIME.include_content != self.include_content
                    or self.provider is not None
                    and _RUNTIME.provider is not self.provider
                ):
                    raise ValueError("Shared LiteLLM instrumentation settings differ")
                _RUNTIME.refs += 1
                self.runtime = _RUNTIME
                return
            runtime = _Runtime(self.include_content, self.provider)
            try:
                runtime.install()
            except BaseException:
                runtime.close()
                raise
            runtime.refs = 1
            _RUNTIME = runtime
            self.runtime = runtime

    def deactivate(self):
        global _RUNTIME
        with _LOCK:
            if self.runtime is None:
                return
            runtime = self.runtime
            self.runtime = None
            runtime.refs -= 1
            if runtime.refs == 0:
                runtime.close()
                if _RUNTIME is runtime:
                    _RUNTIME = None


LitellmInstrumentor = LiteLLMInstrumentor
