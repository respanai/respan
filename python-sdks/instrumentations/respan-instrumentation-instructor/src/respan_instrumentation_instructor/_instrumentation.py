"""Native Instructor calls with shared ownership and detached stream context."""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import logging
import os
import threading
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

from opentelemetry import context, trace
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.exception_attributes import (
    EXCEPTION_MESSAGE,
    EXCEPTION_TYPE,
)
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.decorators.base import _should_send_prompts
from respan_tracing.utils.span_factory import read_propagated_attributes

from ._schema import _response_model_function_schema
from ._serialization import _redact_text, safe_error_message, safe_json_dumps, safe_text

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_RUNTIME = None
_CURRENT_CALL = ContextVar("instructor.call", default=None)
_WRAPPED = "_respan_instructor_instrumented"
_MISSING = object()
_CONTENT_KEYS = (
    SpanAttributes.TRACELOOP_ENTITY_INPUT,
    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
    SpanAttributes.LLM_REQUEST_FUNCTIONS,
    SpanAttributes.LLM_PROMPTS,
    SpanAttributes.LLM_COMPLETIONS,
)


def _object_value(value, key):
    if isinstance(value, dict):
        return value.get(key)
    try:
        return getattr(value, key, None)
    except Exception:  # noqa: BLE001 - telemetry must preserve the native operation
        return None


def _enum_value(value):
    value = _object_value(value, "value") or value
    return safe_text(value) if isinstance(value, str) else None


def _error_status(error):
    seen = set()
    for _ in range(6):
        if error is None or id(error) in seen:
            return None
        seen.add(id(error))
        status = _object_value(error, "status_code")
        if status is None:
            status = _object_value(_object_value(error, "response"), "status_code")
        if (
            isinstance(status, int)
            and not isinstance(status, bool)
            and 400 <= status <= 599
        ):
            return status
        error = _object_value(error, "last_exception") or _object_value(
            error, "__cause__"
        )
    return None


def _count(value):
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _is_respan_tracing_enabled():
    return bool(getattr(getattr(RespanTracer, "_instance", None), "is_enabled", True))


def _content_allowed():
    return _should_send_prompts() and os.getenv(
        "TRACELOOP_TRACE_CONTENT", "true"
    ).strip().lower() not in {"0", "false", "no", "off"}


def _is_wrapped(fn):
    return bool(getattr(fn, _WRAPPED, False)) and bool(
        getattr(getattr(fn, "_respan_instructor_runtime", None), "active", False)
    )


def _arguments(fn, args, kwargs):
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
        values = dict(bound.arguments)
        values.pop("self", None)
        extra = values.pop("kwargs", None)
        if isinstance(extra, Mapping):
            values.update(extra)
        return values
    except (TypeError, ValueError):
        return dict(kwargs)


def _raw_response_from_result(result, captured_responses=None):
    if captured_responses:
        return captured_responses[-1]
    if isinstance(result, tuple) and len(result) == 2:
        candidate = result[1]
        if (
            _object_value(candidate, "choices") is not None
            or _object_value(candidate, "output") is not None
        ):
            return candidate
    raw = _object_value(result, "_raw_response")
    if raw is not None:
        return raw
    if isinstance(result, (list, tuple)):
        for item in reversed(result):
            raw = _object_value(item, "_raw_response")
            if raw is not None:
                return raw
    return None


def _normalized_tool_calls(raw):
    choices = _object_value(raw, "choices")
    if choices:
        calls = _object_value(_object_value(choices[0], "message"), "tool_calls") or []
    else:
        calls = [
            c
            for c in (
                _object_value(raw, "output") or _object_value(raw, "content") or []
            )
            if _object_value(c, "type") in {"function_call", "tool_use"}
        ]
    result = []
    for call in calls:
        function = _object_value(call, "function") or call
        name = _object_value(function, "name")
        if not isinstance(name, str):
            continue
        arguments = _object_value(function, "arguments")
        if arguments is None:
            arguments = _object_value(function, "input")
        arguments = (
            _redact_text(arguments)
            if isinstance(arguments, str)
            else safe_json_dumps(arguments, complete=True)
        )
        item = {
            "type": "function",
            "function": {"name": _redact_text(name), "arguments": arguments},
        }
        identifier = _object_value(call, "call_id") or _object_value(call, "id")
        if isinstance(identifier, str):
            item["id"] = _redact_text(identifier)
        result.append(item)
    return result


def _set_raw_response_attributes(span, raw, *, capture=True, usage_override=None):
    if raw is None and usage_override is None:
        return
    for key, field in [
        (SpanAttributes.LLM_RESPONSE_MODEL, "model"),
        (gen_ai_attributes.GEN_AI_RESPONSE_ID, "id"),
    ]:
        value = _object_value(raw, field)
        if isinstance(value, str):
            span.set_attribute(key, safe_text(value))
    choices = _object_value(raw, "choices")
    reason = (
        _object_value(choices[0], "finish_reason")
        if choices
        else _object_value(raw, "status")
    )
    if isinstance(reason, str):
        span.set_attribute(
            SpanAttributes.GEN_AI_RESPONSE_FINISH_REASON, safe_text(reason)
        )
    if capture:
        calls = _normalized_tool_calls(raw)
        if calls:
            span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls",
                safe_json_dumps(calls, complete=True),
            )
    usage = (
        usage_override if usage_override is not None else _object_value(raw, "usage")
    )
    if usage is None:
        return
    incoming = _count(_object_value(usage, "input_tokens"))
    if incoming is None:
        incoming = _count(_object_value(usage, "prompt_tokens"))
    outgoing = _count(_object_value(usage, "output_tokens"))
    if outgoing is None:
        outgoing = _count(_object_value(usage, "completion_tokens"))
    total = _count(_object_value(usage, "total_tokens"))
    if total is None and incoming is not None and outgoing is not None:
        total = incoming + outgoing
    for key, number in [
        (gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS, incoming),
        (SpanAttributes.LLM_USAGE_PROMPT_TOKENS, incoming),
        (gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS, outgoing),
        (SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, outgoing),
        (SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total),
    ]:
        if number is not None:
            span.set_attribute(key, number)
    prompt_details = _object_value(usage, "prompt_tokens_details") or _object_value(
        usage, "input_tokens_details"
    )
    completion_details = _object_value(
        usage, "completion_tokens_details"
    ) or _object_value(usage, "output_tokens_details")
    for key, number in [
        (
            SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            _count(_object_value(prompt_details, "cached_tokens")),
        ),
        (
            SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
            _count(_object_value(completion_details, "reasoning_tokens")),
        ),
    ]:
        if number is not None:
            span.set_attribute(key, number)


def _build_span_attributes(
    operation_name, arguments, provider=None, mode=None, *, capture=True
):
    attrs = {
        RESPAN_LOG_TYPE: LOG_TYPE_CHAT,
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value,
        SpanAttributes.TRACELOOP_ENTITY_NAME: operation_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: "",
    }
    if provider:
        attrs[SpanAttributes.LLM_SYSTEM] = provider
    if isinstance(arguments.get("model"), str):
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = safe_text(arguments["model"])
    for arg, key in [
        ("temperature", SpanAttributes.LLM_REQUEST_TEMPERATURE),
        ("top_p", SpanAttributes.LLM_REQUEST_TOP_P),
        ("max_tokens", SpanAttributes.LLM_REQUEST_MAX_TOKENS),
    ]:
        value = arguments.get(arg)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            attrs[key] = value
    attrs[SpanAttributes.GEN_AI_IS_STREAMING] = bool(
        arguments.get("stream")
    ) or operation_name.endswith(("create_partial", "create_iterable"))
    if not capture:
        return attrs
    messages = arguments.get("messages", arguments.get("input"))
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json_dumps(
        {"messages": messages}
    )
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            role = _object_value(message, "role")
            if isinstance(role, str):
                attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.role"] = safe_text(role)
            value = _object_value(message, "content")
            if value is not None:
                attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.content"] = (
                    _redact_text(value)
                    if isinstance(value, str)
                    else safe_json_dumps(value)
                )
            calls = _object_value(message, "tool_calls")
            if calls is not None:
                attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.tool_calls"] = (
                    safe_json_dumps(
                        _normalized_tool_calls(
                            {"choices": [{"message": {"tool_calls": calls}}]}
                        ),
                        complete=True,
                    )
                )
            call_id = _object_value(message, "tool_call_id")
            if isinstance(call_id, str):
                attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.tool_call_id"] = (
                    _redact_text(call_id)
                )
    functions = arguments.get("tools") or _response_model_function_schema(
        arguments.get("response_model")
    )
    if functions is not None:
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safe_json_dumps(
            functions, complete=True
        )
    return attrs


def _set_success_attributes(
    span, result, raw_response=None, *, capture=True, raw_attributes=True
):
    raw = (
        raw_response if raw_response is not None else _raw_response_from_result(result)
    )
    parsed = (
        result[0]
        if isinstance(result, tuple) and len(result) == 2 and raw is result[1]
        else result
    )
    if capture:
        output = safe_json_dumps(parsed)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, output)
        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.content", output)
    if raw_attributes:
        _set_raw_response_attributes(
            span,
            raw,
            capture=capture,
            usage_override=_object_value(parsed, "_total_usage"),
        )


_USAGE_KEYS = {
    gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS,
    SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
    gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS,
    SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
    SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
    SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
    SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
}


class _Attributes:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class _RawIterator:
    def __init__(self, native, call):
        self.native, self.call = native, call

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.native)
        self.call.chunk(item)
        return item

    def __getattr__(self, name):
        return getattr(self.native, name)


class _RawAsyncIterator:
    def __init__(self, native, call):
        self.native, self.call = native, call

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.native.__anext__()
        self.call.chunk(item)
        return item

    def __getattr__(self, name):
        return getattr(self.native, name)


class _Call:
    def __init__(self, runtime, operation, arguments, provider, mode):
        self.runtime = runtime
        self.capture = runtime.trace_content and _content_allowed()
        self.operation = operation
        self.finished = False
        self.items = []
        self.observed = False
        self.tool_calls = {}
        self.raw_seen = False
        self.usage_keys = set()
        self.parent_context = context.get_current()
        tracer = (
            runtime.provider.get_tracer(__name__)
            if runtime.provider is not None
            else trace.get_tracer(__name__)
        )
        self.span = tracer.start_span(operation, context=self.parent_context)
        runtime.pending.add(self)
        if not self.span.is_recording():
            return
        try:
            self.start_attributes(operation, arguments, provider, mode)
        except BaseException:
            self.finish()
            raise

    def start_attributes(self, operation, arguments, provider, mode):
        attrs = _build_span_attributes(
            operation, arguments, provider, mode, capture=self.capture
        )
        attrs.update(read_propagated_attributes() or {})
        inherited_metadata = attrs.get(RESPAN_METADATA)
        try:
            metadata = (
                json.loads(inherited_metadata)
                if isinstance(inherited_metadata, str)
                else {}
            )
        except (TypeError, ValueError):
            metadata = {}
        metadata = metadata if isinstance(metadata, dict) else {}
        metadata.update(
            {
                "instructor_api": operation,
                "mode": _enum_value(mode),
                "token_budget": arguments.get("token_budget"),
            }
        )
        attrs[RESPAN_METADATA] = safe_json_dumps(metadata)
        for key, value in attrs.items():
            self.span.set_attribute(key, value)

    def _content(self):
        if not self.span.is_recording():
            return False
        if self.capture and not _content_allowed():
            self.capture = False
            self.items.clear()
            self.tool_calls.clear()
            attributes = getattr(self.span, "_attributes", None)
            if attributes is None:
                attributes = getattr(self.span, "attributes", {})
            for key in list(attributes or {}):
                if any(
                    key == prefix or key.startswith(prefix + ".")
                    for prefix in _CONTENT_KEYS
                ):
                    attributes.pop(key, None)
        return self.capture and self.span.is_recording()

    def content(self):
        try:
            return self._content()
        except BaseException:  # noqa: BLE001 - a policy failure hides telemetry, not the native operation
            self.capture = False
            self.items.clear()
            self.tool_calls.clear()
            try:
                attributes = getattr(self.span, "_attributes", None)
                if self.span.is_recording() and attributes is not None:
                    for key in list(attributes):
                        if any(
                            key == prefix or key.startswith(prefix + ".")
                            for prefix in _CONTENT_KEYS
                        ):
                            attributes.pop(key, None)
            except BaseException:  # noqa: BLE001 - preserve native outcome even with a hostile span implementation
                logger.debug("Instructor content cleanup failed open")
            return False

    @contextmanager
    def scope(self):
        if self.finished:
            yield
            return
        current = context.get_current()
        if not self.content():
            current = context.set_value(ENABLE_CONTENT_TRACING_KEY, False, current)
        token = context.attach(trace.set_span_in_context(self.span, current))
        active = _CURRENT_CALL.set(self)
        try:
            yield
        finally:
            try:
                self.content()
            finally:
                try:
                    _CURRENT_CALL.reset(active)
                finally:
                    context.detach(token)

    def observe(self, item):
        try:
            if self.content():
                value = json.loads(safe_json_dumps(item))
                if self.operation.endswith("create_partial"):
                    self.items[:] = [value]
                else:
                    self.items.append(value)
            raw = _raw_response_from_result(item)
            if raw is not None:
                _set_raw_response_attributes(self.span, raw, capture=self.content())
            self.observed = bool(self.items)
        except Exception:  # noqa: BLE001 - optional snapshots cannot break iterator advancement
            logger.debug("Instructor item observation failed open")

    def raw(self, response):
        if not self.span.is_recording() or self.finished:
            return
        try:
            self.raw_seen = True
            record = _Attributes()
            _set_raw_response_attributes(record, response, capture=self.content())
            self.usage_keys.update(
                key for key in record.attributes if key in _USAGE_KEYS
            )
            for key, value in record.attributes.items():
                self.span.set_attribute(key, value)
            iterator = _object_value(response, "_iterator")
            if type(response).__module__.startswith("openai"):
                if isinstance(iterator, AsyncIterator):
                    response._iterator = _RawAsyncIterator(iterator, self)
                elif isinstance(iterator, Iterator):
                    response._iterator = _RawIterator(iterator, self)
        except Exception:  # noqa: BLE001 - optional observation must preserve the native response
            logger.debug("Instructor raw response observation failed open")

    def chunk(self, chunk):
        if self.finished:
            return
        try:
            final = _object_value(chunk, "response")
            record = _Attributes()
            _set_raw_response_attributes(
                record, final if final is not None else chunk, capture=self.content()
            )
            self.usage_keys.update(
                key for key in record.attributes if key in _USAGE_KEYS
            )
            for key, value in record.attributes.items():
                self.span.set_attribute(key, value)
            if not self.content():
                return
            for choice in _object_value(chunk, "choices") or []:
                for part in (
                    _object_value(_object_value(choice, "delta"), "tool_calls") or []
                ):
                    index = _object_value(part, "index")
                    if not isinstance(index, int) or isinstance(index, bool):
                        continue
                    call = self.tool_calls.setdefault(
                        index, {"type": "function", "function": {"arguments": ""}}
                    )
                    identifier = _object_value(part, "id")
                    function = _object_value(part, "function")
                    name = _object_value(function, "name")
                    arguments = _object_value(function, "arguments")
                    if isinstance(identifier, str):
                        call["id"] = _redact_text(identifier)
                    if isinstance(name, str):
                        call["function"]["name"] = _redact_text(name)
                    if isinstance(arguments, str):
                        call["function"]["arguments"] += arguments
            calls = [
                call for call in self.tool_calls.values() if "name" in call["function"]
            ]
            if calls:
                self.span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls",
                    safe_json_dumps(calls, complete=True),
                )
        except Exception:  # noqa: BLE001 - telemetry cannot interrupt native stream decoding
            logger.debug("Instructor chunk observation failed open")

    def observed_usage(self, usage):
        if usage is None:
            return
        record = _Attributes()
        _set_raw_response_attributes(record, None, capture=False, usage_override=usage)
        for key, value in record.attributes.items():
            if key in self.usage_keys:
                self.span.set_attribute(key, value)

    def finish(self, result=_MISSING, error=None):
        if self.finished:
            return
        self.finished = True
        try:
            capture = self.content()
            if error is not None:
                if self.raw_seen:
                    self.observed_usage(_object_value(error, "total_usage"))
                else:
                    last = _object_value(error, "last_completion")
                    if _object_value(last, "usage") is not None:
                        _set_raw_response_attributes(self.span, last, capture=capture)
                name = type(error).__name__
                message = safe_error_message(error) if capture else name
                self.span.set_status(trace.Status(trace.StatusCode.ERROR, message))
                self.span.set_attribute(ERROR_TYPE, name)
                if capture:
                    self.span.set_attribute(ERROR_MESSAGE, message)
                self.span.add_event(
                    "exception", {EXCEPTION_TYPE: name, EXCEPTION_MESSAGE: message}
                )
                status = _error_status(error)
                if status is not None:
                    self.span.set_attribute(HTTP_RESPONSE_STATUS_CODE, status)
            elif result is not _MISSING:
                _set_success_attributes(
                    self.span, result, capture=capture, raw_attributes=not self.raw_seen
                )
                if self.raw_seen:
                    usage = _object_value(
                        result[0]
                        if isinstance(result, tuple) and len(result) == 2
                        else result,
                        "_total_usage",
                    )
                    self.observed_usage(usage)
            elif self.observed and capture:
                output = (
                    self.items[-1]
                    if self.operation.endswith("create_partial") and self.items
                    else self.items
                )
                _set_success_attributes(self.span, output)
        except Exception:  # noqa: BLE001 - telemetry must preserve the native operation
            logger.debug("Instructor telemetry finalization failed open")
        finally:
            self.items.clear()
            self.tool_calls.clear()
            try:
                self.span.end()
            except BaseException:  # noqa: BLE001 - processor errors cannot change native results
                logger.debug("Instructor span processor failed open")
            finally:
                self.runtime.pending.discard(self)


class _Iterator:
    def __init__(self, native, call):
        self.native, self.call = native, call

    def __iter__(self):
        return self

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _advance(self, method, *args):
        if self.call.finished:
            return method(*args)
        try:
            with self.call.scope():
                item = method(*args)
                self.call.observe(item)
            return item
        except StopIteration:
            self.call.finish()
            raise
        except BaseException as error:
            self.call.finish(error=error)
            raise

    def __next__(self):
        return self._advance(self.native.__next__)

    def send(self, value):
        return self._advance(self.native.send, value)

    def throw(self, *args):
        return self._advance(self.native.throw, *args)

    def close(self):
        try:
            with self.call.scope():
                close = getattr(self.native, "close", None)
                return close() if close is not None else None
        except BaseException as error:
            self.call.finish(error=error)
            raise
        finally:
            self.call.finish()


class _AsyncIterator:
    def __init__(self, native, call):
        self.native, self.call = native, call

    def __aiter__(self):
        return self

    def __getattr__(self, name):
        return getattr(self.native, name)

    async def _advance(self, method, *args):
        if self.call.finished:
            return await method(*args)
        try:
            with self.call.scope():
                item = await method(*args)
                self.call.observe(item)
            return item
        except StopAsyncIteration:
            self.call.finish()
            raise
        except BaseException as error:
            self.call.finish(error=error)
            raise

    async def __anext__(self):
        return await self._advance(self.native.__anext__)

    async def asend(self, value):
        return await self._advance(self.native.asend, value)

    async def athrow(self, *args):
        return await self._advance(self.native.athrow, *args)

    async def aclose(self):
        try:
            with self.call.scope():
                close = getattr(self.native, "aclose", None)
                return await close() if close is not None else None
        except BaseException as error:
            self.call.finish(error=error)
            raise
        finally:
            self.call.finish()


def _hook_kwargs(kwargs, call):
    values = dict(kwargs)
    if not call.span.is_recording():
        return values
    for path in (
        "instructor.v2.core.hooks",
        "instructor.core.hooks",
        "instructor.hooks",
    ):
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        hooks = values.get("hooks")
        if hooks is None:
            hooks = module.Hooks()
        elif hasattr(hooks, "copy"):
            hooks = hooks.copy()
        else:
            return values
        hooks.on("completion:response", call.raw)
        values["hooks"] = hooks
        return values
    return values


class _Runtime:
    def __init__(self, provider, trace_content):
        self.provider, self.trace_content = provider, trace_content
        self.count, self.active = 1, True
        self.patches = []
        self.pending = set()

    def patch(self, owner, name, wrapped):
        original = getattr(owner, name)
        if any(o is owner and n == name for o, n, *_ in self.patches):
            return
        own = vars(owner).get(name, _MISSING)
        setattr(owner, name, wrapped)
        self.patches.append((owner, name, original, own, wrapped))

    def restore(self):
        self.active = False
        for call in tuple(self.pending):
            call.finish()
        self.pending.clear()
        for owner, name, _, own, wrapped in reversed(self.patches):
            if getattr(owner, name, None) is wrapped:
                if own is _MISSING:
                    delattr(owner, name)
                else:
                    setattr(owner, name, own)
        self.patches.clear()

    def invoke(
        self, original, args, kwargs, operation, instance=None, provider=None, mode=None
    ):
        if (
            not self.active
            or _CURRENT_CALL.get() is not None
            or context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY)
            or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
        ):
            return None, kwargs
        call = None
        try:
            arguments = dict(_object_value(instance, "kwargs") or {})
            arguments.update(_arguments(original, args, kwargs))
            if arguments.get("model") is None:
                arguments["model"] = _object_value(instance, "default_model")
            provider = _enum_value(_object_value(instance, "provider")) or provider
            mode = _object_value(instance, "mode") or mode
            call = _Call(self, operation, arguments, provider, mode)
            return call, _hook_kwargs(kwargs, call)
        except Exception:  # noqa: BLE001 - telemetry must preserve the native operation
            if call is not None:
                call.finish()
            logger.debug("Instructor span creation failed open")
            return None, kwargs

    def result(self, result, call):
        if isinstance(result, AsyncIterator):
            return _AsyncIterator(result, call)
        if isinstance(result, Iterator):
            return _Iterator(result, call)
        call.finish(result)
        return result

    def wrap(self, original, operation, *, method=False, provider=None, mode=None):
        if _is_wrapped(original):
            return original
        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def wrapped(*args, **kwargs):
                call, values = self.invoke(
                    original,
                    args,
                    kwargs,
                    operation,
                    args[0] if method and args else None,
                    provider,
                    mode,
                )
                if call is None:
                    return await original(*args, **kwargs)
                try:
                    with call.scope():
                        result = await original(*args, **values)
                except BaseException as error:
                    call.finish(error=error)
                    raise
                return self.result(result, call)
        else:

            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                call, values = self.invoke(
                    original,
                    args,
                    kwargs,
                    operation,
                    args[0] if method and args else None,
                    provider,
                    mode,
                )
                if call is None:
                    return original(*args, **kwargs)
                try:
                    with call.scope():
                        result = original(*args, **values)
                except BaseException as error:
                    call.finish(error=error)
                    raise
                return self.result(result, call)

        setattr(wrapped, _WRAPPED, True)
        wrapped._respan_instructor_runtime = self
        return wrapped

    def wrap_patch(self, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            if not self.active:
                return result
            try:
                arguments = _arguments(original, args, kwargs)
                provider = _enum_value(arguments.get("provider"))
                if provider is None:
                    parameter = inspect.signature(original).parameters.get("provider")
                    provider = (
                        _enum_value(parameter.default)
                        if parameter is not None
                        else "openai"
                    )
                if callable(result):
                    return self.wrap(
                        result,
                        "instructor.patch",
                        provider=provider,
                        mode=arguments.get("mode"),
                    )
                client = arguments.get("client")
                if client is not None:
                    completions = _object_value(
                        _object_value(client, "chat"), "completions"
                    )
                    create = _object_value(completions, "create")
                    if callable(create):
                        completions.create = self.wrap(
                            create,
                            "instructor.patch",
                            provider=provider,
                            mode=arguments.get("mode"),
                        )
                return result
            except BaseException:  # noqa: BLE001 - native patch already succeeded
                logger.debug("Instructor patch observation failed open")
                return result

        setattr(wrapped, _WRAPPED, True)
        wrapped._respan_instructor_runtime = self
        return wrapped


class InstructorInstrumentor:
    """Observe Instructor's public methods and patch callables; shared per process."""

    name = "instructor"

    def __init__(self, *, tracer_provider=None, trace_content=True):
        self._provider, self._trace_content = tracer_provider, bool(trace_content)
        self._is_instrumented = False

    def _wrap_create_callable(
        self, original_create, operation_name, provider=None, mode=None
    ):
        runtime = _RUNTIME or _Runtime(self._provider, self._trace_content)
        return runtime.wrap(
            original_create, operation_name, provider=provider, mode=mode
        )

    def _wrap_instructor_method(self, original_method, operation_name):
        runtime = _RUNTIME or _Runtime(self._provider, self._trace_content)
        return runtime.wrap(original_method, operation_name, method=True)

    def activate(self):
        global _RUNTIME
        with _LOCK:
            if self._is_instrumented:
                return
            if not _is_respan_tracing_enabled():
                logger.info(
                    "Instructor instrumentation skipped because Respan tracing is disabled"
                )
                return
            if _RUNTIME is not None:
                if (
                    _RUNTIME.provider is not self._provider
                    or _RUNTIME.trace_content != self._trace_content
                ):
                    raise ValueError(
                        "Instructor instrumentation already has different provider/privacy settings"
                    )
                _RUNTIME.count += 1
                self._is_instrumented = True
                return
            try:
                module = importlib.import_module("instructor")
            except ImportError:
                logger.warning(
                    "Failed to activate Instructor instrumentation: missing dependency"
                )
                return
            runtime = _Runtime(self._provider, self._trace_content)
            try:
                modules = [module]
                for path in (
                    "instructor.v2.core.patch",
                    "instructor.core.patch",
                    "instructor.patch",
                ):
                    try:
                        modules.append(importlib.import_module(path))
                    except ImportError:
                        pass
                for owner in modules:
                    patch = getattr(owner, "patch", None)
                    if callable(patch) and not _is_wrapped(patch):
                        runtime.patch(owner, "patch", runtime.wrap_patch(patch))
                for cls_name in ("Instructor", "AsyncInstructor"):
                    cls = getattr(module, cls_name)
                    for method in (
                        "create",
                        "create_partial",
                        "create_iterable",
                        "create_with_completion",
                    ):
                        original = getattr(cls, method, None)
                        if callable(original) and not _is_wrapped(original):
                            operation = (
                                "instructor."
                                + ("async_" if cls_name.startswith("Async") else "")
                                + method
                            )
                            runtime.patch(
                                cls,
                                method,
                                runtime.wrap(original, operation, method=True),
                            )
            except BaseException:
                runtime.restore()
                raise
            _RUNTIME = runtime
            self._is_instrumented = True

    def deactivate(self):
        global _RUNTIME
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            if _RUNTIME is not None:
                _RUNTIME.count -= 1
                if _RUNTIME.count == 0:
                    _RUNTIME.restore()
                    _RUNTIME = None
