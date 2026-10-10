"""Small compatibility hooks for data OpenLIT does not retain on its spans.

The hooks run *inside* OpenLIT's OpenAI wrappers.  They do not create spans or
make provider calls; they only enrich the active OpenLIT span with request data,
the provider HTTP status, and whether token usage came from the provider.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from functools import wraps
from types import TracebackType
from typing import Any, Self

from openlit.semcov import SemanticConvention
from opentelemetry import trace
from opentelemetry.semconv.attributes.http_attributes import (
    HTTP_RESPONSE_STATUS_CODE,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from wrapt import FunctionWrapper

from respan_instrumentation_openlit._constants import (
    OPENLIT_PROVIDER_USAGE,
    OPENLIT_RAW_USAGE_OBSERVED,
    OPENLIT_SOURCE_INPUT,
    OPENLIT_SOURCE_OUTPUT,
    OPENLIT_SOURCE_TOOLS,
    OPENLIT_SOURCE_USAGE,
)
from respan_instrumentation_openlit._processor import (
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
)
from respan_instrumentation_openlit._serialization import (
    input_messages,
    json_string,
    safe_text,
    tool_definitions,
)

from ._policy import allowed

logger = logging.getLogger(__name__)
_RAW_SPAN = ContextVar("respan_openlit_raw_span", default=None)

RequestHook = tuple[Any, str, Callable[..., Any], Callable[..., Any]]
ChunkHook = tuple[Any, str, Callable[..., Any], Callable[..., Any]]
FactoryHook = tuple[Any, str, Callable[..., Any], Callable[..., Any]]
OpenAIPatch = tuple[Any, str, Any, Any]

_REQUEST_TARGETS = (
    ("openai.resources.chat.completions", "Completions", "create", False, False),
    ("openai.resources.chat.completions", "AsyncCompletions", "create", True, False),
    ("openai.resources.chat.completions", "Completions", "parse", False, False),
    ("openai.resources.chat.completions", "AsyncCompletions", "parse", True, False),
    ("openai.resources.responses.responses", "Responses", "create", False, False),
    (
        "openai.resources.responses.responses",
        "AsyncResponses",
        "create",
        True,
        False,
    ),
    ("openai.resources.responses.responses", "Responses", "parse", False, False),
    (
        "openai.resources.responses.responses",
        "AsyncResponses",
        "parse",
        True,
        False,
    ),
    ("openai.resources.embeddings", "Embeddings", "create", False, True),
    ("openai.resources.embeddings", "AsyncEmbeddings", "create", True, True),
)

_OPENLIT_OPENAI_MODULES = (
    "openlit.instrumentation.openai.openai",
    "openlit.instrumentation.openai.async_openai",
)

_STREAM_FACTORY_TARGETS = (
    ("chat_completions", False),
    ("async_chat_completions", True),
    ("responses", False),
    ("async_responses", True),
)


def _loaded_patch_owners() -> list[Any]:
    owners: list[Any] = []
    seen: set[int] = set()
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        if id(module) not in seen:
            seen.add(id(module))
            owners.append(module)
        for value in tuple(vars(module).values()):
            if not isinstance(value, type) or id(value) in seen:
                continue
            seen.add(id(value))
            owners.append(value)
    return owners


def _is_openlit_wrapper(value: Any) -> bool:
    if not isinstance(value, FunctionWrapper):
        return False
    wrapper = _safe_get(value, "_self_wrapper")
    module_name = _safe_get(wrapper, "__module__", "")
    return isinstance(module_name, str) and module_name.startswith(
        "openlit.instrumentation."
    )


def snapshot_openai_resource_methods() -> dict[tuple[Any, str], Any]:
    """Capture pre-existing wrappers before OpenLIT patches loaded libraries."""

    snapshot: dict[tuple[Any, str], Any] = {}
    for owner in _loaded_patch_owners():
        for name, value in tuple(vars(owner).items()):
            if isinstance(value, FunctionWrapper):
                snapshot[(owner, name)] = value
    return snapshot


def capture_openai_patches(
    before: Mapping[tuple[Any, str], Any],
) -> list[OpenAIPatch]:
    """Record only the outer wrappers installed by this OpenLIT activation."""

    patches: list[OpenAIPatch] = []
    for owner in _loaded_patch_owners():
        for name, current in tuple(vars(owner).items()):
            if not _is_openlit_wrapper(current):
                continue
            baseline = before.get((owner, name))
            if current is baseline:
                continue
            previous = current
            seen: set[int] = set()
            while (
                _is_openlit_wrapper(previous)
                and previous is not baseline
                and id(previous) not in seen
            ):
                seen.add(id(previous))
                previous = _safe_get(previous, "__wrapped__")
            if previous is current or previous is None:
                continue
            patches.append((owner, name, previous, current))
    return patches


def restore_openai_patches(patches: list[OpenAIPatch]) -> None:
    """Restore owned wrappers without overwriting a later foreign patch."""

    for owner, name, previous, installed in reversed(patches):
        current = vars(owner).get(name)
        if current is installed:
            setattr(owner, name, previous)
            continue
        seen: set[int] = set()
        while isinstance(current, FunctionWrapper) and id(current) not in seen:
            seen.add(id(current))
            wrapped = _safe_get(current, "__wrapped__")
            if wrapped is installed:
                current.__wrapped__ = previous
                break
            current = wrapped


def _openlit_span() -> Any | None:
    span = trace.get_current_span()
    if not getattr(span, "is_recording", lambda: False)():
        return None
    scope = getattr(span, "instrumentation_scope", None) or getattr(
        span, "instrumentation_info", None
    )
    scope_name = str(getattr(scope, "name", "") or "")
    if not scope_name.startswith("openlit.instrumentation.openai"):
        return None
    return span


def _safe_get(value: Any, name: str, default: Any = None) -> Any:
    try:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)
    except Exception:  # noqa: BLE001 - telemetry must preserve native outcomes.  # noqa: BLE001 - SDK/user objects may expose descriptors.
        return default


def _has_provider_usage(response: Any) -> bool:
    usage = _safe_get(response, "usage")
    if usage is None:
        return False
    for name in (
        "prompt_tokens",
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ):
        if _safe_get(usage, name) is not None:
            return True
    return False


def _http_status_code(error: BaseException) -> int | None:
    response = _safe_get(error, "response")
    candidates = (_safe_get(error, "status_code"), _safe_get(response, "status_code"))
    for candidate in candidates:
        try:
            code = int(candidate)
        except (TypeError, ValueError):
            continue
        if 100 <= code <= 599:
            return code
    return None


def _set_request_attributes(
    span: Any,
    kwargs: Mapping[str, Any],
    *,
    capture_content: bool,
    max_content_length: int,
    is_embedding: bool,
) -> None:
    if not allowed(span, capture_content):
        return
    request_input = kwargs.get("messages")
    if request_input is None:
        request_input = kwargs.get("input")
    if request_input is not None:
        if is_embedding:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                json_string(request_input, max_bytes=max_content_length),
            )
        else:
            span.set_attribute(
                GEN_AI_INPUT_MESSAGES,
                json_string(
                    input_messages(request_input),
                    max_bytes=max_content_length,
                    complete=True,
                ),
            )
            span.set_attribute(
                OPENLIT_SOURCE_INPUT, span.attributes[GEN_AI_INPUT_MESSAGES]
            )
    tools = kwargs.get("tools")
    if tools:
        span.set_attribute(
            SemanticConvention.GEN_AI_TOOL_DEFINITIONS,
            json_string(
                tool_definitions(tools),
                max_bytes=max_content_length,
                complete=True,
                schema=True,
            ),
        )
        span.set_attribute(
            OPENLIT_SOURCE_TOOLS,
            span.attributes[SemanticConvention.GEN_AI_TOOL_DEFINITIONS],
        )


def _valid_count(value):
    return value if type(value) is int and value >= 0 else None


def _source_usage(response):
    usage = _safe_get(response, "usage")
    if usage is None:
        return {}
    actual = {}
    fields = {
        "input": ("prompt_tokens", "input_tokens"),
        "output": ("completion_tokens", "output_tokens"),
        "total": ("total_tokens",),
    }
    for target, names in fields.items():
        for name in names:
            value = _valid_count(_safe_get(usage, name))
            if value is not None:
                actual[target] = value
                break
    for details_name in (
        "prompt_tokens_details",
        "input_tokens_details",
        "input_token_details",
    ):
        details = _safe_get(usage, details_name)
        for target, names in {
            "cache_read": ("cached_tokens", "cache_read_tokens"),
            "cache_creation": ("cache_write_tokens", "cache_creation_tokens"),
        }.items():
            for name in names:
                value = _valid_count(_safe_get(details, name))
                if value is not None:
                    actual[target] = value
                    break
    for details_name in (
        "completion_tokens_details",
        "output_tokens_details",
        "output_token_details",
    ):
        value = _valid_count(
            _safe_get(_safe_get(usage, details_name), "reasoning_tokens")
        )
        if value is not None:
            actual["reasoning"] = value
    return actual


def _response_messages(response):
    choices = _safe_get(response, "choices")
    if isinstance(choices, list):
        messages = []
        for choice in choices:
            message = _safe_get(choice, "message")
            if message is None:
                continue
            result = {
                k: _safe_get(message, k)
                for k in ("role", "content")
                if _safe_get(message, k) is not None
            }
            calls = _safe_get(message, "tool_calls")
            if isinstance(calls, list):
                result["tool_calls"] = [
                    {
                        "id": _safe_get(c, "id"),
                        "type": "function",
                        "function": {
                            "name": _safe_get(_safe_get(c, "function"), "name"),
                            "arguments": _safe_get(
                                _safe_get(c, "function"), "arguments"
                            ),
                        },
                    }
                    for c in calls
                ]
            messages.append(result)
        return messages
    output = _safe_get(response, "output")
    if not isinstance(output, list):
        return None
    messages = []
    calls = []
    for item in output:
        kind = _safe_get(item, "type")
        if kind == "message":
            parts = _safe_get(item, "content")
            if isinstance(parts, list):
                text = "".join(
                    _safe_get(part, "text", "")
                    for part in parts
                    if _safe_get(part, "type") == "output_text"
                    and isinstance(_safe_get(part, "text"), str)
                )
                messages.append(
                    {"role": _safe_get(item, "role", "assistant"), "content": text}
                )
        elif kind == "function_call":
            calls.append(
                {
                    "id": _safe_get(item, "call_id"),
                    "type": "function",
                    "function": {
                        "name": _safe_get(item, "name"),
                        "arguments": _safe_get(item, "arguments"),
                    },
                }
            )
    if calls:
        messages.append({"role": "assistant", "tool_calls": calls})
    return messages


def _mark_response(span, response):
    if not span.attributes.get(OPENLIT_RAW_USAGE_OBSERVED):
        span.set_attribute(OPENLIT_PROVIDER_USAGE, _has_provider_usage(response))
        span.set_attribute(
            OPENLIT_SOURCE_USAGE, json_string(_source_usage(response), complete=True)
        )
    if allowed(span):
        messages = _response_messages(response)
        if messages is not None:
            span.set_attribute(
                OPENLIT_SOURCE_OUTPUT,
                json_string(messages, complete=True),
            )


def _mark_error(span: Any, error: BaseException) -> None:
    status_code = _http_status_code(error)
    if status_code is not None:
        span.set_attribute(HTTP_RESPONSE_STATUS_CODE, status_code)
    span.set_attribute(OPENLIT_PROVIDER_USAGE, False)
    span.set_attribute(OPENLIT_SOURCE_USAGE, "{}")


def _observe_request(span, kwargs, capture_content, max_content_length, is_embedding):
    if span is None:
        return
    try:
        _set_request_attributes(
            span,
            kwargs,
            capture_content=capture_content,
            max_content_length=max_content_length,
            is_embedding=is_embedding,
        )
    except Exception:  # noqa: BLE001 - telemetry must preserve native outcomes.
        from ._instrumentation import _PROCESSOR

        if _PROCESSOR is not None:
            _PROCESSOR.veto(span)


def _observe_end(span, response=None, error=None):
    if span is None:
        return
    try:
        allowed(span)
        if error is not None:
            _mark_error(span, error)
        else:
            _mark_response(span, response)
    except Exception as exc:  # noqa: BLE001 - preserve provider outcomes.
        logger.debug("OpenLIT observer skipped: %s", type(exc).__name__)


def _sync_wrapper(original, *, capture_content, max_content_length, is_embedding):
    @wraps(original)
    def wrapped(*args, **kwargs):
        span = _openlit_span() if wrapped._respan_active else None
        _observe_request(
            span, kwargs, capture_content, max_content_length, is_embedding
        )
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            _observe_end(span, error=exc)
            raise
        _observe_end(span, response=result)
        return result

    wrapped._respan_active = True
    return wrapped


def _async_wrapper(original, *, capture_content, max_content_length, is_embedding):
    @wraps(original)
    async def wrapped(*args, **kwargs):
        span = _openlit_span() if wrapped._respan_active else None
        _observe_request(
            span, kwargs, capture_content, max_content_length, is_embedding
        )
        try:
            result = await original(*args, **kwargs)
        except BaseException as exc:
            _observe_end(span, error=exc)
            raise
        _observe_end(span, response=result)
        return result

    wrapped._respan_active = True
    return wrapped


def _observe_raw_usage(body, span):
    if span is None or not span.is_recording() or not isinstance(body, dict):
        return
    span.set_attribute(OPENLIT_RAW_USAGE_OBSERVED, True)
    actual = _source_usage(
        body.get("response") if isinstance(body.get("response"), dict) else body
    )
    existing = span.attributes.get(OPENLIT_SOURCE_USAGE)
    if existing is None:
        span.set_attribute(OPENLIT_SOURCE_USAGE, "{}")
    if actual or existing is None:
        span.set_attribute(OPENLIT_SOURCE_USAGE, json_string(actual, complete=True))
    span.set_attribute(
        OPENLIT_PROVIDER_USAGE, bool(actual) or bool(existing and existing != "{}")
    )


def _install_raw_usage_hooks(hooks):
    for module_name, class_name, asynchronous in [
        ("openai._response", "APIResponse", False),
        ("openai._response", "AsyncAPIResponse", True),
        ("openai._legacy_response", "LegacyAPIResponse", False),
    ]:
        owner = getattr(importlib.import_module(module_name), class_name)
        original = owner.parse
        if asynchronous:

            def factory(original):
                @wraps(original)
                async def parse(instance, *args, **kwargs):
                    span = _openlit_span() if parse._respan_active else None
                    result = await original(instance, *args, **kwargs)
                    _observe_http_usage(instance, span)
                    return result

                parse._respan_active = True
                return parse
        else:

            def factory(original):
                @wraps(original)
                def parse(instance, *args, **kwargs):
                    span = _openlit_span() if parse._respan_active else None
                    result = original(instance, *args, **kwargs)
                    _observe_http_usage(instance, span)
                    return result

                parse._respan_active = True
                return parse

        replacement = factory(original)
        owner.parse = replacement
        hooks.append((owner, "parse", original, replacement))
    owner = importlib.import_module("openai._streaming").ServerSentEvent
    original = owner.json

    @wraps(original)
    def event_json(instance, *args, **kwargs):
        result = original(instance, *args, **kwargs)
        if event_json._respan_active:
            try:
                _observe_raw_usage(result, _RAW_SPAN.get())
            except Exception as exc:  # noqa: BLE001 - native SSE values/errors are preserved.
                logger.debug("OpenLIT raw SSE usage skipped: %s", type(exc).__name__)
        return result

    event_json._respan_active = True
    owner.json = event_json
    hooks.append((owner, "json", original, event_json))


def _observe_http_usage(instance, span):
    if span is None or not span.is_recording():
        return
    try:
        response = instance.http_response
        if (
            "json" in response.headers.get("content-type", "")
            and response.is_stream_consumed
        ):
            _observe_raw_usage(response.json(), span)
    except Exception as exc:  # noqa: BLE001 - source observation never changes parsing.
        logger.debug("OpenLIT raw HTTP usage skipped: %s", type(exc).__name__)


def install_openai_request_hooks(
    *,
    capture_content: bool,
    max_content_length: int,
) -> list[RequestHook]:
    """Install hooks before OpenLIT wraps the OpenAI resource methods."""

    hooks: list[RequestHook] = []
    try:
        for (
            module_name,
            class_name,
            method_name,
            is_async,
            is_embedding,
        ) in _REQUEST_TARGETS:
            try:
                module = importlib.import_module(module_name)
                owner = getattr(module, class_name)
                original = getattr(owner, method_name)
            except (AttributeError, ImportError):
                continue
            replacement = (
                _async_wrapper(
                    original,
                    capture_content=capture_content,
                    max_content_length=max_content_length,
                    is_embedding=is_embedding,
                )
                if is_async
                else _sync_wrapper(
                    original,
                    capture_content=capture_content,
                    max_content_length=max_content_length,
                    is_embedding=is_embedding,
                )
            )
            setattr(owner, method_name, replacement)
            hooks.append((owner, method_name, original, replacement))
        _install_raw_usage_hooks(hooks)
    except Exception:
        remove_openai_request_hooks(hooks)
        raise
    return hooks


def remove_openai_request_hooks(hooks: list[RequestHook]) -> None:
    """Restore only OpenAI methods which still contain this adapter's hook."""

    for owner, method_name, original, replacement in reversed(hooks):
        replacement._respan_active = False
        current = vars(owner).get(method_name)
        if current is replacement:
            setattr(owner, method_name, original)
            continue
        seen: set[int] = set()
        while isinstance(current, FunctionWrapper) and id(current) not in seen:
            seen.add(id(current))
            wrapped = _safe_get(current, "__wrapped__")
            if wrapped is replacement:
                current.__wrapped__ = original
                break
            current = wrapped


def _checkpoint(span, configured=True):
    try:
        return allowed(span, configured)
    except Exception as exc:  # noqa: BLE001 - policy observers preserve native outcomes.
        logger.debug("OpenLIT checkpoint skipped: %s", type(exc).__name__)
        return False


def _finish_early_stream(
    scope: Any,
    *,
    capture_content: bool,
    max_content_length: int,
    error: BaseException | None = None,
) -> None:
    """End an OpenLIT stream span that upstream leaves open on early close."""

    span = _safe_get(scope, "_span")
    is_recording = _safe_get(span, "is_recording")
    if span is None or not callable(is_recording) or not is_recording():
        return
    permitted = _checkpoint(span, capture_content)
    if error is None and permitted:
        partial_text = _safe_get(scope, "_llmresponse")
        if isinstance(partial_text, str) and partial_text:
            span.set_attribute(
                GEN_AI_OUTPUT_MESSAGES,
                json_string(
                    [
                        {
                            "role": "assistant",
                            "parts": [
                                {
                                    "type": "text",
                                    "content": safe_text(
                                        partial_text,
                                        limit=4_000,
                                    ),
                                }
                            ],
                        }
                    ],
                    max_bytes=max_content_length,
                ),
            )
    if error is None:
        span.set_status(Status(StatusCode.OK))
    else:
        message = safe_text(type(error).__name__, default="stream cancelled")
        status_code = _http_status_code(error)
        if status_code is not None:
            span.set_attribute(HTTP_RESPONSE_STATUS_CODE, status_code)
        span.set_status(Status(StatusCode.ERROR, message))
    span.end()


class _SyncStreamProxy:
    def __init__(
        self,
        wrapped: Any,
        *,
        capture_content: bool,
        max_content_length: int = 16_000,
    ) -> None:
        self._wrapped = wrapped
        self._capture_content = capture_content
        self._max_content_length = max_content_length
        self._finished = False
        self._source_closed = False

    def __iter__(self) -> _SyncStreamProxy:
        return self

    def __next__(self) -> Any:
        if self._finished:
            raise StopIteration
        try:
            _checkpoint(_safe_get(self._wrapped, "_span"), self._capture_content)
            token = _RAW_SPAN.set(_safe_get(self._wrapped, "_span"))
            try:
                result = next(self._wrapped)
            finally:
                _RAW_SPAN.reset(token)
            _checkpoint(_safe_get(self._wrapped, "_span"), self._capture_content)
            return result
        except StopIteration:
            self._finished = True
            raise
        except Exception as exc:
            self._close(error=exc)
            raise

    def __enter__(self) -> Self:
        enter = _safe_get(self._wrapped, "__enter__")
        try:
            if callable(enter):
                enter()
        except BaseException as exc:
            self._close(error=exc)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        suppress = False
        exit_method = _safe_get(self._wrapped, "__exit__")
        try:
            if callable(exit_method):
                suppress = bool(exit_method(exc_type, exc, traceback))
                self._source_closed = True
        except BaseException as exit_error:
            self._finish(error=exc or exit_error)
            raise
        else:
            self._finish(error=exc if isinstance(exc, BaseException) else None)
        return suppress

    def _close_source(self) -> None:
        if self._source_closed:
            return
        close = _safe_get(self._wrapped, "close")
        if callable(close):
            close()
        self._source_closed = True

    def _finish(self, *, error: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        _finish_early_stream(
            self._wrapped,
            capture_content=self._capture_content,
            max_content_length=self._max_content_length,
            error=error,
        )

    def _close(self, *, error: BaseException | None = None) -> None:
        close_error: BaseException | None = None
        try:
            self._close_source()
        except BaseException as exc:  # noqa: BLE001 - preserve cancellation.
            close_error = exc
        finally:
            self._finish(error=error or close_error)
        if error is None and close_error is not None:
            raise close_error

    def close(self) -> None:
        self._close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class _AsyncStreamProxy:
    def __init__(
        self,
        wrapped: Any,
        *,
        capture_content: bool,
        max_content_length: int = 16_000,
    ) -> None:
        self._wrapped = wrapped
        self._capture_content = capture_content
        self._max_content_length = max_content_length
        self._finished = False
        self._source_closed = False

    def __aiter__(self) -> _AsyncStreamProxy:
        return self

    async def __anext__(self) -> Any:
        if self._finished:
            raise StopAsyncIteration
        try:
            _checkpoint(_safe_get(self._wrapped, "_span"), self._capture_content)
            token = _RAW_SPAN.set(_safe_get(self._wrapped, "_span"))
            try:
                result = await self._wrapped.__anext__()
            finally:
                _RAW_SPAN.reset(token)
            _checkpoint(_safe_get(self._wrapped, "_span"), self._capture_content)
            return result
        except StopAsyncIteration:
            self._finished = True
            raise
        except asyncio.CancelledError as exc:
            await self._close(error=exc)
            raise
        except Exception as exc:
            await self._close(error=exc)
            raise

    async def __aenter__(self) -> Self:
        enter = _safe_get(self._wrapped, "__aenter__")
        try:
            if callable(enter):
                await enter()
        except BaseException as exc:
            await self._close(error=exc)
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        suppress = False
        exit_method = _safe_get(self._wrapped, "__aexit__")
        try:
            if callable(exit_method):
                suppress = bool(await exit_method(exc_type, exc, traceback))
                self._source_closed = True
        except BaseException as exit_error:
            self._finish(error=exc or exit_error)
            raise
        else:
            self._finish(error=exc if isinstance(exc, BaseException) else None)
        return suppress

    async def _close_source(self) -> None:
        if self._source_closed:
            return
        close = _safe_get(self._wrapped, "close")
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result
        self._source_closed = True

    def _finish(self, *, error: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        _finish_early_stream(
            self._wrapped,
            capture_content=self._capture_content,
            max_content_length=self._max_content_length,
            error=error,
        )

    async def _close(self, *, error: BaseException | None = None) -> None:
        close_error: BaseException | None = None
        try:
            await self._close_source()
        except BaseException as exc:  # noqa: BLE001 - preserve cancellation.
            close_error = exc
        finally:
            self._finish(error=error or close_error)
        if error is None and close_error is not None:
            raise close_error

    async def close(self) -> None:
        await self._close()

    async def aclose(self) -> None:
        await self._close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def _factory_capture_content(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> bool:
    if len(args) > 5:
        return bool(args[5])
    return bool(kwargs.get("capture_message_content", False))


def _sync_stream_factory(
    original: Callable[..., Any], *, max_content_length: int
) -> Callable[..., Any]:
    @wraps(original)
    def factory(*factory_args: Any, **factory_kwargs: Any) -> Callable[..., Any]:
        upstream_wrapper = original(*factory_args, **factory_kwargs)
        capture_content = _factory_capture_content(factory_args, factory_kwargs)
        processor = (
            _safe_get(factory_args[3], "processor") if len(factory_args) > 3 else None
        )

        @wraps(upstream_wrapper)
        def wrapper(
            wrapped: Callable[..., Any],
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            if processor is not None and not processor.active:
                return wrapped(*args, **kwargs)
            result = upstream_wrapper(wrapped, instance, args, kwargs)
            if kwargs.get("stream") and _safe_get(result, "_span") is not None:
                return _SyncStreamProxy(
                    result,
                    capture_content=capture_content,
                    max_content_length=max_content_length,
                )
            return result

        return wrapper

    return factory


def _async_stream_factory(
    original: Callable[..., Any], *, max_content_length: int
) -> Callable[..., Any]:
    @wraps(original)
    def factory(*factory_args: Any, **factory_kwargs: Any) -> Callable[..., Any]:
        upstream_wrapper = original(*factory_args, **factory_kwargs)
        capture_content = _factory_capture_content(factory_args, factory_kwargs)
        processor = (
            _safe_get(factory_args[3], "processor") if len(factory_args) > 3 else None
        )

        @wraps(upstream_wrapper)
        async def wrapper(
            wrapped: Callable[..., Any],
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            if processor is not None and not processor.active:
                return await wrapped(*args, **kwargs)
            result = await upstream_wrapper(wrapped, instance, args, kwargs)
            if kwargs.get("stream") and _safe_get(result, "_span") is not None:
                return _AsyncStreamProxy(
                    result,
                    capture_content=capture_content,
                    max_content_length=max_content_length,
                )
            return result

        return wrapper

    return factory


def install_openai_stream_factory_hooks(
    *, max_content_length: int = 16_000
) -> list[FactoryHook]:
    """Patch wrapper factories only while OpenLIT constructs its wrappers."""

    try:
        module = importlib.import_module("openlit.instrumentation.openai")
    except ImportError:
        return []
    hooks: list[FactoryHook] = []
    try:
        for name, is_async in _STREAM_FACTORY_TARGETS:
            original = getattr(module, name, None)
            if not callable(original):
                continue
            replacement = (
                _async_stream_factory(original, max_content_length=max_content_length)
                if is_async
                else _sync_stream_factory(
                    original, max_content_length=max_content_length
                )
            )
            setattr(module, name, replacement)
            hooks.append((module, name, original, replacement))
    except Exception:
        remove_openai_stream_factory_hooks(hooks)
        raise
    return hooks


def remove_openai_stream_factory_hooks(hooks: list[FactoryHook]) -> None:
    """Restore wrapper factories without overwriting later foreign patches."""

    for module, name, original, replacement in reversed(hooks):
        if getattr(module, name, None) is replacement:
            setattr(module, name, original)


def install_openai_stream_usage_hooks() -> list[ChunkHook]:
    """Mark streaming usage as provider-supplied when the usage chunk arrives."""

    hooks: list[ChunkHook] = []
    try:
        for module_name in _OPENLIT_OPENAI_MODULES:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            for name in ("process_chat_chunk", "process_response_chunk"):
                original = getattr(module, name, None)
                if not callable(original):
                    continue

                @wraps(original)
                def process_chunk(
                    scope: Any,
                    chunk: Any,
                    *,
                    __original: Callable[..., Any] = original,
                ) -> Any:
                    result = __original(scope, chunk)
                    response = _safe_get(chunk, "response", chunk)
                    if _has_provider_usage(response):
                        span = getattr(scope, "_span", None)
                        if span is not None:
                            try:
                                _checkpoint(span)
                                if not span.attributes.get(OPENLIT_RAW_USAGE_OBSERVED):
                                    span.set_attribute(OPENLIT_PROVIDER_USAGE, True)
                                    span.set_attribute(
                                        OPENLIT_SOURCE_USAGE,
                                        json_string(
                                            _source_usage(response), complete=True
                                        ),
                                    )
                            except Exception as exc:  # noqa: BLE001 - preserve chunk identity/error.
                                logger.debug(
                                    "OpenLIT chunk observation skipped: %s",
                                    type(exc).__name__,
                                )
                    return result

                setattr(module, name, process_chunk)
                hooks.append((module, name, original, process_chunk))
    except Exception:
        remove_openai_stream_usage_hooks(hooks)
        raise
    return hooks


def remove_openai_stream_usage_hooks(hooks: list[ChunkHook]) -> None:
    """Restore only OpenLIT chunk processors still owned by this adapter."""

    for module, name, original, replacement in reversed(hooks):
        if getattr(module, name, None) is replacement:
            setattr(module, name, original)
