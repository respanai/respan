"""Observe native Anthropic calls while preserving SDK return objects."""

from __future__ import annotations

import importlib
import inspect
import logging
from contextvars import ContextVar
from functools import wraps
from threading import RLock
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace

from respan_instrumentation_anthropic._messages import CallState
from respan_instrumentation_anthropic._privacy import (
    PrivacyObserver,
    explicit_capture,
    suppressed,
)

logger = logging.getLogger(__name__)
_LOCK = RLock()
_OWNERS: set[Any] = set()
_PATCHES: list[tuple[Any, str, Any, Any]] = []
_CONFIG = None
_ACTIVE = ContextVar("respan_anthropic_native_call", default=False)


def _detach_wrapper(original: Any, instrumentor: Any) -> Any:
    @wraps(original)
    def detach(*args: Any, **kwargs: Any) -> Any:
        # Export processors temporarily suppress themselves. That scope must
        # not revoke a pending sibling request's capture policy.
        if instrumentor._patches_active and not explicit_capture():
            try:
                instrumentor._observer.notice()
            except Exception:  # noqa: BLE001, S110
                pass
        return original(*args, **kwargs)

    return detach


def _manager_entry(original: Any, asynchronous: bool) -> Any:
    def observe(manager: Any, stream: Any) -> Any:
        state = getattr(manager, "__respan_state__", None)
        # SDK 0.x helpers rebuild a MessageStream from the response; SDK 1.x
        # helpers consume the returned raw stream. Observe only the consumed one.
        raw_stream = getattr(stream, "_raw_stream", None)
        if (
            state is not None
            and getattr(raw_stream, "__respan_state__", None) is not state
        ):
            try:
                _observe_stream(raw_stream if raw_stream is not None else stream, state)
                close = stream.close
                if asynchronous:

                    async def aclose(*args: Any, **kwargs: Any):
                        try:
                            return await close(*args, **kwargs)
                        finally:
                            state.safe("finish")

                    stream.close = aclose
                else:

                    def sync_close(*args: Any, **kwargs: Any):
                        try:
                            return close(*args, **kwargs)
                        finally:
                            state.safe("finish")

                    stream.close = sync_close
            except Exception:  # noqa: BLE001
                state.safe("finish")
        return stream

    if asynchronous:

        @wraps(original)
        async def enter(manager: Any) -> Any:
            return observe(manager, await original(manager))
    else:

        @wraps(original)
        def enter(manager: Any) -> Any:
            return observe(manager, original(manager))

    return enter


def _observe_stream(stream: Any, state: CallState) -> Any:
    """Keep the actual Stream/AsyncStream and observe its native iterator."""
    iterator = stream._iterator
    close = stream.close
    stream.__respan_state__ = state
    if inspect.isasyncgen(iterator) or hasattr(iterator, "__anext__"):

        async def iterate():
            try:
                async for event in iterator:
                    state.safe("event", event)
                    yield event
            except BaseException as exc:
                if not isinstance(exc, GeneratorExit):
                    state.safe("failure", exc)
                raise
            finally:
                state.safe("finish")

        @wraps(close)
        async def aclose(*args: Any, **kwargs: Any):
            try:
                return await close(*args, **kwargs)
            finally:
                state.safe("finish")

        stream._iterator = iterate()
        stream.close = aclose
    else:

        def iterate():
            try:
                for event in iterator:
                    state.safe("event", event)
                    yield event
            except BaseException as exc:
                if not isinstance(exc, GeneratorExit):
                    state.safe("failure", exc)
                raise
            finally:
                state.safe("finish")

        @wraps(close)
        def sync_close(*args: Any, **kwargs: Any):
            try:
                return close(*args, **kwargs)
            finally:
                state.safe("finish")

        stream._iterator = iterate()
        stream.close = sync_close
    return stream


def _wrapper(
    original: Any,
    instrumentor: Any,
    *,
    helper: bool = False,
    managed: bool = False,
    asynchronous: bool = False,
) -> Any:
    def start(args: Any, kwargs: Any) -> CallState | None:
        if (
            not instrumentor._patches_active
            or _ACTIVE.get()
            or suppressed()
            or (instrumentor.context is not None and suppressed(instrumentor.context))
        ):
            return None
        try:
            instrumentor._ensure_provider()
            session = (
                kwargs.get("session_id", args[0] if args else None) if managed else None
            )
            state = CallState.__new__(CallState)
            try:
                state.__init__(instrumentor, kwargs, session_id=session)
            except Exception:
                span = getattr(state, "span", None)
                if span is not None:
                    try:
                        span.end()
                    except Exception:  # noqa: BLE001, S110
                        pass
                raise
            return state
        except Exception:  # noqa: BLE001
            return None

    def observe(result: Any, state: CallState | None) -> Any:
        if state is None:
            return result
        try:
            if helper:
                result.__respan_state__ = state
                class_name = type(result).__name__
                request_key = f"_{class_name}__api_request"
                if class_name in ("MessageStreamManager", "BetaMessageStreamManager"):
                    request = getattr(result, request_key)

                    def send():
                        token = _ACTIVE.set(True)
                        try:
                            result = request()
                        except BaseException as exc:
                            state.safe("failure", exc)
                            state.safe("finish")
                            raise
                        finally:
                            _ACTIVE.reset(token)

                        try:
                            return _observe_stream(result, state)
                        except Exception:  # noqa: BLE001
                            state.safe("finish")
                            return result

                    setattr(result, request_key, send)
                elif class_name in (
                    "AsyncMessageStreamManager",
                    "BetaAsyncMessageStreamManager",
                    "AsyncBetaMessageStreamManager",
                ):
                    request = getattr(result, request_key)

                    async def send_async():
                        token = _ACTIVE.set(True)
                        try:
                            result = await request
                        except BaseException as exc:
                            state.safe("failure", exc)
                            state.safe("finish")
                            raise
                        finally:
                            _ACTIVE.reset(token)

                        try:
                            return _observe_stream(result, state)
                        except Exception:  # noqa: BLE001
                            state.safe("finish")
                            return result

                    setattr(result, request_key, send_async())
                else:
                    state.safe("finish")
            else:
                from anthropic import AsyncStream, Stream

                if isinstance(result, (Stream, AsyncStream)):
                    _observe_stream(result, state)
                else:
                    state.safe("finish", result)
        except Exception:  # noqa: BLE001
            state.safe("finish")
        return result

    if asynchronous:

        @wraps(original)
        async def async_call(self: Any, *args: Any, **kwargs: Any) -> Any:
            state = start(args, kwargs)
            try:
                result = await original(self, *args, **kwargs)
            except BaseException as exc:
                if state:
                    state.safe("failure", exc)
                    state.safe("finish")
                raise
            return observe(result, state)

        return async_call

    @wraps(original)
    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        state = start(args, kwargs)
        try:
            result = original(self, *args, **kwargs)
        except BaseException as exc:
            if state:
                state.safe("failure", exc)
                state.safe("finish")
            raise
        return observe(result, state)

    return call


class AnthropicInstrumentor:
    """Instrument stable/beta Messages and available managed-session streams.

    Capture policy is intersected with OTel context and observed local parents.
    Optional SDK surfaces are discovered without deploying agent resources.
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        capture_content: bool = True,
        tracer_provider: Any = None,
        context: Any = None,
    ) -> None:
        self.capture_content = capture_content
        self.tracer_provider = tracer_provider
        self.context = context
        self._observer = PrivacyObserver()
        self._providers: list[Any] = []
        self._tracer = None
        self._is_instrumented = False
        self._patches_active = False

    def _ensure_provider(self) -> None:
        provider = self.tracer_provider or trace.get_tracer_provider()
        self._tracer = provider.get_tracer(__name__)
        if provider not in self._providers and hasattr(provider, "add_span_processor"):
            provider.add_span_processor(self._observer)
            self._providers.append(provider)

    def activate(self, *, tracer_provider: Any = None) -> None:
        global _CONFIG
        with _LOCK:
            if self._is_instrumented:
                return
            if tracer_provider is not None:
                self.tracer_provider = tracer_provider
            config = (self.capture_content, self.tracer_provider, self.context)
            if _OWNERS:
                if config != _CONFIG:
                    logger.warning(
                        "Anthropic instrumentation is active with a different configuration"
                    )
                    return
                _OWNERS.add(self)
                self._is_instrumented = True
                return
            try:
                resources = importlib.import_module("anthropic.resources.messages")
            except ImportError:
                return
            staged = []
            try:
                self._ensure_provider()
                runtime = context_api._RUNTIME_CONTEXT
                original = runtime.detach
                replacement = _detach_wrapper(original, self)
                replacement.__respan_owner__ = self
                staged.append((runtime, "detach", original, replacement))
                runtime.detach = replacement
                modules = [(resources, False)]
                for path, managed in (
                    ("anthropic.resources.beta.messages.messages", False),
                    ("anthropic.resources.beta.sessions.events", True),
                ):
                    try:
                        modules.append((importlib.import_module(path), managed))
                    except ImportError:
                        pass
                for module, managed in modules:
                    for name in (
                        ("Events", "AsyncEvents")
                        if managed
                        else ("Messages", "AsyncMessages")
                    ):
                        cls = getattr(module, name, None)
                        if cls is None:
                            continue
                        for method in (
                            ("stream",) if managed else ("create", "parse", "stream")
                        ):
                            original = getattr(cls, method, None)
                            if original is None:
                                continue
                            helper = method == "stream" and not managed
                            replacement = _wrapper(
                                original,
                                self,
                                helper=helper,
                                managed=managed,
                                asynchronous=name.startswith("Async") and not helper,
                            )
                            replacement.__respan_owner__ = self
                            staged.append((cls, method, original, replacement))
                            setattr(cls, method, replacement)
                streaming = importlib.import_module("anthropic.lib.streaming")
                for name in (
                    "MessageStreamManager",
                    "AsyncMessageStreamManager",
                    "BetaMessageStreamManager",
                    "BetaAsyncMessageStreamManager",
                    "AsyncBetaMessageStreamManager",
                ):
                    cls = getattr(streaming, name, None)
                    if cls is None:
                        continue
                    asynchronous = "Async" in name
                    method = "__aenter__" if asynchronous else "__enter__"
                    original = getattr(cls, method)
                    replacement = _manager_entry(original, asynchronous)
                    replacement.__respan_owner__ = self
                    staged.append((cls, method, original, replacement))
                    setattr(cls, method, replacement)
            except Exception:  # noqa: BLE001
                for cls, method, original, replacement in reversed(staged):
                    if getattr(cls, method, None) is replacement:
                        setattr(cls, method, original)
                self._remove_processors()
                return
            _PATCHES.extend(staged)
            self._patches_active = True
            _CONFIG = config
            _OWNERS.add(self)
            self._is_instrumented = True

    def _remove_processors(self) -> None:
        for provider in self._providers:
            processor = getattr(provider, "_active_span_processor", None)
            if processor is not None and hasattr(processor, "_span_processors"):
                with processor._lock:
                    processor._span_processors = tuple(
                        item
                        for item in processor._span_processors
                        if item is not self._observer
                    )
        self._providers.clear()

    def deactivate(self) -> None:
        global _CONFIG
        with _LOCK:
            if self not in _OWNERS:
                return
            _OWNERS.remove(self)
            self._is_instrumented = False
            if _OWNERS:
                return
            owners = {
                getattr(replacement, "__respan_owner__", None)
                for _, _, _, replacement in _PATCHES
            }
            for cls, method, original, replacement in reversed(_PATCHES):
                if getattr(cls, method, None) is replacement:
                    setattr(cls, method, original)
            for owner in owners:
                if owner is not None:
                    owner._patches_active = False
                    owner._remove_processors()
            self._remove_processors()
            _PATCHES.clear()
            _CONFIG = None
