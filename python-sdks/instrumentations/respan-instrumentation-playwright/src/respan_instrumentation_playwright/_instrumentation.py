"""One ``tool`` span per public Playwright API call, sync and async.

Playwright is not an agent: it is what an agent (or a test, or a script) drives.
So this instrumentation adds no root of its own. Each public call that reaches
the browser (``Page.goto``, ``Locator.click``, ``expect(...).to_have_text``, ...)
becomes a span under whatever span is active - a Respan workflow, an LLM agent's
span - and calls made with no active span are not traced at all.

Seams (Playwright's own generated public classes, patched from outside):

* ``playwright.async_api._generated.<Class>.<method>`` for every ``async def``
  public method: exactly the calls that round-trip to the browser, the same
  granularity as Playwright's trace viewer (``page.locator()`` builds a locator
  locally and is not a span).
* ``playwright.sync_api._generated.<Class>.<method>`` for the same names: the
  sync API mirrors the async one method for method.
* ``AsyncBase._wrap_handler`` / ``SyncBase._wrap_handler``: every user callback
  (``page.on``, ``page.route``, ``expose_function``, ...) passes through it. The
  sync API runs callbacks in a fresh greenlet with an empty context, so spans
  opened inside a route handler would lose their parent and the customer's
  propagated attributes; callbacks are run in the context they were registered in.

Probes this instrumentation makes itself (page title, screenshots, the
password-field check) are marked internal, so Playwright's own trace viewer
does not show them as the user's calls.
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import importlib
import inspect
import logging
import threading
from collections.abc import Callable, Coroutine, Generator, Iterator, Mapping
from contextlib import contextmanager
from functools import cache, partial, wraps
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.llm_logging import LOG_TYPE_TOOL
from respan_tracing import is_content_tracing_enabled
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.agent_instrumentation import (
    SCREENSHOT_JPEG_QUALITY,
    AgentScope,
    MethodPatcher,
    agent_span,
    Replacements,
    compact_image_data_url,
    safely,
    secret_replacements,
)

from respan_instrumentation_playwright._constants import (
    ACTION_METHODS,
    ASYNC_API_MODULE,
    ASYNC_BASE_MODULE,
    ATTR_API_NAME,
    ATTR_PAGE_TITLE,
    ATTR_PAGE_URL,
    INSTRUMENTATION_NAME,
    LEARNED_SECRET_NAME,
    MASKED_TEXT,
    MAX_LEARNED_SECRETS,
    NAVIGATION_METHODS,
    PAGE_IMPL_MODULE,
    SCREENSHOT_METHOD,
    SCREENSHOT_MODES,
    SCREENSHOT_NEVER,
    SCREENSHOT_ON_NAVIGATION,
    SYNC_API_MODULE,
    SYNC_BASE_MODULE,
    TEXT_ENTRY_ARGS,
    TRACER_NAME,
    WRAP_HANDLER,
)
from respan_instrumentation_playwright._serialization import (
    call_arguments,
    call_result,
    json_string,
)

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_RUNTIME: _Runtime | None = None


class _Scope(AgentScope):
    """Declared secrets, plus text the page took into a password field.

    A typed password can come back in a URL (a GET form), an error or a later
    ``evaluate``; once seen, it is masked for the rest of the caller's context.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._learned: contextvars.ContextVar[Replacements] = contextvars.ContextVar(
            f"{name}_learned", default=()
        )

    def learn(self, value: str) -> None:
        learned = dict(self._learned.get())
        learned.update(secret_replacements({LEARNED_SECRET_NAME: value}))
        pairs = sorted(learned.items(), key=lambda pair: -len(pair[0]))
        self._learned.set(tuple(pairs[:MAX_LEARNED_SECRETS]))

    def mask(self, text: str) -> str:
        text = super().mask(text)
        for secret, placeholder in self._learned.get():
            text = text.replace(secret, placeholder)
        return text

    def mask_span(self, span: Any) -> None:
        if not self.has_secrets:
            return
        for key, value in list((span.attributes or {}).items()):
            if isinstance(value, str) and (masked := self.mask(value)) != value:
                span.set_attribute(key, masked)

    @property
    def has_secrets(self) -> bool:
        return super().has_secrets or bool(self._learned.get())


# Secrets declared with ``secrets(...)`` and learned password text: masked out of every span.
_SCOPE = _Scope("respan_playwright")
# The caller's context just outside the call span, for callbacks registered by that call.
_CALLER_CONTEXT: contextvars.ContextVar[Optional[contextvars.Context]] = contextvars.ContextVar(
    "respan_playwright_caller_context", default=None
)
_HANDLER_ATTR = "__respan_context_handler__"
_PROBE_TIMEOUT_MS = 5000
_FAILED = object()

_IS_PASSWORD_ANY = (
    "els => els.some(e => (e.getAttribute('type') || '').toLowerCase() === 'password')"
)
_IS_PASSWORD_ONE = "e => (e.getAttribute('type') || '').toLowerCase() === 'password'"
# Typed text goes to the focused element; inside an iframe we cannot see which one.
_IS_PASSWORD_FOCUSED = (
    "() => { const e = document.activeElement;"
    " return !!e && (e.tagName === 'IFRAME' || e.tagName === 'FRAME'"
    " || (e.getAttribute('type') || '').toLowerCase() === 'password'); }"
)


@contextmanager
def secrets(sensitive_data: Mapping[str, Any]) -> Iterator[None]:
    """Mask these values (``{name: value}``) out of every Playwright span.

    Each value is exported as ``<secret>name</secret>``, wherever it shows up:
    typed text, URLs, ``evaluate`` arguments, error messages.
    """
    with _SCOPE.run(None, sensitive_data):
        yield


def _has_parent_span() -> bool:
    return trace.get_current_span().get_span_context().is_valid


@cache
def _page_impl_class() -> type | None:
    try:
        return importlib.import_module(PAGE_IMPL_MODULE).Page
    except (ImportError, AttributeError):
        return None


def _owner_page(impl: Any) -> Any:
    """The impl Page a call acts on, from attributes Playwright keeps locally."""
    page_class = _page_impl_class()
    for _ in range(5):
        if impl is None or page_class is None:
            return None
        if isinstance(impl, page_class):
            return impl
        channel = getattr(impl, "_channel", None)
        channel_object = getattr(channel, "_object", None)
        impl = (
            getattr(impl, "_page", None)
            or getattr(impl, "_actual_page", None)
            or getattr(impl, "_actual_locator", None)
            or getattr(impl, "_frame", None)
            # Keyboard / Mouse / Touchscreen talk over their page's channel.
            or (channel_object if channel_object is not impl else None)
        )
    return None


def _connection(impl: Any) -> Any:
    return getattr(impl, "_connection", None) or getattr(
        getattr(impl, "_channel", None), "_connection", None
    )


def _internal(impl: Any, factory: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    """Run an impl coroutine as an internal call (hidden from Playwright's trace)."""
    return _connection(impl).wrap_api_call(factory, is_internal=True)


class _Call:
    """What one public call records: input, page context, output."""

    def __init__(
        self,
        runtime: _Runtime,
        span: Any,
        method: str,
        public: Any,
        original: Callable[..., Any],
        args: tuple,
        kwargs: dict,
    ) -> None:
        self.runtime = runtime
        self.span = span
        self.method = method
        self.impl = getattr(public, "_impl_obj", None)
        self.content = runtime.content_allowed()
        self.arguments: dict[str, Any] = {}
        if self.content:
            self.arguments = call_arguments(original, public, args, kwargs)
        text_arg = TEXT_ENTRY_ARGS.get(method)
        self.text_arg = text_arg if self.arguments.get(text_arg) else None

    def record_input(self, *, is_password: bool) -> None:
        if not self.content:
            return
        arguments = dict(self.arguments)
        if self.text_arg and is_password:
            safely(partial(_SCOPE.learn, arguments[self.text_arg]))
            arguments[self.text_arg] = MASKED_TEXT
        if arguments:
            self.span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string(arguments))

    def after(self, result: Any) -> Generator[Any, Any, None]:
        """Post-call recording. Yields internal impl coroutines for the caller to run
        (awaited in the async API, ``_sync``-ed in the sync API) and receives each
        result, or ``_FAILED``."""
        is_password = False
        if self.text_arg:
            # Fail closed: a check that cannot run masks the text.
            probe = self._password_probe()
            is_password = True if probe is None else (yield probe) is not False
        self.record_input(is_password=is_password)
        if not self.content:
            return
        page = _owner_page(self.impl)
        if page is not None and not page.is_closed():
            self.span.set_attribute(ATTR_PAGE_URL, _SCOPE.mask(page.url))
            if self.method in NAVIGATION_METHODS:
                title = yield partial(_internal, page, page.title)
                if isinstance(title, str) and title:
                    self.span.set_attribute(ATTR_PAGE_TITLE, title)
        output = call_result(result, screenshot=self._screenshot_part)
        if page is not None and self._wants_page_screenshot(page):
            image = yield partial(
                _internal,
                page,
                partial(
                    page.screenshot,
                    type="jpeg",
                    quality=SCREENSHOT_JPEG_QUALITY,
                    timeout=_PROBE_TIMEOUT_MS,
                ),
            )
            if isinstance(image, bytes):
                output = {"result": output} if output is not None else {}
                output["screenshot"] = self._screenshot_part(image)
        if output is not None:
            self.span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(output))

    def _wants_page_screenshot(self, page: Any) -> bool:
        mode = self.runtime.screenshot_on
        if not self.runtime.capture_screenshots or mode == SCREENSHOT_NEVER:
            return False
        if page.is_closed() or self.method == SCREENSHOT_METHOD:
            return False
        methods = NAVIGATION_METHODS if mode == SCREENSHOT_ON_NAVIGATION else ACTION_METHODS
        return self.method in methods

    def _screenshot_part(self, image: bytes) -> Optional[dict[str, Any]]:
        if not self.runtime.capture_screenshots:
            return None
        media = "jpeg" if image[:3] == b"\xff\xd8\xff" else "png"
        url = f"data:image/{media};base64,{base64.b64encode(image).decode()}"
        # A data-URI image part: Respan's ingest lifts it into media storage.
        return {"type": "image_url", "image_url": {"url": compact_image_data_url(url)}}

    def _password_probe(self) -> Optional[Callable[[], Any]]:
        impl = owner = self.impl
        selector = self.arguments.get("selector")
        cls = type(impl).__name__
        if cls == "Locator":
            owner = impl._frame  # a Locator is not a channel owner
            evaluate = partial(owner.eval_on_selector_all, impl._selector, _IS_PASSWORD_ANY)
        elif cls == "Page" and isinstance(selector, str):
            evaluate = partial(impl._main_frame.eval_on_selector_all, selector, _IS_PASSWORD_ANY)
        elif cls == "Frame" and isinstance(selector, str):
            evaluate = partial(impl.eval_on_selector_all, selector, _IS_PASSWORD_ANY)
        elif cls == "ElementHandle":
            evaluate = partial(impl.evaluate, _IS_PASSWORD_ONE)
        elif cls == "Keyboard" and (page := _owner_page(impl)) is not None:
            evaluate = partial(page._main_frame.evaluate, _IS_PASSWORD_FOCUSED)
        else:
            return None
        return partial(_internal, owner, evaluate)


def _drive_sync(steps: Generator[Any, Any, None], public: Any) -> None:
    """Run a _Call.after generator in the sync API (probes go through ``_sync``)."""
    try:
        factory = next(steps)
        while True:
            try:
                value = public._sync(factory())
            except Exception:
                logger.debug("playwright probe failed", exc_info=True)
                value = _FAILED
            factory = steps.send(value)
    except StopIteration:
        pass
    except Exception:
        logger.debug("playwright telemetry failed open", exc_info=True)


async def _drive_async(steps: Generator[Any, Any, None]) -> None:
    try:
        factory = next(steps)
        while True:
            try:
                value = await factory()
            except Exception:
                logger.debug("playwright probe failed", exc_info=True)
                value = _FAILED
            factory = steps.send(value)
    except StopIteration:
        pass
    except Exception:
        logger.debug("playwright telemetry failed open", exc_info=True)


async def _await_in(context: contextvars.Context, coroutine: Any) -> Any:
    return await asyncio.get_running_loop().create_task(coroutine, context=context)


class _Runtime:
    def __init__(
        self, *, capture_content: bool, capture_screenshots: bool, screenshot_on: str
    ) -> None:
        self.capture_content = capture_content
        self.capture_screenshots = capture_screenshots
        self.screenshot_on = screenshot_on
        self.active = True
        self.count = 1
        self.patcher = MethodPatcher()

    @property
    def tracer(self) -> trace.Tracer:
        # Resolved per call so a provider installed after activation is honoured.
        return trace.get_tracer(TRACER_NAME)

    def content_allowed(self) -> bool:
        return self.capture_content and is_content_tracing_enabled()

    def span(self, name: str) -> Any:
        return agent_span(
            self.tracer,
            name,
            LOG_TYPE_TOOL,
            scope=_SCOPE,
            content_allowed=self.content_allowed,
        )

    def traced(self) -> bool:
        return self.active and _has_parent_span()

    @contextmanager
    def call(
        self, api_name: str, method: str, public: Any, original: Any, args: tuple, kwargs: dict
    ) -> Iterator[_Call]:
        caller = _CALLER_CONTEXT.set(contextvars.copy_context())
        try:
            with self.span(api_name) as span:
                span.set_attribute(ATTR_API_NAME, api_name)
                record = _Call(self, span, method, public, original, args, kwargs)
                try:
                    yield record
                except BaseException:
                    # Fail closed: a failed call never shows the text it was typing.
                    safely(partial(record.record_input, is_password=True))
                    raise
        finally:
            _CALLER_CONTEXT.reset(caller)

    # -- public API calls ----------------------------------------------------

    def wrap_async(self, api_name: str, method: str, original: Any) -> Any:
        runtime = self

        @wraps(original)
        async def call(public: Any, *args: Any, **kwargs: Any) -> Any:
            if not runtime.traced():
                return await original(public, *args, **kwargs)
            with runtime.call(api_name, method, public, original, args, kwargs) as record:
                result = await original(public, *args, **kwargs)
                await _drive_async(record.after(result))
                return result

        return call

    def wrap_sync(self, api_name: str, method: str, original: Any) -> Any:
        runtime = self

        @wraps(original)
        def call(public: Any, *args: Any, **kwargs: Any) -> Any:
            if not runtime.traced():
                return original(public, *args, **kwargs)
            with runtime.call(api_name, method, public, original, args, kwargs) as record:
                result = original(public, *args, **kwargs)
                _drive_sync(record.after(result), public)
                return result

        return call

    # -- callbacks -----------------------------------------------------------

    def wrap_handler_factory(self, original: Any) -> Any:
        runtime = self

        @wraps(original)
        def _wrap_handler(public: Any, handler: Any) -> Any:
            mapped = original(public, handler)
            if not runtime.active or not callable(mapped):
                return mapped
            # Playwright caches the mapped callback per handler so remove_listener
            # finds it again; cache ours on it for the same reason.
            existing = getattr(mapped, _HANDLER_ATTR, None)
            if existing is not None:
                return existing
            # Callbacks belong to the span the caller was in, not to the
            # (already finished) span of the call that registered them.
            registered = _CALLER_CONTEXT.get() or contextvars.copy_context()

            def in_registration_context(*args: Any) -> Any:
                context = registered.copy()
                result = context.run(mapped, *args)
                if inspect.iscoroutine(result):
                    return _await_in(context, result)
                return result

            try:
                setattr(mapped, _HANDLER_ATTR, in_registration_context)
            except (AttributeError, TypeError):
                return mapped
            return in_registration_context

        return _wrap_handler


def _public_io_methods(async_module: Any) -> Iterator[tuple[str, str]]:
    """(class name, method) for every public method that round-trips to the browser."""
    base = importlib.import_module(ASYNC_BASE_MODULE).AsyncBase
    for class_name, cls in vars(async_module).items():
        if not (inspect.isclass(cls) and issubclass(cls, base)):
            continue
        if cls.__module__ != async_module.__name__:
            continue
        for method, function in vars(cls).items():
            if not method.startswith("_") and inspect.iscoroutinefunction(function):
                yield class_name, method


class PlaywrightInstrumentor:
    """Trace Playwright API calls (sync and async) under the active span."""

    name = INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        capture_content: bool = True,
        capture_screenshots: bool = True,
        screenshot_on: str = SCREENSHOT_NEVER,
    ) -> None:
        if screenshot_on not in SCREENSHOT_MODES:
            raise ValueError(f"screenshot_on must be one of {sorted(SCREENSHOT_MODES)}")
        self._settings = (capture_content, capture_screenshots, screenshot_on)
        self._runtime: _Runtime | None = None

    def activate(self) -> None:
        global _RUNTIME
        with _LOCK:
            if self._runtime is not None:
                return
            tracer = getattr(RespanTracer, "_instance", None)
            if tracer is not None and not getattr(tracer, "is_enabled", True):
                return
            if _RUNTIME is not None:
                current = (
                    _RUNTIME.capture_content,
                    _RUNTIME.capture_screenshots,
                    _RUNTIME.screenshot_on,
                )
                if current != self._settings:
                    raise ValueError(
                        "Playwright instrumentation is already active with different settings"
                    )
                _RUNTIME.count += 1
                self._runtime = _RUNTIME
                return
            try:
                async_api = importlib.import_module(ASYNC_API_MODULE)
                sync_api = importlib.import_module(SYNC_API_MODULE)
                async_base = importlib.import_module(ASYNC_BASE_MODULE)
                sync_base = importlib.import_module(SYNC_BASE_MODULE)
            except ImportError:
                logger.debug("playwright is not installed; instrumentation skipped")
                return
            capture_content, capture_screenshots, screenshot_on = self._settings
            runtime = _Runtime(
                capture_content=capture_content,
                capture_screenshots=capture_screenshots,
                screenshot_on=screenshot_on,
            )
            try:
                for class_name, method in _public_io_methods(async_api):
                    api_name = f"{class_name}.{method}"
                    runtime.patcher.patch(
                        getattr(async_api, class_name),
                        method,
                        partial(runtime.wrap_async, api_name, method),
                    )
                    if sync_class := getattr(sync_api, class_name, None):
                        runtime.patcher.patch(
                            sync_class, method, partial(runtime.wrap_sync, api_name, method)
                        )
                for base in (async_base.AsyncBase, sync_base.SyncBase):
                    runtime.patcher.patch(base, WRAP_HANDLER, runtime.wrap_handler_factory)
            except Exception:
                runtime.patcher.restore()
                raise
            _RUNTIME = runtime
            self._runtime = runtime

    def deactivate(self) -> None:
        global _RUNTIME
        with _LOCK:
            runtime, self._runtime = self._runtime, None
            if runtime is None:
                return
            runtime.count -= 1
            if runtime.count:
                return
            runtime.active = False
            runtime.patcher.restore()
            if _RUNTIME is runtime:
                _RUNTIME = None
