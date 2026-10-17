"""Kernel browser infrastructure as Respan spans: one ``tool`` span per API call.

Kernel is not an agent: it hosts the browsers (and their VMs) that an agent
drives. The agent's own spans come from its instrumentation (browser-use,
Playwright, a computer-use loop's LLM SDK); this package adds what happens on
the Kernel side. Seams (Kernel's Stainless-generated resources, patched from
outside, sync and async):

* ``browsers.create / retrieve / update / list / delete_by_id / load_extensions
  / curl / repl``  -> ``tool`` span (``create`` records the session id and live view)
* ``browsers.computer.*`` (screenshot, mouse, keyboard, clipboard, batch)
  -> ``tool`` span; ``capture_screenshot`` output is a compact JPEG
* ``browsers.process.*`` (exec, spawn, status, stdin, stdout_stream, kill)
  -> ``tool`` span; stdout/stderr decoded, env values masked
* ``browsers.playwright.execute`` -> ``tool`` span (code in, result out)
* ``browsers.replays.*`` -> ``tool`` span with the replay id and view URL
* ``KernelApp.action`` handlers -> ``agent`` span ``kernel.action.<name>``,
  flushed when the handler returns (the Kernel runtime may stop right after)

Grouping: a Kernel browser session is the default thread / session / trace
group of its spans. Inside an action, the first browser the action creates
groups the whole action run (its invocation id when it creates none), so one
trace has one thread. Values the customer set always win.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import re
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.llm_logging import LOG_TYPE_AGENT, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import (
    RESPAN_SESSION_ID,
    RESPAN_SPAN_ATTRIBUTES_MAP,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing import is_content_tracing_enabled
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import propagate_attributes, read_propagated_attributes
from respan_tracing.utils.agent_instrumentation import (
    AgentScope,
    MethodPatcher,
    agent_span,
    json_string,
    safely,
    secret_replacements,
    set_error,
)

from respan_instrumentation_kernel._constants import (
    API_KEY_SECRET_NAME,
    APP_MODULE,
    ATTR_ACTION_NAME,
    ATTR_API_NAME,
    ATTR_APP_NAME,
    ATTR_EXIT_CODE,
    ATTR_INVOCATION_ID,
    ATTR_LIVE_VIEW_URL,
    ATTR_PROCESS_ID,
    ATTR_REPLAY_ID,
    ATTR_REPLAY_VIEW_URL,
    ATTR_SESSION_ID,
    ENV_ARG,
    GROUP_BROWSERS,
    GROUP_PLAYWRIGHT,
    GROUP_PROCESS,
    GROUP_REPLAYS,
    INSTRUMENTATION_NAME,
    MASKED_TEXT,
    METHOD_BATCH,
    METHOD_CAPTURE_SCREENSHOT,
    METHOD_CREATE,
    METHOD_EXECUTE,
    METHOD_START,
    METHOD_TYPE_TEXT,
    MIN_LEARNED_SECRET_LENGTH,
    PASSWORD_PROBE_CODE,
    PASSWORD_PROBE_TIMEOUT_SEC,
    PLAYWRIGHT_EXECUTION_ERROR,
    PLAYWRIGHT_MODULE,
    RESOURCES,
    SESSION_ARG,
    SPAN_ACTION_PREFIX,
    SPAN_PREFIX,
    STDIN_ARG,
    TOKEN_QUERY_PARAMS,
    TOOL_DETAIL_PREFIX,
    TRACER_NAME,
    TYPED_TEXT_SECRET_NAME,
)
from respan_instrumentation_kernel._serialization import (
    call_arguments,
    image_part,
    response_bytes,
    response_data,
)

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_RUNTIME: Optional[_Runtime] = None

_TOKEN_IN_URL = re.compile(r"([?&](?:%s)=)[^&#\s\"'\\]+" % "|".join(TOKEN_QUERY_PARAMS))


class _Scope(AgentScope):
    """Secrets the customer declared, plus ones each call brings.

    Per call: the client's API key, process environment values and text typed
    into a password field. Always: the bearer tokens in the URLs Kernel returns
    (``cdp_ws_url?jwt=...``), which grant control of the browser.
    """

    @contextmanager
    def session(self, session_id: Optional[str]) -> Iterator[None]:
        token = self._session.set(session_id)
        try:
            yield
        finally:
            self._session.reset(token)

    @contextmanager
    def adding(self, secrets: Mapping[str, Any]) -> Iterator[None]:
        token = self._secrets.set(self._merged(secrets))
        try:
            yield
        finally:
            self._secrets.reset(token)

    def add(self, secrets: Mapping[str, Any]) -> None:
        """Mask these too until the enclosing ``adding`` ends."""
        self._secrets.set(self._merged(secrets))

    def _merged(self, secrets: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
        merged = dict(self._secrets.get())
        merged.update(secret_replacements(secrets))
        return tuple(sorted(merged.items(), key=lambda pair: -len(pair[0])))

    def mask(self, text: str) -> str:
        return _TOKEN_IN_URL.sub(r"\1" + MASKED_TEXT, super().mask(text))

    def mask_span(self, span: Any) -> None:
        for key, value in list((span.attributes or {}).items()):
            if isinstance(value, str) and (masked := self.mask(value)) != value:
                span.set_attribute(key, masked)

    @property
    def has_secrets(self) -> bool:
        # URL tokens can be anywhere, so error events are always masked.
        return True


_SCOPE = _Scope("respan_kernel")


@dataclass
class _Run:
    """One KernelApp action invocation."""

    span: Any
    session_id: Optional[str] = None
    # Propagates the session to every span the action's code starts after the
    # browser exists (browser-use, Playwright, LLM calls), so they share its thread.
    propagation: Any = None


_RUN: ContextVar[Optional[_Run]] = ContextVar("respan_kernel_run", default=None)
# propagate_attributes() keyword for each grouping attribute.
_GROUPING_KEYWORDS = {
    attribute: keyword
    for keyword, attribute in RESPAN_SPAN_ATTRIBUTES_MAP.items()
    if attribute in (RESPAN_THREADS_ID, RESPAN_SESSION_ID, RESPAN_TRACE_GROUP_ID)
}


@contextmanager
def secrets(sensitive_data: Mapping[str, Any]) -> Iterator[None]:
    """Mask these values (``{name: value}``) out of every Kernel span.

    Each value is exported as ``<secret>name</secret>``, wherever it shows up:
    typed text, process commands and output, Playwright code, error messages.
    """
    with _SCOPE.run(None, sensitive_data):
        yield


def _learnable(secrets: Mapping[str, Any]) -> dict[str, str]:
    """Values long enough to mask everywhere without mangling unrelated text."""
    return {
        name: value
        for name, value in secrets.items()
        if isinstance(value, str) and len(value) >= MIN_LEARNED_SECRET_LENGTH
    }


class _Call:
    """What one Kernel API call records: input, grouping, output."""

    def __init__(
        self,
        runtime: _Runtime,
        group: str,
        method: str,
        resource: Any,
        original: Callable[..., Any],
        args: tuple,
        kwargs: dict,
    ) -> None:
        self.runtime = runtime
        self.group = group
        self.method = method
        self.api_name = f"{group}.{method}"
        self.client = getattr(resource, "_client", None)
        self.content = runtime.content_allowed()
        self.arguments = call_arguments(original, args, kwargs)
        session = self.arguments.get(SESSION_ARG)
        self.session_id = session if isinstance(session, str) else None
        # Fail closed: with probing on, typed text is masked unless the probe
        # shows it did not go to a password field.
        self.mask_typed = runtime.probe_password_fields and method in (
            METHOD_TYPE_TEXT,
            METHOD_BATCH,
        )
        self.input_recorded = False
        self.span: Any = None

    @property
    def wants_probe(self) -> bool:
        return (
            self.mask_typed
            and self.method == METHOD_TYPE_TEXT
            and self.session_id is not None
            and self.client is not None
        )

    def on_probe(self, result: Any) -> None:
        if getattr(result, "success", False) and getattr(result, "result", None) is False:
            self.mask_typed = False

    def call_secrets(self) -> dict[str, Any]:
        secrets: dict[str, Any] = {}
        if isinstance(api_key := getattr(self.client, "api_key", None), str):
            secrets[API_KEY_SECRET_NAME] = api_key
        env = self.arguments.get(ENV_ARG)
        if isinstance(env, Mapping):
            secrets.update(_learnable(env))
        return secrets

    @contextmanager
    def opened(self) -> Iterator[_Call]:
        run = _RUN.get()
        session = (run.session_id if run else None) or self.session_id
        with (
            _SCOPE.adding(self.call_secrets()),
            _SCOPE.session(session),
            self.runtime.span(
                SPAN_PREFIX + self.api_name,
                LOG_TYPE_TOOL,
                detail=f"{TOOL_DETAIL_PREFIX}{self.group}_{self.method}",
            ) as span,
        ):
            self.span = span
            span.set_attribute(ATTR_API_NAME, self.api_name)
            if self.session_id:
                span.set_attribute(ATTR_SESSION_ID, self.session_id)
            try:
                yield self
            except BaseException:
                safely(self.record_input)
                raise

    # -- input ---------------------------------------------------------------

    def record_input(self) -> None:
        if self.input_recorded or not self.content:
            return
        self.input_recorded = True
        arguments = {k: v for k, v in self.arguments.items() if k != SESSION_ARG}
        if isinstance(env := arguments.get(ENV_ARG), Mapping):
            arguments[ENV_ARG] = {name: MASKED_TEXT for name in env}
        if isinstance(data := arguments.get(STDIN_ARG), str):
            arguments[STDIN_ARG] = {"type": "bytes", "length": len(data) * 3 // 4}
        if self.mask_typed:
            arguments = self._mask_typed_text(arguments)
        if arguments:
            self.span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string(arguments))

    def _mask_typed_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        typed: list[str] = []
        if self.method == METHOD_TYPE_TEXT and isinstance(arguments.get("text"), str):
            typed.append(arguments["text"])
            arguments["text"] = MASKED_TEXT
        elif self.method == METHOD_BATCH:
            # A batch can move focus between actions: mask all of its typing.
            actions = []
            for action in arguments.get("actions") or ():
                payload = action.get(METHOD_TYPE_TEXT) if isinstance(action, Mapping) else None
                if isinstance(payload, Mapping) and isinstance(payload.get("text"), str):
                    typed.append(payload["text"])
                    action = {**action, METHOD_TYPE_TEXT: {**payload, "text": MASKED_TEXT}}
                actions.append(action)
            arguments["actions"] = actions
        # The same text may come back later in this call (an error, an echo).
        for text in typed:
            _SCOPE.add(_learnable({TYPED_TEXT_SECRET_NAME: text}))
        return arguments

    # -- output --------------------------------------------------------------

    def finish(self, result: Any) -> None:
        self.record_input()
        span = self.span
        if self.group == GROUP_BROWSERS and self.method == METHOD_CREATE:
            self._on_browser(result)
        if self.group == GROUP_PROCESS:
            if isinstance(process_id := getattr(result, "process_id", None), str):
                span.set_attribute(ATTR_PROCESS_ID, process_id)
            elif isinstance(process_id := self.arguments.get("process_id"), str):
                span.set_attribute(ATTR_PROCESS_ID, process_id)
            if isinstance(exit_code := getattr(result, "exit_code", None), int):
                span.set_attribute(ATTR_EXIT_CODE, exit_code)
        if self.group == GROUP_REPLAYS and self.method == METHOD_START:
            self._on_replay(result)
        if self.group == GROUP_PLAYWRIGHT and self.method == METHOD_EXECUTE:
            if getattr(result, "success", True) is False:
                set_error(
                    span,
                    PLAYWRIGHT_EXECUTION_ERROR,
                    getattr(result, "error", None),
                    with_message=self.content,
                    mask=_SCOPE.mask,
                )
        if self.content and (output := self._output(result)) is not None:
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(output))

    def _output(self, result: Any) -> Any:
        if result is None:
            return None
        data = response_bytes(result)
        if data is None:
            return response_data(result)
        if self.method == METHOD_CAPTURE_SCREENSHOT and self.runtime.capture_screenshots:
            return {"screenshot": image_part(data)}
        return {"type": "bytes", "length": len(data)}

    def _on_browser(self, browser: Any) -> None:
        session = getattr(browser, "session_id", None)
        if not isinstance(session, str):
            return
        self.span.set_attribute(ATTR_SESSION_ID, session)
        run = _RUN.get()
        if run is not None and run.session_id is None:
            # The action's first browser groups the whole action run.
            run.session_id = session
            with _SCOPE.session(session):
                _SCOPE.apply_grouping(run.span)
            customer = read_propagated_attributes()
            missing = {k: session for a, k in _GROUPING_KEYWORDS.items() if not customer.get(a)}
            if missing:
                # Left open until the action returns (it runs in the action's context).
                run.propagation = propagate_attributes(**missing)
                run.propagation.__enter__()
        with _SCOPE.session(run.session_id if run else session):
            _SCOPE.apply_grouping(self.span)
        if self.content and isinstance(url := getattr(browser, "browser_live_view_url", None), str):
            self.span.set_attribute(ATTR_LIVE_VIEW_URL, url)

    def _on_replay(self, replay: Any) -> None:
        run = _RUN.get()
        spans = [self.span] + ([run.span] if run is not None else [])
        replay_id = getattr(replay, "replay_id", None)
        view_url = getattr(replay, "replay_view_url", None) if self.content else None
        for span in spans:
            if isinstance(replay_id, str):
                span.set_attribute(ATTR_REPLAY_ID, replay_id)
            if isinstance(view_url, str):
                span.set_attribute(ATTR_REPLAY_VIEW_URL, _SCOPE.mask(view_url))


class _Runtime:
    def __init__(
        self,
        *,
        capture_content: bool,
        capture_screenshots: bool,
        probe_password_fields: bool,
    ) -> None:
        self.capture_content = capture_content
        self.capture_screenshots = capture_screenshots
        self.probe_password_fields = probe_password_fields
        self.active = True
        self.count = 1
        self.patcher = MethodPatcher()
        # Unpatched playwright.execute (sync, async) for the password probe.
        self.execute: dict[bool, tuple[type, Callable[..., Any]]] = {}

    @property
    def tracer(self) -> trace.Tracer:
        # Resolved per call so a provider installed after activation is honoured.
        return trace.get_tracer(TRACER_NAME)

    @property
    def settings(self) -> tuple[bool, bool, bool]:
        return (self.capture_content, self.capture_screenshots, self.probe_password_fields)

    def content_allowed(self) -> bool:
        return self.capture_content and is_content_tracing_enabled()

    def span(self, name: str, log_type: str, *, detail: str) -> Any:
        return agent_span(
            self.tracer,
            name,
            log_type,
            scope=_SCOPE,
            content_allowed=self.content_allowed,
            detail=detail,
        )

    @staticmethod
    def flush() -> None:
        """Send the spans now: a Kernel action's process may stop when it returns."""
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is not None:
            safely(tracer.flush)

    # -- API calls -----------------------------------------------------------

    def _probe_sync(self, record: _Call) -> None:
        if (probe := self.execute.get(False)) is None:
            return  # this Kernel has no playwright.execute: the text stays masked
        owner, execute = probe
        try:
            result = execute(
                owner(record.client),
                record.session_id,
                code=PASSWORD_PROBE_CODE,
                timeout_sec=PASSWORD_PROBE_TIMEOUT_SEC,
            )
        except Exception:  # noqa: BLE001 - a failed probe keeps the text masked
            logger.debug("kernel password probe failed", exc_info=True)
            return
        record.on_probe(result)

    async def _probe_async(self, record: _Call) -> None:
        if (probe := self.execute.get(True)) is None:
            return  # this Kernel has no playwright.execute: the text stays masked
        owner, execute = probe
        try:
            result = await execute(
                owner(record.client),
                record.session_id,
                code=PASSWORD_PROBE_CODE,
                timeout_sec=PASSWORD_PROBE_TIMEOUT_SEC,
            )
        except Exception:  # noqa: BLE001 - a failed probe keeps the text masked
            logger.debug("kernel password probe failed", exc_info=True)
            return
        record.on_probe(result)

    def wrap_sync(self, group: str, method: str, original: Any) -> Any:
        runtime = self

        @wraps(original)
        def call(resource: Any, *args: Any, **kwargs: Any) -> Any:
            if not runtime.active:
                return original(resource, *args, **kwargs)
            record = _Call(runtime, group, method, resource, original, args, kwargs)
            with record.opened():
                if record.wants_probe:
                    runtime._probe_sync(record)
                result = original(resource, *args, **kwargs)
                safely(partial(record.finish, result))
                return result

        return call

    def wrap_async(self, group: str, method: str, original: Any) -> Any:
        runtime = self

        @wraps(original)
        async def call(resource: Any, *args: Any, **kwargs: Any) -> Any:
            if not runtime.active:
                return await original(resource, *args, **kwargs)
            record = _Call(runtime, group, method, resource, original, args, kwargs)
            with record.opened():
                if record.wants_probe:
                    await runtime._probe_async(record)
                result = await original(resource, *args, **kwargs)
                safely(partial(record.finish, result))
                return result

        return call

    # -- KernelApp actions ---------------------------------------------------

    @contextmanager
    def action(self, app_name: Any, name: str, args: tuple) -> Iterator[Callable[[Any], None]]:
        context = args[0] if args else None
        invocation_id = getattr(context, "invocation_id", None)
        run = _Run(span=None)
        with self.span(SPAN_ACTION_PREFIX + name, LOG_TYPE_AGENT, detail=name) as span:
            run.span = span
            token = _RUN.set(run)
            try:
                span.set_attribute(ATTR_ACTION_NAME, name)
                if isinstance(app_name, str):
                    span.set_attribute(ATTR_APP_NAME, app_name)
                if isinstance(invocation_id, str):
                    span.set_attribute(ATTR_INVOCATION_ID, invocation_id)
                content = self.content_allowed()
                if content and len(args) > 1:
                    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string(args[1]))

                def record_output(result: Any) -> None:
                    if content and result is not None:
                        span.set_attribute(
                            SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(result)
                        )

                yield record_output
            finally:
                _RUN.reset(token)
                if run.propagation is not None:
                    safely(partial(run.propagation.__exit__, None, None, None))
                # An action that created no browser is grouped by its invocation.
                fallback = run.session_id or (
                    invocation_id if isinstance(invocation_id, str) else None
                )
                with _SCOPE.session(fallback):
                    safely(partial(_SCOPE.apply_grouping, span))

    def wrap_action(self, app_name: Any, name: str, handler: Callable[..., Any]) -> Any:
        runtime = self

        if inspect.iscoroutinefunction(handler):

            @wraps(handler)
            async def traced_async(*args: Any, **kwargs: Any) -> Any:
                if not runtime.active:
                    return await handler(*args, **kwargs)
                try:
                    with runtime.action(app_name, name, args) as record_output:
                        result = await handler(*args, **kwargs)
                        safely(partial(record_output, result))
                        return result
                finally:
                    runtime.flush()

            return traced_async

        @wraps(handler)
        def traced(*args: Any, **kwargs: Any) -> Any:
            if not runtime.active:
                return handler(*args, **kwargs)
            try:
                with runtime.action(app_name, name, args) as record_output:
                    result = handler(*args, **kwargs)
                    safely(partial(record_output, result))
                    return result
            finally:
                runtime.flush()

        return traced

    def wrap_action_decorator(self, original: Any) -> Any:
        runtime = self

        @wraps(original)
        def action(app: Any, name: str, *args: Any, **kwargs: Any) -> Any:
            register = original(app, name, *args, **kwargs)

            def traced_register(handler: Callable[..., Any]) -> Any:
                return register(runtime.wrap_action(getattr(app, "name", None), name, handler))

            return traced_register

        return action


def _import_first(candidates: tuple[str, ...]) -> Any:
    """The first of these modules that imports, or None (a resource this Kernel lacks)."""
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    logger.debug("kernel has none of %s; those calls are not traced", candidates)
    return None


class KernelInstrumentor:
    """Trace Kernel browser, computer, process, Playwright and replay calls, and app actions."""

    name = INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        capture_content: bool = True,
        capture_screenshots: bool = True,
        probe_password_fields: bool = False,
    ) -> None:
        self._settings = (capture_content, capture_screenshots, probe_password_fields)
        self._runtime: Optional[_Runtime] = None

    def activate(self) -> None:
        global _RUNTIME
        with _LOCK:
            if self._runtime is not None:
                return
            tracer = getattr(RespanTracer, "_instance", None)
            if tracer is not None and not getattr(tracer, "is_enabled", True):
                return
            if _RUNTIME is not None:
                if _RUNTIME.settings != self._settings:
                    raise ValueError(
                        "Kernel instrumentation is already active with different settings"
                    )
                _RUNTIME.count += 1
                self._runtime = _RUNTIME
                return
            try:
                app_module = importlib.import_module(APP_MODULE)
            except ImportError:
                logger.debug("kernel is not installed; instrumentation skipped")
                return
            capture_content, capture_screenshots, probe_password_fields = self._settings
            runtime = _Runtime(
                capture_content=capture_content,
                capture_screenshots=capture_screenshots,
                probe_password_fields=probe_password_fields,
            )
            try:
                for candidates, sync_name, async_name, group, methods in RESOURCES:
                    module = _import_first(candidates)
                    for class_name, is_async in ((sync_name, False), (async_name, True)):
                        owner = getattr(module, class_name, None)
                        if owner is None:
                            continue
                        if candidates == PLAYWRIGHT_MODULE:
                            runtime.execute[is_async] = (owner, owner.__dict__[METHOD_EXECUTE])
                        wrap = runtime.wrap_async if is_async else runtime.wrap_sync
                        for method in methods:
                            runtime.patcher.patch(owner, method, partial(wrap, group, method))
                runtime.patcher.patch(app_module.KernelApp, "action", runtime.wrap_action_decorator)
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
