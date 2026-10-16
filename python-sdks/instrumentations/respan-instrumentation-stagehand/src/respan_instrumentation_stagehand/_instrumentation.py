"""Respan spans for Stagehand v4 (Python): session -> operation -> LLM call / browser action.

Stagehand v4 runs its browser logic inside a Chrome extension that the SDK
drives over JSON-RPC; the Python SDK is a peer of the TypeScript one and embeds
the same extension. This instrumentation wraps the SDK's public seams from
outside:

* ``Stagehand.create`` ... ``close()``     -> ``agent`` span (the session)
* ``act`` / ``observe`` / ``extract``      -> ``task`` span per operation
* the client model callable (``model=``)   -> ``chat`` span per LLM call
* ``Page`` / ``Locator`` / ``BrowserContext`` actions -> ``tool`` spans

The Python SDK opens no OpenTelemetry spans of its own (it only propagates
``traceparent`` into the extension), so there is nothing native to filter out.
"""

from __future__ import annotations

import asyncio
import contextvars
import importlib
import inspect
import json
import logging
import threading
import uuid
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import partial, wraps
from typing import Any, Optional

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.semconv_ai import (
    SpanAttributes,
    TraceloopSpanKindValues,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_INTERNAL_SPAN_NAME_DETAIL,
    RESPAN_LOG_TYPE,
)
from respan_tracing import is_content_tracing_enabled
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.agent_instrumentation import (
    AgentScope,
    MethodPatcher,
    Replacements,
    agent_span,
    record_error,
    safely,
    secret_replacements,
    set_error,
    suppress_provider_llm_spans,
)

from respan_instrumentation_stagehand._constants import (
    ACT,
    ATTR_ACTION_ID,
    ATTR_CACHE_STATUS,
    ATTR_PAGE_URL,
    ATTR_SESSION_ID,
    ATTR_USAGE_SOURCE,
    CLIENT_MODEL,
    CLIENT_MODEL_NAME_ATTR,
    CONTEXT_ACTIONS,
    CONTEXT_MODULE,
    CONTEXT_OWNER,
    EXTRACT,
    INSTRUMENTATION_NAME,
    LLM_NAME_DETAIL,
    LOCATOR_ACTIONS,
    LOCATOR_MODULE,
    LOCATOR_OWNER,
    MIN_SECRET_LENGTH,
    MODELS_MODULE,
    NAVIGATION_ACTIONS,
    OPERATION_PREFIX,
    OPERATIONS,
    PAGE_ACTIONS,
    PAGE_MODULE,
    PAGE_OWNER,
    REDACTED,
    SCREENSHOT_DEADLINE_SECONDS,
    SCREENSHOT_OPTIONS,
    SENSITIVE_FIELD,
    SENSITIVE_SELECTOR,
    SESSION_NAME_DETAIL,
    SPAN_LLM,
    SPAN_SESSION,
    STAGEHAND_MODULE,
    TRACER_NAME,
    TYPED_TEXT,
    USAGE_FIELDS,
    USAGE_SOURCE_OPERATION,
    XPATH_PREFIX,
)
from respan_instrumentation_stagehand._serialization import (
    chat_messages,
    completion,
    content_text,
    functions,
    image_bytes_part,
    json_string,
    jsonable,
)

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_RUNTIME: _Runtime | None = None


class _Session:
    """One Stagehand instance, from ``Stagehand.create`` to ``close()``."""

    def __init__(self, session_id: str, span: Any, model_name: Optional[str]) -> None:
        self.id = session_id
        self.span = span
        self.model_name = model_name
        self.context: contextvars.Context = contextvars.copy_context()
        # Secret value -> "%name%", from variables passed to act / observe.
        self.replacements: Replacements = ()
        self.ops: list[_Operation] = []
        self.operations = 0
        self.failed = 0
        self.usage: dict[str, int] = {}
        self.closed = False

    def add_variables(self, variables: Any) -> None:
        if not isinstance(variables, Mapping) or not variables:
            return
        pairs = dict(self.replacements)
        for name, raw in variables.items():
            value = raw.get("value") if isinstance(raw, Mapping) else getattr(raw, "value", raw)
            # Booleans and tiny values would mask unrelated text.
            if value is None or isinstance(value, bool) or len(str(value)) < MIN_SECRET_LENGTH:
                continue
            # The kit's encoded variants (a value typed into a URL appears encoded),
            # with Stagehand's own placeholder syntax.
            for secret, _ in secret_replacements({name: str(value)}):
                pairs[secret] = f"%{name}%"
        self.replacements = tuple(sorted(pairs.items(), key=lambda pair: -len(pair[0])))

    def find_operation(self, params: Mapping[str, Any]) -> Optional[_Operation]:
        """The operation an llm.generate call belongs to.

        The extension's callback carries no operation id or trace context, so
        with several operations running at once the prompt is matched against
        each operation's instruction.
        """
        if len(self.ops) <= 1:
            return self.ops[0] if self.ops else None
        prompt = json.dumps([params.get("system_prompt"), params.get("messages")])
        matches = [op for op in self.ops if op.needle and op.needle in prompt]
        return (matches or self.ops)[-1]


class _Operation:
    def __init__(self, kind: str, span: Any, instruction: Any) -> None:
        self.kind = kind
        self.span = span
        self.llm_calls = 0
        # Set once the operation's span is current: its LLM calls run in it.
        self.context: Optional[contextvars.Context] = None
        text = instruction if isinstance(instruction, str) else _field(instruction, "description")
        # The instruction as it appears inside a JSON-serialised prompt.
        self.needle = (
            json.dumps(text.strip())[1:-1] if isinstance(text, str) and text.strip() else None
        )


class _Scope(AgentScope):
    """AgentScope whose secrets are the current Stagehand session's variables."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._current: contextvars.ContextVar[Optional[_Session]] = contextvars.ContextVar(
            f"{name}_current", default=None
        )

    @contextmanager
    def session(self, session: _Session) -> Iterator[None]:
        token = self._current.set(session)
        try:
            with self.run(session.id):
                yield
        finally:
            self._current.reset(token)

    def mask(self, text: str) -> str:
        session = self._current.get()
        for secret, placeholder in session.replacements if session else ():
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
        session = self._current.get()
        return bool(session and session.replacements)


_SCOPE = _Scope("respan_stagehand")
# The operation running in this context (act / observe / extract).
_OPERATION: contextvars.ContextVar[Optional[_Operation]] = contextvars.ContextVar(
    "respan_stagehand_operation", default=None
)
# Set while the instrumentation itself, or a traced action, drives the browser.
_INTERNAL: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "respan_stagehand_internal", default=False
)
_SESSIONS: weakref.WeakKeyDictionary[Any, _Session] = weakref.WeakKeyDictionary()
# Pages, locators and contexts hold the session's RPC client, not the Stagehand.
_SESSIONS_BY_RPC: weakref.WeakKeyDictionary[Any, _Session] = weakref.WeakKeyDictionary()


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _non_negative_int(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


@contextmanager
def _parent(span: Any) -> Iterator[None]:
    """Make ``span`` the parent of spans started here (the caller's other context stays)."""
    token = context_api.attach(trace.set_span_in_context(span))
    try:
        yield
    finally:
        context_api.detach(token)


class _Runtime:
    def __init__(
        self,
        *,
        capture_content: bool,
        capture_screenshots: bool,
        client_model_name: Optional[str],
    ) -> None:
        self.capture_content = capture_content
        self.capture_screenshots = capture_screenshots
        self.client_model_name = client_model_name
        self.active = True
        self.count = 1
        self.patcher = MethodPatcher()
        self.page_class: Any = None
        self.page_ref_class: Any = None

    @property
    def tracer(self) -> trace.Tracer:
        # Resolved per call so a provider installed after activation is honoured.
        return trace.get_tracer(TRACER_NAME)

    def content_allowed(self) -> bool:
        return self.capture_content and is_content_tracing_enabled()

    def span(self, name: str, log_type: str, *, detail: Optional[str] = None) -> Any:
        return agent_span(
            self.tracer,
            name,
            log_type,
            scope=_SCOPE,
            content_allowed=self.content_allowed,
            detail=detail,
        )

    # -- session -------------------------------------------------------------

    def wrap_create(self, original: Any) -> Any:
        runtime = self
        function = original.__func__

        @wraps(function)
        async def create(cls: Any, *args: Any, **kwargs: Any) -> Any:
            if not runtime.active:
                return await function(cls, *args, **kwargs)
            browser = kwargs.get("browser")
            browser_session = getattr(browser, "session_id", None)
            session_id = (
                browser_session
                if isinstance(browser_session, str) and browser_session
                else str(uuid.uuid4())
            )
            model = kwargs.get("model")
            is_client_model = callable(model)
            if is_client_model:
                model_name = (
                    getattr(model, CLIENT_MODEL_NAME_ATTR, None) or runtime.client_model_name
                )
            else:
                model_name = model if isinstance(model, str) else None
            span = runtime.tracer.start_span(
                SPAN_SESSION,
                attributes={
                    RESPAN_LOG_TYPE: LOG_TYPE_AGENT,
                    SpanAttributes.TRACELOOP_ENTITY_NAME: SPAN_SESSION,
                    SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
                    RESPAN_INTERNAL_SPAN_NAME_DETAIL: SESSION_NAME_DETAIL,
                    ATTR_SESSION_ID: session_id,
                },
            )
            session = _Session(session_id, span, model_name)
            with _SCOPE.session(session):
                safely(partial(_SCOPE.apply_grouping, span))
            safely(partial(runtime.session_start, session, kwargs, is_client_model, browser))
            if is_client_model:
                kwargs = {**kwargs, "model": runtime.wrap_generate(session, model)}
            try:
                with _parent(span), _SCOPE.session(session):
                    # The context the session's LLM calls fall back to.
                    session.context = contextvars.copy_context()
                    stagehand = await function(cls, *args, **kwargs)
            except BaseException as error:
                session.closed = True
                runtime.end_session(session, error=error)
                raise
            safely(partial(runtime.register, session, stagehand))
            return stagehand

        return classmethod(create)

    def session_start(
        self, session: _Session, kwargs: Mapping[str, Any], is_client_model: bool, browser: Any
    ) -> None:
        settings: dict[str, Any] = {
            "model": (session.model_name or CLIENT_MODEL)
            if is_client_model
            else session.model_name,
            "browser": getattr(browser, "provider", None),
        }
        for key in ("self_heal", "dom_settle_timeout_ms", "cache"):
            if kwargs.get(key) is not None:
                settings[key] = jsonable(kwargs[key])
        if self.content_allowed() and isinstance(kwargs.get("system_prompt"), str):
            settings["system_prompt"] = kwargs["system_prompt"]
        session.span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string(settings))

    def register(self, session: _Session, stagehand: Any) -> None:
        _SESSIONS[stagehand] = session
        if (rpc := getattr(stagehand, "_rpc_client", None)) is not None:
            _SESSIONS_BY_RPC[rpc] = session

    def end_session(self, session: _Session, *, error: Optional[BaseException] = None) -> None:
        with _SCOPE.session(session):
            if error is not None:
                safely(
                    partial(
                        record_error,
                        session.span,
                        error,
                        with_message=self.content_allowed(),
                        scope=_SCOPE,
                    )
                )
            else:
                safely(
                    lambda: session.span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        json_string(
                            {
                                "operations": session.operations,
                                "failed_operations": session.failed,
                                "usage": session.usage,
                            }
                        ),
                    )
                )
            safely(partial(_SCOPE.mask_span, session.span))
        session.span.end()

    def wrap_close(self, original: Any) -> Any:
        runtime = self

        @wraps(original)
        async def close(stagehand: Any, *args: Any, **kwargs: Any) -> Any:
            session = _SESSIONS.get(stagehand)
            if session is None or session.closed:
                return await original(stagehand, *args, **kwargs)
            session.closed = True
            try:
                with _parent(session.span), _SCOPE.session(session):
                    result = await original(stagehand, *args, **kwargs)
            except BaseException as error:
                runtime.end_session(session, error=error)
                raise
            runtime.end_session(session)
            return result

        return close

    # -- act / observe / extract ----------------------------------------------

    def wrap_operation(self, kind: str, original: Any) -> Any:
        runtime = self

        @wraps(original)
        async def operation(stagehand: Any, *args: Any, **kwargs: Any) -> Any:
            session = _SESSIONS.get(stagehand)
            if not runtime.active or session is None or session.closed:
                return await original(stagehand, *args, **kwargs)
            safely(partial(session.add_variables, kwargs.get("variables")))
            instruction = args[0] if args else kwargs.get("instruction")
            with (
                _parent(session.span),
                _SCOPE.session(session),
                runtime.span(f"{OPERATION_PREFIX}{kind}", LOG_TYPE_TASK, detail=kind) as span,
            ):
                op = _Operation(kind, span, instruction)
                token = _OPERATION.set(op)
                op.context = contextvars.copy_context()
                session.ops.append(op)
                session.operations += 1
                safely(partial(runtime.operation_start, op, args, kwargs))
                try:
                    result = await original(stagehand, *args, **kwargs)
                except BaseException:
                    session.failed += 1
                    raise
                finally:
                    session.ops.remove(op)
                    _OPERATION.reset(token)
                try:
                    await runtime.operation_end(session, op, stagehand, result, kwargs)
                except Exception:
                    logger.debug("stagehand telemetry failed open", exc_info=True)
                return result

        return operation

    def operation_start(self, op: _Operation, args: tuple, kwargs: Mapping[str, Any]) -> None:
        options: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in ("page", "instruction", "schema") or value is None:
                continue
            if key == "variables":
                options[key] = list(value) if isinstance(value, Mapping) else []
            elif key == "model":
                options[key] = _field(value, "model_name") or CLIENT_MODEL
            elif key == "locator":
                options[key] = getattr(value, "selector", None)
            elif key == "ignore_locators":
                options[key] = [getattr(locator, "selector", None) for locator in value]
            else:
                options[key] = jsonable(value)
        if not self.content_allowed():
            op.span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string({"options": options})
            )
            return
        instruction = args[0] if args else kwargs.get("instruction")
        payload: dict[str, Any] = {"instruction": jsonable(instruction), "options": options}
        if op.kind == EXTRACT:
            schema = args[1] if len(args) > 1 else kwargs.get("schema")
            if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
                payload["schema"] = schema.model_json_schema()
        page = kwargs.get("page")
        if isinstance(url := _field(getattr(page, "ref", None), "url"), str):
            payload["url"] = url
            op.span.set_attribute(ATTR_PAGE_URL, url)
        op.span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string(payload))

    async def operation_end(
        self,
        session: _Session,
        op: _Operation,
        stagehand: Any,
        result: Any,
        kwargs: Mapping[str, Any],
    ) -> None:
        span = op.span
        metadata = jsonable(getattr(result, "metadata", None)) or {}
        usage = metadata.get("usage") or {}
        for key in USAGE_FIELDS:
            if (count := _non_negative_int(usage.get(key))) is not None:
                session.usage[key] = session.usage.get(key, 0) + count
        cache_status = (metadata.get("cache") or {}).get("status")
        if isinstance(cache_status, str):
            span.set_attribute(ATTR_CACHE_STATUS, cache_status)
        if isinstance(action_id := metadata.get("action_id"), str):
            span.set_attribute(ATTR_ACTION_ID, action_id)
        # A built-in model runs inside the browser extension, out of reach; the
        # operation's aggregate usage is all Stagehand reports. Record it as one
        # content-free chat span so tokens and cost still show up.
        if op.llm_calls == 0 and (
            (usage.get("input_tokens") or 0) > 0 or (usage.get("output_tokens") or 0) > 0
        ):
            self.usage_only_chat(session, usage, kwargs)
        data = getattr(result, "data", None)
        content = self.content_allowed()
        if op.kind == ACT and _field(data, "success") is False:
            session.failed += 1
            set_error(
                span,
                "act reported failure",
                _field(data, "message"),
                with_message=content,
                mask=_SCOPE.mask,
            )
        output: dict[str, Any] = {"metadata": metadata}
        if content:
            output["data"] = jsonable(data)
            shot = None
            if op.kind == ACT and self.capture_screenshots:
                shot = await self.screenshot(stagehand, kwargs.get("page"))
            if shot is not None:
                output["screenshot"] = shot
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(output))

    def usage_only_chat(
        self, session: _Session, usage: Mapping[str, Any], kwargs: Mapping[str, Any]
    ) -> None:
        model_name = _field(kwargs.get("model"), "model_name") or session.model_name
        with self.span(SPAN_LLM, LOG_TYPE_CHAT, detail=model_name or LLM_NAME_DETAIL) as span:
            span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, LOG_TYPE_CHAT)
            span.set_attribute(ATTR_USAGE_SOURCE, USAGE_SOURCE_OPERATION)
            if model_name:
                span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model_name)
            _set_usage(span, usage)

    async def screenshot(self, stagehand: Any, page: Any) -> Optional[dict[str, Any]]:
        async def take() -> Optional[bytes]:
            token = _INTERNAL.set(True)
            try:
                target = page or await stagehand.browser.context.active_page()
                return None if target is None else await target.screenshot(**SCREENSHOT_OPTIONS)
            finally:
                _INTERNAL.reset(token)

        try:
            image = await asyncio.wait_for(take(), SCREENSHOT_DEADLINE_SECONDS)
        except Exception:  # noqa: BLE001 - a missing screenshot never fails the act
            logger.debug("stagehand act screenshot failed", exc_info=True)
            return None
        return image_bytes_part(image, keep=True) if isinstance(image, bytes) else None

    # -- LLM -------------------------------------------------------------------

    def wrap_generate(self, session: _Session, generate: Callable[..., Any]) -> Callable[..., Any]:
        runtime = self

        @wraps(generate)
        async def traced(params: Any, *args: Any, **kwargs: Any) -> Any:
            if not runtime.active or session.closed:
                return await generate(params, *args, **kwargs)
            dumped = jsonable(params)
            dumped = dumped if isinstance(dumped, dict) else {}
            op = session.find_operation(dumped)
            if op is not None:
                op.llm_calls += 1
            # The extension's callback runs in Stagehand's RPC reader task; run
            # the span in the operation's context instead, so it nests under
            # the operation and carries the caller's propagated attributes.
            context = (op.context if op is not None and op.context else session.context).copy()
            call = runtime.llm_call(session, generate, params, dumped, args, kwargs)
            return await asyncio.get_running_loop().create_task(call, context=context)

        return traced

    async def llm_call(
        self,
        session: _Session,
        generate: Callable[..., Any],
        params: Any,
        dumped: dict[str, Any],
        args: tuple,
        kwargs: dict,
    ) -> Any:
        with self.span(
            SPAN_LLM, LOG_TYPE_CHAT, detail=session.model_name or LLM_NAME_DETAIL
        ) as span:
            span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, LOG_TYPE_CHAT)
            if session.model_name:
                span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, session.model_name)
            safely(partial(self.llm_start, span, dumped))
            with suppress_provider_llm_spans():
                result = await generate(params, *args, **kwargs)
            safely(partial(self.llm_end, span, result))
            return result

    def llm_start(self, span: Any, params: dict[str, Any]) -> None:
        if not self.content_allowed():
            return
        messages = chat_messages(params, keep_images=self.capture_screenshots)
        for index, message in enumerate(messages):
            prefix = f"{SpanAttributes.LLM_PROMPTS}.{index}"
            span.set_attribute(f"{prefix}.role", message["role"])
            span.set_attribute(f"{prefix}.content", content_text(message.get("content")))
            if calls := message.get("tool_calls"):
                span.set_attribute(f"{prefix}.tool_calls", json_string(calls))
        payload: dict[str, Any] = {"messages": messages}
        if response_format := params.get("response_format"):
            payload["response_format"] = response_format
        if tools := functions(params.get("tools")):
            payload["tools"] = tools
            span.set_attribute(SpanAttributes.LLM_REQUEST_FUNCTIONS, json_string(tools))
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string(payload))

    def llm_end(self, span: Any, result: Any) -> None:
        dumped = jsonable(result)
        if not isinstance(dumped, dict):
            return
        _set_usage(span, dumped.get("usage") or {})
        if not self.content_allowed():
            return
        message = completion(dumped)
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
        span.set_attribute(f"{prefix}.role", message["role"])
        span.set_attribute(f"{prefix}.content", content_text(message["content"]))
        if calls := message.get("tool_calls"):
            span.set_attribute(f"{prefix}.tool_calls", json_string(calls))
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(message))

    # -- browser actions ---------------------------------------------------------

    def wrap_action(self, owner: str, method: str, original: Any) -> Any:
        runtime = self
        name = f"{owner}.{method}"

        @wraps(original)
        async def action(target: Any, *args: Any, **kwargs: Any) -> Any:
            session = _SESSIONS_BY_RPC.get(getattr(target, "_rpc_client", None))
            if not runtime.active or session is None or session.closed or _INTERNAL.get():
                return await original(target, *args, **kwargs)
            op = _OPERATION.get()
            parent = op.span if op is not None and op in session.ops else session.span
            with (
                _parent(parent),
                _SCOPE.session(session),
                runtime.span(name, LOG_TYPE_TOOL, detail=f"{owner}_{method}") as span,
            ):
                content = runtime.content_allowed()
                # Nested public calls (a locator helper using its page, the
                # credential probe) are part of this action, not spans of their own.
                token = _INTERNAL.set(True)
                try:
                    if content:
                        await runtime.action_start(
                            span, owner, method, target, original, args, kwargs
                        )
                    result = await original(target, *args, **kwargs)
                finally:
                    _INTERNAL.reset(token)
                if content:
                    safely(partial(runtime.action_end, span, owner, method, target, result))
                return result

        return action

    async def action_start(
        self,
        span: Any,
        owner: str,
        method: str,
        target: Any,
        original: Any,
        args: tuple,
        kwargs: dict,
    ) -> None:
        payload: dict[str, Any] = {}
        try:
            if owner == LOCATOR_OWNER:
                payload["selector"] = target.selector
                if target.nth_index is not None:
                    payload["nth"] = target.nth_index
            elif owner == PAGE_OWNER and isinstance(url := _field(target.ref, "url"), str):
                payload["url"] = url
            arguments = _bind(original, target, args, kwargs)
            text_arg = TYPED_TEXT.get((owner, method))
            if text_arg and text_arg in arguments:
                # Typed text is often a credential; a field that is one is never exported.
                selector = payload.get("selector") or ""
                if SENSITIVE_SELECTOR.search(selector) or await self.types_into_secret(
                    owner, target
                ):
                    arguments[text_arg] = REDACTED
            if method == "set_active_page":
                arguments = {"page_id": getattr(arguments.get("page"), "page_id", None)}
            payload["arguments"] = {
                key: {"selector": getattr(value, "selector", None)}
                if hasattr(value, "descriptor")
                else jsonable(value)
                for key, value in arguments.items()
            }
        except Exception:  # noqa: BLE001 - telemetry must fail open
            logger.debug("stagehand action input failed", exc_info=True)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string(payload))

    async def types_into_secret(self, owner: str, target: Any) -> bool:
        """Whether text typed by this action lands in a credential field.

        Selectors from observe() are XPaths that say nothing about the field, so
        the page is asked for the target's type / name / id / autocomplete.
        Unknown = secret.
        """
        try:
            if owner == LOCATOR_OWNER:
                page = self.page_class(
                    target._rpc_client, self.page_ref_class(page_id=target.page_id)
                )
                described = await page.evaluate(_describe_field(target.selector))
            else:
                described = await target.evaluate(_describe_field(None))
        except Exception:  # noqa: BLE001 - fail closed
            return True
        return (
            not isinstance(described, str)
            or not described
            or bool(SENSITIVE_FIELD.search(described))
        )

    def action_end(self, span: Any, owner: str, method: str, target: Any, result: Any) -> None:
        url = (
            _field(target.ref, "url")
            if owner == PAGE_OWNER
            else _field(getattr(result, "ref", None), "url")
        )
        if isinstance(url, str):
            span.set_attribute(ATTR_PAGE_URL, url)
        if owner == PAGE_OWNER and method in NAVIGATION_ACTIONS:
            output: Any = {"url": url, "status": getattr(result, "status", None)}
        elif isinstance(result, bytes):
            output = (
                image_bytes_part(result, keep=self.capture_screenshots)
                if method == "screenshot"
                else jsonable(result)
            )
        elif owner == CONTEXT_OWNER and method == "new_page":
            output = {"page_id": getattr(result, "page_id", None), "url": url}
        else:
            output = jsonable(result)
        if output is not None:
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(output))


def _bind(function: Any, target: Any, args: tuple, kwargs: dict) -> dict[str, Any]:
    """The call's arguments by name (``self`` dropped, unset defaults left out)."""
    bound = inspect.signature(function).bind(target, *args, **kwargs)
    arguments = dict(bound.arguments)
    arguments.pop(next(iter(arguments)), None)
    return {key: value for key, value in arguments.items() if value is not None}


def _describe_field(selector: Optional[str]) -> str:
    """Page-side expression describing the field a selector (or, for None, the focus) points at."""
    return f"""(() => {{
      const selector = {json.dumps(selector)};
      try {{
        const el = selector === null
          ? document.activeElement
          : selector.startsWith({json.dumps(XPATH_PREFIX)})
            ? document.evaluate(selector.slice({len(XPATH_PREFIX)}), document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue
            : document.querySelector(selector);
        if (!el) return "";
        return ["type=" + (el.type || ""), el.name, el.id, el.autocomplete, el.getAttribute("aria-label")].join(" ");
      }} catch (error) {{
        return "";
      }}
    }})()"""


def _set_usage(span: Any, usage: Mapping[str, Any]) -> None:
    prompt = _non_negative_int(usage.get("input_tokens"))
    completion_tokens = _non_negative_int(usage.get("output_tokens"))
    total = _non_negative_int(usage.get("total_tokens"))
    if total is None and prompt is not None and completion_tokens is not None:
        total = prompt + completion_tokens
    for key, value in (
        (SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt),
        (SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens),
        (SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total),
    ):
        if value is not None:
            span.set_attribute(key, value)
    if cached := _non_negative_int(usage.get("cached_input_tokens")):
        span.set_attribute(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, cached)
    if reasoning := _non_negative_int(usage.get("reasoning_tokens")):
        span.set_attribute(SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS, reasoning)


def with_model_name(generate: Callable[..., Any], model_name: str) -> Callable[..., Any]:
    """Name the model a client model calls, so its chat spans report it and Respan can price them.

    Stagehand tells the client model nothing about which model it is. Takes
    precedence over ``StagehandInstrumentor(client_model_name=...)``::

        Stagehand.create(browser=browser, model=with_model_name(generate, "gpt-4.1-mini"))
    """
    try:
        setattr(generate, CLIENT_MODEL_NAME_ATTR, model_name)
        return generate
    except (AttributeError, TypeError):
        # Bound methods and builtins take no attributes; name a thin wrapper instead.
        @wraps(generate)
        async def named(*args: Any, **kwargs: Any) -> Any:
            return await generate(*args, **kwargs)

        setattr(named, CLIENT_MODEL_NAME_ATTR, model_name)
        return named


class StagehandInstrumentor:
    """Trace Stagehand v4 sessions: operations, LLM calls and browser actions."""

    name = INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        capture_content: bool = True,
        capture_screenshots: bool = True,
        client_model_name: Optional[str] = None,
    ) -> None:
        self._settings = (capture_content, capture_screenshots, client_model_name)
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
                    _RUNTIME.client_model_name,
                )
                if current != self._settings:
                    raise ValueError(
                        "Stagehand instrumentation is already active with different settings"
                    )
                _RUNTIME.count += 1
                self._runtime = _RUNTIME
                return
            try:
                stagehand_module = importlib.import_module(STAGEHAND_MODULE)
                page_module = importlib.import_module(PAGE_MODULE)
                locator_module = importlib.import_module(LOCATOR_MODULE)
                context_module = importlib.import_module(CONTEXT_MODULE)
                models_module = importlib.import_module(MODELS_MODULE)
            except ImportError:
                logger.debug("stagehand is not installed; instrumentation skipped")
                return
            stagehand_class = stagehand_module.Stagehand
            if not isinstance(stagehand_class.__dict__.get("create"), classmethod):
                raise RuntimeError(
                    "StagehandInstrumentor requires Stagehand v4 (Stagehand.create)."
                )
            capture_content, capture_screenshots, client_model_name = self._settings
            runtime = _Runtime(
                capture_content=capture_content,
                capture_screenshots=capture_screenshots,
                client_model_name=client_model_name,
            )
            runtime.page_class = page_module.Page
            runtime.page_ref_class = models_module.PageRef
            patch = runtime.patcher.patch
            try:
                patch(stagehand_class, "create", runtime.wrap_create)
                for kind in OPERATIONS:
                    patch(stagehand_class, kind, partial(runtime.wrap_operation, kind))
                patch(stagehand_class, "close", runtime.wrap_close)
                for owner, cls, methods in (
                    (PAGE_OWNER, page_module.Page, PAGE_ACTIONS),
                    (LOCATOR_OWNER, locator_module.Locator, LOCATOR_ACTIONS),
                    (CONTEXT_OWNER, context_module.BrowserContext, CONTEXT_ACTIONS),
                ):
                    for method in methods:
                        patch(cls, method, partial(runtime.wrap_action, owner, method))
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
