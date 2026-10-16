"""Native Skyvern spans: run -> step / workflow block -> LLM call / browser action.

Skyvern ships its own OpenTelemetry spans (tracer ``skyvern``) and a Laminar hook,
but its spans carry no Respan span kind, so Respan drops them. This
instrumentation wraps the seams directly:

* ``Skyvern.run_task`` / ``run_workflow`` / ``login``,
  ``AsyncSkyvern.download_files``  -> ``agent`` span (cloud and local mode)
* ``AsyncSkyvern.run_sdk_action`` -> ``agent`` span per ``page.ai_*`` call
  (``task`` when it runs inside another Skyvern span)

With the in-process engine (``Skyvern.local()``, ``pip install "skyvern[local]"``):

* ``ForgeAgent.agent_step``       -> ``task`` span per step (page URL, actions, screenshot)
* ``Block.execute_safe``          -> ``task`` span per workflow block
* ``ActionHandler.handle_action`` -> ``tool`` span per browser action (click, input_text, ...)
* ``litellm.acompletion`` / ``Router.acompletion`` -> ``chat`` span for every model
  call Skyvern makes (all of its ~18 prompt handlers end there), named by its
  prompt name

Local mode runs the engine behind an in-process ASGI app; the work happens in a
task whose context is copied from the caller, so the Respan context reaches it.
Skyvern's own spans sit between ours, so every span here is parented to the
nearest *Respan* span instead of the current OTel span.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cache, partial, wraps
from typing import Any

from opentelemetry import context, trace
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
)
from respan_tracing import is_content_tracing_enabled
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.agent_instrumentation import (
    AgentScope,
    MethodPatcher,
    agent_span,
    safely,
    secret_replacements,
    set_error,
)

from respan_instrumentation_skyvern._constants import (
    ACTION_FIELDS_DROPPED,
    ACTIONS_MODULE,
    AGENT_MODULE,
    API_CLIENT_MODULE,
    ATTR_ACTION_TYPE,
    ATTR_APP_URL,
    ATTR_BLOCK_LABEL,
    ATTR_BLOCK_TYPE,
    ATTR_ENGINE,
    ATTR_PAGE_URL,
    ATTR_PROMPT_NAME,
    ATTR_RECORDING_URL,
    ATTR_RUN_ID,
    ATTR_STEP_ID,
    ATTR_STEP_NUMBER,
    ATTR_STEP_RETRY,
    ATTR_TASK_ID,
    ATTR_WORKFLOW_RUN_ID,
    BLOCK_MODULE,
    CLIENT_MODULE,
    CONTEXT_MODULE,
    FAILED_RUN_STATUSES,
    FORGE_APP_MODULE,
    INSTRUMENTATION_NAME,
    LITELLM_MODULE,
    LLM_HANDLER_MODULE,
    NATIVE_PROMPT_NAME_ATTR,
    NATIVE_TRACER_NAME,
    PAGE_AGENT_MODULE,
    RUN_NAME_DETAILS,
    SPAN_ACTION,
    SPAN_BLOCK,
    SPAN_DOWNLOAD_FILES,
    SPAN_LLM,
    SPAN_LOGIN,
    SPAN_RUN_TASK,
    SPAN_RUN_WORKFLOW,
    SPAN_SDK_ACTION,
    SPAN_STEP,
    STEP_NAME_DETAIL,
    TRACER_NAME,
)
from respan_instrumentation_skyvern._serialization import (
    content_text,
    json_string,
    jsonable,
    prompt_messages,
    screenshot_part,
)

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_RUNTIME: _Runtime | None = None

# Client-side inputs worth showing; the rest are transport options.
_RUN_INPUT_CONTENT = (
    "prompt",
    "url",
    "title",
    "data_extraction_schema",
    "parameters",
    "workflow_id",
    "credential_type",
    "credential_id",
    "navigation_goal",
)
_RUN_INPUT_SETTINGS = (
    "engine",
    "max_steps",
    "max_steps_override",
    "wait_for_completion",
    "browser_session_id",
    "proxy_location",
)
_RUN_OUTPUT_FIELDS = (
    "run_id",
    "status",
    "run_type",
    "step_count",
    "failure_reason",
    "output",
    "downloaded_files",
    "errors",
)


@dataclass
class _RunState:
    """One Skyvern run, shared by its client span and the engine's copied context.

    ``asyncio`` copies the context binding, not the object: the engine task sees
    the same instance and reports the run id and secrets back to the client span.
    """

    run_id: str | None = None
    secrets: dict[str, str] = field(default_factory=dict)
    _replacements: tuple[tuple[str, str], ...] = ()

    def add_secrets(self, named: dict[str, str]) -> None:
        if named.items() <= self.secrets.items():
            return
        self.secrets.update(named)
        self._replacements = secret_replacements(self.secrets)

    @property
    def replacements(self) -> tuple[tuple[str, str], ...]:
        return self._replacements


_STATE: ContextVar[_RunState | None] = ContextVar("respan_skyvern_run", default=None)
# The innermost open Respan span: Skyvern's own spans sit between ours.
_PARENT: ContextVar[Any] = ContextVar("respan_skyvern_parent", default=None)
# A client run in progress: page.agent.run_task calls Skyvern.run_task, one span.
_IN_RUN: ContextVar[bool] = ContextVar("respan_skyvern_in_run", default=False)
# The prompt name of the Skyvern LLM caller in progress (v3 engine, CUA, OpenRouter).
_PROMPT_NAME: ContextVar[str | None] = ContextVar("respan_skyvern_prompt_name", default=None)
# Router.acompletion calls litellm.acompletion; only the outer call is a span.
_IN_LLM: ContextVar[bool] = ContextVar("respan_skyvern_in_llm", default=False)


@cache
def _module(name: str) -> Any:
    return importlib.import_module(name)


def _skyvern_context() -> Any:
    try:
        return _module(CONTEXT_MODULE).current()
    except Exception:  # noqa: BLE001 - the engine may not be installed
        return None


def _engine_run_id() -> str | None:
    ctx = _skyvern_context()
    if ctx is None:
        return None
    return ctx.run_id or ctx.workflow_run_id or ctx.task_id


def _engine_secrets() -> dict[str, str]:
    """The running workflow's resolved secrets, by the placeholder the model sees.

    Credential / secret parameters are resolved while the run executes, so they
    are read live rather than once at the start of the run.
    """
    ctx = _skyvern_context()
    if ctx is None:
        return {}
    named: dict[str, str] = {}
    try:
        manager = _module(FORGE_APP_MODULE).app.WORKFLOW_CONTEXT_MANAGER
        run_context = manager.workflow_run_contexts.get(ctx.workflow_run_id)
        if run_context is not None:
            named.update(
                (key, value)
                for key, value in run_context.secrets.items()
                if isinstance(value, str) and value
            )
    except Exception:  # noqa: BLE001 - engine internals moved or app not started
        pass
    runtime = getattr(ctx, "runtime_secret_values", None) or ()
    for index, value in enumerate(sorted(v for v in runtime if isinstance(v, str))):
        named.setdefault(f"runtime_secret_{index}", value)
    return named


def _run_replacements() -> tuple[tuple[str, str], ...]:
    state = _STATE.get()
    live = _engine_secrets()
    if state is None:
        return secret_replacements(live)
    if live:
        state.add_secrets(live)
    return state.replacements


class _SkyvernScope(AgentScope):
    """AgentScope whose secrets are the Skyvern run's, discovered as it resolves them."""

    def mask(self, text: str) -> str:
        text = super().mask(text)
        for secret, placeholder in _run_replacements():
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
        return super().has_secrets or bool(_run_replacements())


_SCOPE = _SkyvernScope("respan_skyvern")


def _parent_context() -> Any:
    parent = _PARENT.get()
    if parent is not None and parent.is_recording():
        return trace.set_span_in_context(parent)
    current = trace.get_current_span()
    scope = getattr(current, "instrumentation_scope", None)
    if getattr(scope, "name", None) == NATIVE_TRACER_NAME:
        # Started under a Skyvern span with no Respan ancestor: that parent is
        # never exported, so start a new trace instead of an orphan.
        return trace.set_span_in_context(trace.INVALID_SPAN)
    return None


@contextmanager
def _run_scope() -> Iterator[_RunState]:
    """Reuse the enclosing run's state, or open one for a new run."""
    state = _STATE.get()
    if state is not None:
        yield state
        return
    state = _RunState()
    token = _STATE.set(state)
    try:
        yield state
    finally:
        _STATE.reset(token)


@contextmanager
def _restore(var: ContextVar[Any], token: Any) -> Iterator[None]:
    try:
        yield
    finally:
        var.reset(token)


def _content_allowed(runtime: _Runtime) -> bool:
    return runtime.capture_content and is_content_tracing_enabled()


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _bound_arguments(original: Any, args: tuple, kwargs: dict) -> dict[str, Any]:
    try:
        bound = inspect.signature(original).bind(*args, **kwargs)
    except TypeError:
        return dict(kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    arguments.pop("self", None)
    return arguments


def _is_failed_status(status: Any) -> bool:
    return str(_enum_value(status) or "").lower() in FAILED_RUN_STATUSES


class _Runtime:
    def __init__(self, *, capture_content: bool, capture_screenshots: bool) -> None:
        self.capture_content = capture_content
        self.capture_screenshots = capture_screenshots
        self.active = True
        self.count = 1
        self.patcher = MethodPatcher()

    @property
    def tracer(self) -> trace.Tracer:
        # Resolved per call so a provider installed after activation is honoured.
        return trace.get_tracer(TRACER_NAME)

    @contextmanager
    def span(
        self,
        name: str,
        log_type: str,
        *,
        detail: str | None = None,
    ) -> Iterator[Any]:
        state = _STATE.get()
        run_id = (state.run_id if state else None) or _engine_run_id()
        if state is not None and state.run_id is None and run_id:
            state.run_id = run_id
        parent = _parent_context()
        token = context.attach(parent) if parent is not None else None
        try:
            with (
                _SCOPE.run(run_id),
                agent_span(
                    self.tracer,
                    name,
                    log_type,
                    scope=_SCOPE,
                    content_allowed=partial(_content_allowed, self),
                    detail=detail,
                ) as span,
            ):
                previous = _PARENT.set(span)
                try:
                    yield span
                finally:
                    _PARENT.reset(previous)
        finally:
            if token is not None:
                context.detach(token)

    # -- client: runs ------------------------------------------------------------

    def wrap_run(self, span_name: str) -> Callable[[Any], Any]:
        runtime = self

        def factory(original: Any) -> Any:
            @wraps(original)
            async def run(client: Any, *args: Any, **kwargs: Any) -> Any:
                if not runtime.active or _IN_RUN.get():
                    return await original(client, *args, **kwargs)
                arguments = _bound_arguments(original, (client, *args), kwargs)
                in_run = _IN_RUN.set(True)
                with (
                    _restore(_IN_RUN, in_run),
                    _run_scope() as state,
                    runtime.span(
                        span_name, LOG_TYPE_AGENT, detail=RUN_NAME_DETAILS[span_name]
                    ) as span,
                ):
                    safely(lambda: runtime._run_start(span, arguments))
                    response = await original(client, *args, **kwargs)
                    safely(lambda: runtime._run_end(span, state, response))
                    return response

            return run

        return factory

    def _run_start(self, span: Any, arguments: dict[str, Any]) -> None:
        if engine := arguments.get("engine"):
            span.set_attribute(ATTR_ENGINE, str(_enum_value(engine)))
        data = {
            key: _enum_value(arguments[key])
            for key in _RUN_INPUT_SETTINGS
            if arguments.get(key) is not None
        }
        if _content_allowed(self):
            data.update(
                (key, arguments[key])
                for key in _RUN_INPUT_CONTENT
                if arguments.get(key) is not None
            )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string(data))

    def _run_end(self, span: Any, state: _RunState, response: Any) -> None:
        run_id = state.run_id or getattr(response, "run_id", None)
        if run_id:
            span.set_attribute(ATTR_RUN_ID, str(run_id))
            # The run id is the default thread / session / trace group; it is
            # only known once Skyvern created the run.
            with _SCOPE.run(run_id):
                _SCOPE.apply_grouping(span)
        output = {
            key: _enum_value(value)
            for key in _RUN_OUTPUT_FIELDS
            if (value := getattr(response, key, None)) is not None
        }
        if not _content_allowed(self):
            for key in ("output", "failure_reason", "errors", "downloaded_files"):
                output.pop(key, None)
        else:
            # Presigned recording / app URLs carry access tokens: content only.
            if recording := getattr(response, "recording_url", None):
                span.set_attribute(ATTR_RECORDING_URL, str(recording))
                output["recording_url"] = recording
            if app_url := getattr(response, "app_url", None):
                span.set_attribute(ATTR_APP_URL, str(app_url))
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(output))
        status = getattr(response, "status", None)
        if _is_failed_status(status):
            set_error(
                span,
                f"run {_enum_value(status)}",
                getattr(response, "failure_reason", None),
                with_message=_content_allowed(self),
                mask=_SCOPE.mask,
            )

    # -- client: page.ai_* actions --------------------------------------------

    def wrap_sdk_action(self, original: Any) -> Any:
        runtime = self

        @wraps(original)
        async def run_sdk_action(client: Any, *args: Any, **kwargs: Any) -> Any:
            if not runtime.active:
                return await original(client, *args, **kwargs)
            action = kwargs.get("action")
            action_type = str(getattr(action, "type", "") or "") or None
            name = f"skyvern.{action_type}" if action_type else SPAN_SDK_ACTION
            log_type = LOG_TYPE_TASK if _PARENT.get() is not None else LOG_TYPE_AGENT
            with (
                _run_scope() as state,
                runtime.span(name, log_type, detail=action_type) as span,
            ):
                if action_type:
                    span.set_attribute(ATTR_ACTION_TYPE, action_type)
                if _content_allowed(runtime):
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_INPUT,
                        json_string({"url": kwargs.get("url"), "action": action}),
                    )
                response = await original(client, *args, **kwargs)
                safely(lambda: runtime._sdk_action_end(span, state, response))
                return response

        return run_sdk_action

    def _sdk_action_end(self, span: Any, state: _RunState, response: Any) -> None:
        run_id = state.run_id or getattr(response, "workflow_run_id", None)
        if run_id:
            span.set_attribute(ATTR_RUN_ID, str(run_id))
            with _SCOPE.run(run_id):
                _SCOPE.apply_grouping(span)
        if _content_allowed(self):
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json_string({"result": getattr(response, "result", None)}),
            )

    # -- engine: steps, blocks, actions ---------------------------------------

    def wrap_agent_step(self, original: Any) -> Any:
        runtime = self

        @wraps(original)
        async def agent_step(agent: Any, task: Any, step: Any, *args: Any, **kwargs: Any) -> Any:
            if not runtime.active:
                return await original(agent, task, step, *args, **kwargs)
            with runtime.span(SPAN_STEP, LOG_TYPE_TASK, detail=STEP_NAME_DETAIL) as span:
                safely(lambda: runtime._step_start(span, task, step, kwargs))
                result = await original(agent, task, step, *args, **kwargs)
                safely(lambda: runtime._step_end(span, result))
                return result

        return agent_step

    def _step_start(self, span: Any, task: Any, step: Any, kwargs: dict) -> None:
        if (number := _non_negative_int(getattr(step, "order", None))) is not None:
            span.set_attribute(ATTR_STEP_NUMBER, number)
        if (retry := _non_negative_int(getattr(step, "retry_index", None))) is not None:
            span.set_attribute(ATTR_STEP_RETRY, retry)
        for attr, value in (
            (ATTR_STEP_ID, getattr(step, "step_id", None)),
            (ATTR_TASK_ID, getattr(task, "task_id", None)),
            (ATTR_WORKFLOW_RUN_ID, getattr(task, "workflow_run_id", None)),
            (ATTR_ENGINE, _enum_value(kwargs.get("engine"))),
        ):
            if value:
                span.set_attribute(attr, str(value))

    def _step_end(self, span: Any, result: Any) -> None:
        step, detailed = result
        status = str(_enum_value(getattr(step, "status", None)) or "")
        step_exception = getattr(detailed, "step_exception", None)
        if status == "failed":
            set_error(
                span,
                "step failed",
                step_exception,
                with_message=_content_allowed(self),
                mask=_SCOPE.mask,
            )
        output: dict[str, Any] = {"status": status}
        usage = {
            "input_tokens": getattr(step, "input_token_count", None),
            "output_tokens": getattr(step, "output_token_count", None),
            "cost": getattr(step, "step_cost", None),
        }
        output["usage"] = {k: v for k, v in usage.items() if v is not None}
        if not _content_allowed(self):
            # URLs carry magic-link tokens and session ids in query strings.
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(output))
            return
        scraped = getattr(detailed, "scraped_page", None)
        if url := getattr(scraped, "url", None):
            span.set_attribute(ATTR_PAGE_URL, str(url))
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string({"url": url}))
        output["actions"] = [
            {
                "action": _action_dict(action),
                "results": [jsonable(r) for r in results or ()],
            }
            for action, results in getattr(detailed, "actions_and_results", None) or ()
        ]
        if step_exception:
            output["step_exception"] = step_exception
        screenshots = getattr(scraped, "screenshots", None) or ()
        if self.capture_screenshots and screenshots and isinstance(screenshots[0], bytes):
            output["screenshot"] = screenshot_part(screenshots[0])
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(output))

    def wrap_execute_safe(self, original: Any) -> Any:
        runtime = self

        @wraps(original)
        async def execute_safe(block: Any, *args: Any, **kwargs: Any) -> Any:
            if not runtime.active:
                return await original(block, *args, **kwargs)
            block_type = str(_enum_value(getattr(block, "block_type", "")) or "") or None
            name = f"{SPAN_BLOCK}.{block_type}" if block_type else SPAN_BLOCK
            with runtime.span(name, LOG_TYPE_TASK, detail=block_type) as span:
                if block_type:
                    span.set_attribute(ATTR_BLOCK_TYPE, block_type)
                if label := getattr(block, "label", None):
                    span.set_attribute(ATTR_BLOCK_LABEL, str(label))
                result = await original(block, *args, **kwargs)
                safely(lambda: runtime._block_end(span, result))
                return result

        return execute_safe

    def _block_end(self, span: Any, result: Any) -> None:
        status = _enum_value(getattr(result, "status", None))
        output: dict[str, Any] = {"success": getattr(result, "success", None), "status": status}
        if _content_allowed(self):
            output["output"] = getattr(result, "output_parameter_value", None)
            if reason := getattr(result, "failure_reason", None):
                output["failure_reason"] = reason
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(output))
        if _is_failed_status(status):
            set_error(
                span,
                f"block {status}",
                getattr(result, "failure_reason", None),
                with_message=_content_allowed(self),
                mask=_SCOPE.mask,
            )

    def wrap_handle_action(self, original: Any) -> Any:
        runtime = self
        func = original.__func__ if isinstance(original, staticmethod) else original

        @wraps(func)
        async def handle_action(
            scraped_page: Any, task: Any, step: Any, page: Any, action: Any, *args: Any, **kwargs: Any
        ) -> Any:
            if not runtime.active:
                return await func(scraped_page, task, step, page, action, *args, **kwargs)
            action_type = str(_enum_value(getattr(action, "action_type", "")) or "") or None
            with runtime.span(action_type or SPAN_ACTION, LOG_TYPE_TOOL) as span:
                if action_type:
                    span.set_attribute(ATTR_ACTION_TYPE, action_type)
                if _content_allowed(runtime):
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string(_action_dict(action))
                    )
                results = await func(scraped_page, task, step, page, action, *args, **kwargs)
                safely(lambda: runtime._action_end(span, results))
                return results

        return staticmethod(handle_action) if isinstance(original, staticmethod) else handle_action

    def _action_end(self, span: Any, results: Any) -> None:
        results = list(results or ())
        # Skyvern reports action failures as ActionFailure results, not exceptions.
        failed = [r for r in results if getattr(r, "success", True) is False]
        if failed:
            first = failed[0]
            set_error(
                span,
                "action failed",
                f"{first.exception_type}: {first.exception_message}",
                with_message=_content_allowed(self),
                mask=_SCOPE.mask,
            )
        if _content_allowed(self):
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json_string({"results": [jsonable(r) for r in results]}),
            )

    # -- engine: llm -------------------------------------------------------------

    def wrap_llm_caller_call(self, original: Any) -> Any:
        @wraps(original)
        async def call(caller: Any, *args: Any, **kwargs: Any) -> Any:
            name = kwargs.get("prompt_name", args[1] if len(args) > 1 else None)
            token = _PROMPT_NAME.set(str(name) if name else None)
            try:
                return await original(caller, *args, **kwargs)
            finally:
                _PROMPT_NAME.reset(token)

        return call

    def wrap_acompletion(self, original: Any) -> Any:
        runtime = self

        @wraps(original)
        async def acompletion(*args: Any, **kwargs: Any) -> Any:
            return await runtime.traced_llm_call(original, None, args, kwargs)

        return acompletion

    def wrap_router_acompletion(self, original: Any) -> Any:
        runtime = self

        @wraps(original)
        async def acompletion(router: Any, *args: Any, **kwargs: Any) -> Any:
            return await runtime.traced_llm_call(original, router, args, kwargs)

        return acompletion

    async def traced_llm_call(self, original: Any, router: Any, args: tuple, kwargs: dict) -> Any:
        call_args = args if router is None else (router, *args)
        # Only model calls Skyvern makes inside a traced run become spans.
        if not self.active or _IN_LLM.get() or _PARENT.get() is None:
            return await original(*call_args, **kwargs)
        native = trace.get_current_span()
        prompt_name = (getattr(native, "attributes", None) or {}).get(
            NATIVE_PROMPT_NAME_ATTR
        ) or _PROMPT_NAME.get()
        model = kwargs.get("model", args[0] if args else None)
        messages = kwargs.get("messages", args[1] if len(args) > 1 else None)
        with self.span(SPAN_LLM, LOG_TYPE_CHAT) as span:
            safely(lambda: self._llm_start(span, model, messages, prompt_name, kwargs))
            in_llm = _IN_LLM.set(True)
            try:
                with _suppress_provider_llm_spans():
                    response = await original(*call_args, **kwargs)
            finally:
                _IN_LLM.reset(in_llm)
            safely(lambda: self._llm_end(span, response))
            return response

    def _llm_start(
        self, span: Any, model: Any, messages: Any, prompt_name: Any, kwargs: dict
    ) -> None:
        span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, "chat")
        if prompt_name:
            span.set_attribute(ATTR_PROMPT_NAME, str(prompt_name))
        if model:
            name, provider = _split_litellm_model(str(model), kwargs)
            span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, name)
            if provider:
                span.set_attribute(SpanAttributes.LLM_SYSTEM, provider)
        for key, attr in (
            ("temperature", SpanAttributes.LLM_REQUEST_TEMPERATURE),
            ("max_tokens", SpanAttributes.LLM_REQUEST_MAX_TOKENS),
        ):
            if isinstance(kwargs.get(key), int | float) and not isinstance(kwargs.get(key), bool):
                span.set_attribute(attr, kwargs[key])
        if not _content_allowed(self):
            return
        for index, message in enumerate(
            prompt_messages(messages, keep_images=self.capture_screenshots)
        ):
            prefix = f"{SpanAttributes.LLM_PROMPTS}.{index}"
            span.set_attribute(f"{prefix}.role", str(message.get("role") or "user"))
            span.set_attribute(f"{prefix}.content", content_text(message.get("content")))

    def _llm_end(self, span: Any, response: Any) -> None:
        if model := getattr(response, "model", None):
            span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, str(model))
        usage = getattr(response, "usage", None)
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            for value, key in (
                (getattr(usage, "prompt_tokens", None), SpanAttributes.LLM_USAGE_PROMPT_TOKENS),
                (
                    getattr(usage, "completion_tokens", None),
                    SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                ),
                (getattr(usage, "total_tokens", None), SpanAttributes.LLM_USAGE_TOTAL_TOKENS),
                (
                    getattr(details, "cached_tokens", None),
                    SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
                ),
            ):
                if (count := _non_negative_int(value)) is not None:
                    span.set_attribute(key, count)
        if not _content_allowed(self):
            return
        choices = getattr(response, "choices", None) or ()
        message = getattr(choices[0], "message", None) if choices else None
        if message is None:
            return
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
        span.set_attribute(f"{prefix}.role", "assistant")
        span.set_attribute(f"{prefix}.content", content_text(getattr(message, "content", None)))
        if tool_calls := getattr(message, "tool_calls", None):
            span.set_attribute(f"{prefix}.tool_calls", json_string(tool_calls))


def _split_litellm_model(model: str, kwargs: dict) -> tuple[str, str | None]:
    """litellm's "<provider>/<model>" -> (model, provider), as litellm resolves it."""
    try:
        litellm = _module(LITELLM_MODULE)
        name, provider, _, _ = litellm.get_llm_provider(
            model=model,
            custom_llm_provider=kwargs.get("custom_llm_provider"),
            api_base=kwargs.get("api_base"),
        )
        return str(name), str(provider) if provider else None
    except Exception:  # noqa: BLE001 - unknown provider: keep litellm's name
        return model, None


def _action_dict(action: Any) -> dict[str, Any]:
    data = jsonable(action)
    if not isinstance(data, dict):
        return {"action": data}
    for key in ACTION_FIELDS_DROPPED:
        data.pop(key, None)
    return data


@cache
def _openai_suppressor() -> Callable[[], Any]:
    # Optional dependency, resolved once: a failed import is not cached by Python.
    try:
        from respan_instrumentation_openai._instrumentation import (
            suppress_openai_instrumentation,
        )
    except ImportError:
        return nullcontext
    return suppress_openai_instrumentation


@contextmanager
def _suppress_provider_llm_spans() -> Iterator[None]:
    """litellm / OpenAI instrumentations would log the call this span already carries.

    The LiteLLM instrumentation honours OTel's language-model suppression key;
    Respan's OpenAI instrumentation (litellm's OpenAI-compatible path) has its
    own switch.
    """
    suppress_openai_instrumentation = _openai_suppressor()
    token = context.attach(context.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True))
    try:
        with suppress_openai_instrumentation():
            yield
    finally:
        context.detach(token)


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


class SkyvernInstrumentor:
    """Trace Skyvern runs: tasks, workflows, steps, LLM calls and browser actions."""

    name = INSTRUMENTATION_NAME

    def __init__(self, *, capture_content: bool = True, capture_screenshots: bool = True) -> None:
        self._capture_content = capture_content
        self._capture_screenshots = capture_screenshots
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
                if (_RUNTIME.capture_content, _RUNTIME.capture_screenshots) != (
                    self._capture_content,
                    self._capture_screenshots,
                ):
                    raise ValueError(
                        "Skyvern instrumentation is already active with different settings"
                    )
                _RUNTIME.count += 1
                self._runtime = _RUNTIME
                return
            client_module = _import(CLIENT_MODULE)
            api_module = _import(API_CLIENT_MODULE)
            if client_module is None or api_module is None:
                logger.debug("skyvern is not installed; instrumentation skipped")
                return
            runtime = _Runtime(
                capture_content=self._capture_content,
                capture_screenshots=self._capture_screenshots,
            )
            patch = runtime.patcher.patch
            try:
                skyvern_cls = client_module.Skyvern
                patch(skyvern_cls, "run_task", runtime.wrap_run(SPAN_RUN_TASK))
                patch(skyvern_cls, "run_workflow", runtime.wrap_run(SPAN_RUN_WORKFLOW))
                patch(skyvern_cls, "login", runtime.wrap_run(SPAN_LOGIN))
                api_cls = api_module.AsyncSkyvern
                patch(api_cls, "download_files", runtime.wrap_run(SPAN_DOWNLOAD_FILES))
                patch(api_cls, "run_sdk_action", runtime.wrap_sdk_action)
                # The in-process engine is optional (`skyvern[local]`).
                agent_module = _import(AGENT_MODULE)
                actions_module = _import(ACTIONS_MODULE)
                block_module = _import(BLOCK_MODULE)
                litellm = _import(LITELLM_MODULE)
                page_agent_module = _import(PAGE_AGENT_MODULE)
                llm_module = _import(LLM_HANDLER_MODULE)
                if agent_module is not None:
                    patch(agent_module.ForgeAgent, "agent_step", runtime.wrap_agent_step)
                if actions_module is not None:
                    patch(
                        actions_module.ActionHandler, "handle_action", runtime.wrap_handle_action
                    )
                if block_module is not None:
                    patch(block_module.Block, "execute_safe", runtime.wrap_execute_safe)
                if page_agent_module is not None:
                    page_agent = page_agent_module.SkyvernBrowserPageAgent
                    for method, span_name in (
                        ("run_task", SPAN_RUN_TASK),
                        ("run_workflow", SPAN_RUN_WORKFLOW),
                        ("login", SPAN_LOGIN),
                        ("download_files", SPAN_DOWNLOAD_FILES),
                    ):
                        patch(page_agent, method, runtime.wrap_run(span_name))
                if llm_module is not None:
                    patch(llm_module.LLMCaller, "call", runtime.wrap_llm_caller_call)
                if agent_module is not None and litellm is not None:
                    patch(litellm, "acompletion", runtime.wrap_acompletion)
                    patch(litellm.Router, "acompletion", runtime.wrap_router_acompletion)
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
