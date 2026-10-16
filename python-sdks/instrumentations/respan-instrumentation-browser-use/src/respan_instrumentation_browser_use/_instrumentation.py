"""Native browser-use spans: agent run -> step -> LLM call / browser action.

browser-use ships its own ``observe`` decorators, but they only emit when the
Laminar SDK is installed. This instrumentation wraps the same seams directly:

* ``Agent.run``            -> ``agent`` span (task in, final result out)
* ``Agent.step``           -> ``task`` span per step (page URL, reasoning, screenshot)
* ``Tools.act``            -> ``tool`` span per browser action (click, navigate, ...)
* ``<llm class>.ainvoke``  -> ``chat`` span for every model call, any provider

LLM classes are patched just before ``TokenCost.register_llm`` captures the bound
``ainvoke``: every model browser-use uses (main, extraction, judge, fallback,
compaction) passes through it, and patching later would miss the captured method.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from contextvars import ContextVar
from functools import partial, wraps
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv_ai import (
    SpanAttributes,
)
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
)
from respan_tracing import is_content_tracing_enabled
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.agent_instrumentation import (
    PATCH_MARKER,
    AgentScope,
    MethodPatcher,
    agent_span,
    compact_image_data_url,
    safely,
    set_error,
    suppress_provider_llm_spans,
)

from respan_instrumentation_browser_use._constants import (
    AGENT_MODULE,
    AGENT_NAME_DETAIL,
    ANTHROPIC_USAGE_MODULES,
    ATTR_ACTION_NAME,
    ATTR_AGENT_ID,
    ATTR_PAGE_TITLE,
    ATTR_PAGE_URL,
    ATTR_SESSION_ID,
    ATTR_STEP_NUMBER,
    INSTRUMENTATION_NAME,
    RERUN_NAME_DETAIL,
    SPAN_ACTION,
    SPAN_AGENT,
    SPAN_LLM,
    SPAN_RERUN,
    SPAN_STEP,
    TOKENS_MODULE,
    TOOLS_MODULE,
    TRACER_NAME,
)
from respan_instrumentation_browser_use._serialization import (
    content_text,
    json_string,
    jsonable,
    message_dicts,
)

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_RUNTIME: _Runtime | None = None
# Fallback / wrapper models call an inner model's ainvoke; only the outer call is a span.
_IN_LLM: ContextVar[bool] = ContextVar("respan_browser_use_in_llm", default=False)
# The running Agent's session id (default thread / session / trace group of every
# span, so follow-up runs of the same Agent group together) and its sensitive_data.
# browser-use masks secrets only in what it sends the model; its action results and
# errors (e.g. "Navigation to https://...?token=<real value> blocked") carry them.
_SCOPE = AgentScope("respan_browser_use")


def _content_allowed(runtime: _Runtime) -> bool:
    return runtime.capture_content and is_content_tracing_enabled()


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _agent_scope(agent: Any) -> Any:
    return _SCOPE.run(agent.session_id, getattr(agent, "sensitive_data", None))


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

    def span(self, name: str, log_type: str, *, detail: str | None = None) -> Any:
        return agent_span(
            self.tracer,
            name,
            log_type,
            scope=_SCOPE,
            content_allowed=partial(_content_allowed, self),
            detail=detail,
        )

    # -- agent ---------------------------------------------------------------

    def wrap_run(self, original: Any) -> Any:
        runtime = self

        @wraps(original)
        async def run(agent: Any, *args: Any, **kwargs: Any) -> Any:
            if not runtime.active:
                return await original(agent, *args, **kwargs)
            with (
                _agent_scope(agent),
                runtime.span(SPAN_AGENT, LOG_TYPE_AGENT, detail=AGENT_NAME_DETAIL) as span,
            ):
                # Follow-up runs (add_new_task) append to the same history.
                before = len(agent.history.history)
                safely(lambda: runtime._agent_start(span, agent))
                history = await original(agent, *args, **kwargs)
                safely(lambda: runtime._agent_end(span, history, before))
                return history

        return run

    def wrap_rerun(self, original: Any) -> Any:
        runtime = self

        @wraps(original)
        async def rerun_history(agent: Any, history: Any, *args: Any, **kwargs: Any) -> Any:
            if not runtime.active:
                return await original(agent, history, *args, **kwargs)
            # A replay executes the recorded actions directly; without its own
            # root span each action would export as a separate one-span trace.
            with (
                _agent_scope(agent),
                runtime.span(SPAN_RERUN, LOG_TYPE_AGENT, detail=RERUN_NAME_DETAIL) as span,
            ):
                safely(lambda: runtime._rerun_start(span, agent, history))
                results = await original(agent, history, *args, **kwargs)
                safely(lambda: runtime._rerun_end(span, results))
                return results

        return rerun_history

    def _rerun_start(self, span: Any, agent: Any, history: Any) -> None:
        span.set_attribute(ATTR_AGENT_ID, str(agent.id))
        span.set_attribute(ATTR_SESSION_ID, str(agent.session_id))
        replayed = {"steps": len(getattr(history, "history", ()) or ())}
        if _content_allowed(self):
            replayed["task"] = agent.task
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string(replayed))

    def _rerun_end(self, span: Any, results: Any) -> None:
        results = list(results or ())
        errors = [r.error for r in results if getattr(r, "error", None)]
        output: dict[str, Any] = {"actions": len(results), "failed_actions": len(errors)}
        if _content_allowed(self) and errors:
            output["errors"] = errors
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(output))

    def _agent_start(self, span: Any, agent: Any) -> None:
        span.set_attribute(ATTR_AGENT_ID, str(agent.id))
        span.set_attribute(ATTR_SESSION_ID, str(agent.session_id))
        llm = getattr(agent, "llm", None)
        if model := getattr(llm, "model", None):
            span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, str(model))
        if _content_allowed(self):
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string({"task": agent.task})
            )

    def _agent_end(self, span: Any, history: Any, before: int) -> None:
        is_successful = history.is_successful()
        items = history.history[before:]
        output: dict[str, Any] = {
            "is_done": history.is_done(),
            "is_successful": is_successful,
            "steps": len(items),
            "duration_seconds": sum(
                item.metadata.duration_seconds for item in items if item.metadata
            ),
        }
        # browser-use's built-in judge (use_judge=True by default) grades the run.
        if (validated := history.is_validated()) is not None:
            output["judge_verdict"] = validated
        if _content_allowed(self):
            output["final_result"] = history.final_result()
            if judgement := history.judgement():
                output["judgement"] = judgement
            errors = [e for e in history.errors() if e]
            if errors:
                output["errors"] = errors
        if (usage := getattr(history, "usage", None)) is not None:
            output["usage"] = jsonable(usage)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(output))
        if is_successful is False:
            span.set_status(Status(StatusCode.ERROR, "agent reported failure"))

    # -- step ----------------------------------------------------------------

    def wrap_step(self, original: Any) -> Any:
        runtime = self

        @wraps(original)
        async def step(agent: Any, *args: Any, **kwargs: Any) -> Any:
            if not runtime.active:
                return await original(agent, *args, **kwargs)
            history = getattr(agent, "history", None)
            before = len(getattr(history, "history", ()) or ())
            number = _non_negative_int(
                getattr(getattr(agent, "state", None), "n_steps", None)
            )
            with runtime.span(SPAN_STEP, LOG_TYPE_TASK) as span:
                if number is not None:
                    span.set_attribute(ATTR_STEP_NUMBER, number)
                result = await original(agent, *args, **kwargs)
                safely(lambda: runtime._step_end(span, agent, before))
                return result

        return step

    def _step_end(self, span: Any, agent: Any, before: int) -> None:
        items = agent.history.history
        if len(items) <= before:
            return  # the step failed before browser-use recorded it
        item = items[-1]
        state = item.state
        # browser-use catches step failures and records them as result errors.
        if item.result and all(result.error for result in item.result):
            set_error(
                span,
                "step failed",
                item.result[0].error,
                with_message=_content_allowed(self),
                mask=_SCOPE.mask,
            )
        if not _content_allowed(self):
            return  # URLs carry magic-link tokens and session ids in query strings
        if state.url:
            span.set_attribute(ATTR_PAGE_URL, state.url)
        if state.title:
            span.set_attribute(ATTR_PAGE_TITLE, state.title)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json_string({"url": state.url, "title": state.title}),
        )
        output: dict[str, Any] = {}
        model_output = item.model_output
        if model_output is not None:
            for field in (
                "thinking",
                "evaluation_previous_goal",
                "memory",
                "next_goal",
            ):
                if value := getattr(model_output, field, None):
                    output[field] = value
            output["actions"] = [
                jsonable(action) for action in getattr(model_output, "action", ()) or ()
            ]
        output["results"] = [_action_result(result) for result in item.result]
        if self.capture_screenshots and (screenshot := state.get_screenshot()):
            # A data-URI image part: Respan's ingest lifts it into media storage and
            # the log view renders it, so no separate upload API is needed.
            output["screenshot"] = {
                "type": "image_url",
                "image_url": {"url": compact_image_data_url(f"data:image/png;base64,{screenshot}")},
            }
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(output))

    # -- action --------------------------------------------------------------

    def wrap_act(self, original: Any) -> Any:
        runtime = self

        @wraps(original)
        async def act(tools: Any, action: Any, *args: Any, **kwargs: Any) -> Any:
            if not runtime.active:
                return await original(tools, action, *args, **kwargs)
            name, params = _action_call(action)
            with runtime.span(name or SPAN_ACTION, LOG_TYPE_TOOL) as span:
                if name:
                    span.set_attribute(ATTR_ACTION_NAME, name)
                if _content_allowed(runtime):
                    # Params are captured before browser-use substitutes sensitive_data,
                    # so secrets stay as their <secret>name</secret> placeholders.
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string(params)
                    )
                result = await original(tools, action, *args, **kwargs)
                safely(lambda: runtime._act_end(span, result))
                return result

        return act

    def _act_end(self, span: Any, result: Any) -> None:
        # browser-use returns action failures as ActionResult.error, not exceptions.
        if error := getattr(result, "error", None):
            set_error(
                span, "action error", error, with_message=_content_allowed(self), mask=_SCOPE.mask
            )
        if _content_allowed(self):
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json_string(_action_result(result)),
            )

    # -- llm -----------------------------------------------------------------

    def wrap_register_llm(self, original: Any) -> Any:
        runtime = self

        @wraps(original)
        def register_llm(service: Any, llm: Any, *args: Any, **kwargs: Any) -> Any:
            if runtime.active and llm is not None:
                safely(lambda: runtime.patch_llm_class(type(llm)))
            result = original(service, llm, *args, **kwargs)
            if runtime.active and llm is not None:
                safely(lambda: runtime.wrap_llm_instance(llm))
            return result

        return register_llm

    def wrap_llm_instance(self, llm: Any) -> None:
        """Trace calls through the ainvoke browser-use stored on the instance.

        TokenCost replaces ``llm.ainvoke`` with a closure over whatever bound method
        existed when the model was first registered. A model registered before
        activation (or before a deactivate/reactivate) therefore never reaches the
        class patch; wrapping the instance attribute covers it, and _IN_LLM keeps
        the class patch from opening a second span when both are on the path.
        """
        current = llm.__dict__.get("ainvoke")
        if current is None or getattr(current, PATCH_MARKER, None) is self:
            return
        runtime = self

        @wraps(current)
        async def ainvoke(messages: Any, *args: Any, **kwargs: Any) -> Any:
            return await runtime.traced_llm_call(llm, current, messages, args, kwargs)

        setattr(ainvoke, PATCH_MARKER, self)
        # Pydantic-backed models reject plain setattr; browser-use patches the same way.
        object.__setattr__(llm, "ainvoke", ainvoke)

    def patch_llm_class(self, cls: type) -> None:
        # Patch the class that defines ainvoke, so subclasses share one wrapper.
        for owner in cls.__mro__:
            if "ainvoke" in owner.__dict__:
                if inspect.iscoroutinefunction(owner.__dict__["ainvoke"]):
                    self.patch(owner, "ainvoke", self.wrap_ainvoke)
                return

    def wrap_ainvoke(self, original: Any) -> Any:
        runtime = self

        @wraps(original)
        async def ainvoke(llm: Any, messages: Any, *args: Any, **kwargs: Any) -> Any:
            return await runtime.traced_llm_call(
                llm, partial(original, llm), messages, args, kwargs
            )

        return ainvoke

    async def traced_llm_call(
        self, llm: Any, call: Any, messages: Any, args: tuple, kwargs: dict
    ) -> Any:
        if not self.active or _IN_LLM.get():
            return await call(messages, *args, **kwargs)
        with self.span(SPAN_LLM, LOG_TYPE_CHAT) as span:
            safely(lambda: self._llm_start(span, llm, messages))
            in_llm = _IN_LLM.set(True)
            try:
                with suppress_provider_llm_spans():
                    response = await call(messages, *args, **kwargs)
            finally:
                _IN_LLM.reset(in_llm)
            safely(lambda: self._llm_end(span, llm, response))
            return response

    def _llm_start(self, span: Any, llm: Any, messages: Any) -> None:
        span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, "chat")
        if model := getattr(llm, "model", None):
            span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, str(model))
        if provider := getattr(llm, "provider", None):
            span.set_attribute(SpanAttributes.LLM_SYSTEM, str(provider))
        if not _content_allowed(self):
            return
        prompts = message_dicts(messages, keep_images=self.capture_screenshots)
        for index, message in enumerate(prompts):
            prefix = f"{SpanAttributes.LLM_PROMPTS}.{index}"
            span.set_attribute(f"{prefix}.role", str(message.get("role") or "user"))
            span.set_attribute(
                f"{prefix}.content", content_text(message.get("content"))
            )
            if tool_calls := message.get("tool_calls"):
                span.set_attribute(f"{prefix}.tool_calls", json_string(tool_calls))

    def _llm_end(self, span: Any, llm: Any, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens, total_tokens = usage.prompt_tokens, usage.total_tokens
            if _reports_anthropic_usage(llm):
                # browser-use's native Anthropic clients leave cache writes out of
                # prompt_tokens (and all cache tokens out of total_tokens). Respan
                # prices prompt_tokens as the full prompt minus both cache counts,
                # so without this cache-write calls are under-billed.
                prompt_tokens += usage.prompt_cache_creation_tokens or 0
                total_tokens = prompt_tokens + usage.completion_tokens
            for value, key in (
                (prompt_tokens, SpanAttributes.LLM_USAGE_PROMPT_TOKENS),
                (usage.completion_tokens, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS),
                (total_tokens, SpanAttributes.LLM_USAGE_TOTAL_TOKENS),
                (
                    usage.prompt_cached_tokens,
                    SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
                ),
                (
                    usage.prompt_cache_creation_tokens,
                    SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
                ),
            ):
                if (count := _non_negative_int(value)) is not None:
                    span.set_attribute(key, count)
        if not _content_allowed(self):
            return
        completion = getattr(response, "completion", None)
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
        span.set_attribute(f"{prefix}.role", "assistant")
        span.set_attribute(
            f"{prefix}.content",
            completion if isinstance(completion, str) else json_string(completion),
        )


def _reports_anthropic_usage(llm: Any) -> bool:
    return any(
        owner.__module__ in ANTHROPIC_USAGE_MODULES for owner in type(llm).__mro__
    )


def _action_call(action: Any) -> tuple[str | None, Any]:
    """An ActionModel holds exactly one set field: {action_name: params}."""
    dumped = action.model_dump(exclude_unset=True)
    for name, params in dumped.items():
        if params is not None:
            return name, params
    return None, dumped


def _action_result(result: Any) -> dict[str, Any]:
    data = jsonable(result)
    if not isinstance(data, dict):
        return {"result": data}
    # Inline attachment images are already in the step screenshot / LLM prompt.
    if images := data.pop("images", None):
        data["images"] = [image.get("name") for image in images]
    return data


class BrowserUseInstrumentor:
    """Trace browser-use agents: runs, steps, LLM calls and browser actions."""

    name = INSTRUMENTATION_NAME

    def __init__(
        self, *, capture_content: bool = True, capture_screenshots: bool = True
    ) -> None:
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
                        "browser-use instrumentation is already active with different settings"
                    )
                _RUNTIME.count += 1
                self._runtime = _RUNTIME
                return
            try:
                agent_module = importlib.import_module(AGENT_MODULE)
                tools_module = importlib.import_module(TOOLS_MODULE)
                tokens_module = importlib.import_module(TOKENS_MODULE)
            except ImportError:
                logger.debug("browser-use is not installed; instrumentation skipped")
                return
            runtime = _Runtime(
                capture_content=self._capture_content,
                capture_screenshots=self._capture_screenshots,
            )
            try:
                runtime.patcher.patch(agent_module.Agent, "run", runtime.wrap_run)
                runtime.patcher.patch(agent_module.Agent, "rerun_history", runtime.wrap_rerun)
                runtime.patcher.patch(agent_module.Agent, "step", runtime.wrap_step)
                runtime.patcher.patch(tools_module.Tools, "act", runtime.wrap_act)
                runtime.patcher.patch(
                    tokens_module.TokenCost, "register_llm", runtime.wrap_register_llm
                )
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
