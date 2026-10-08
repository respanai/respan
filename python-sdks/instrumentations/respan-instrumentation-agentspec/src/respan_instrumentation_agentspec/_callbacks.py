"""Owned supplements to AgentSpec's native LangGraph callback events."""

from __future__ import annotations

import inspect
import json
import logging
import math
from contextvars import ContextVar
from functools import wraps

logger = logging.getLogger(__name__)

_CAPTURE = ContextVar("respan_agentspec_callback", default=None)
_PATCHES = []
_STATE = None


def token_count(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        return None
    try:
        count = int(value)
        return count if count >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def usage(response):
    generations = getattr(response, "generations", None) or []
    message = (
        getattr(generations[0][0], "message", None)
        if generations and generations[0]
        else None
    )
    result = {}
    sources = [
        getattr(message, "usage_metadata", None),
        getattr(message, "response_metadata", None),
        getattr(response, "llm_output", None),
    ]
    for source in list(sources):
        if isinstance(source, dict):
            sources.extend(source.get(key) for key in ("token_usage", "usage"))
    for source in sources:
        if not isinstance(source, dict):
            continue
        for target, names in [
            ("input_tokens", ("input_tokens", "prompt_tokens", "prompt")),
            ("output_tokens", ("output_tokens", "completion_tokens", "completion")),
        ]:
            for name in names:
                value = token_count(source.get(name))
                if value is not None:
                    result.setdefault(target, value)
                    break
        for target, container, name in [
            ("cache", "input_token_details", "cache_read"),
            ("reasoning", "output_token_details", "reasoning"),
            ("cache", "prompt_tokens_details", "cached_tokens"),
            ("reasoning", "completion_tokens_details", "reasoning_tokens"),
        ]:
            details = source.get(container)
            value = (
                token_count(details.get(name)) if isinstance(details, dict) else None
            )
            if value is not None:
                result.setdefault(target, value)
    return result


def messages(value):
    result = []
    for message in value:
        role = {"human": "user", "ai": "assistant"}.get(
            getattr(message, "type", ""), getattr(message, "type", "user")
        )
        row = {"role": role, "content": getattr(message, "content", "")}
        calls = getattr(message, "tool_calls", None)
        if calls:
            row["tool_calls"] = [
                {
                    "id": c.get("id"),
                    "type": "function",
                    "function": {
                        "name": c.get("name"),
                        "arguments": json.dumps(c.get("args", {})),
                    },
                }
                for c in calls
            ]
        call_id = getattr(message, "tool_call_id", None)
        if call_id is not None:
            row["tool_call_id"] = call_id
        result.append(row)
    return result


def install():
    global _STATE
    if _STATE is not None:
        return
    try:
        import pyagentspec.adapters.langgraph.tracing as module
    except ImportError:
        return
    cls = getattr(module, "AgentSpecLlmCallbackHandler", None) or getattr(
        module, "AgentSpecCallbackHandler", None
    )
    if cls is None:
        return
    state = {"active": True}
    _STATE = state

    def patch(owner, name, replacement):
        own = name in owner.__dict__
        original = inspect.getattr_static(owner, name, None)
        setattr(owner, name, replacement)
        _PATCHES.append((owner, name, original, replacement, own))

    def capture_method(original, kind):
        def data(args, kwargs):
            if kind == "start":
                source = kwargs.get("messages", args[1] if len(args) > 1 else [])
                return {"messages": messages(source[0]) if source else []}
            if kind == "end":
                response = args[0] if args else kwargs.get("response")
                generations = getattr(response, "generations", None) or []
                generation = (
                    generations[0][0] if generations and generations[0] else None
                )
                message = getattr(generation, "message", None)
                metadata = getattr(message, "response_metadata", None) or {}
                info = getattr(generation, "generation_info", None) or {}
                response_id = metadata.get("id") or getattr(message, "id", None)
                if not isinstance(response_id, str) or response_id.startswith(
                    ("lc_run-", "run-")
                ):
                    response_id = None
                return {
                    "usage": usage(response),
                    "completion": messages([message])[0] if message is not None else {},
                    "response_id": response_id,
                    "finish_reason": metadata.get("finish_reason")
                    or info.get("finish_reason"),
                }
            return {"error": args[0] if args else kwargs.get("error")}

        def safe_data(args, kwargs):
            try:
                return data(args, kwargs)
            except Exception:  # noqa: BLE001 - telemetry cannot change callback behavior
                return {}

        def close_error(self, kwargs):
            from pyagentspec.tracing.events import ExceptionRaised

            run_id = str(kwargs["run_id"])
            span = self.agentspec_spans_registry.get(run_id)
            if span is None:
                return None
            error = _CAPTURE.get()["error"]
            from ._processor import safe_error

            detail = "; ".join(
                arg for arg in getattr(error, "args", ()) if isinstance(arg, str)
            )
            event = ExceptionRaised(
                exception_type=type(error).__name__,
                exception_message=safe_error(detail or type(error).__name__),
                exception_stacktrace="",
            )
            return run_id, span, event

        @wraps(original)
        def wrapper(self, *args, **kwargs):
            if not state["active"]:
                return original(self, *args, **kwargs)
            token = _CAPTURE.set(safe_data(args, kwargs))
            try:
                return original(self, *args, **kwargs)
            finally:
                try:
                    if kind in {"error", "tool_error"} and (
                        closed := close_error(self, kwargs)
                    ):
                        run_id, span, event = closed
                        self._add_event(run_id, span, event)
                        self._end_span(run_id, span)
                        self.agentspec_spans_registry.pop(run_id, None)
                        self.messages_in_process.pop(run_id, None)
                except Exception:
                    logger.debug("Could not close failed AgentSpec span", exc_info=True)
                finally:
                    _CAPTURE.reset(token)

        @wraps(original)
        async def async_wrapper(self, *args, **kwargs):
            if not state["active"]:
                return await original(self, *args, **kwargs)
            token = _CAPTURE.set(safe_data(args, kwargs))
            try:
                return await original(self, *args, **kwargs)
            finally:
                try:
                    if kind in {"error", "tool_error"} and (
                        closed := close_error(self, kwargs)
                    ):
                        run_id, span, event = closed
                        await self._add_event_async(run_id, span, event)
                        await self._end_span_async(run_id, span)
                        self.agentspec_spans_registry.pop(run_id, None)
                        self.messages_in_process.pop(run_id, None)
                except Exception:
                    logger.debug("Could not close failed AgentSpec span", exc_info=True)
                finally:
                    _CAPTURE.reset(token)

        return async_wrapper if inspect.iscoroutinefunction(original) else wrapper

    try:
        for name, kind in [
            ("on_chat_model_start", "start"),
            ("on_llm_end", "end"),
            ("on_llm_error", "error"),
        ]:
            for suffix in ("", "_async"):
                method = name + suffix
                original = getattr(cls, method, None)
                if original is None and kind == "error" and suffix == "_async":

                    async def original(self, *args, **kwargs):
                        return None

                if original is not None:
                    patch(cls, method, capture_method(original, kind))
        tool_cls = getattr(module, "AgentSpecToolCallbackHandler", cls)
        for method in ("on_tool_error", "on_tool_error_async"):
            original = getattr(tool_cls, method, None)
            if original is not None:
                patch(tool_cls, method, capture_method(original, "tool_error"))
        # Older SDK handlers re-enter a Context captured at request start.
        # Carry callback-only response metadata through that native boundary.
        context_cls = getattr(module, "AgentSpecCallbackHandler", cls)
        original_context = getattr(context_cls, "_run_in_ctx", None)
        if original_context is not None:

            @wraps(original_context)
            def run_in_context(self, run_id, function, *args, **kwargs):
                if not state["active"]:
                    return original_context(self, run_id, function, *args, **kwargs)
                captured = _CAPTURE.get()

                def invoke(*inner_args, **inner_kwargs):
                    token = _CAPTURE.set(captured)
                    try:
                        return function(*inner_args, **inner_kwargs)
                    finally:
                        _CAPTURE.reset(token)

                return original_context(self, run_id, invoke, *args, **kwargs)

            patch(context_cls, "_run_in_ctx", run_in_context)
        # Unlike native success callbacks, on_llm_error is absent from the
        # SDK's dispatch table and must be opted into async dispatch.
        original_init = cls.__init__

        @wraps(original_init)
        def initialize(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if state["active"]:
                getattr(self, "_events_handled", set()).add("on_llm_error")

        patch(cls, "__init__", initialize)
    except BaseException:
        restore()
        raise


def restore():
    global _STATE
    if _STATE is not None:
        _STATE["active"] = False
    for owner, name, original, wrapper, own in reversed(_PATCHES):
        if inspect.getattr_static(owner, name) is wrapper:
            if own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
    _PATCHES.clear()
    _STATE = None
