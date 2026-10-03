"""IBM watsonx Orchestrate ADK instrumentation plugin for Respan."""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
)
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import (
    build_readable_span,
    read_propagated_attributes,
)

from respan_instrumentation_watson_orchestrate_adk import _otel_emitter
from respan_instrumentation_watson_orchestrate_adk._constants import (
    AGENT_BUILDER_CLIENT_CLASS,
    AGENT_BUILDER_CLIENT_MODULE,
    ASYNC_RUN_METHODS,
    CHAT_METHODS,
    CHAT_REFINEMENT_METHODS,
    CPE_CLIENT_CLASS,
    CPE_CLIENT_MODULE,
    LLM_CHAT_METHODS,
    PYTHON_TOOL_CLASS,
    PYTHON_TOOL_MODULE,
    RUN_CLIENT_CLASS,
    RUN_CLIENT_MODULE,
    RUN_METHODS,
    TOOL_CALL_METHOD,
    WATSON_ORCHESTRATE_ADK_INSTRUMENTATION_NAME,
    WATSON_ORCHESTRATE_CHAT_SPAN_NAME,
    WATSON_ORCHESTRATE_RUN_SPAN_NAME,
    WATSON_ORCHESTRATE_TOOL_SPAN_NAME,
    WATSONX_AI_CLIENT_CLASS,
    WATSONX_AI_CLIENT_MODULE,
)
from respan_instrumentation_watson_orchestrate_adk._serialization import (
    provider_status_code,
    safe_exception_message,
    safe_text,
)

logger = logging.getLogger(__name__)

_LOCK = RLock()
_ACTIVATION_COUNT = 0
_PATCH_EPOCH = 0


@dataclass(frozen=True)
class _Patch:
    cls: type[Any]
    method_name: str
    original: Any
    wrapper: Any


_PATCHES: list[_Patch] = []


def _load_class(module_name: str, class_name: str) -> type[Any]:
    module = importlib.import_module(module_name)
    value = getattr(module, class_name, None)
    if value is None:
        raise AttributeError(f"{module_name}.{class_name}")
    return value


def _current_trace_parent_ids() -> tuple[str | None, str | None]:
    try:
        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return None, None
        return format_trace_id(context.trace_id), format_span_id(context.span_id)
    except BaseException:  # noqa: BLE001
        return None, None


def _safe_attr(instance: Any, name: str) -> Any:
    try:
        return getattr(instance, name, None)
    except BaseException:  # noqa: BLE001
        return None


def _tool_name(instance: Any) -> str:
    for key in ("name", "display_name"):
        value = _safe_attr(instance, key)
        if value:
            return safe_text(value, max_bytes=256)
    spec = _safe_attr(instance, "__tool_spec__")
    value = _safe_attr(spec, "name")
    if value:
        return safe_text(value, max_bytes=256)
    fn = _safe_attr(instance, "fn")
    value = _safe_attr(fn, "__name__")
    if value:
        return safe_text(value, max_bytes=256)
    return safe_text(type(instance).__name__, max_bytes=256)


def _call_kwargs(
    *,
    original: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        bound = inspect.signature(original).bind_partial(instance, *args, **kwargs)
    except (TypeError, ValueError):
        result = dict(kwargs)
        if args:
            result["_args"] = list(args[:50])
        return result
    result = {key: value for key, value in bound.arguments.items() if key != "self"}
    nested_kwargs = result.pop("kwargs", None)
    if isinstance(nested_kwargs, dict):
        nested = dict(nested_kwargs)
        nested.update(result)
        result = nested
    positional = result.pop("args", None)
    if isinstance(positional, tuple) and positional:
        result["_args"] = list(positional[:50])
    return result


def _emit_error_kwargs(exc: BaseException) -> dict[str, Any]:
    return {
        "error_message": safe_exception_message(exc),
        "status_code": provider_status_code(exc),
    }


@contextmanager
def _operation(kind):
    name = {
        "chat": WATSON_ORCHESTRATE_CHAT_SPAN_NAME,
        "tool": WATSON_ORCHESTRATE_TOOL_SPAN_NAME,
    }.get(kind, WATSON_ORCHESTRATE_RUN_SPAN_NAME)
    # Snapshot policy and attribution before provider callbacks can change context.
    include_content = context_api.get_value(
        ENABLE_CONTENT_TRACING_KEY
    ) is not False and os.getenv(
        "TRACELOOP_TRACE_CONTENT", "true"
    ).strip().lower() not in {"false", "0", "no", "off"}
    trace_id, parent_id = _current_trace_parent_ids()
    skeleton = build_readable_span(
        name=name,
        trace_id=trace_id,
        parent_id=parent_id,
        merge_propagated=False,
    )
    sampler = getattr(trace.get_tracer_provider(), "sampler", None)
    if (
        sampler is not None
        and not sampler.should_sample(
            context_api.get_current(), skeleton.context.trace_id, name
        ).decision.is_sampled()
    ):
        dropped = trace.SpanContext(
            skeleton.context.trace_id,
            skeleton.context.span_id,
            False,
            trace.TraceFlags(0),
        )
        with trace.use_span(trace.NonRecordingSpan(dropped), end_on_exit=False):
            yield None
        return
    snapshot = {
        "trace_id": format_trace_id(skeleton.context.trace_id),
        "parent_id": parent_id,
        "span_id": format_span_id(skeleton.context.span_id),
        "propagated": read_propagated_attributes(),
        "include_content": include_content,
    }
    with trace.use_span(trace.NonRecordingSpan(skeleton.context), end_on_exit=False):
        yield snapshot


def _enabled(epoch: int) -> bool:
    parent = trace.get_current_span().get_span_context()
    if parent.is_valid and not parent.trace_flags.sampled:
        return False
    return (
        _ACTIVATION_COUNT > 0
        and epoch == _PATCH_EPOCH
        and WatsonOrchestrateADKInstrumentor._is_respan_tracing_enabled()
        and not context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
    )


def _emit(emitter, values, snapshot, response=None, error=None):
    if snapshot is None:
        return
    try:
        error_kwargs = _emit_error_kwargs(error) if error is not None else {}
        if error is not None and not snapshot["include_content"]:
            error_kwargs["error_message"] = type(error).__name__
        emitter(response=response, **values, **snapshot, **error_kwargs)
    except Exception:
        logger.debug("Failed to capture Watson call", exc_info=True)


def _wrap_method(method_name, original, kind):
    epoch = _PATCH_EPOCH

    def values(instance, args, kwargs):
        common = {"start_ns": time.time_ns()}
        if kind == "tool":
            return _otel_emitter.emit_tool_span, {
                **common,
                "tool_name": _tool_name(instance),
                "args": args,
                "kwargs": kwargs,
            }
        common.update(
            method_name=method_name,
            call_kwargs=_call_kwargs(
                original=original, instance=instance, args=args, kwargs=kwargs
            ),
        )
        if kind == "chat":
            if method_name == "generate_response":
                common["call_kwargs"] = {
                    key: value
                    for key, value in common["call_kwargs"].items()
                    if key in {"input", "instructions", "model"}
                }
            common["instance"] = instance
            return _otel_emitter.emit_chat_span, common
        return _otel_emitter.emit_agent_run_span, common

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        if not _enabled(epoch):
            return original(self, *args, **kwargs)
        emitter, call = values(self, args, kwargs)
        with _operation(kind) as snapshot:
            try:
                response = original(self, *args, **kwargs)
            except BaseException as exc:
                _emit(emitter, call, snapshot, error=exc)
                raise
            _emit(emitter, call, snapshot, response=response)
            return response

    @functools.wraps(original)
    async def async_wrapper(self, *args, **kwargs):
        if not _enabled(epoch):
            return await original(self, *args, **kwargs)
        emitter, call = values(self, args, kwargs)
        with _operation(kind) as snapshot:
            try:
                response = await original(self, *args, **kwargs)
            except BaseException as exc:
                _emit(emitter, call, snapshot, error=exc)
                raise
            _emit(emitter, call, snapshot, response=response)
            return response

    return async_wrapper if inspect.iscoroutinefunction(original) else wrapper


def _wrap_tool_call(original):
    return _wrap_method("__call__", original, "tool")


def _wrap_agent_run(method_name, original):
    return _wrap_method(method_name, original, "agent")


def _wrap_async_agent_run(method_name, original):
    return _wrap_method(method_name, original, "agent")


def _wrap_chat_method(method_name, original):
    return _wrap_method(method_name, original, "chat")


def _install(
    installed: list[_Patch],
    cls: type[Any],
    method_name: str,
    factory: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> None:
    original = getattr(cls, method_name, None)
    if original is None:
        return
    wrapper = factory(original)
    setattr(cls, method_name, wrapper)
    installed.append(_Patch(cls, method_name, original, wrapper))


def _optional_class(module_name: str, class_name: str) -> type[Any] | None:
    try:
        return _load_class(module_name, class_name)
    except (ImportError, AttributeError):
        return None


def _restore_owned(patches: list[_Patch]) -> None:
    for patch in reversed(patches):
        if getattr(patch.cls, patch.method_name, None) is patch.wrapper:
            setattr(patch.cls, patch.method_name, patch.original)


class WatsonOrchestrateADKInstrumentor:
    """Respan instrumentor for IBM watsonx Orchestrate ADK."""

    name = WATSON_ORCHESTRATE_ADK_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        return tracer is None or bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        global _ACTIVATION_COUNT, _PATCH_EPOCH
        with _LOCK:
            if self._is_instrumented:
                return
            if not self._is_respan_tracing_enabled():
                return
            if _ACTIVATION_COUNT:
                _ACTIVATION_COUNT += 1
                self._is_instrumented = True
                return
            _PATCH_EPOCH += 1
            installed: list[_Patch] = []
            try:
                tool = _optional_class(PYTHON_TOOL_MODULE, PYTHON_TOOL_CLASS)
                if tool is not None:
                    _install(installed, tool, TOOL_CALL_METHOD, _wrap_tool_call)
                run_client = _optional_class(RUN_CLIENT_MODULE, RUN_CLIENT_CLASS)
                if run_client is not None:
                    for method_name in RUN_METHODS:
                        _install(
                            installed,
                            run_client,
                            method_name,
                            lambda original, name=method_name: _wrap_agent_run(
                                name, original
                            ),
                        )
                    for method_name in ASYNC_RUN_METHODS:
                        _install(
                            installed,
                            run_client,
                            method_name,
                            lambda original, name=method_name: _wrap_async_agent_run(
                                name, original
                            ),
                        )
                flow_client = _optional_class(
                    "ibm_watsonx_orchestrate_clients.tools.tempus_client",
                    "TempusClient",
                )
                if flow_client is not None:
                    for method_name in ("run_flow", "arun_flow"):
                        _install(
                            installed,
                            flow_client,
                            method_name,
                            lambda original, name=method_name: _wrap_agent_run(
                                name, original
                            ),
                        )
                for module_name, class_name, methods in (
                    (
                        AGENT_BUILDER_CLIENT_MODULE,
                        AGENT_BUILDER_CLIENT_CLASS,
                        CHAT_METHODS,
                    ),
                    (
                        CPE_CLIENT_MODULE,
                        CPE_CLIENT_CLASS,
                        (*CHAT_METHODS, *CHAT_REFINEMENT_METHODS),
                    ),
                    (
                        "ibm_watsonx_orchestrate.client.autodiscover.groq.groq_client",
                        "GroqClient",
                        LLM_CHAT_METHODS,
                    ),
                    (
                        "ibm_watsonx_orchestrate.client.autodiscover.ai_gateway.ai_gateway_client",
                        "AIGatewayClient",
                        LLM_CHAT_METHODS,
                    ),
                    (
                        WATSONX_AI_CLIENT_MODULE,
                        WATSONX_AI_CLIENT_CLASS,
                        LLM_CHAT_METHODS,
                    ),
                ):
                    cls = _optional_class(module_name, class_name)
                    if cls is None:
                        continue
                    for method_name in methods:
                        _install(
                            installed,
                            cls,
                            method_name,
                            lambda original, name=method_name: _wrap_chat_method(
                                name, original
                            ),
                        )
            except BaseException:
                _restore_owned(installed)
                raise
            if not installed:
                logger.warning(
                    "Watson Orchestrate ADK instrumentation found no supported SDK classes"
                )
                return
            _PATCHES[:] = installed
            _ACTIVATION_COUNT = 1
            self._is_instrumented = True

    def deactivate(self) -> None:
        global _ACTIVATION_COUNT
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _ACTIVATION_COUNT = max(0, _ACTIVATION_COUNT - 1)
            if _ACTIVATION_COUNT:
                return
            _restore_owned(_PATCHES)
            _PATCHES.clear()


def _restore_methods() -> None:
    """Compatibility test helper; restore only wrappers owned by this runtime."""
    global _ACTIVATION_COUNT
    with _LOCK:
        _restore_owned(_PATCHES)
        _PATCHES.clear()
        _ACTIVATION_COUNT = 0
