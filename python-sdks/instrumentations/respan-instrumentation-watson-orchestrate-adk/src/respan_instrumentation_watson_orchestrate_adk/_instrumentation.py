"""IBM watsonx Orchestrate ADK instrumentation plugin for Respan."""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import time
from typing import Any, Callable

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
    WATSONX_AI_CLIENT_CLASS,
    WATSONX_AI_CLIENT_MODULE,
    WATSON_ORCHESTRATE_ADK_INSTRUMENTATION_NAME,
)
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_OriginalKey = tuple[type[Any], str]
_original_methods: dict[_OriginalKey, Any] = {}


def _load_class(module_name: str, class_name: str) -> type[Any]:
    module = importlib.import_module(module_name)
    class_value = getattr(module, class_name, None)
    if class_value is None:
        raise AttributeError(f"{module_name}.{class_name}")
    return class_value


def _tool_name(instance: Any) -> str:
    for key in ("name", "display_name"):
        value = getattr(instance, key, None)
        if value:
            return str(value)

    spec = getattr(instance, "__tool_spec__", None)
    value = getattr(spec, "name", None)
    if value:
        return str(value)

    fn = getattr(instance, "fn", None)
    value = getattr(fn, "__name__", None)
    if value:
        return str(value)

    return instance.__class__.__name__


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
            result["_args"] = list(args)
        return result

    result = {
        key: value
        for key, value in bound.arguments.items()
        if key != "self"
    }
    if "kwargs" in result and isinstance(result["kwargs"], dict):
        nested = dict(result.pop("kwargs"))
        nested.update(result)
        result = nested
    if "args" in result and isinstance(result["args"], tuple):
        positional = result.pop("args")
        if positional:
            result["_args"] = list(positional)
    return result


def _wrap_tool_call(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        tool_name = _tool_name(self)
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            _otel_emitter.emit_tool_span(
                tool_name=tool_name,
                args=args,
                kwargs=kwargs,
                start_ns=start_ns,
                error_message=str(exc),
            )
            raise

        _otel_emitter.emit_tool_span(
            tool_name=tool_name,
            args=args,
            kwargs=kwargs,
            start_ns=start_ns,
            response=response,
        )
        return response

    return wrapper


def _wrap_agent_run(
    *,
    method_name: str,
    original: Callable[..., Any],
) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        call_kwargs = _call_kwargs(
            original=original,
            instance=self,
            args=args,
            kwargs=kwargs,
        )
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            _otel_emitter.emit_agent_run_span(
                method_name=method_name,
                call_kwargs=call_kwargs,
                start_ns=start_ns,
                error_message=str(exc),
            )
            raise

        _otel_emitter.emit_agent_run_span(
            method_name=method_name,
            call_kwargs=call_kwargs,
            start_ns=start_ns,
            response=response,
        )
        return response

    return wrapper


def _wrap_async_agent_run(
    *,
    method_name: str,
    original: Callable[..., Any],
) -> Callable[..., Any]:
    @functools.wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        call_kwargs = _call_kwargs(
            original=original,
            instance=self,
            args=args,
            kwargs=kwargs,
        )
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            _otel_emitter.emit_agent_run_span(
                method_name=method_name,
                call_kwargs=call_kwargs,
                start_ns=start_ns,
                error_message=str(exc),
            )
            raise

        _otel_emitter.emit_agent_run_span(
            method_name=method_name,
            call_kwargs=call_kwargs,
            start_ns=start_ns,
            response=response,
        )
        return response

    return wrapper


def _wrap_chat_method(
    *,
    method_name: str,
    original: Callable[..., Any],
) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        call_kwargs = _call_kwargs(
            original=original,
            instance=self,
            args=args,
            kwargs=kwargs,
        )
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            _otel_emitter.emit_chat_span(
                method_name=method_name,
                call_kwargs=call_kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                instance=self,
            )
            raise

        _otel_emitter.emit_chat_span(
            method_name=method_name,
            call_kwargs=call_kwargs,
            start_ns=start_ns,
            response=response,
            instance=self,
        )
        return response

    return wrapper


def _patch_method(
    cls: type[Any],
    method_name: str,
    wrapper_factory: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> bool:
    original = getattr(cls, method_name, None)
    if original is None:
        return False

    key = (cls, method_name)
    if key in _original_methods:
        return False

    _original_methods[key] = original
    setattr(cls, method_name, wrapper_factory(original))
    return True


def _restore_methods() -> None:
    for (cls, method_name), original in list(_original_methods.items()):
        setattr(cls, method_name, original)
    _original_methods.clear()


class WatsonOrchestrateADKInstrumentor:
    """Respan instrumentor for IBM watsonx Orchestrate ADK.

    The IBM package exposes a mix of local Python tool objects and generated
    REST clients. This instrumentor patches only stable public methods and
    emits canonical Respan/OpenTelemetry spans without requiring the SDK at
    import time.
    """

    name = WATSON_ORCHESTRATE_ADK_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def _patch_optional_class(
        self,
        *,
        module_name: str,
        class_name: str,
        patcher: Callable[[type[Any]], int],
    ) -> int:
        try:
            cls = _load_class(module_name, class_name)
        except (ImportError, AttributeError) as exc:
            logger.debug(
                "Watson Orchestrate ADK instrumentation skipped %s.%s: %s",
                module_name,
                class_name,
                exc,
            )
            return 0
        return patcher(cls)

    def activate(self) -> None:
        """Activate native Watson Orchestrate ADK instrumentation."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Watson Orchestrate ADK instrumentation skipped because Respan tracing is disabled"
            )
            return

        patched_count = 0
        try:
            patched_count += self._patch_optional_class(
                module_name=PYTHON_TOOL_MODULE,
                class_name=PYTHON_TOOL_CLASS,
                patcher=lambda cls: int(
                    _patch_method(cls, TOOL_CALL_METHOD, _wrap_tool_call)
                ),
            )
            patched_count += self._patch_optional_class(
                module_name=RUN_CLIENT_MODULE,
                class_name=RUN_CLIENT_CLASS,
                patcher=self._patch_run_client,
            )
            patched_count += self._patch_optional_class(
                module_name=AGENT_BUILDER_CLIENT_MODULE,
                class_name=AGENT_BUILDER_CLIENT_CLASS,
                patcher=self._patch_chat_client,
            )
            patched_count += self._patch_optional_class(
                module_name=CPE_CLIENT_MODULE,
                class_name=CPE_CLIENT_CLASS,
                patcher=self._patch_cpe_client,
            )
            patched_count += self._patch_optional_class(
                module_name=WATSONX_AI_CLIENT_MODULE,
                class_name=WATSONX_AI_CLIENT_CLASS,
                patcher=self._patch_llm_client,
            )
        except Exception:
            _restore_methods()
            self._is_instrumented = False
            logger.exception("Failed to activate Watson Orchestrate ADK instrumentation")
            return

        self._is_instrumented = patched_count > 0
        if not self._is_instrumented:
            logger.warning(
                "Watson Orchestrate ADK instrumentation found no supported SDK classes to patch"
            )
            return
        logger.info("Watson Orchestrate ADK instrumentation activated")

    def _patch_run_client(self, cls: type[Any]) -> int:
        patched = 0
        for method_name in RUN_METHODS:
            patched += int(
                _patch_method(
                    cls,
                    method_name,
                    lambda original, method_name=method_name: _wrap_agent_run(
                        method_name=method_name,
                        original=original,
                    ),
                )
            )
        for method_name in ASYNC_RUN_METHODS:
            patched += int(
                _patch_method(
                    cls,
                    method_name,
                    lambda original, method_name=method_name: _wrap_async_agent_run(
                        method_name=method_name,
                        original=original,
                    ),
                )
            )
        return patched

    def _patch_chat_client(self, cls: type[Any]) -> int:
        patched = 0
        for method_name in CHAT_METHODS:
            patched += int(
                _patch_method(
                    cls,
                    method_name,
                    lambda original, method_name=method_name: _wrap_chat_method(
                        method_name=method_name,
                        original=original,
                    ),
                )
            )
        return patched

    def _patch_cpe_client(self, cls: type[Any]) -> int:
        patched = 0
        for method_name in (*CHAT_METHODS, *CHAT_REFINEMENT_METHODS):
            patched += int(
                _patch_method(
                    cls,
                    method_name,
                    lambda original, method_name=method_name: _wrap_chat_method(
                        method_name=method_name,
                        original=original,
                    ),
                )
            )
        return patched

    def _patch_llm_client(self, cls: type[Any]) -> int:
        patched = 0
        for method_name in LLM_CHAT_METHODS:
            patched += int(
                _patch_method(
                    cls,
                    method_name,
                    lambda original, method_name=method_name: _wrap_chat_method(
                        method_name=method_name,
                        original=original,
                    ),
                )
            )
        return patched

    def deactivate(self) -> None:
        """Deactivate the instrumentation and restore original SDK methods."""
        if self._is_instrumented:
            _restore_methods()
        self._is_instrumented = False
        logger.info("Watson Orchestrate ADK instrumentation deactivated")
