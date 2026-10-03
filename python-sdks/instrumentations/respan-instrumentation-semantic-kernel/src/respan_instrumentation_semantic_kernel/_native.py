"""Scoped supplements for gaps in Semantic Kernel's native diagnostics."""

from __future__ import annotations

import contextvars
import importlib
import inspect
import json
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
from typing import Any

from opentelemetry import context, trace
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.llm_logging import LOG_TYPE_TASK, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.decorators.base import _should_send_prompts
from wrapt import FunctionWrapper

from ._constants import (
    SEMANTIC_KERNEL_MODEL_DIAGNOSTICS_LOGGER,
    SK_CONTENT_CAPTURE_ATTRIBUTE,
)

logger = logging.getLogger(__name__)


@dataclass
class _HookState:
    active: bool = True


_EMBEDDING_SPAN: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "respan_sk_embedding_span", default=None
)


def suppressed() -> bool:
    return bool(context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY))


def capture_content(enabled: bool) -> bool:
    snapshot = getattr(trace.get_current_span(), SK_CONTENT_CAPTURE_ATTRIBUTE, True)
    return enabled and snapshot and _should_send_prompts() and not suppressed()


def _usage(span: Any, value: Any) -> None:
    try:
        _record_usage(span, value)
    except Exception:
        logger.debug("Could not capture provider usage", exc_info=True)


def _record_usage(span: Any, value: Any) -> None:
    if value is None or not span.is_recording():
        return
    for name, key in (
        ("prompt_tokens", gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS),
        ("completion_tokens", gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS),
    ):
        count = getattr(value, name, None)
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            span.set_attribute(key, count)
    for details, name, key in (
        (
            "prompt_tokens_details",
            "cached_tokens",
            SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
        ),
        (
            "completion_tokens_details",
            "reasoning_tokens",
            SpanAttributes.LLM_USAGE_REASONING_TOKENS,
        ),
    ):
        count = getattr(getattr(value, details, None), name, None)
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            span.set_attribute(key, count)


def _content_json(value: Any) -> str:
    return json.dumps(value)


def install_native_hooks(enabled: bool) -> list[tuple[Any, str, Any, Any]]:
    state = _HookState()
    patches: list[tuple[Any, str, Any, Any]] = [(state, "active", False, True)]

    def patch(owner, name, replacement):
        original = inspect.getattr_static(owner, name)
        setattr(owner, name, replacement)
        patches.append((owner, name, original, replacement))
        return original

    try:
        model = importlib.import_module(
            "semantic_kernel.utils.telemetry.model_diagnostics.decorators"
        )
        function = importlib.import_module(
            "semantic_kernel.utils.telemetry.model_diagnostics.function_tracer"
        )
        agent = importlib.import_module(
            "semantic_kernel.utils.telemetry.agent_diagnostics.decorators"
        )
        for module in (model, function, agent):
            for name in (
                "are_model_diagnostics_enabled",
                "are_sensitive_events_enabled",
            ):
                if not hasattr(module, name):
                    continue
                original = getattr(module, name)
                sensitive = name == "are_sensitive_events_enabled"

                def check(original=original, sensitive=sensitive):
                    if not state.active:
                        return original()
                    return (
                        original()
                        and not suppressed()
                        and (not sensitive or capture_content(enabled))
                    )

                patch(module, name, check)

        original_function_span = function.start_as_current_span

        @wraps(original_function_span)
        def function_span(*args, **kwargs):
            if not state.active:
                return original_function_span(*args, **kwargs)
            if suppressed():
                return nullcontext(
                    trace.NonRecordingSpan(trace.get_current_span().get_span_context())
                )

            @contextmanager
            def classified_span():
                with original_function_span(*args, **kwargs) as span:
                    kernel_function = (
                        args[1] if len(args) > 1 else kwargs.get("function")
                    )
                    metadata = getattr(kernel_function, "metadata", None)
                    if state.active and metadata is not None:
                        span.set_attribute(
                            RESPAN_LOG_TYPE,
                            LOG_TYPE_TASK if metadata.is_prompt else LOG_TYPE_TOOL,
                        )
                    yield span

            return classified_span()

        patch(function, "start_as_current_span", function_span)
        chat_base = importlib.import_module(
            "semantic_kernel.connectors.ai.chat_completion_client_base"
        )
        original_loop_tracer = chat_base.tracer

        class LoopTracer:
            def __getattr__(self, name):
                return getattr(original_loop_tracer, name)

            def start_span(self, *args, **kwargs):
                if state.active and suppressed():
                    return trace.NonRecordingSpan(
                        trace.get_current_span().get_span_context()
                    )
                return original_loop_tracer.start_span(*args, **kwargs)

            def start_as_current_span(self, *args, **kwargs):
                if state.active and suppressed():
                    return nullcontext(
                        trace.NonRecordingSpan(
                            trace.get_current_span().get_span_context()
                        )
                    )
                return original_loop_tracer.start_as_current_span(*args, **kwargs)

        patch(chat_base, "tracer", LoopTracer())

        original_response = model._set_completion_response

        @wraps(original_response)
        def completion_response(span, completions, provider):
            if not state.active:
                return original_response(span, completions, provider)
            if not completions:
                return None
            result = original_response(span, completions, provider)
            if completions:
                _usage(span, completions[0].metadata.get("usage"))
            return result

        patch(model, "_set_completion_response", completion_response)
        handler = importlib.import_module(
            "semantic_kernel.connectors.ai.open_ai.services.open_ai_handler"
        ).OpenAIHandler
        original_usage = handler.store_usage

        @wraps(original_usage)
        def store_usage(service, response):
            result = original_usage(service, response)
            if not state.active:
                return result
            current = _EMBEDDING_SPAN.get()
            if current is not None and current[0] is service:
                _usage(current[1], getattr(response, "usage", None))
            else:
                span = trace.get_current_span()
                scope = getattr(
                    getattr(span, "instrumentation_scope", None), "name", ""
                )
                if scope == SEMANTIC_KERNEL_MODEL_DIAGNOSTICS_LOGGER:
                    _usage(span, getattr(response, "usage", None))
            return result

        patch(handler, "store_usage", store_usage)
        original_embedding = handler._send_embedding_request

        @wraps(original_embedding)
        async def embedding(service, settings):
            if not state.active or suppressed():
                return await original_embedding(service, settings)
            tracer = trace.get_tracer("semantic_kernel.respan.embeddings")
            attrs = {
                gen_ai_attributes.GEN_AI_OPERATION_NAME: "embeddings",
                SpanAttributes.LLM_REQUEST_MODEL: settings.ai_model_id
                or service.ai_model_id,
                SpanAttributes.LLM_SYSTEM: "azure"
                if type(service).__name__.startswith("Azure")
                else "openai",
            }
            with tracer.start_as_current_span("embeddings", attributes=attrs) as span:
                token = _EMBEDDING_SPAN.set((service, span))
                try:
                    if span.is_recording() and capture_content(enabled):
                        try:
                            span.set_attribute(
                                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                                _content_json(settings.input),
                            )
                        except Exception:
                            logger.debug(
                                "Could not capture embedding input", exc_info=True
                            )
                    result = await original_embedding(service, settings)
                    if (
                        state.active
                        and span.is_recording()
                        and capture_content(enabled)
                    ):
                        try:
                            span.set_attribute(
                                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                                _content_json(result),
                            )
                        except Exception:
                            logger.debug(
                                "Could not capture embedding output", exc_info=True
                            )
                    return result
                finally:
                    _EMBEDDING_SPAN.reset(token)

        patch(handler, "_send_embedding_request", embedding)

        # Native agent decorators read messages only from positional args. Keep
        # the SDK's original coroutine/iterator while normalizing this argument.
        def normalize_agent_method(method):
            try:
                parameters = tuple(inspect.signature(method).parameters.values())
                positional_messages = (
                    len(parameters) > 1
                    and parameters[1].name == "messages"
                    and parameters[1].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                )
            except (TypeError, ValueError):
                positional_messages = False

            def normalize_agent_call(wrapped, instance, args, kwargs):
                if (
                    state.active
                    and positional_messages
                    and not args
                    and "messages" in kwargs
                ):
                    kwargs = dict(kwargs)
                    args = (kwargs.pop("messages"),)
                return wrapped(*args, **kwargs)

            wrapped = FunctionWrapper(method, normalize_agent_call)
            patches.append((base, "__respan_agent_method__", method, wrapped))
            return wrapped

        base = importlib.import_module("semantic_kernel.agents.agent").Agent
        for name in (
            "trace_agent_get_response",
            "trace_agent_invocation",
            "trace_agent_streaming_invocation",
        ):
            decorator = getattr(agent, name)

            def decorate(method, decorator=decorator):
                decorated = decorator(method)
                return normalize_agent_method(decorated) if state.active else decorated

            patch(agent, name, decorate)

        pending = list(base.__subclasses__())
        visited = set()
        while pending:
            cls = pending.pop()
            if cls in visited:
                continue
            visited.add(cls)
            pending.extend(cls.__subclasses__())
            for name in ("get_response", "invoke", "invoke_stream"):
                method = cls.__dict__.get(name)
                if method is not None and getattr(
                    method, "__agent_diagnostics__", False
                ):
                    patch(cls, name, normalize_agent_method(method))
        return patches
    except Exception:
        restore_native_hooks(patches)
        raise


def restore_native_hooks(patches: list[tuple[Any, str, Any, Any]]) -> None:
    for owner, _, _, _ in patches:
        if isinstance(owner, _HookState):
            owner.active = False
    for owner, name, original, replacement in reversed(patches):
        if name == "__respan_agent_method__":
            pending = list(owner.__subclasses__())
            visited = set()
            while pending:
                cls = pending.pop()
                if cls in visited:
                    continue
                visited.add(cls)
                pending.extend(cls.__subclasses__())
                for method_name, value in list(cls.__dict__.items()):
                    if value is replacement:
                        setattr(cls, method_name, original)
        elif inspect.getattr_static(owner, name, None) is replacement:
            setattr(owner, name, original)
