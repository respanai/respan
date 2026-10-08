"""Own AutoGen boundaries while reusing OpenInference's payload serializers."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import os
from typing import Any

from openai import APIStatusError
from openinference.instrumentation import OITracer, TraceConfig, get_output_attributes
from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor
from openinference.instrumentation.autogen_agentchat import _wrappers as oi
from openinference.semconv.trace import OpenInferenceSpanKindValues as Kind
from openinference.semconv.trace import SpanAttributes as OI
from opentelemetry import context, trace
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_MESSAGE,
    ERROR_TYPE,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_TOOL_CALL_ID,
)
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)


def capture_content(parent_context=None):
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT", "true").lower() != "false"
        and context.get_value(ENABLE_CONTENT_TRACING_KEY, parent_context) is not False
    )


def _safe_attributes(function, *args, **kwargs):
    try:
        return dict(function(*args, **kwargs))
    except Exception:
        logger.debug("AutoGen telemetry serialization failed", exc_info=True)
        return {}


def _error(span, exc):
    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
    span.set_attribute(ERROR_TYPE, type(exc).__name__)
    span.set_attribute(ERROR_MESSAGE, str(exc))
    if isinstance(exc, APIStatusError):
        span.set_attribute(HTTP_RESPONSE_STATUS_CODE, exc.status_code)
    span.record_exception(exc)


def _arguments(wrapped, args, kwargs):
    # Binding observes the native signature; do not silently drop invalid kwargs.
    return inspect.signature(wrapped).bind(*args, **kwargs).arguments


def _model_attributes(instance, arguments, content):
    attrs = {OI.OPENINFERENCE_SPAN_KIND: Kind.LLM.value}
    attrs.update(_safe_attributes(oi._llm_model_name, instance))
    if content:
        attrs.update(
            _safe_attributes(
                oi._llm_messages_attributes, arguments.get("messages"), "input"
            )
        )
        from respan_instrumentation_autogen._instrumentation import (
            _respan_get_llm_tool_attributes,
        )

        attrs.update(
            _safe_attributes(_respan_get_llm_tool_attributes, arguments.get("tools"))
        )
    return attrs


def _model_output(result, content):
    attrs = {}
    usage = getattr(result, "usage", None)
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    valid = lambda value: (
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
    )
    if valid(prompt):
        attrs[OI.LLM_TOKEN_COUNT_PROMPT] = prompt
    if valid(completion):
        attrs[OI.LLM_TOKEN_COUNT_COMPLETION] = completion
    if valid(prompt) and valid(completion):
        attrs[OI.LLM_TOKEN_COUNT_TOTAL] = prompt + completion
    if content:
        attrs.update(_safe_attributes(get_output_attributes, result))
        attrs.update(_safe_attributes(oi._extract_output_message_attributes, result))
        attrs.update(_safe_attributes(oi._extract_output_tool_calls, result))
    return attrs


class _Stream:
    """Advance/close an SDK iterator in its captured context, never the consumer's."""

    def __init__(
        self,
        iterator,
        *,
        tracer=None,
        name=None,
        attributes=None,
        output=None,
        terminal=None,
        active=lambda: True,
    ):
        self._iterator = iterator
        self._active = active
        self._context = contextvars.copy_context()
        if not capture_content():
            self._context.run(
                lambda: context.attach(
                    context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
                )
            )
        self._tracer, self._name, self._attributes = tracer, name, attributes
        self._output, self._terminal = output, terminal
        self._span = None
        self._closed = False
        self._failed = False
        self._busy = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._operate("__anext__")

    async def asend(self, value):
        return await self._operate("asend", value)

    async def athrow(self, *args):
        return await self._operate("athrow", *args)

    async def aclose(self):
        return await self._operate("aclose")

    async def _operate(self, method, *args):
        if self._closed:
            if method == "aclose":
                return None
            raise StopAsyncIteration
        if self._busy:
            raise RuntimeError("AutoGen stream is already running")
        self._busy = True
        try:
            return await asyncio.create_task(
                self._advance(method, args), context=self._context
            )
        finally:
            self._busy = False

    async def _finish(self):
        self._closed = True
        try:
            await self._iterator.aclose()
        except BaseException as exc:
            if self._span is not None and not isinstance(exc, GeneratorExit):
                self._failed = True
                _error(self._span, exc)
            raise
        finally:
            if self._span is not None:
                if not self._failed:
                    self._span.set_status(trace.StatusCode.OK)
                self._span.end()

    async def _advance(self, method, args):
        if method == "aclose":
            await self._finish()
            return None
        if self._span is None and not self._active():
            self._tracer = None
        if self._span is None and self._tracer is not None:
            self._span = self._tracer.start_span(
                self._name, attributes=self._attributes
            )
            context.attach(trace.set_span_in_context(self._span))
        try:
            result = await getattr(self._iterator, method)(*args)
        except StopAsyncIteration:
            await self._finish()
            raise
        except BaseException as exc:
            if self._span is not None and not isinstance(exc, GeneratorExit):
                self._failed = True
                _error(self._span, exc)
            try:
                await self._finish()
            except BaseException:
                # Cleanup must not mask the original iteration/cancellation error.
                logger.debug(
                    "AutoGen cleanup failed after iteration error", exc_info=True
                )
            raise
        if self._terminal is not None and isinstance(result, self._terminal):
            if self._span is not None and self._output is not None:
                self._span.set_attributes(_safe_attributes(self._output, result))
            await self._finish()
        return result


def _stream_wrapper(tracer, kind, generation):
    def call(wrapped, instance, args, kwargs):
        iterator = wrapped(*args, **kwargs)
        if not generation["enabled"] or context.get_value(
            context._SUPPRESS_INSTRUMENTATION_KEY
        ):
            return iterator
        from autogen_agentchat.base import Response, TaskResult
        from autogen_core.models import CreateResult

        if kind is None:
            return _Stream(
                iterator, terminal=TaskResult, active=lambda: generation["enabled"]
            )
        content = capture_content()
        arguments = _arguments(wrapped, args, kwargs)
        if kind is Kind.LLM:
            attrs = _model_attributes(instance, arguments, content)
            output = lambda result: _model_output(result, content)
            terminal = CreateResult
            name = f"{type(instance).__name__}.create_stream"
        else:
            name = getattr(instance, "name", type(instance).__name__)
            attrs = {OI.OPENINFERENCE_SPAN_KIND: kind.value}
            if content:
                attrs.update(
                    _safe_attributes(
                        lambda: {
                            OI.INPUT_VALUE: oi._get_input_value(
                                wrapped, *args, **kwargs
                            )
                        }
                    )
                )
            output = get_output_attributes if content else lambda result: {}
            terminal = Response if kind is Kind.AGENT else TaskResult
        return _Stream(
            iterator,
            tracer=tracer,
            name=name,
            attributes=attrs,
            output=output,
            terminal=terminal,
            active=lambda: generation["enabled"],
        )

    return call


def _model_wrapper(tracer, generation):
    async def call(wrapped, instance, args, kwargs):
        if not generation["enabled"] or context.get_value(
            context._SUPPRESS_INSTRUMENTATION_KEY
        ):
            return await wrapped(*args, **kwargs)
        content = capture_content()
        arguments = _arguments(wrapped, args, kwargs)
        with tracer.start_as_current_span(
            f"{type(instance).__name__}.create",
            attributes=_model_attributes(instance, arguments, content),
        ) as span:
            try:
                result = await wrapped(*args, **kwargs)
            except BaseException as exc:
                _error(span, exc)
                raise
            span.set_attributes(_model_output(result, content))
            span.set_status(trace.StatusCode.OK)
            return result

    return call


def _tool_wrapper(tracer, generation):
    async def call(wrapped, instance, args, kwargs):
        if not generation["enabled"] or context.get_value(
            context._SUPPRESS_INSTRUMENTATION_KEY
        ):
            return await wrapped(*args, **kwargs)
        arguments = _arguments(wrapped, args, kwargs)
        tool_call = arguments.get("tool_call")
        if tool_call is None:
            return await wrapped(*args, **kwargs)
        content = capture_content()
        attrs = {
            OI.OPENINFERENCE_SPAN_KIND: Kind.TOOL.value,
            OI.TOOL_NAME: tool_call.name,
        }
        if tool_call.id:
            attrs[GEN_AI_TOOL_CALL_ID] = tool_call.id
        if content:
            attrs[OI.INPUT_VALUE] = tool_call.arguments
        with tracer.start_as_current_span(tool_call.name, attributes=attrs) as span:
            try:
                result = await wrapped(*args, **kwargs)
            except BaseException as exc:
                _error(span, exc)
                raise
            response = result[1] if isinstance(result, tuple) else result
            if content and hasattr(response, "content"):
                span.set_attribute(OI.OUTPUT_VALUE, response.content)
            if getattr(response, "is_error", False):
                message = str(response.content)
                span.set_status(trace.Status(trace.StatusCode.ERROR, message))
                span.set_attribute(ERROR_MESSAGE, message)
            else:
                span.set_status(trace.StatusCode.OK)
            return result

    return call


class ModernAutoGenInstrumentor(AutogenAgentChatInstrumentor):
    """Reuse the OI contract with transactional ownership and current SDK boundaries."""

    _instance = None

    def _instrument(self, **kwargs: Any):
        from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
        from autogen_agentchat.teams import BaseGroupChat
        from autogen_ext.models.openai import BaseOpenAIChatCompletionClient

        self._generation = {"enabled": False}
        self._owned = []
        if AutogenAgentChatInstrumentor().is_instrumented_by_opentelemetry:
            raise RuntimeError(
                "Deactivate the existing AutoGen OpenInference instrumentor first"
            )
        config = kwargs.get("config") or TraceConfig()
        if not isinstance(config, TraceConfig):
            raise TypeError("config must be TraceConfig")
        tracer = OITracer(
            trace.get_tracer(
                oi.__package__, tracer_provider=kwargs.get("tracer_provider")
            ),
            config=config,
        )
        self._generation = {"enabled": False}
        self._owned = []
        targets = [
            (
                AssistantAgent,
                "on_messages_stream",
                _stream_wrapper(tracer, Kind.AGENT, self._generation),
            ),
            (
                BaseChatAgent,
                "on_messages_stream",
                _stream_wrapper(tracer, Kind.AGENT, self._generation),
            ),
            (
                BaseChatAgent,
                "run_stream",
                _stream_wrapper(tracer, None, self._generation),
            ),
            (
                BaseGroupChat,
                "run_stream",
                _stream_wrapper(tracer, Kind.CHAIN, self._generation),
            ),
            (
                AssistantAgent,
                "_execute_tool_call",
                _tool_wrapper(tracer, self._generation),
            ),
            (
                BaseOpenAIChatCompletionClient,
                "create",
                _model_wrapper(tracer, self._generation),
            ),
            (
                BaseOpenAIChatCompletionClient,
                "create_stream",
                _stream_wrapper(tracer, Kind.LLM, self._generation),
            ),
        ]
        try:
            for owner, name, wrapper in targets:
                original = inspect.getattr_static(owner, name)
                wrap_function_wrapper(owner, name, wrapper)
                self._owned.append(
                    (owner, name, original, inspect.getattr_static(owner, name))
                )
            self._generation["enabled"] = True
        except BaseException:
            self._uninstrument()
            raise

    def _uninstrument(self, **kwargs):
        self._generation["enabled"] = False
        for owner, name, original, installed in reversed(self._owned):
            if inspect.getattr_static(owner, name) is installed:
                setattr(owner, name, original)
        self._owned.clear()
