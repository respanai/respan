"""Optional ADK 2.x Workflow nodes absent from OpenInference's agent hooks."""

from __future__ import annotations

import inspect
import json
import logging
import threading
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any

from openinference.semconv.trace import SpanAttributes as OIAttributes
from opentelemetry import context, trace
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_TOOL_CALL_ID,
)
from pydantic import BaseModel
from respan_instrumentation_openinference._serialization import bounded_json
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.decorators.base import _should_send_prompts
from wrapt import FunctionWrapper, ObjectProxy

from ._compat import _ContextPreservingIterator
from ._processor import _content_policy_override

_ADVISOR_START: ContextVar[Any] = ContextVar("respan_adk_advisor_start", default=None)

_NODE_TOOL: ContextVar[Any] = ContextVar("respan_adk_node_tool", default=None)


def _json(value: Any) -> str:
    """Bound telemetry without invoking arbitrary repr or changing SDK results."""
    try:
        if isinstance(value, BaseModel):
            value = value.model_dump(exclude_none=True)
        return bounded_json(value)
    except Exception:  # noqa: BLE001 -- telemetry serialization cannot alter SDK results
        return json.dumps({"unsupported_type": type(value).__name__})


def patch_workflow_nodes(tracer):
    if tracer is None:
        return None
    try:
        from google.adk.agents import BaseAgent
        from google.adk.workflow import BaseNode, Workflow
    except ImportError:
        return None
    patches = []
    patched_tools = set()
    lock = threading.RLock()
    active = True

    def patch(owner, name, callback):
        original = inspect.getattr_static(owner, name)
        replacement = FunctionWrapper(original, callback)
        setattr(owner, name, replacement)
        patches.append((owner, name, original, replacement))

    async def call_tool(wrapped, instance, args, kwargs):
        if (
            not active
            or _NODE_TOOL.get() is not instance
            or context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY)
        ):
            return await wrapped(*args, **kwargs)
        attributes = {
            OIAttributes.OPENINFERENCE_SPAN_KIND: "TOOL",
            RESPAN_LOG_TYPE: LOG_TYPE_TOOL,
            OIAttributes.TOOL_NAME: instance.name,
            OIAttributes.INPUT_VALUE: _json(
                {"name": instance.name, "arguments": kwargs.get("args", {})}
            ),
        }
        ctx = kwargs.get("tool_context")
        call_id = getattr(ctx, "function_call_id", None)
        if call_id:
            attributes[GEN_AI_TOOL_CALL_ID] = call_id
        with tracer.start_as_current_span(instance.name, attributes=attributes) as span:
            result = await wrapped(*args, **kwargs)
            if result is not None:
                span.set_attribute(OIAttributes.OUTPUT_VALUE, _json(result))
            span.set_status(trace.Status(trace.StatusCode.OK))
            return result

    def call_node(wrapped, instance, args, kwargs):
        source = wrapped(*args, **kwargs)
        if not active or context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY):
            return source
        # The agent wrapper delegates to the already-instrumented BaseAgent.
        if isinstance(instance, BaseAgent):
            return source
        tool = getattr(instance, "tool", None)
        if tool is not None:
            cls = type(tool)
            with lock:
                if cls not in patched_tools:
                    patch(cls, "run_async", call_tool)
                    patched_tools.add(cls)
        current = trace.get_current_span()
        native_span = (
            current
            if getattr(current, "name", "")
            in {f"invoke_node {instance.name}", f"invoke_workflow {instance.name}"}
            else None
        )
        attributes = {
            OIAttributes.OPENINFERENCE_SPAN_KIND: "CHAIN",
            RESPAN_LOG_TYPE: LOG_TYPE_WORKFLOW
            if isinstance(instance, Workflow)
            else LOG_TYPE_TASK,
            OIAttributes.INPUT_VALUE: _json(kwargs.get("node_input")),
        }

        async def traced():
            token = _NODE_TOOL.set(tool)
            try:
                manager = (
                    nullcontext(native_span)
                    if native_span is not None
                    else tracer.start_as_current_span(
                        instance.name, attributes=attributes
                    )
                )
                with manager as span:
                    span.set_attributes(attributes)
                    try:
                        async for event in source:
                            if getattr(event, "output", None) is not None:
                                span.set_attribute(
                                    OIAttributes.OUTPUT_VALUE, _json(event.output)
                                )
                            yield event
                    except GeneratorExit:
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        raise
                    else:
                        span.set_status(trace.Status(trace.StatusCode.OK))
                    finally:
                        await source.aclose()
            finally:
                _NODE_TOOL.reset(token)

        return _ContextPreservingIterator(traced())

    async def advisor_call(wrapped, instance, args, kwargs):
        token = _ADVISOR_START.set((context.get_current(), _should_send_prompts()))
        try:
            return await wrapped(*args, **kwargs)
        finally:
            _ADVISOR_START.reset(token)

    def advisor_attempt(wrapped, instance, args, kwargs):
        result = wrapped(*args, **kwargs)
        if not active or context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY):
            return result
        # ADK reports each real advisor attempt after consumption, including a
        # retry failure. AdvisorResult.usage synthesizes zeroes when absent, so
        # only actual LlmResponse metadata is translated here.
        snapshot = _ADVISOR_START.get() or (
            context.get_current(),
            _should_send_prompts(),
        )
        context_token = context.attach(snapshot[0])
        policy_token = _content_policy_override.set(snapshot[1])
        try:
            from openinference.instrumentation.google_adk._wrappers import (
                _get_attributes_from_llm_response,
                _TraceCallLlm,
            )

            elapsed = max(0.0, kwargs.get("elapsed_s", 0.0))
            start = time.time_ns() - int(elapsed * 1e9)
            with tracer.start_as_current_span(
                "model_consult",
                start_time=start,
                attributes={OIAttributes.OPENINFERENCE_SPAN_KIND: "LLM"},
            ) as span:
                request = kwargs.get("request")
                if request is not None:
                    _TraceCallLlm._set_request_attributes(span, request)
                responses = kwargs.get("responses", ())
                if responses:
                    from google.genai import types

                    # Match ADK's advisor result: concatenate visible non-partial
                    # text, retaining the last actual cumulative usage/model.
                    text = "".join(
                        part.text
                        for response in responses
                        if not response.partial
                        for part in (response.content.parts if response.content else ())
                        or ()
                        if part.text and not getattr(part, "thought", False)
                    ).strip()
                    usage = next(
                        (
                            response.usage_metadata
                            for response in reversed(responses)
                            if response.usage_metadata is not None
                        ),
                        None,
                    )
                    model = next(
                        (
                            response.model_version
                            for response in reversed(responses)
                            if response.model_version
                        ),
                        None,
                    )
                    assembled = responses[-1].model_copy(
                        update={
                            "content": types.Content(
                                role="model", parts=[types.Part(text=text)]
                            )
                            if text
                            else None,
                            "usage_metadata": usage,
                            "model_version": model,
                        }
                    )
                    span.set_attributes(
                        dict(_get_attributes_from_llm_response(assembled))
                    )
                error = kwargs.get("error")
                if error is not None:
                    span.record_exception(error)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))
                else:
                    span.set_status(trace.Status(trace.StatusCode.OK))
        except Exception:
            logging.getLogger(__name__).debug(
                "Could not capture model consultation", exc_info=True
            )
        finally:
            _content_policy_override.reset(policy_token)
            context.detach(context_token)
        return result

    try:
        from google.adk.telemetry import node_tracing

        original_tracer = node_tracing.tracer

        class NodeTracer(ObjectProxy):
            @contextmanager
            def start_as_current_span(self, name, *args, **kwargs):
                if not active:
                    with self.__wrapped__.start_as_current_span(
                        name, *args, **kwargs
                    ) as span:
                        yield span
                    return
                with tracer.start_as_current_span(name, *args, **kwargs) as span:
                    yield span

        replacement_tracer = NodeTracer(original_tracer)
        node_tracing.tracer = replacement_tracer
        patches.append((node_tracing, "tracer", original_tracer, replacement_tracer))
        patch(BaseNode, "run", call_node)
        try:
            from google.adk.tools.model_consult import _advisor
        except ImportError:
            pass
        else:
            patch(_advisor, "_record_telemetry", advisor_attempt)
            patch(_advisor, "call_advisor", advisor_call)
            from google.adk.tools.model_consult import _model_consult_tool

            patch(_model_consult_tool, "call_advisor", advisor_call)
    except BaseException:
        for owner, name, original, replacement in reversed(patches):
            if inspect.getattr_static(owner, name) is replacement:
                setattr(owner, name, original)
        raise

    def undo():
        nonlocal active
        active = False
        with lock:
            for owner, name, original, replacement in reversed(patches):
                if inspect.getattr_static(owner, name) is replacement:
                    setattr(owner, name, original)
            patches.clear()
            patched_tools.clear()

    return undo
