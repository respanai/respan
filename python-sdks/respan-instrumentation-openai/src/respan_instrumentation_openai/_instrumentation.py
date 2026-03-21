"""OpenAI SDK instrumentation plugin for Respan.

Self-contained plugin that enables ``opentelemetry-instrumentation-openai``
and applies a sync prompt-capture patch. Spans carry proper GenAI semantic
conventions and pass ``is_processable_span()`` natively — no conversion needed.
"""

import copy
import json
import logging
import traceback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sync prompt-capture patch
# ---------------------------------------------------------------------------


def _patch_chat_prompt_capture():
    """Replace the async chat _handle_request with a sync version.

    Root cause: opentelemetry-instrumentation-openai v0.52+ has _handle_request
    as async def (for optional base64 image upload). In sync contexts, it runs
    through run_async() which either calls asyncio.run() or spawns a thread.
    Both paths can silently lose prompt attributes when:
      - _set_request_attributes (NOT @dont_throw) raises on response_format
        handling, killing the entire _handle_request before _set_prompts runs
      - asyncio.run() / thread path has environment-specific issues (Lambda, etc.)

    The embeddings wrapper is fully sync and works correctly. This patch makes
    the chat path match the embeddings path: fully synchronous with fault
    isolation between each section.

    The only async code in _set_prompts was for Config.upload_base64_image
    (rarely used). For list content (multimodal), we json.dumps as-is — the
    base64 data stays inline, which is the default behavior anyway.
    """
    try:
        from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw
        from opentelemetry.instrumentation.openai.shared import (
            _set_request_attributes,
            _set_client_attributes,
            _set_functions_attributes,
            _set_span_attribute,
            set_tools_attributes,
            model_as_dict,
            propagate_trace_context,
        )
        from opentelemetry.instrumentation.openai.shared.config import Config
        from opentelemetry.instrumentation.openai.utils import (
            should_send_prompts,
            should_emit_events,
            is_openai_v1,
        )
        from opentelemetry.semconv._incubating.attributes import (
            gen_ai_attributes as GenAIAttributes,
        )
        from opentelemetry.semconv_ai import SpanAttributes

        def _set_prompts_sync(span, messages):
            if not span.is_recording() or messages is None:
                return

            for i, msg in enumerate(messages):
                prefix = f"{GenAIAttributes.GEN_AI_PROMPT}.{i}"
                msg = msg if isinstance(msg, dict) else model_as_dict(msg)

                _set_span_attribute(span, f"{prefix}.role", msg.get("role"))
                if msg.get("content"):
                    content = copy.deepcopy(msg.get("content"))
                    if isinstance(content, list):
                        content = json.dumps(content)
                    _set_span_attribute(span, f"{prefix}.content", content)
                if msg.get("tool_call_id"):
                    _set_span_attribute(
                        span, f"{prefix}.tool_call_id", msg.get("tool_call_id")
                    )
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for j, tool_call in enumerate(tool_calls):
                        if is_openai_v1():
                            tool_call = model_as_dict(tool_call)
                        function = tool_call.get("function")
                        _set_span_attribute(
                            span, f"{prefix}.tool_calls.{j}.id", tool_call.get("id")
                        )
                        _set_span_attribute(
                            span, f"{prefix}.tool_calls.{j}.name", function.get("name")
                        )
                        _set_span_attribute(
                            span,
                            f"{prefix}.tool_calls.{j}.arguments",
                            function.get("arguments"),
                        )

        def _handle_request_sync(span, kwargs, instance):
            # Section 1: Request attributes (fault-isolated from prompts)
            try:
                _set_request_attributes(span, kwargs, instance)
            except Exception:
                logging.warning(
                    "respan: _set_request_attributes failed (response_format may be incompatible). "
                    "Request attributes like model/temperature may be incomplete on this span. "
                    "Error: %s",
                    traceback.format_exc(),
                )

            try:
                _set_client_attributes(span, instance)
            except Exception:
                pass

            # Section 2: Prompt/event capture
            try:
                if should_emit_events():
                    from opentelemetry.instrumentation.openai.shared.event_emitter import emit_event
                    from opentelemetry.instrumentation.openai.shared.event_models import MessageEvent
                    for message in kwargs.get("messages", []):
                        emit_event(
                            MessageEvent(
                                content=message.get("content"),
                                role=message.get("role"),
                                tool_calls=cw._parse_tool_calls(
                                    message.get("tool_calls", None)
                                ),
                            )
                        )
                else:
                    if should_send_prompts():
                        _set_prompts_sync(span, kwargs.get("messages"))
                        if kwargs.get("functions"):
                            _set_functions_attributes(span, kwargs.get("functions"))
                        elif kwargs.get("tools"):
                            set_tools_attributes(span, kwargs.get("tools"))
            except Exception:
                logging.warning(
                    "respan: chat prompt capture failed. "
                    "Input messages may not appear on the dashboard for this span. "
                    "Error: %s",
                    traceback.format_exc(),
                )

            # Section 3: Trace propagation + reasoning
            try:
                if Config.enable_trace_context_propagation:
                    propagate_trace_context(span, kwargs)
                reasoning_effort = kwargs.get("reasoning_effort")
                _set_span_attribute(
                    span,
                    SpanAttributes.LLM_REQUEST_REASONING_EFFORT,
                    reasoning_effort or (),
                )
            except Exception:
                pass

        async def _noop():
            pass

        def _patched_handle_request(span, kwargs, instance):
            _handle_request_sync(span, kwargs, instance)
            return _noop()

        cw._handle_request = _patched_handle_request
        logger.debug("Patched chat prompt capture to sync path")

    except Exception as e:
        logger.warning("Failed to patch chat prompt capture: %s", e)


# ---------------------------------------------------------------------------
# Instrumentor
# ---------------------------------------------------------------------------


class OpenAIInstrumentor:
    """Respan instrumentor for direct OpenAI SDK usage.

    Activates OTEL auto-instrumentation for the ``openai`` package so
    that all ``ChatCompletion``, ``Completion``, and ``Embedding`` calls
    are traced automatically.

    Usage::

        from respan import Respan
        from respan_instrumentation_openai import OpenAIInstrumentor

        respan = Respan(instrumentations=[OpenAIInstrumentor()])
    """

    name = "openai"

    def __init__(self) -> None:
        self._instrumented = False

    def activate(self) -> None:
        """Instrument the OpenAI SDK via OTEL and patch prompt capture."""
        try:
            from opentelemetry.instrumentation.openai import OpenAIInstrumentor as OTELOpenAI

            instrumentor = OTELOpenAI()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            self._instrumented = True

            # Apply sync prompt-capture patch
            try:
                _patch_chat_prompt_capture()
            except Exception as exc:
                logger.debug("Prompt capture patch not applied: %s", exc)

            logger.info("OpenAI SDK instrumentation activated")
        except ImportError as exc:
            logger.warning(
                "Failed to activate OpenAI instrumentation — missing dependency: %s", exc
            )

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        if self._instrumented:
            try:
                from opentelemetry.instrumentation.openai import OpenAIInstrumentor as OTELOpenAI
                OTELOpenAI().uninstrument()
            except Exception:
                pass
            self._instrumented = False
        logger.info("OpenAI SDK instrumentation deactivated")
