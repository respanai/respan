"""Per-emission content decisions for CrewAI's asynchronous event delivery."""

from contextvars import ContextVar
from typing import Any

from opentelemetry.instrumentation.utils import is_instrumentation_enabled
from opentelemetry.semconv_ai import SpanAttributes
from respan_tracing.decorators.base import _should_send_prompts

CONTENT_POLICY_ATTRIBUTE = "_respan_crewai_content_allowed"
TOOL_CALL_ID_ATTRIBUTE = "_respan_crewai_tool_call_id"
ENABLED_POLICY_ATTRIBUTE = "_respan_crewai_instrumentation_enabled"
EVENT_CONTENT: ContextVar[bool] = ContextVar(
    "respan_crewai_event_content", default=True
)


def content_allowed() -> bool:
    return EVENT_CONTENT.get() and _should_send_prompts()


def event_enabled(event: Any) -> bool:
    return getattr(event, ENABLED_POLICY_ATTRIBUTE, is_instrumentation_enabled())


def strip_content(attributes: Any) -> None:
    for key in list(attributes):
        if key in (
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            SpanAttributes.LLM_REQUEST_FUNCTIONS,
        ) or key.startswith(
            (f"{SpanAttributes.LLM_PROMPTS}.", f"{SpanAttributes.LLM_COMPLETIONS}.")
        ):
            del attributes[key]
