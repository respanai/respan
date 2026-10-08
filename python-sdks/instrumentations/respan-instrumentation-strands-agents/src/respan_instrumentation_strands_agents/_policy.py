"""Content policy evaluated at native span creation and again at completion."""

from __future__ import annotations

import os
from typing import Any

from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

from respan_instrumentation_strands_agents._serialization import payload_text, safe_text


def content_enabled() -> bool:
    return context_api.get_value(ENABLE_CONTENT_TRACING_KEY) is not False and os.getenv(
        "TRACELOOP_TRACE_CONTENT", "true"
    ).strip().lower() not in {"false", "0", "off", "no"}


def suppressed() -> bool:
    return bool(
        context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY)
        or context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    )


def is_payload_key(key: str) -> bool:
    return key in {
        SpanAttributes.TRACELOOP_ENTITY_INPUT,
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        SpanAttributes.LLM_REQUEST_FUNCTIONS,
        gen_ai.GEN_AI_INPUT_MESSAGES,
        gen_ai.GEN_AI_OUTPUT_MESSAGES,
        gen_ai.GEN_AI_SYSTEM_INSTRUCTIONS,
        gen_ai.GEN_AI_TOOL_DEFINITIONS,
        gen_ai.GEN_AI_TOOL_CALL_ARGUMENTS,
        gen_ai.GEN_AI_TOOL_CALL_RESULT,
        "gen_ai.agent.tools",
        "gen_ai.tool.json_schema",
        "gen_ai.tool.description",
        "system_prompt",
        "exception.message",
        "exception.stacktrace",
        "error.message",
    } or key.startswith(
        (
            f"{SpanAttributes.LLM_PROMPTS}.",
            f"{SpanAttributes.LLM_COMPLETIONS}.",
            "memory.",
        )
    )


def filter_attributes(attributes: Any, allowed: bool) -> dict[str, Any]:
    result = {}
    for key, value in (attributes or {}).items():
        if is_payload_key(key):
            if not allowed:
                continue
            result[key] = payload_text(
                value,
                complete=key == gen_ai.GEN_AI_TOOL_DEFINITIONS
                or key == SpanAttributes.LLM_REQUEST_FUNCTIONS
                or key.endswith(".tool_calls"),
            )
        else:
            result[key] = safe_text(value) if isinstance(value, str) else value
    return result
