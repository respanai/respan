"""Remove payload attributes when the active Respan content policy disables them."""

from typing import Any

from openinference.semconv.trace import SpanAttributes as OIAttributes
from opentelemetry.semconv_ai import SpanAttributes


def without_content(attributes: dict[str, Any]) -> dict[str, Any]:
    exact = {
        SpanAttributes.TRACELOOP_ENTITY_INPUT,
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        SpanAttributes.LLM_REQUEST_FUNCTIONS,
        OIAttributes.INPUT_VALUE,
        OIAttributes.OUTPUT_VALUE,
        OIAttributes.INPUT_IMAGES,
        OIAttributes.OUTPUT_IMAGES,
        OIAttributes.LLM_TOOLS,
        OIAttributes.LLM_PROMPT_TEMPLATE,
        OIAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
        OIAttributes.LLM_INVOCATION_PARAMETERS,
        OIAttributes.TOOL_PARAMETERS,
    }
    prefixes = (
        SpanAttributes.LLM_PROMPTS + ".",
        SpanAttributes.LLM_COMPLETIONS + ".",
        OIAttributes.LLM_INPUT_MESSAGES + ".",
        OIAttributes.LLM_OUTPUT_MESSAGES + ".",
        OIAttributes.EMBEDDING_EMBEDDINGS + ".",
        OIAttributes.RETRIEVAL_DOCUMENTS + ".",
        "haystack.component.input",
        "haystack.component.output",
    )
    return {
        key: value
        for key, value in attributes.items()
        if key not in exact and not key.startswith(prefixes)
    }
