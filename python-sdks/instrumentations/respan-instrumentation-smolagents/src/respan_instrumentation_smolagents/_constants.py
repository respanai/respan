"""Internal constants for smolagents instrumentation."""

from opentelemetry.semconv.attributes import otel_attributes
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from openinference.semconv.trace import (
    MessageAttributes as OIMessageAttributes,
    MessageContentAttributes as OIMessageContentAttributes,
    SpanAttributes as OISpanAttributes,
)

SMOLAGENTS_INSTRUMENTATION_NAME = "smolagents"
ASSISTANT_ROLE = "assistant"
SMOLAGENTS_FINAL_ANSWER_ARGUMENT = "answer"
SMOLAGENTS_FINAL_ANSWER_TOOL_NAME = "final_answer"
TOOL_CALL_FUNCTION_ARGUMENTS_FIELD = "arguments"
TOOL_CALL_FUNCTION_FIELD = "function"
TOOL_CALL_FUNCTION_NAME_FIELD = "name"
OPENINFERENCE_INSTRUMENTATION_MODULE = "openinference.instrumentation"
OPENINFERENCE_SMOLAGENTS_MODULE = (
    f"{OPENINFERENCE_INSTRUMENTATION_MODULE}.smolagents"
)
OTEL_SCOPE_NAME = otel_attributes.OTEL_SCOPE_NAME

GEN_AI_COMPLETION_ROLE_ATTR = f"{TLSpanAttributes.LLM_COMPLETIONS}.0.role"
GEN_AI_COMPLETION_CONTENT_ATTR = f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"
GEN_AI_COMPLETION_TOOL_CALLS_ATTR = f"{TLSpanAttributes.LLM_COMPLETIONS}.0.tool_calls"
LLM_REQUEST_FUNCTIONS_ATTR = TLSpanAttributes.LLM_REQUEST_FUNCTIONS

OPENINFERENCE_INPUT_MESSAGES_ATTR = OISpanAttributes.LLM_INPUT_MESSAGES
OPENINFERENCE_OUTPUT_MESSAGES_ATTR = OISpanAttributes.LLM_OUTPUT_MESSAGES
OPENINFERENCE_MESSAGE_ROLE_ATTR = OIMessageAttributes.MESSAGE_ROLE
OPENINFERENCE_MESSAGE_CONTENT_ATTR = OIMessageAttributes.MESSAGE_CONTENT
OPENINFERENCE_MESSAGE_CONTENTS_ATTR = OIMessageAttributes.MESSAGE_CONTENTS
OPENINFERENCE_MESSAGE_CONTENT_TYPE_ATTR = (
    OIMessageContentAttributes.MESSAGE_CONTENT_TYPE
)
OPENINFERENCE_MESSAGE_CONTENT_TEXT_ATTR = (
    OIMessageContentAttributes.MESSAGE_CONTENT_TEXT
)

# Off-contract OpenInference source attributes. The smolagents processor
# translates these into the canonical Respan/GenAI contract and then strips the
# raw aliases before export so they do not leak to the backend.
SPAN_ALIAS_MODEL = OISpanAttributes.LLM_MODEL_NAME
SPAN_ALIAS_PROMPT_TOKENS = OISpanAttributes.LLM_TOKEN_COUNT_PROMPT
SPAN_ALIAS_COMPLETION_TOKENS = OISpanAttributes.LLM_TOKEN_COUNT_COMPLETION
SPAN_ALIAS_TOTAL_REQUEST_TOKENS = OISpanAttributes.LLM_TOKEN_COUNT_TOTAL
SPAN_ALIAS_TOOLS = OISpanAttributes.LLM_TOOLS
SPAN_ALIAS_TOOL_CALLS = OIMessageAttributes.MESSAGE_TOOL_CALLS
