"""Semantic Kernel raw telemetry constants local to this instrumentation."""

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

SEMANTIC_KERNEL_INSTRUMENTATION_NAME = "semantic-kernel"
SEMANTIC_KERNEL_ROOT_MODULE = "semantic_kernel"
SEMANTIC_KERNEL_SCOPE_PREFIX = "semantic_kernel"
SEMANTIC_KERNEL_MODEL_DIAGNOSTICS_LOGGER = (
    "semantic_kernel.utils.telemetry.model_diagnostics.decorators"
)

SEMANTIC_KERNEL_ENABLE_OTEL_DIAGNOSTICS_ENV = (
    "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS"
)
SEMANTIC_KERNEL_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE_ENV = (
    "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE"
)

SEMANTIC_KERNEL_MODEL_DIAGNOSTICS_MODULES = (
    "semantic_kernel.utils.telemetry.model_diagnostics.decorators",
    "semantic_kernel.utils.telemetry.model_diagnostics.function_tracer",
)

SK_CHAT_COMPLETION_OPERATION = "chat.completions"
SK_CHAT_STREAMING_COMPLETION_OPERATION = "chat.streaming_completions"
SK_TEXT_COMPLETION_OPERATION = "text.completions"
SK_TEXT_STREAMING_COMPLETION_OPERATION = "text.streaming_completions"
SK_TOOL_OPERATION = "execute_tool"
SK_AUTO_FUNCTION_INVOCATION_SPAN_NAME = "AutoFunctionInvocationLoop"

SK_EVENT_NAME_ATTR = "event.name"
SK_CHAT_MESSAGE_INDEX_ATTR = "CHAT_MESSAGE_INDEX"
SK_SYSTEM_MESSAGE_EVENT = "gen_ai.system.message"
SK_USER_MESSAGE_EVENT = "gen_ai.user.message"
SK_ASSISTANT_MESSAGE_EVENT = "gen_ai.assistant.message"
SK_TOOL_MESSAGE_EVENT = "gen_ai.tool.message"
SK_PROMPT_EVENT = "gen_ai.prompt"
SK_CHOICE_EVENT = "gen_ai.choice"
SK_CONTENT_PROMPT_EVENT = "gen_ai.content.prompt"
SK_CONTENT_COMPLETION_EVENT = "gen_ai.content.completion"

SK_PROMPT_ATTR = GenAIAttributes.GEN_AI_PROMPT
SK_COMPLETION_ATTR = GenAIAttributes.GEN_AI_COMPLETION
SK_RESPONSE_PROMPT_TOKENS_ATTR = "gen_ai.response.prompt_tokens"
SK_RESPONSE_COMPLETION_TOKENS_ATTR = "gen_ai.response.completion_tokens"
SK_AVAILABLE_FUNCTIONS_ATTR = "sk.available_functions"
