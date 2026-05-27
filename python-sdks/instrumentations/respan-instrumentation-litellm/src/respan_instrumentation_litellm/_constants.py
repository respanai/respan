"""LiteLLM instrumentation constants."""

from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)

LITELLM_INSTRUMENTATION_NAME = "litellm"
LITELLM_CHAT_SPAN_NAME = "litellm.completion"

ASSISTANT_ROLE = "assistant"
FUNCTION_KEY = "function"
FUNCTION_TOOL_TYPE = "function"
USER_ROLE = "user"

API_BASE_KEY = "api_base"
ARGUMENTS_KEY = "arguments"
CACHE_HIT_KEY = "cache_hit"
CALL_TYPE_KEY = "call_type"
CHOICES_KEY = "choices"
COMPLETE_STREAMING_RESPONSE_KEY = "complete_streaming_response"
COMPLETION_TOKENS_KEY = "completion_tokens"
CONTENT_KEY = "content"
COST_KEY = "response_cost"
DELTA_KEY = "delta"
EXCEPTION_KEY = "exception"
FUNCTIONS_KEY = "functions"
ID_KEY = "id"
LITELLM_CALL_ID_KEY = "litellm_call_id"
LITELLM_PARAMS_KEY = "litellm_params"
MESSAGE_KEY = "message"
MESSAGES_KEY = "messages"
METADATA_KEY = "metadata"
MODEL_KEY = "model"
NAME_KEY = "name"
PROMPT_TOKENS_KEY = "prompt_tokens"
RESPONSE_KEY = "response"
RESPAN_PARAMS_KEY = "respan_params"
RESPAN_SKIP_CALLBACK_KEY = "_respan_litellm_streaming_wrapper"
ROLE_KEY = "role"
STANDARD_LOGGING_OBJECT_KEY = "standard_logging_object"
STREAM_KEY = "stream"
TEXT_KEY = "text"
TOOL_CALLS_KEY = "tool_calls"
TOOL_CHOICE_KEY = "tool_choice"
TOOLS_KEY = "tools"
TOTAL_TOKENS_KEY = "total_tokens"
TRACEBACK_EXCEPTION_KEY = "traceback_exception"
TYPE_KEY = "type"
USAGE_KEY = "usage"

PROVIDER_MODEL_PREFIXES = {
    "anthropic",
    "azure",
    "bedrock",
    "cohere",
    "deepseek",
    "gemini",
    "google",
    "groq",
    "mistral",
    "ollama",
    "openai",
    "perplexity",
    "together_ai",
    "vertex_ai",
    "xai",
}

OPENAI_MODEL_PREFIXES = (
    "gpt-",
    "o1",
    "o3",
    "o4",
    "chatgpt-",
)

OFF_CONTRACT_ALIASES = {
    "completion_tokens",
    "has_tool_calls",
    "model",
    "parallel_tool_calls",
    "prompt_tokens",
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
    "span_tools",
    "tool_calls",
    "tools",
    "total_request_tokens",
}
