"""LlamaIndex instrumentation-local constants."""

from __future__ import annotations

CHAT_EVENT_KEY = "chat"
COMPLETION_EVENT_KEY = "completion"
EMBEDDING_EVENT_KEY = "embedding"

LLAMA_INDEX_INSTRUMENTATION_NAME = "llama-index"
LLAMA_INDEX_ROOT_MODULE = "llama_index_instrumentation"

LLAMA_INDEX_CHAT_SPAN_NAME = "llama_index.chat"
LLAMA_INDEX_COMPLETION_SPAN_NAME = "llama_index.completion"
LLAMA_INDEX_EMBEDDING_SPAN_NAME = "llama_index.embedding"
LLAMA_INDEX_TOOL_SPAN_PREFIX = "llama_index.tool."
LLAMA_INDEX_DEFAULT_TOOL_NAME = "llama_index_tool"

LLAMA_INDEX_RUN_ID_TAG = "llamaindex.run_id"
LLAMA_INDEX_START_EVENT_TAG = "llamaindex.start_event"
LLAMA_INDEX_STEP_INPUT_EVENT_TAG = "llamaindex.step.input_event"
LLAMA_INDEX_STEP_INPUT_SUMMARY_TAG = "llamaindex.step.input_summary"

MESSAGE_ROLE_ASSISTANT = "assistant"
MESSAGE_ROLE_SYSTEM = "system"
MESSAGE_ROLE_USER = "user"

# Not available in the supported Traceloop GenAI semantic-convention package.
LLAMA_INDEX_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
LLAMA_INDEX_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
STATUS_CODE_ATTR = "status_code"
