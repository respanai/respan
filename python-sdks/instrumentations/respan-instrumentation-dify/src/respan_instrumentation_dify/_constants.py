"""Dify instrumentation constants."""

from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)

DIFY_INSTRUMENTATION_NAME = "dify"
DIFY_CLIENT_MODULE = "dify_client.client"
DIFY_CHAT_SPAN_NAME = "dify.chat"
DIFY_COMPLETION_SPAN_NAME = "dify.completion"
DIFY_WORKFLOW_SPAN_NAME = "dify.workflow"
DIFY_API_SPAN_NAME = "dify.request"

ANSWER_KEY = "answer"
CONVERSATION_ID_KEY = "conversation_id"
CREATED_AT_KEY = "created_at"
DATA_KEY = "data"
ENDPOINT_KEY = "endpoint"
EVENT_KEY = "event"
FILES_KEY = "files"
ID_KEY = "id"
INPUTS_KEY = "inputs"
LATENCY_KEY = "latency"
MESSAGE_ID_KEY = "message_id"
METADATA_KEY = "metadata"
METHOD_KEY = "method"
MODE_KEY = "mode"
PARAMS_KEY = "params"
QUERY_KEY = "query"
RESPONSE_MODE_KEY = "response_mode"
STATUS_KEY = "status"
TASK_ID_KEY = "task_id"
TOTAL_TOKENS_KEY = "total_tokens"
USAGE_KEY = "usage"
USER_KEY = "user"
WORKFLOW_RUN_ID_KEY = "workflow_run_id"

PROMPT_TOKENS_KEY = "prompt_tokens"
COMPLETION_TOKENS_KEY = "completion_tokens"
OUTPUTS_KEY = "outputs"
ERROR_KEY = "error"

CHAT_MESSAGES_ENDPOINT = "/chat-messages"
COMPLETION_MESSAGES_ENDPOINT = "/completion-messages"
WORKFLOW_RUN_ENDPOINT = "/workflows/run"
FILES_UPLOAD_ENDPOINT = "/files/upload"
PARAMETERS_ENDPOINT = "/parameters"
MESSAGES_ENDPOINT = "/messages"
CONVERSATIONS_ENDPOINT = "/conversations"

STREAMING_RESPONSE_MODE = "streaming"

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
