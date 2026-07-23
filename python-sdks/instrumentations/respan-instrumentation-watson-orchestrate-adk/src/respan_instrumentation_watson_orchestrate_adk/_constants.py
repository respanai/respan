"""Package-local constants for IBM watsonx Orchestrate ADK instrumentation."""

from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)

WATSON_ORCHESTRATE_ADK_INSTRUMENTATION_NAME = "watson-orchestrate-adk"
WATSON_ORCHESTRATE_ADK_SYSTEM_NAME = "watsonx_orchestrate"

WATSON_ORCHESTRATE_TOOL_SPAN_NAME = "watson-orchestrate-adk.tool"
WATSON_ORCHESTRATE_RUN_SPAN_NAME = "watson-orchestrate-adk.run"
WATSON_ORCHESTRATE_CHAT_SPAN_NAME = "watson-orchestrate-adk.chat"

PYTHON_TOOL_MODULE = "ibm_watsonx_orchestrate.agent_builder.tools.python_tool"
PYTHON_TOOL_CLASS = "PythonTool"

RUN_CLIENT_MODULE = "ibm_watsonx_orchestrate_clients.chat.run_client"
RUN_CLIENT_CLASS = "RunClient"

AGENT_BUILDER_CLIENT_MODULE = (
    "ibm_watsonx_orchestrate_clients.ai_builder.agent_builder_client"
)
AGENT_BUILDER_CLIENT_CLASS = "AgentBuilderClient"

CPE_CLIENT_MODULE = "ibm_watsonx_orchestrate_clients.ai_builder.cpe.cpe_client"
CPE_CLIENT_CLASS = "CPEClient"

WATSONX_AI_CLIENT_MODULE = (
    "ibm_watsonx_orchestrate.client.autodiscover.watsonx_ai.watsonx_ai_client"
)
WATSONX_AI_CLIENT_CLASS = "WatsonxAIClient"

TOOL_CALL_METHOD = "__call__"
RUN_METHODS = (
    "create_run",
    "create_run_with_files",
    "wait_for_run_completion",
)
ASYNC_RUN_METHODS = ("stream_run_with_websocket",)
CHAT_METHODS = ("submit_chat", "submit_chat_with_agent_architect")
CHAT_REFINEMENT_METHODS = ("submit_refine_agent_with_chats",)
LLM_CHAT_METHODS = ("generate_response",)

AGENT_ID_KEY = "agent_id"
ARGUMENTS_KEY = "arguments"
ASSISTANT_ROLE = "assistant"
CHAT_LLM_KEY = "chat_llm"
CHAT_MODEL_NAME_KEY = "chat_model_name"
CHOICES_KEY = "choices"
COMPLETION_TOKENS_KEY = "completion_tokens"
CONTENT_KEY = "content"
DELTA_KEY = "delta"
EVENT_KEY = "event"
FUNCTION_KEY = "function"
FUNCTION_TOOL_TYPE = "function"
INPUT_KEY = "input"
INSTRUCTION_KEY = "instruction"
MESSAGE_KEY = "message"
MESSAGES_KEY = "messages"
MODEL_KEY = "model"
MODEL_ID_KEY = "model_id"
NAME_KEY = "name"
OUTPUT_TOKENS_KEY = "output_tokens"
PROMPT_TOKENS_KEY = "prompt_tokens"
ROLE_KEY = "role"
RUN_ID_KEY = "run_id"
STATUS_KEY = "status"
TEXT_KEY = "text"
THREAD_ID_KEY = "thread_id"
TOKEN_USAGE_KEY = "token_usage"
TOOLS_KEY = "tools"
TOTAL_TOKENS_KEY = "total_tokens"
TYPE_KEY = "type"
USAGE_KEY = "usage"
USER_MESSAGE_KEY = "user_message"
USER_ROLE = "user"

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
