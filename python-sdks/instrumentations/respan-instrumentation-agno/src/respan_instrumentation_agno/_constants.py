"""Agno-owned constants for Agno instrumentation."""

AGNO_INSTRUMENTATION_NAME = "agno"
AGNO_AGENT_MODULE = "agno.agent.agent"
AGNO_TEAM_MODULE = "agno.team.team"
AGNO_AGENT_CLASS_NAME = "Agent"
AGNO_TEAM_CLASS_NAME = "Team"
AGNO_TARGET_AGENT = "agent"
AGNO_TARGET_TEAM = "team"
AGNO_EVENT_SPAN_NAME = "agno.event"
AGNO_MODEL_REQUEST_SPAN_NAME = "agno.model_request"
AGNO_TOOL_SPAN_NAME = "agno.tool"

RUN_METHOD_NAME = "run"
ARUN_METHOD_NAME = "arun"
RESPAN_AGNO_ORIGINALS_ATTR = "_respan_agno_originals"
RESPAN_AGNO_WRAPPED_ATTR = "_respan_agno_wrapped"

ASSISTANT_ROLE = "assistant"
FUNCTION_TYPE = "function"
USER_ROLE = "user"

AGENT_ID_KEY = "agent_id"
AGENT_NAME_KEY = "agent_name"
ARGUMENTS_KEY = "arguments"
CACHE_READ_TOKENS_KEY = "cache_read_tokens"
CANCELLED_STATUS = "cancelled"
CHAT_SPAN_SEED_PART = "chat"
COMPLETED_EVENT_SUFFIX = "Completed"
COMPLETED_STATUS = "completed"
CONTENT_KEY = "content"
CONTENT_EVENT_SUFFIX = "Content"
CONTENT_TYPE_KEY = "content_type"
DEFAULT_AGNO_EVENT_NAME = "agno_event"
DEFAULT_AGENT_NAME = "agno_agent"
DEFAULT_EVENT_NAME = "event"
DEFAULT_TEAM_NAME = "agno_team"
DEFAULT_TOOL_NAME = "tool"
DESCRIPTION_KEY = "description"
ERROR_STATUS = "error"
EVENT_KEY = "event"
EVENT_SPAN_SEED_PART = "event"
FUNCTION_KEY = "function"
ID_KEY = "id"
INPUT_KEY = "input"
INPUT_TOKENS_KEY = "input_tokens"
MESSAGES_KEY = "messages"
METADATA_KEY = "metadata"
METRICS_KEY = "metrics"
MODEL_KEY = "model"
MODEL_PROVIDER_KEY = "model_provider"
MODEL_REQUEST_ENTITY_NAME = "model_request"
NAME_KEY = "name"
OUTPUT_TOKENS_KEY = "output_tokens"
PARAMETERS_JSON_SCHEMA_KEY = "parameters_json_schema"
PARAMETERS_KEY = "parameters"
PROVIDER_KEY = "provider"
RESULT_KEY = "result"
ROLE_KEY = "role"
ROOT_SPAN_SEED_PART = "root"
RUN_COMPLETED_EVENT_SUFFIX = "RunCompleted"
RUN_ID_KEY = "run_id"
SESSION_ID_KEY = "session_id"
STATUS_KEY = "status"
STRICT_KEY = "strict"
TEAM_ID_KEY = "team_id"
TEAM_NAME_KEY = "team_name"
TOOL_ARGS_KEY = "tool_args"
TOOL_CALL_ERROR_KEY = "tool_call_error"
TOOL_CALL_ID_KEY = "tool_call_id"
TOOL_CALLS_KEY = "tool_calls"
TOOL_NAME_KEY = "tool_name"
TOOL_SPAN_SEED_PART = "tool"
TOOLS_KEY = "tools"
TOTAL_TOKENS_KEY = "total_tokens"
TYPE_KEY = "type"
USER_ID_KEY = "user_id"
VALUE_KEY = "value"

FUNCTION_TOOL_SCHEMA_KEYS = frozenset(
    (NAME_KEY, DESCRIPTION_KEY, PARAMETERS_KEY, STRICT_KEY)
)
ROOT_OUTPUT_PAYLOAD_KEYS = frozenset(
    (
        RUN_ID_KEY,
        STATUS_KEY,
        MODEL_KEY,
        MODEL_PROVIDER_KEY,
        CONTENT_TYPE_KEY,
        METADATA_KEY,
    )
)
RUN_OUTPUT_MARKER_KEYS = (
    RUN_ID_KEY,
    CONTENT_KEY,
    MODEL_KEY,
    METRICS_KEY,
    STATUS_KEY,
)

AGNO_RUN_ID_ATTR = "agno.run.id"
AGNO_SESSION_ID_ATTR = "agno.session.id"
AGNO_USER_ID_ATTR = "agno.user.id"
AGNO_AGENT_ID_ATTR = "agno.agent.id"
AGNO_AGENT_NAME_ATTR = "agno.agent.name"
AGNO_TEAM_ID_ATTR = "agno.team.id"
AGNO_TEAM_NAME_ATTR = "agno.team.name"
AGNO_EVENT_NAME_ATTR = "agno.event.name"
AGNO_TOOL_CALL_ID_ATTR = "agno.tool.call.id"
AGNO_TOOL_NAME_ATTR = "agno.tool.name"
AGNO_STATUS_ATTR = "agno.status"
