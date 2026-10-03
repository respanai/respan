"""Google Gen AI instrumentation constants."""

GOOGLE_GENAI_INSTRUMENTATION_NAME = "google-genai"
GOOGLE_GENAI_SYSTEM_NAME = "google"
GOOGLE_GENAI_CHAT_SPAN_NAME = "google_genai.generate_content"

GOOGLE_GENAI_MODELS_MODULE = "google.genai.models"
MODELS_CLASS_NAME = "Models"
ASYNC_MODELS_CLASS_NAME = "AsyncModels"
GENERATE_CONTENT_METHOD_NAME = "generate_content"
EMBED_CONTENT_METHOD_NAME = "embed_content"
GENERATE_CONTENT_STREAM_METHOD_NAME = "generate_content_stream"

ASSISTANT_ROLE = "assistant"
FUNCTION_KEY = "function"
FUNCTION_TOOL_TYPE = "function"
MODEL_ROLE = "model"
SYSTEM_ROLE = "system"
TOOL_ROLE = "tool"
USER_ROLE = "user"

ARGS_KEY = "args"
CANDIDATES_KEY = "candidates"
CONFIG_KEY = "config"
CONTENT_KEY = "content"
CONTENTS_KEY = "contents"
DESCRIPTION_KEY = "description"
FUNCTION_CALL_KEY = "function_call"
FUNCTION_DECLARATIONS_KEY = "function_declarations"
FUNCTION_RESPONSE_KEY = "function_response"
ID_KEY = "id"
MODEL_KEY = "model"
NAME_KEY = "name"
PARAMETERS_KEY = "parameters"
PARAMETERS_JSON_SCHEMA_KEY = "parameters_json_schema"
PARTS_KEY = "parts"
RESPONSE_KEY = "response"
ROLE_KEY = "role"
TEXT_KEY = "text"
TOOLS_KEY = "tools"
TYPE_KEY = "type"
USAGE_METADATA_KEY = "usage_metadata"

AUTOMATIC_FUNCTION_CALLING_HISTORY_KEY = "automatic_function_calling_history"
CANDIDATES_TOKEN_COUNT_KEY = "candidates_token_count"
THOUGHTS_TOKEN_COUNT_KEY = "thoughts_token_count"
PROMPT_TOKEN_COUNT_KEY = "prompt_token_count"
TOTAL_TOKEN_COUNT_KEY = "total_token_count"

BUILTIN_TOOL_FIELDS = (
    "google_search",
    "google_search_retrieval",
    "exa_ai_search",
    "google_maps",
    "code_execution",
    "url_context",
    "retrieval",
    "file_search",
    "enterprise_web_search",
    "computer_use",
    "mcp_servers",
    "parallel_ai_search",
)
