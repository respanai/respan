"""AWS Bedrock Runtime instrumentation constants."""

AWS_BEDROCK_INSTRUMENTATION_NAME = "aws-bedrock"
AWS_BEDROCK_SYSTEM_NAME = "bedrock"
AWS_BEDROCK_CHAT_SPAN_NAME = "aws_bedrock.chat"

BEDROCK_RUNTIME_SERVICE_NAME = "bedrock-runtime"

INVOKE_MODEL_OPERATION = "InvokeModel"
INVOKE_MODEL_STREAM_OPERATION = "InvokeModelWithResponseStream"
CONVERSE_OPERATION = "Converse"
CONVERSE_STREAM_OPERATION = "ConverseStream"
SUPPORTED_OPERATIONS = frozenset(
    {
        INVOKE_MODEL_OPERATION,
        INVOKE_MODEL_STREAM_OPERATION,
        CONVERSE_OPERATION,
        CONVERSE_STREAM_OPERATION,
    }
)
STREAMING_OPERATIONS = frozenset(
    {
        INVOKE_MODEL_STREAM_OPERATION,
        CONVERSE_STREAM_OPERATION,
    }
)

ASSISTANT_ROLE = "assistant"
FUNCTION_KEY = "function"
FUNCTION_TOOL_TYPE = "function"
SYSTEM_ROLE = "system"
TOOL_ROLE = "tool"
USER_ROLE = "user"

ACCEPT_KEY = "accept"
ANTHROPIC_VERSION_KEY = "anthropic_version"
BODY_KEY = "body"
CONTENT_KEY = "content"
CONTENT_TYPE_KEY = "contentType"
DESCRIPTION_KEY = "description"
INPUT_KEY = "input"
INPUT_SCHEMA_KEY = "input_schema"
MAX_TOKENS_KEY = "max_tokens"
MESSAGE_KEY = "message"
MESSAGES_KEY = "messages"
MODEL_ID_KEY = "modelId"
NAME_KEY = "name"
OUTPUT_KEY = "output"
ROLE_KEY = "role"
SYSTEM_KEY = "system"
TEXT_KEY = "text"
TOOL_CALLS_KEY = "tool_calls"
TOOLS_KEY = "tools"
TOOL_CONFIG_KEY = "toolConfig"
TYPE_KEY = "type"
USAGE_KEY = "usage"
