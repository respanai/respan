export const AWS_BEDROCK_INSTRUMENTATION_NAME = "aws-bedrock";
export const AWS_BEDROCK_INSTRUMENTATION_PACKAGE = "@respan/instrumentation-aws-bedrock";
export const AWS_BEDROCK_CHAT_SPAN_NAME = "aws_bedrock.chat";
export const AWS_BEDROCK_SYSTEM_NAME = "bedrock";
export const PACKAGE_VERSION = "1.0.0";
export const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";
export const STATUS_CODE_ATTR = "status_code";

export const INVOKE_MODEL_OPERATION = "InvokeModel";
export const INVOKE_MODEL_STREAM_OPERATION = "InvokeModelWithResponseStream";
export const CONVERSE_OPERATION = "Converse";
export const CONVERSE_STREAM_OPERATION = "ConverseStream";

export const SUPPORTED_OPERATIONS = new Set([
  INVOKE_MODEL_OPERATION,
  INVOKE_MODEL_STREAM_OPERATION,
  CONVERSE_OPERATION,
  CONVERSE_STREAM_OPERATION,
]);

export const STREAMING_OPERATIONS = new Set([
  INVOKE_MODEL_STREAM_OPERATION,
  CONVERSE_STREAM_OPERATION,
]);

export const BODY_KEY = "body";
export const CONTENT_KEY = "content";
export const DESCRIPTION_KEY = "description";
export const FUNCTION_KEY = "function";
export const FUNCTION_TOOL_TYPE = "function";
export const INPUT_KEY = "input";
export const INPUT_SCHEMA_KEY = "inputSchema";
export const MESSAGE_KEY = "message";
export const MESSAGES_KEY = "messages";
export const MODEL_ID_KEY = "modelId";
export const NAME_KEY = "name";
export const OUTPUT_KEY = "output";
export const ROLE_KEY = "role";
export const SYSTEM_KEY = "system";
export const TEXT_KEY = "text";
export const TOOL_CONFIG_KEY = "toolConfig";
export const TOOLS_KEY = "tools";
export const TYPE_KEY = "type";
export const USAGE_KEY = "usage";

export const ASSISTANT_ROLE = "assistant";
export const SYSTEM_ROLE = "system";
export const TOOL_ROLE = "tool";
export const USER_ROLE = "user";
