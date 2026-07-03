"""AWS SageMaker Runtime instrumentation constants."""

SAGEMAKER_INSTRUMENTATION_NAME = "sagemaker"
SAGEMAKER_SYSTEM_NAME = "sagemaker"
SAGEMAKER_CHAT_SPAN_NAME = "sagemaker.chat"
SAGEMAKER_COMPLETION_SPAN_NAME = "sagemaker.completion"

SAGEMAKER_RUNTIME_SERVICE_NAME = "sagemaker-runtime"

INVOKE_ENDPOINT_OPERATION = "InvokeEndpoint"
INVOKE_ENDPOINT_STREAM_OPERATION = "InvokeEndpointWithResponseStream"
INVOKE_ENDPOINT_ASYNC_OPERATION = "InvokeEndpointAsync"
SUPPORTED_OPERATIONS = frozenset(
    {
        INVOKE_ENDPOINT_OPERATION,
        INVOKE_ENDPOINT_STREAM_OPERATION,
        INVOKE_ENDPOINT_ASYNC_OPERATION,
    }
)
STREAMING_OPERATIONS = frozenset({INVOKE_ENDPOINT_STREAM_OPERATION})
ASYNC_OPERATIONS = frozenset({INVOKE_ENDPOINT_ASYNC_OPERATION})

ASSISTANT_ROLE = "assistant"
FUNCTION_KEY = "function"
FUNCTION_TOOL_TYPE = "function"
SYSTEM_ROLE = "system"
TOOL_ROLE = "tool"
USER_ROLE = "user"

ACCEPT_KEY = "Accept"
BODY_KEY = "Body"
BYTES_KEY = "Bytes"
CHOICES_KEY = "choices"
CONTENT_KEY = "content"
CONTENT_TYPE_KEY = "ContentType"
CUSTOM_ATTRIBUTES_KEY = "CustomAttributes"
DESCRIPTION_KEY = "description"
ENDPOINT_NAME_KEY = "EndpointName"
FUNCTIONS_KEY = "functions"
GENERATED_TEXT_KEY = "generated_text"
INPUT_KEY = "input"
INPUT_LOCATION_KEY = "InputLocation"
INPUTS_KEY = "inputs"
MESSAGE_KEY = "message"
MESSAGES_KEY = "messages"
MODEL_KEY = "model"
NAME_KEY = "name"
OUTPUT_LOCATION_KEY = "OutputLocation"
OUTPUTS_KEY = "outputs"
PARAMETERS_KEY = "parameters"
PAYLOAD_PART_KEY = "PayloadPart"
PROMPT_KEY = "prompt"
RESPAN_MODEL_ATTRIBUTE = "respan_model"
ROLE_KEY = "role"
SYSTEM_KEY = "system"
TARGET_MODEL_KEY = "TargetModel"
TEXT_KEY = "text"
TOOL_CALLS_KEY = "tool_calls"
TOOLS_KEY = "tools"
TYPE_KEY = "type"
USAGE_KEY = "usage"
