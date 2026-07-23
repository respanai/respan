"""Microsoft Agent Framework raw attribute keys."""

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

AGENT_FRAMEWORK_INSTRUMENTATION_NAME = "microsoft-agent-framework"
AGENT_FRAMEWORK_SCOPE_PREFIX = "agent_framework"
AGENT_FRAMEWORK_SYSTEM = "microsoft.agent_framework"

# Agent Framework emits these as native OTEL GenAI attributes. Import them
# from upstream semconv so this package does not shadow shared constants.
ATTR_GEN_AI_OPERATION_NAME = GenAIAttributes.GEN_AI_OPERATION_NAME
ATTR_GEN_AI_PROVIDER_NAME = GenAIAttributes.GEN_AI_PROVIDER_NAME
ATTR_GEN_AI_AGENT_NAME = GenAIAttributes.GEN_AI_AGENT_NAME
ATTR_GEN_AI_TOOL_NAME = GenAIAttributes.GEN_AI_TOOL_NAME
ATTR_GEN_AI_TOOL_CALL_ARGUMENTS = GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS
ATTR_GEN_AI_TOOL_CALL_RESULT = GenAIAttributes.GEN_AI_TOOL_CALL_RESULT
ATTR_GEN_AI_TOOL_CALL_ID = GenAIAttributes.GEN_AI_TOOL_CALL_ID
ATTR_GEN_AI_TOOL_DEFINITIONS = GenAIAttributes.GEN_AI_TOOL_DEFINITIONS
ATTR_GEN_AI_INPUT_MESSAGES = GenAIAttributes.GEN_AI_INPUT_MESSAGES
ATTR_GEN_AI_OUTPUT_MESSAGES = GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
ATTR_GEN_AI_SYSTEM_INSTRUCTIONS = GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS
ATTR_GEN_AI_CONVERSATION_ID = GenAIAttributes.GEN_AI_CONVERSATION_ID
ATTR_GEN_AI_RESPONSE_MODEL = GenAIAttributes.GEN_AI_RESPONSE_MODEL
ATTR_GEN_AI_USAGE_INPUT_TOKENS = GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS

ATTR_WORKFLOW_NAME = "workflow.name"
ATTR_WORKFLOW_ID = "workflow.id"
ATTR_WORKFLOW_EXECUTOR_ID = "workflow.executor.id"
ATTR_WORKFLOW_EDGE_GROUP_ID = "workflow.edge_group.id"

OPERATION_CHAT = "chat"
OPERATION_INVOKE_AGENT = "invoke_agent"
OPERATION_CREATE_AGENT = "create_agent"
OPERATION_EXECUTE_TOOL = "execute_tool"

WORKFLOW_SPAN_PREFIXES = (
    "workflow.run",
    "workflow.start",
    "workflow.resume",
)

TASK_SPAN_PREFIXES = (
    "executor.process",
    "executor.send_message",
    "executor.yield_output",
    "edge_group.process",
)

TOP_LEVEL_ALIAS_ATTRS = frozenset(
    {
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "tools",
        "tool_calls",
        "span_tools",
        "has_tool_calls",
        "parallel_tool_calls",
    }
)
