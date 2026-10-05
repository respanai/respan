"""Microsoft Agent Framework raw attribute keys."""

AGENT_FRAMEWORK_INSTRUMENTATION_NAME = "microsoft-agent-framework"
AGENT_FRAMEWORK_SCOPE_PREFIX = "agent_framework"
AGENT_FRAMEWORK_SYSTEM = "microsoft.agent_framework"

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
