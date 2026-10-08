"""AgentOps translator classification, with canonical Respan log values."""

from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_GUARDRAIL,
    LOG_TYPE_TASK,
    LOG_TYPE_TEXT,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)

AGENTOPS_INSTRUMENTATION_NAME = "agentops"
# Default prefix used by AgentOps' public update_trace_metadata API.
AGENTOPS_METADATA_PREFIX = "trace.metadata."
AGENTOPS_KIND_LOG_TYPES = {
    "session": LOG_TYPE_WORKFLOW,
    "workflow": LOG_TYPE_WORKFLOW,
    "agent": LOG_TYPE_AGENT,
    "task": LOG_TYPE_TASK,
    "operation": LOG_TYPE_TASK,
    "chain": LOG_TYPE_TASK,
    "tool": LOG_TYPE_TOOL,
    "guardrail": LOG_TYPE_GUARDRAIL,
    "http": LOG_TYPE_TASK,
    "llm": LOG_TYPE_CHAT,
    "text": LOG_TYPE_TEXT,
    "embedding": LOG_TYPE_EMBEDDING,
}
# Removal list only: these source aliases are explicitly prohibited by span-contract.
OFF_CONTRACT_ALIASES = {
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "span_tools",
    "has_tool_calls",
    "parallel_tool_calls",
    "respan.span.tools",
    "respan.span.tool_calls",
    "respan.span.handoffs",
    "status_code",
    "error.message",
}
