"""Braintrust native span type mapping."""

from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)

BRAINTRUST_SPAN_TYPE_TO_LOG_TYPE = {
    "llm": LOG_TYPE_CHAT,
    "chat": LOG_TYPE_CHAT,
    "tool": LOG_TYPE_TOOL,
    "function": LOG_TYPE_TOOL,
    "eval": LOG_TYPE_WORKFLOW,
    "automation": LOG_TYPE_WORKFLOW,
    "agent": LOG_TYPE_AGENT,
    "embedding": LOG_TYPE_EMBEDDING,
    "task": LOG_TYPE_TASK,
    "score": LOG_TYPE_TASK,
    "facet": LOG_TYPE_TASK,
    "preprocessor": LOG_TYPE_TASK,
}
