"""Integration-local Exa operation metadata.

Only Exa-owned translator keys live here. Canonical GenAI, Traceloop, and
Respan attribute keys are imported from their owning packages.
"""

from __future__ import annotations

from dataclasses import dataclass

EXA_INSTRUMENTATION_NAME = "exa"
EXA_INSTRUMENTATION_SCOPE = "respan-instrumentation-exa"
EXA_SYSTEM = "exa"

EXA_METADATA_NAMESPACE = "exa"
METADATA_OPERATION = "operation"
METADATA_LANGUAGE = "language"
METADATA_STREAM = "stream"
METADATA_STREAM_COMPLETED = "stream_completed"
METADATA_RESULT_COUNT = "result_count"
METADATA_REQUEST_ID = "request_id"
METADATA_RESOLVED_SEARCH_TYPE = "resolved_search_type"
METADATA_COST_TOTAL_USD = "cost_total_usd"
METADATA_CITATIONS = "citations"
METADATA_RESEARCH_LEGACY = "research_legacy"
STATUS_CODE_ATTR = "status_code"

FAMILY_TOOL = "tool"
FAMILY_CHAT = "chat"
FAMILY_AGENT = "agent"
FAMILY_TASK = "task"


@dataclass(frozen=True)
class OperationConfig:
    """Static mapping for one public Exa SDK method."""

    entity_name: str
    family: str
    operation: str
    always_streaming: bool = False
    stream_flag: str | None = None
    stream_family: str | None = None
    legacy_research: bool = False


OFF_CONTRACT_ALIASES = frozenset(
    {
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
    }
)
