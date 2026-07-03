"""Cursor hook field names owned by this instrumentation package."""

from __future__ import annotations

from pathlib import Path

CURSOR_HOOK_EVENT_NAME = "hook_event_name"
CURSOR_CONVERSATION_ID = "conversation_id"
CURSOR_GENERATION_ID = "generation_id"
CURSOR_MODEL = "model"
CURSOR_VERSION = "cursor_version"

CURSOR_EVENT_BEFORE_SUBMIT_PROMPT = "beforeSubmitPrompt"
CURSOR_EVENT_AFTER_AGENT_THOUGHT = "afterAgentThought"
CURSOR_EVENT_AFTER_SHELL_EXECUTION = "afterShellExecution"
CURSOR_EVENT_AFTER_FILE_EDIT = "afterFileEdit"
CURSOR_EVENT_AFTER_MCP_EXECUTION = "afterMCPExecution"
CURSOR_EVENT_AFTER_AGENT_RESPONSE = "afterAgentResponse"
CURSOR_EVENT_STOP = "stop"

CURSOR_SUPPORTED_EVENTS = frozenset(
    {
        CURSOR_EVENT_BEFORE_SUBMIT_PROMPT,
        CURSOR_EVENT_AFTER_AGENT_THOUGHT,
        CURSOR_EVENT_AFTER_SHELL_EXECUTION,
        CURSOR_EVENT_AFTER_FILE_EDIT,
        CURSOR_EVENT_AFTER_MCP_EXECUTION,
        CURSOR_EVENT_AFTER_AGENT_RESPONSE,
        CURSOR_EVENT_STOP,
    }
)

DEFAULT_CURSOR_STATE_FILE = (
    Path.home() / ".cursor" / "state" / "respan_cursor_sdk_state.json"
)
