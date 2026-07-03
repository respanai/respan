"""Convert Cursor hook events into canonical Respan OTEL spans."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_sdk.utils.serialization import serialize_value
from respan_tracing.utils.span_factory import build_readable_span, inject_span

from ._constants import (
    CURSOR_CONVERSATION_ID,
    CURSOR_EVENT_AFTER_AGENT_RESPONSE,
    CURSOR_EVENT_AFTER_AGENT_THOUGHT,
    CURSOR_EVENT_AFTER_FILE_EDIT,
    CURSOR_EVENT_AFTER_MCP_EXECUTION,
    CURSOR_EVENT_AFTER_SHELL_EXECUTION,
    CURSOR_EVENT_BEFORE_SUBMIT_PROMPT,
    CURSOR_EVENT_STOP,
    CURSOR_GENERATION_ID,
    CURSOR_HOOK_EVENT_NAME,
    CURSOR_MODEL,
    CURSOR_SUPPORTED_EVENTS,
    CURSOR_VERSION,
    DEFAULT_CURSOR_STATE_FILE,
)

logger = logging.getLogger(__name__)

_MAX_ENTITY_OUTPUT_LENGTH = 8_000
_MAX_METADATA_VALUE_LENGTH = 1_000


@dataclass(frozen=True)
class CursorHookResult:
    """Processing result for one Cursor hook event."""

    event_name: str
    emitted: bool
    span_name: str | None = None
    trace_id: str | None = None
    span_id: str | None = None


class CursorStateStore:
    """Small JSON state store used across Cursor hook process invocations."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_CURSOR_STATE_FILE

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to read Cursor hook state from %s", self.path)
            return {}
        return payload if isinstance(payload, dict) else {}

    def save(self, state: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(serialize_value(dict(state)), indent=2, sort_keys=True),
            encoding="utf-8",
        )


class CursorHookProcessor:
    """Process Cursor hook JSON payloads and emit Respan spans."""

    def __init__(self, *, state_path: str | Path | None = None) -> None:
        self._state = CursorStateStore(path=state_path)

    @property
    def state_path(self) -> Path:
        return self._state.path

    def process_event(self, event: Mapping[str, Any]) -> CursorHookResult:
        event_name = _string(event.get(CURSOR_HOOK_EVENT_NAME))
        if not event_name:
            return CursorHookResult(event_name="", emitted=False)
        if event_name not in CURSOR_SUPPORTED_EVENTS:
            logger.debug("Ignoring unsupported Cursor hook event: %s", event_name)
            return CursorHookResult(event_name=event_name, emitted=False)

        state = self._state.load()
        if event_name == CURSOR_EVENT_BEFORE_SUBMIT_PROMPT:
            self._handle_before_submit_prompt(event=event, state=state)
            return CursorHookResult(event_name=event_name, emitted=False)
        if event_name == CURSOR_EVENT_STOP:
            self._handle_stop(event=event, state=state)
            return CursorHookResult(event_name=event_name, emitted=False)

        if event_name == CURSOR_EVENT_AFTER_AGENT_THOUGHT:
            return self._emit_agent_thought(event=event, state=state)
        if event_name == CURSOR_EVENT_AFTER_SHELL_EXECUTION:
            return self._emit_shell_execution(event=event, state=state)
        if event_name == CURSOR_EVENT_AFTER_FILE_EDIT:
            return self._emit_file_edit(event=event, state=state)
        if event_name == CURSOR_EVENT_AFTER_MCP_EXECUTION:
            return self._emit_mcp_execution(event=event, state=state)
        if event_name == CURSOR_EVENT_AFTER_AGENT_RESPONSE:
            return self._emit_agent_response(event=event, state=state)

        return CursorHookResult(event_name=event_name, emitted=False)

    def _handle_before_submit_prompt(
        self,
        *,
        event: Mapping[str, Any],
        state: dict[str, Any],
    ) -> None:
        generation_id = _generation_id(event)
        attachments = event.get("attachments")
        state[generation_id] = {
            "prompt": _string(event.get("prompt")),
            "attachments_count": len(attachments) if isinstance(attachments, list) else 0,
            "start_time": _event_time(event),
            "child_count": 0,
        }
        self._state.save(state)

    def _handle_stop(self, *, event: Mapping[str, Any], state: dict[str, Any]) -> None:
        generation_id = _string(event.get(CURSOR_GENERATION_ID))
        if generation_id and generation_id in state:
            del state[generation_id]
            self._state.save(state)

    def _emit_agent_thought(
        self,
        *,
        event: Mapping[str, Any],
        state: dict[str, Any],
    ) -> CursorHookResult:
        child_index = self._next_child_index(event=event, state=state)
        text = _string(event.get("text"))
        duration_ms = _duration_ms(event)
        end_time = _event_time(event)
        start_time = _subtract_ms(end_time, duration_ms)
        span_name = f"Cursor thinking {child_index}"
        return self._emit_span(
            event=event,
            span_name=span_name,
            span_id=f"{_generation_id(event)}-thinking-{child_index}",
            parent_id=_root_span_id(event),
            log_type=LOG_TYPE_TASK,
            entity_path=f"thinking.{child_index}",
            entity_input={
                "type": "reasoning",
                "index": child_index,
            },
            entity_output=text,
            metadata={
                "cursor.event": CURSOR_EVENT_AFTER_AGENT_THOUGHT,
                "cursor.duration_ms": duration_ms,
                "cursor.index": child_index,
            },
            start_time=start_time,
            end_time=end_time,
        )

    def _emit_shell_execution(
        self,
        *,
        event: Mapping[str, Any],
        state: dict[str, Any],
    ) -> CursorHookResult:
        child_index = self._next_child_index(event=event, state=state)
        command = _string(event.get("command"))
        duration_ms = _duration_ms(event)
        end_time = _event_time(event)
        start_time = _subtract_ms(end_time, duration_ms)
        span_name = _compact_name("Cursor shell", command)
        return self._emit_span(
            event=event,
            span_name=span_name,
            span_id=f"{_generation_id(event)}-shell-{child_index}",
            parent_id=_root_span_id(event),
            log_type=LOG_TYPE_TOOL,
            entity_path=f"shell.{child_index}",
            entity_input={"command": command},
            entity_output=_string(event.get("output")),
            metadata={
                "cursor.event": CURSOR_EVENT_AFTER_SHELL_EXECUTION,
                "cursor.command": _metadata_string(command),
                "cursor.duration_ms": duration_ms,
                "cursor.index": child_index,
            },
            start_time=start_time,
            end_time=end_time,
        )

    def _emit_file_edit(
        self,
        *,
        event: Mapping[str, Any],
        state: dict[str, Any],
    ) -> CursorHookResult:
        child_index = self._next_child_index(event=event, state=state)
        file_path = _string(event.get("file_path"))
        edits = event.get("edits")
        edit_count = len(edits) if isinstance(edits, list) else 0
        end_time = _event_time(event)
        start_time = _subtract_ms(end_time, _duration_ms(event, default_ms=100))
        span_name = _compact_name("Cursor edit", Path(file_path).name or file_path)
        return self._emit_span(
            event=event,
            span_name=span_name,
            span_id=f"{_generation_id(event)}-file-{child_index}",
            parent_id=_root_span_id(event),
            log_type=LOG_TYPE_TOOL,
            entity_path=f"file_edit.{child_index}",
            entity_input={
                "file_path": file_path,
                "edit_count": edit_count,
            },
            entity_output=_format_edits(edits),
            metadata={
                "cursor.event": CURSOR_EVENT_AFTER_FILE_EDIT,
                "cursor.file_path": _metadata_string(file_path),
                "cursor.edit_count": edit_count,
                "cursor.index": child_index,
            },
            start_time=start_time,
            end_time=end_time,
        )

    def _emit_mcp_execution(
        self,
        *,
        event: Mapping[str, Any],
        state: dict[str, Any],
    ) -> CursorHookResult:
        child_index = self._next_child_index(event=event, state=state)
        tool_name = _string(event.get("tool_name"))
        duration_ms = _duration_ms(event)
        end_time = _event_time(event)
        start_time = _subtract_ms(end_time, duration_ms)
        span_name = _compact_name("Cursor MCP", tool_name)
        return self._emit_span(
            event=event,
            span_name=span_name,
            span_id=f"{_generation_id(event)}-mcp-{child_index}",
            parent_id=_root_span_id(event),
            log_type=LOG_TYPE_TOOL,
            entity_path=f"mcp.{child_index}",
            entity_input=_jsonish(event.get("tool_input")),
            entity_output=_jsonish(event.get("result_json")),
            metadata={
                "cursor.event": CURSOR_EVENT_AFTER_MCP_EXECUTION,
                "cursor.tool_name": _metadata_string(tool_name),
                "cursor.duration_ms": duration_ms,
                "cursor.index": child_index,
            },
            start_time=start_time,
            end_time=end_time,
        )

    def _emit_agent_response(
        self,
        *,
        event: Mapping[str, Any],
        state: dict[str, Any],
    ) -> CursorHookResult:
        generation_id = _generation_id(event)
        generation_state = state.get(generation_id)
        if not isinstance(generation_state, Mapping):
            generation_state = {}

        user_prompt = _string(generation_state.get("prompt")) or "[No prompt captured]"
        response_text = _string(event.get("text"))
        child_count = generation_state.get("child_count")
        start_time = _string(generation_state.get("start_time")) or _event_time(event)
        end_time = _event_time(event)

        result = self._emit_span(
            event=event,
            span_name=f"Cursor generation {_short_id(generation_id)}",
            span_id=_root_span_id(event),
            parent_id=None,
            log_type=LOG_TYPE_AGENT,
            entity_path="",
            entity_input=[{"role": "user", "content": user_prompt}],
            entity_output=[{"role": "assistant", "content": response_text}],
            metadata={
                "cursor.event": CURSOR_EVENT_AFTER_AGENT_RESPONSE,
                "cursor.child_count": child_count if isinstance(child_count, int) else 0,
                "cursor.attachments_count": generation_state.get("attachments_count", 0),
            },
            start_time=start_time,
            end_time=end_time,
        )

        if generation_id in state:
            del state[generation_id]
            self._state.save(state)

        return result

    def _next_child_index(
        self,
        *,
        event: Mapping[str, Any],
        state: dict[str, Any],
    ) -> int:
        generation_id = _generation_id(event)
        generation_state = state.get(generation_id)
        if not isinstance(generation_state, dict):
            generation_state = {}
        child_index = int(generation_state.get("child_count") or 0) + 1
        generation_state["child_count"] = child_index
        state[generation_id] = generation_state
        self._state.save(state)
        return child_index

    def _emit_span(
        self,
        *,
        event: Mapping[str, Any],
        span_name: str,
        span_id: str,
        parent_id: str | None,
        log_type: str,
        entity_path: str,
        entity_input: Any,
        entity_output: Any,
        metadata: Mapping[str, Any],
        start_time: str,
        end_time: str,
    ) -> CursorHookResult:
        trace_id = _trace_id(event)
        attrs = _base_attributes(
            event=event,
            log_type=log_type,
            entity_name=span_name,
            entity_path=entity_path,
            entity_input=entity_input,
            entity_output=entity_output,
            metadata=metadata,
        )
        span = build_readable_span(
            name=span_name,
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            start_time_iso=start_time,
            end_time_iso=end_time,
            attributes=attrs,
        )
        emitted = inject_span(span=span)
        return CursorHookResult(
            event_name=_string(event.get(CURSOR_HOOK_EVENT_NAME)),
            emitted=bool(emitted),
            span_name=span_name,
            trace_id=trace_id,
            span_id=span_id,
        )


def _base_attributes(
    *,
    event: Mapping[str, Any],
    log_type: str,
    entity_name: str,
    entity_path: str,
    entity_input: Any,
    entity_output: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    conversation_id = _conversation_id(event)
    workflow_name = _workflow_name(event)
    attrs: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: log_type,
        RESPAN_THREADS_ID: f"cursor_{conversation_id}",
        RESPAN_TRACE_GROUP_ID: workflow_name,
        SpanAttributes.TRACELOOP_WORKFLOW_NAME: workflow_name,
        SpanAttributes.TRACELOOP_ENTITY_NAME: entity_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: entity_path,
        SpanAttributes.TRACELOOP_ENTITY_INPUT: _json_string(entity_input),
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT: _json_string(entity_output),
    }

    common_metadata = {
        "cursor.conversation_id": conversation_id,
        "cursor.generation_id": _generation_id(event),
        "cursor.model": _string(event.get(CURSOR_MODEL)),
        "cursor.version": _string(event.get(CURSOR_VERSION)),
    }
    for key, value in {**common_metadata, **metadata}.items():
        if value in (None, ""):
            continue
        attrs[f"{RESPAN_METADATA}.{key}"] = _metadata_value(value)

    return attrs


def _format_edits(edits: Any) -> list[dict[str, Any]] | str:
    if not isinstance(edits, list):
        return "No edits"

    formatted: list[dict[str, Any]] = []
    for index, edit in enumerate(edits, start=1):
        if isinstance(edit, Mapping):
            formatted.append(
                {
                    "index": index,
                    "start_line": edit.get("startLine")
                    or _nested_get(edit, ("start", "line")),
                    "end_line": edit.get("endLine") or _nested_get(edit, ("end", "line")),
                    "old": _truncate(_string(edit.get("oldText") or edit.get("old")), 500),
                    "new": _truncate(_string(edit.get("newText") or edit.get("new")), 500),
                }
            )
        else:
            formatted.append({"index": index, "value": _truncate(str(edit), 500)})
    return formatted


def _conversation_id(event: Mapping[str, Any]) -> str:
    return _string(event.get(CURSOR_CONVERSATION_ID)) or "unknown"


def _generation_id(event: Mapping[str, Any]) -> str:
    return _string(event.get(CURSOR_GENERATION_ID)) or "unknown"


def _trace_id(event: Mapping[str, Any]) -> str:
    return f"cursor:{_conversation_id(event)}:{_generation_id(event)}"


def _root_span_id(event: Mapping[str, Any]) -> str:
    return f"{_generation_id(event)}-root"


def _workflow_name(event: Mapping[str, Any]) -> str:
    return f"cursor_{_conversation_id(event)}"


def _event_time(event: Mapping[str, Any]) -> str:
    timestamp = event.get("timestamp") or event.get("time")
    if isinstance(timestamp, str) and timestamp:
        return timestamp
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _subtract_ms(timestamp: str, duration_ms: int) -> str:
    try:
        normalized = timestamp.replace("Z", "+00:00")
        end_time = datetime.fromisoformat(normalized)
    except ValueError:
        end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(milliseconds=max(duration_ms, 0))
    return start_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _duration_ms(event: Mapping[str, Any], *, default_ms: int = 100) -> int:
    raw_value = event.get("duration_ms")
    if raw_value is None:
        raw_value = event.get("duration")
    if isinstance(raw_value, int | float):
        if raw_value < 0:
            return default_ms
        return int(raw_value)
    return default_ms


def _json_string(value: Any) -> str:
    if isinstance(value, str):
        return _truncate(value, _MAX_ENTITY_OUTPUT_LENGTH)
    return json.dumps(serialize_value(value), default=str)


def _jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped[:1] not in {"{", "["}:
        return stripped
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _metadata_value(value: Any) -> str | int | float | bool:
    if isinstance(value, bool | int | float):
        return value
    return _metadata_string(str(value))


def _metadata_string(value: str) -> str:
    return _truncate(value, _MAX_METADATA_VALUE_LENGTH)


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def _short_id(value: str) -> str:
    return value if len(value) <= 24 else f"{value[:21]}..."


def _compact_name(prefix: str, value: str) -> str:
    value = value.strip()
    if not value:
        return prefix
    return f"{prefix}: {_truncate(value, 64)}"


def _nested_get(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = payload
    for part in path:
        if not isinstance(cursor, Mapping):
            return None
        cursor = cursor.get(part)
    return cursor


def monotonic_ns() -> int:
    """Expose monotonic time for tests without making it part of the API."""

    return time.monotonic_ns()
