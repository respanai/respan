"""Translate version-one Cursor hook observations into canonical spans."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_TOOL_CALL_ID,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
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
from respan_sdk.utils.data_processing.id_processing import ensure_trace_id
from respan_tracing.utils.span_factory import build_readable_span, inject_span

from ._constants import CURSOR_SUPPORTED_EVENTS, DEFAULT_CURSOR_STATE_FILE
from ._policy import Policy, permitted, span_key, suppressed
from ._serialization import dumps, safe

_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class CursorHookResult:
    event_name: str
    emitted: bool
    span_name: str | None = None
    trace_id: str | None = None
    span_id: str | None = None


class CursorStateStore:
    """Atomic owner-only state with per-file process and thread serialization."""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else DEFAULT_CURSOR_STATE_FILE
        with _LOCKS_GUARD:
            self.lock = _LOCKS.setdefault(str(self.path.absolute()), threading.RLock())

    def load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if type(data) is dict else {}
        except (OSError, ValueError):
            return {}

    def save(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, name = tempfile.mkstemp(prefix=".respan-cursor-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(state, stream, separators=(",", ":"), allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @contextmanager
    def transaction(self):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX)
                elif os.name == "nt":
                    import msvcrt

                    if os.fstat(fd).st_size == 0:
                        os.write(fd, b"0")
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                state = self.load()
                yield state
                self.save(state)
            finally:
                if os.name == "posix":
                    fcntl.flock(fd, fcntl.LOCK_UN)
                elif os.name == "nt":
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                os.close(fd)


def _string(value):
    return value if type(value) is str else ""


def _event_time(event):
    value = event.get("timestamp", event.get("time"))
    try:
        stamp = (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            if type(value) is str
            else datetime.now(timezone.utc)
        )
        return (
            stamp.replace(tzinfo=stamp.tzinfo or timezone.utc)
            .astimezone(timezone.utc)
            .isoformat()
        )
    except ValueError:
        return datetime.now(timezone.utc).isoformat()


def _duration(event):
    value = event.get("duration_ms", event.get("duration"))
    return (
        value
        if type(value) in (int, float) and value >= 0 and value < float("inf")
        else 0
    )


def _name(value, fallback):
    value = re.sub(r"[^\w.\-]", "_", _string(value)).strip("_")
    return value[:80] or fallback


def _jsonish(value):
    if type(value) is str:
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


class CursorHookProcessor:
    def __init__(self, *, state_path=None, capture_content=True):
        self._state = CursorStateStore(state_path)
        self.capture_content = capture_content
        self._owned_policy = None
        self._closed = False
        self._used_state = False
        self._bounds = {}

    def close(self, *, discard_pending=True):
        self._closed = True
        if discard_pending and self._used_state and self.state_path.exists():
            with self._state.transaction() as state:
                for record in state.values():
                    if type(record) is dict and not record.get("finished"):
                        record["allowed"] = False
                        for key in ("prompt", "responses", "subagents", "attachments"):
                            record.pop(key, None)
        if self._owned_policy is not None:
            self._owned_policy.close()
            self._owned_policy = None
        self._bounds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def state_path(self):
        return self._state.path

    def process_event(self, event: Mapping[str, Any]):
        name = _string(event.get("hook_event_name")) if type(event) is dict else ""
        if self._closed or name not in CURSOR_SUPPORTED_EVENTS:
            return CursorHookResult(name, False)
        from ._native import current_policy

        policy = current_policy()
        if policy is None:
            if self._owned_policy is None:
                provider = trace.get_tracer_provider()
                if not hasattr(provider, "add_span_processor"):
                    return CursorHookResult(name, False)
                self._owned_policy = Policy()
                provider.add_span_processor(self._owned_policy)
            policy = self._owned_policy
        parent_span = trace.get_current_span()
        parent = span_key(parent_span)
        if not permitted():
            policy.deny(parent)
        allowed = self.capture_content and permitted() and policy.enroll(parent_span)
        sampled = not suppressed()
        if name in (
            "workspaceOpen",
            "sessionStart",
            "sessionEnd",
            "beforeTabFileRead",
            "afterTabFileEdit",
        ):
            if not sampled or not self._sampled(None):
                return CursorHookResult(name, False)
            data = (
                {
                    k: event[k]
                    for k in (
                        "session_id",
                        "reason",
                        "duration_ms",
                        "is_background_agent",
                        "composer_mode",
                        "final_status",
                        "workspace_roots",
                        "file_path",
                        "content",
                        "edits",
                    )
                    if k in event
                }
                if allowed
                else None
            )
            return self._emit(
                event,
                _name(name, "hook"),
                LOG_TYPE_TOOL if name == "afterTabFileEdit" else LOG_TYPE_TASK,
                None,
                None,
                None,
                data,
                None,
                allowed,
                error=_string(event.get("error_message"))
                if name == "sessionEnd" and event.get("reason") == "error"
                else None,
            )
        conversation = _string(event.get("conversation_id"))
        generation = _string(event.get("generation_id"))
        if not conversation or not generation:
            return CursorHookResult(name, False)
        key = json.dumps([conversation, generation], separators=(",", ":"))
        trace_id = f"cursor:{conversation}:{generation}"
        root_id = f"{conversation}:{generation}:root"
        self._used_state = True
        with self._state.transaction() as state:
            record = state.get(key)
            if type(record) is not dict:
                record = {
                    "start_time": _event_time(event),
                    "allowed": allowed,
                    "sampled": self._sampled(trace_id) and sampled,
                    "parent": list(parent) if parent else None,
                    "child_count": 0,
                    "responses": [],
                }
                state[key] = record
            original_parent = tuple(record["parent"]) if record.get("parent") else None
            bound = self._bounds.setdefault(key, policy.watch(original_parent))
            record["allowed"] = bool(
                record.get("allowed")
                and allowed
                and bound.allowed
                and (policy is None or policy.ancestors(original_parent))
            )
            record["sampled"] = bool(record.get("sampled") and sampled)
            capture = record["allowed"] and record["sampled"]
            if not capture:
                for field in ("prompt", "responses", "subagents"):
                    record.pop(field, None)
            if record.get("finished"):
                return CursorHookResult(name, False)
            if name == "beforeSubmitPrompt":
                if capture:
                    record["prompt"] = safe(event.get("prompt"))
                    record["attachments"] = safe(event.get("attachments"))
                return CursorHookResult(name, False)
            if name == "stop":
                status = _string(event.get("status"))
                result = self._emit(
                    event,
                    "agent",
                    LOG_TYPE_AGENT,
                    trace_id,
                    root_id,
                    None,
                    {
                        "prompt": record.get("prompt"),
                        "attachments": record.get("attachments"),
                    },
                    record.get("responses") or None,
                    capture,
                    start=record["start_time"],
                    error=_string(event.get("error_message")) or status
                    if status
                    in ("error", "aborted", "cancelled", "canceled", "stopped")
                    else None,
                    metadata={
                        "cursor.stop_status": status,
                        "cursor.child_count": record["child_count"],
                        "cursor.loop_count": event.get("loop_count"),
                    },
                    emit=record["sampled"],
                )
                if result.emitted or not record["sampled"]:
                    state[key] = {
                        "finished": True,
                        "allowed": record["allowed"],
                        "sampled": record["sampled"],
                    }
                    self._bounds.pop(key, None)
                self._bound(state)
                return result
            record["child_count"] += 1
            span_id = f"{root_id}:event:{record['child_count']}"
            metadata = {"cursor.index": record["child_count"]}
            if name == "afterAgentResponse":
                output = event.get("text")
                if capture:
                    record.setdefault("responses", []).append(safe(output))
                return self._emit(
                    event,
                    "response",
                    LOG_TYPE_TASK,
                    trace_id,
                    span_id,
                    root_id,
                    None,
                    output,
                    capture,
                    metadata=metadata,
                    emit=record["sampled"],
                )
            if name == "afterAgentThought":
                entity, output, kind = None, event.get("text"), LOG_TYPE_TASK
                label = "reasoning"
            elif name == "afterShellExecution":
                entity, output, kind = (
                    {"command": event.get("command")},
                    event.get("output"),
                    LOG_TYPE_TOOL,
                )
                label = "Shell"
            elif name == "afterFileEdit":
                entity, output, kind = (
                    {"file_path": event.get("file_path")},
                    event.get("edits"),
                    LOG_TYPE_TOOL,
                )
                label = "Write"
            elif name in ("afterMCPExecution", "postToolUse", "postToolUseFailure"):
                entity = _jsonish(event.get("tool_input"))
                output = _jsonish(
                    event.get(
                        "result_json" if name == "afterMCPExecution" else "tool_output"
                    )
                )
                kind, label = LOG_TYPE_TOOL, _name(event.get("tool_name"), "tool")
                metadata["cursor.failure_type"] = (
                    event.get("failure_type") if name == "postToolUseFailure" else None
                )
            elif name == "subagentStart":
                subkey = _string(event.get("subagent_id")) or _string(
                    event.get("subagent_type")
                )
                record.setdefault("subagents", {})[subkey] = {
                    "span_id": span_id,
                    "start": _event_time(event),
                    "input": safe(event.get("task")) if capture else None,
                    "tool_call_id": _string(event.get("tool_call_id")),
                }
                return CursorHookResult(name, False)
            elif name == "subagentStop":
                subkey = _string(event.get("subagent_id")) or _string(
                    event.get("subagent_type")
                )
                sub = record.get("subagents", {}).pop(subkey, {})
                return self._emit(
                    event,
                    _name(event.get("subagent_type"), "subagent"),
                    LOG_TYPE_AGENT,
                    trace_id,
                    sub.get("span_id", span_id),
                    root_id,
                    sub.get("input"),
                    event.get("summary"),
                    capture,
                    start=sub.get("start"),
                    error=_string(event.get("error_message"))
                    or _string(event.get("status"))
                    if event.get("status") in ("error", "aborted")
                    else None,
                    call_id=sub.get("tool_call_id"),
                    metadata=metadata,
                    emit=record["sampled"],
                )
            else:
                fields = {
                    "preToolUse": ("tool_name", "tool_input"),
                    "beforeShellExecution": ("command", "cwd"),
                    "beforeMCPExecution": (
                        "tool_name",
                        "tool_input",
                        "mcp_server_name",
                    ),
                    "beforeReadFile": ("file_path", "content", "attachments"),
                    "preCompact": (
                        "trigger",
                        "context_usage_percent",
                        "context_tokens",
                        "context_window_size",
                        "message_count",
                        "messages_to_compact",
                        "is_first_compaction",
                    ),
                }
                entity = (
                    {k: event[k] for k in fields.get(name, ()) if k in event}
                    if capture
                    else None
                )
                output, kind, label = None, LOG_TYPE_TASK, name
            result = self._emit(
                event,
                label,
                kind,
                trace_id,
                span_id,
                root_id,
                entity,
                output,
                capture,
                error=_string(event.get("error_message"))
                or _string(event.get("failure_type"))
                if name == "postToolUseFailure"
                else None,
                call_id=_string(event.get("tool_use_id")),
                metadata=metadata,
                emit=record["sampled"],
            )
            self._bound(state)
            return result

    @staticmethod
    def _bound(state):
        if len(state) > 1024:
            for key in list(state):
                if type(state[key]) is dict and state[key].get("finished"):
                    del state[key]
                    if len(state) <= 1024:
                        break

    @staticmethod
    def _sampled(trace_id):
        provider = trace.get_tracer_provider()
        sampler = getattr(provider, "sampler", None)
        if sampler is None:
            return False
        try:
            return sampler.should_sample(
                None, ensure_trace_id(trace_id), "agent", attributes={}
            ).decision.is_sampled()
        except Exception:  # noqa: BLE001 - Fail closed for custom sampler faults.
            return False

    def _emit(
        self,
        event,
        name,
        kind,
        trace_id,
        span_id,
        parent,
        entity_input,
        entity_output,
        allowed,
        *,
        start=None,
        error=None,
        call_id=None,
        metadata=None,
        emit=True,
    ):
        event_name = _string(event.get("hook_event_name"))
        if not emit:
            return CursorHookResult(event_name, False)
        attrs = {
            RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
            RESPAN_LOG_TYPE: kind,
            SpanAttributes.TRACELOOP_ENTITY_NAME: name,
            SpanAttributes.TRACELOOP_ENTITY_PATH: "" if parent is None else name,
            SpanAttributes.TRACELOOP_WORKFLOW_NAME: "cursor-sdk",
        }
        if _string(event.get("conversation_id")):
            attrs[RESPAN_THREADS_ID] = "cursor_" + event["conversation_id"]
            attrs[RESPAN_TRACE_GROUP_ID] = "cursor_" + event["conversation_id"]
        meta = {
            k: v
            for k, v in {
                "cursor.event": event_name,
                "cursor.version": event.get("cursor_version"),
                "cursor.model_id": event.get("model_id", event.get("model")),
                **(metadata or {}),
            }.items()
            if v is not None and v != "" and type(v) in (str, int, float, bool)
        }
        attrs[RESPAN_METADATA] = dumps(meta)
        attrs.update({f"{RESPAN_METADATA}.{k}": v for k, v in safe(meta).items()})
        if kind == LOG_TYPE_TOOL and call_id:
            attrs[GEN_AI_TOOL_CALL_ID] = call_id
        if allowed and permitted():
            if entity_input is not None:
                attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = dumps(entity_input)
            if entity_output is not None and error is None:
                attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = dumps(entity_output)
            if error:
                attrs[ERROR_MESSAGE] = safe(error)
        end = _event_time(event)
        if start is None:
            start = (
                datetime.fromisoformat(end) - timedelta(milliseconds=_duration(event))
            ).isoformat()
        span = build_readable_span(
            name=name if kind != LOG_TYPE_TOOL else "tool." + name,
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent,
            start_time_iso=start,
            end_time_iso=end,
            attributes=attrs,
            merge_propagated=True,
        )
        if error:
            span._status = Status(StatusCode.ERROR, safe(error) if allowed else None)
        emitted = inject_span(span=span)
        return CursorHookResult(event_name, bool(emitted), name, trace_id, span_id)
