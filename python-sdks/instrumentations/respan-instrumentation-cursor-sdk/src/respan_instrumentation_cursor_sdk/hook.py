"""Command-line hook runner for Cursor."""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from respan_tracing import RespanTelemetry

from ._constants import DEFAULT_CURSOR_STATE_FILE
from ._processor import CursorHookProcessor

logger = logging.getLogger(__name__)


def read_stdin() -> dict[str, Any] | None:
    payload = sys.stdin.read()
    if not payload.strip():
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        logger.exception("Cursor hook input was not valid JSON")
        return None
    return value if isinstance(value, dict) else None


def main() -> int:
    event = read_stdin()
    if event is None:
        return 0

    name = event.get("hook_event_name")
    if name == "beforeSubmitPrompt":
        print(json.dumps({"continue": True}))
    elif name in {
        "preToolUse",
        "beforeShellExecution",
        "beforeMCPExecution",
        "beforeReadFile",
        "beforeTabFileRead",
        "subagentStart",
    }:
        print(json.dumps({"permission": "allow"}))
    if os.getenv("TRACE_TO_RESPAN", "true").strip().lower() in {
        "false",
        "0",
        "off",
        "no",
    }:
        return 0

    state_path = Path(
        os.getenv("RESPAN_CURSOR_STATE_FILE", str(DEFAULT_CURSOR_STATE_FILE))
    )

    try:
        with redirect_stdout(sys.stderr):
            telemetry = RespanTelemetry(
                app_name=os.getenv("RESPAN_CURSOR_APP_NAME", "cursor-sdk"),
                api_key=os.getenv("RESPAN_API_KEY"),
                base_url=os.getenv("RESPAN_BASE_URL"),
                is_auto_instrument=False,
                is_batching_enabled=os.getenv("RESPAN_CURSOR_BATCHING", "false")
                .strip()
                .lower()
                == "true",
            )
            processor = CursorHookProcessor(state_path=state_path)
            try:
                processor.process_event(event)
                telemetry.flush()
            finally:
                processor.close(discard_pending=False)
    except Exception:  # noqa: BLE001 - An observation hook must never block Cursor.
        logger.warning("Cursor telemetry failed; observation hook remains non-blocking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
