"""Command-line hook runner for Cursor."""

from __future__ import annotations

import json
import logging
import os
import sys
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
    if os.getenv("TRACE_TO_RESPAN", "true").lower() == "false":
        return 0

    event = read_stdin()
    if event is None:
        return 0

    state_path = Path(
        os.getenv("RESPAN_CURSOR_STATE_FILE", str(DEFAULT_CURSOR_STATE_FILE))
    )

    telemetry = RespanTelemetry(
        app_name=os.getenv("RESPAN_CURSOR_APP_NAME", "cursor-sdk"),
        api_key=os.getenv("RESPAN_API_KEY"),
        base_url=os.getenv("RESPAN_BASE_URL"),
        is_auto_instrument=False,
        is_batching_enabled=os.getenv("RESPAN_CURSOR_BATCHING", "false").lower()
        == "true",
    )
    processor = CursorHookProcessor(state_path=state_path)
    processor.process_event(event)
    telemetry.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
