"""Cursor SDK hook instrumentation plugin for Respan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._processor import CursorHookProcessor


class CursorSDKInstrumentor:
    """Respan instrumentor for Cursor hook events.

    Cursor's Python-facing surface is hook JSON delivered to a command over
    stdin. There is no long-lived Python SDK object to monkey-patch, so the
    lifecycle methods intentionally only mark this plugin active. Use
    :meth:`create_processor` or :meth:`process_event` to emit spans.
    """

    name = "cursor-sdk"

    def __init__(self, *, state_path: str | Path | None = None) -> None:
        self._state_path = Path(state_path) if state_path is not None else None
        self._processor: CursorHookProcessor | None = None
        self._is_instrumented = False

    def activate(self) -> None:
        self._processor = CursorHookProcessor(state_path=self._state_path)
        self._is_instrumented = True

    def deactivate(self) -> None:
        self._processor = None
        self._is_instrumented = False

    @property
    def is_instrumented(self) -> bool:
        return self._is_instrumented

    def process_event(self, event: dict[str, Any]):
        if self._processor is None:
            self.activate()
        assert self._processor is not None
        return self._processor.process_event(event)

    @staticmethod
    def create_processor(
        *,
        state_path: str | Path | None = None,
    ) -> CursorHookProcessor:
        return CursorHookProcessor(state_path=state_path)
