"""Preserve operation content while honoring the SDK's deferred sink routing."""

from __future__ import annotations

from typing import Any

from ai import experimental_telemetry as telemetry

from ._constants import AI_OPERATION_CONTENT
from ._translator import json_value


class _OperationSink:
    """Forward snapshots, delaying only this operation's completed snapshot.

    The SDK snapshots an operation before its function returns the full result.
    Holding its terminal snapshot lets us retain that result in serializable
    SDK data without emitting OpenTelemetry spans during durable execution.
    """

    def __init__(self, sink: Any, kind: str) -> None:
        self._sink = sink
        self._kind = kind
        self._target_id: str | None = None
        self._finished: Any = None

    async def on_push(self, span: Any, /) -> None:
        kind = span.data.get("kind") if isinstance(span.data, dict) else span.data.kind
        if self._target_id is None and kind == self._kind:
            self._target_id = span.id
        if span.id == self._target_id and span.ended_at is not None:
            self._finished = span
            return
        await self._sink.on_push(span)

    async def finish(
        self, *, capture_content: bool, input_payload: str | None, result: Any
    ) -> None:
        if self._finished is None:
            return
        snapshot = self._finished
        self._finished = None
        if capture_content:
            content = {
                "span_id": snapshot.id,
                "kind": self._kind,
                "input": input_payload,
            }
            if result is not None:
                content["output"] = json_value(result.value)
            snapshot.trace_attrs = {
                **snapshot.trace_attrs,
                AI_OPERATION_CONTENT: content,
            }
        # Span.push preserves the SDK's snapshot copying and non-fatal sink
        # failure behavior; the application's sink and live span stay untouched.
        async with telemetry.use_sink(self._sink):
            await snapshot.push()
