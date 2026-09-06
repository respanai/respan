"""Reconcile SDK tool outcomes before upstream's unmatched-span cleanup."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from opentelemetry.instrumentation.claude_agent_sdk._constants import (
    ERROR_TYPE,
    GEN_AI_TOOL_CALL_RESULT,
)
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)


class _InvocationMessageStream:
    """Allow a completed standalone turn to stop upstream without GeneratorExit.

    Streaming prompts can wait for the caller to consume a ResultMessage before
    supplying more input. Never read ahead or delay that result. When the caller
    closes at a result, stop the source iterator and resume upstream just enough
    for its async-for loop to finish normally, then close the SDK source.
    """

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self._source: Any = None
        self._stop = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._source = aiter(self._wrapped(*args, **kwargs))
        return self

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        if self._stop:
            raise StopAsyncIteration
        return await anext(self._source)

    async def close(self, upstream: Any, *, after_result: bool) -> None:
        try:
            if after_result:
                self._stop = True
                try:
                    await anext(upstream)
                except StopAsyncIteration:
                    pass
        finally:
            try:
                await upstream.aclose()
            finally:
                close_source = getattr(self._source, "aclose", None)
                if close_source is not None:
                    await close_source()


def _capture_failure_output(span: Any, error: str, *, capture_content: bool) -> None:
    """Annotate only: the original failure hook still owns its pop and end."""
    if capture_content:
        _set_result_content(span, {"error": error})


def _set_result_content(span: Any, content: Any) -> None:
    try:
        span.set_attribute(GEN_AI_TOOL_CALL_RESULT, json.dumps(content, default=str))
    except (TypeError, ValueError):
        logger.debug("Could not serialize Claude tool result", exc_info=True)


def _finish_tool_span(
    ctx: Any,
    tool_use_id: str,
    *,
    content: Any,
    is_error: bool,
    permission_denied: bool = False,
) -> None:
    # No await between pop and end: a late hook or repeated result sees no
    # pending span and cannot finish the same invocation twice.
    span = ctx.active_tool_spans.pop(tool_use_id, None)
    if span is None:
        return
    try:
        if is_error:
            span.set_attribute(
                ERROR_TYPE, "permission_denied" if permission_denied else "tool_error"
            )
            span.set_status(
                StatusCode.ERROR,
                "Tool permission denied" if permission_denied else "Tool execution failed",
            )
        if ctx.capture_content:
            _set_result_content(span, content)
    finally:
        span.end()


def _reconcile_sdk_message(ctx: Any, message: Any) -> None:
    """Close only tool IDs for which the SDK supplies a terminal outcome.

    Unknown tool/subagent spans remain pending so upstream can report genuine
    cancellation or lifecycle errors. Never infer success from agent completion.
    """
    if ctx is None or not getattr(ctx, "active_tool_spans", None):
        return

    from claude_agent_sdk import ResultMessage, ToolResultBlock, UserMessage

    if isinstance(message, UserMessage) and isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, ToolResultBlock) and block.tool_use_id:
                _finish_tool_span(
                    ctx, block.tool_use_id, content=block.content,
                    is_error=block.is_error is True,
                )

    if isinstance(message, ResultMessage):
        # Some CLI permission failures have no post-hook or ToolResultBlock.
        # This is an explicit SDK denial record, not fabricated tool stderr.
        denials = getattr(message, "permission_denials", None)
        if not isinstance(denials, list):
            return
        for denial in denials:
            tool_use_id = denial.get("tool_use_id") if isinstance(denial, Mapping) else None
            if isinstance(tool_use_id, str) and tool_use_id:
                _finish_tool_span(
                    ctx, tool_use_id,
                    content={"error": "permission_denied", "source": "ResultMessage.permission_denials"},
                    is_error=True, permission_denied=True,
                )
