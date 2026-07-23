"""Shared runtime helpers for Claude Agent SDK examples."""

import asyncio
import os
import sys
from collections.abc import Callable
from typing import Any

import claude_agent_sdk
from claude_agent_sdk import ResultMessage

QUERY_TIMEOUT_SECONDS = int(os.getenv("RESPAN_QUERY_TIMEOUT_SECONDS", "90"))


def suppress_stderr() -> Callable[[], None]:
    """Redirect fd 2 to /dev/null. Returns a restore function."""
    stderr_fd = sys.stderr.fileno()
    saved = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)

    def restore() -> None:
        os.dup2(saved, stderr_fd)
        os.close(saved)

    return restore


async def query_for_result(
    *,
    prompt: str,
    options: Any,
    on_message: Callable[[Any], None] | None = None,
    timeout_seconds: int = QUERY_TIMEOUT_SECONDS,
) -> ResultMessage:
    """
    Run query() with stderr suppression and timeout protection.

    Claude Code can emit noisy hook callback errors to stderr and sometimes
    exit non-zero after already yielding a final ResultMessage.

    query() is auto-instrumented by ``ClaudeAgentSDKInstrumentor`` — no
    exporter argument needed.
    """
    result: ResultMessage | None = None
    restore_stderr = suppress_stderr()
    try:
        try:
            async with asyncio.timeout(timeout_seconds):
                async for message in claude_agent_sdk.query(prompt=prompt, options=options):
                    if on_message:
                        on_message(message)
                    if isinstance(message, ResultMessage):
                        result = message
        except TimeoutError as exc:
            raise TimeoutError(
                f"Timed out after {timeout_seconds}s waiting for query result."
            ) from exc
        except Exception:
            if result is None:
                raise
    finally:
        await asyncio.sleep(0.5)
        restore_stderr()

    if result is None:
        raise RuntimeError("Query completed without a ResultMessage.")

    return result
