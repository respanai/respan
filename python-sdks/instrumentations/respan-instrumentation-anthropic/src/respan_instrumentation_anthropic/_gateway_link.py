"""Link a span to the Respan gateway log row that already records its call.

The Respan gateway returns the unique_id of the log row it wrote in the
``X-Respan-Log-Id`` response header. That row carries the call's usage, cost and
content, so a span for the same call carries the id and is exported as a link
only (the server shows the row in its place). Mirrors the OpenAI
instrumentation's handling; constants are restated here so the package works on
the released respan-sdk floor.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from typing import Any

from opentelemetry.semconv_ai import SpanAttributes

RESPAN_LOG_ID_RESPONSE_HEADER = "X-Respan-Log-Id"
RESPAN_GATEWAY_LOG_ID = "respan.gateway.log_id"

BASE_CLIENT_MODULE = "anthropic._base_client"
SYNC_BASE_CLIENT_CLASS = "SyncAPIClient"
ASYNC_BASE_CLIENT_CLASS = "AsyncAPIClient"
PROCESS_RESPONSE_METHOD = "_process_response"

# Dropped from a linked span: usage (the gateway row has it) and content (the
# server estimates tokens from content when usage is absent).
_LINKED_DROPPED_PREFIXES = (
    "gen_ai.usage.",
    "llm.usage.",
    f"{SpanAttributes.LLM_PROMPTS}.",
    f"{SpanAttributes.LLM_COMPLETIONS}.",
)
_LINKED_DROPPED_KEYS = frozenset(
    {SpanAttributes.TRACELOOP_ENTITY_INPUT, SpanAttributes.TRACELOOP_ENTITY_OUTPUT}
)

# Set by each instrumented call for its duration; the patched
# ``_process_response`` writes the gateway log id of the response into it.
GATEWAY_LOG_ID_SLOT: ContextVar[list[str] | None] = ContextVar(
    "respan_anthropic_gateway_log_id_slot", default=None
)


def _header_log_id(response: Any) -> str | None:
    try:
        value = response.headers.get(RESPAN_LOG_ID_RESPONSE_HEADER)
    except Exception:  # noqa: BLE001 - header capture must never break a call
        return None
    return value if isinstance(value, str) and value else None


def _record(response: Any) -> None:
    slot = GATEWAY_LOG_ID_SLOT.get()
    if slot is None:
        return
    log_id = _header_log_id(response)
    if log_id:
        slot[:] = [log_id]


def gateway_log_id(slot: list[str], exc: BaseException | None = None) -> str | None:
    """The gateway log id of this call's response, if it went through it."""
    if slot:
        return slot[0]
    response = getattr(exc, "response", None) if exc is not None else None
    return _header_log_id(response) if response is not None else None


def link_to_gateway_log(attrs: dict[str, Any], log_id: str) -> dict[str, Any]:
    linked = {
        key: value
        for key, value in attrs.items()
        if key not in _LINKED_DROPPED_KEYS
        and not key.startswith(_LINKED_DROPPED_PREFIXES)
    }
    linked[RESPAN_GATEWAY_LOG_ID] = log_id
    return linked


def wrap_sync_process_response(original: Any) -> Any:
    @wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        _record(kwargs.get("response"))
        return original(self, *args, **kwargs)

    return wrapper


def wrap_async_process_response(original: Any) -> Any:
    @wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        _record(kwargs.get("response"))
        return await original(self, *args, **kwargs)

    return wrapper
