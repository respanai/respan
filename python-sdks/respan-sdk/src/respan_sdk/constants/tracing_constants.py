"""Tracing endpoint constants for Respan SDK."""

from typing import Optional

from respan_sdk.constants.api_constants import (
    TRACES_INGEST_PATH,
    build_api_endpoint,
    DEFAULT_RESPAN_API_BASE_URL,
)

RESPAN_TRACING_INGEST_ENDPOINT = f"{DEFAULT_RESPAN_API_BASE_URL}/{TRACES_INGEST_PATH}"

# Anti-recursion header: SDK exporter sends this on every export request.
# Server-side tracing decorators MUST check for this header and skip
# EMITTING NEW TRACES for the ingest request — but still PROCESS the
# payload (store the trace in CH). The goal is to prevent infinite loops
# when the ingest endpoint is itself observed, not to drop trace data.
RESPAN_DOGFOOD_HEADER = "X-Respan-Dogfood"


def resolve_tracing_ingest_endpoint(base_url: Optional[str] = None) -> str:
    """Build tracing ingest endpoint from an optional base URL."""
    if not base_url:
        return RESPAN_TRACING_INGEST_ENDPOINT
    return build_api_endpoint(base_url, TRACES_INGEST_PATH)
