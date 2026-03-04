"""API-related constants for Respan SDK.

These constants are shared across all Respan exporters and integrations.
"""

from typing import Optional

# Base API URLs
DEFAULT_RESPAN_API_BASE_URL = "https://api.respan.ai/api"

# API Paths
TRACES_INGEST_PATH = "v1/traces/ingest"
CHAT_COMPLETIONS_PATH = "chat/completions"


def build_api_endpoint(base_url: str, relative_path: str) -> str:
    """Build a full API endpoint URL from base URL and relative path.

    Handles base URLs with or without trailing /api (e.g. https://api.respan.ai
    or https://api.respan.ai/api) so all SDK consumers normalize URLs the same way.
    """
    normalized = base_url.rstrip("/")
    if normalized.endswith("/api"):
        return f"{normalized}/{relative_path}"
    return f"{normalized}/api/{relative_path}"


def resolve_chat_completions_endpoint(base_url: Optional[str] = None) -> str:
    """Resolve the chat completions endpoint URL from an optional base URL."""
    if not base_url:
        return f"{DEFAULT_RESPAN_API_BASE_URL}/{CHAT_COMPLETIONS_PATH}"
    return build_api_endpoint(base_url, CHAT_COMPLETIONS_PATH)


# Full endpoint URLs (convenience)
DEFAULT_TRACES_INGEST_ENDPOINT = f"{DEFAULT_RESPAN_API_BASE_URL}/{TRACES_INGEST_PATH}"
