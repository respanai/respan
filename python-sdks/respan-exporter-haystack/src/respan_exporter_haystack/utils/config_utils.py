"""Configuration and endpoint helper utilities."""

import os
from urllib.parse import urlparse
from typing import Optional

from respan_sdk.constants.api_constants import DEFAULT_RESPAN_API_BASE_URL

DEFAULT_RESPAN_BASE_URL = DEFAULT_RESPAN_API_BASE_URL.removesuffix("/api")


def resolve_platform_logs_url(base_url: str, platform_url: Optional[str] = None) -> str:
    """Resolve the platform logs URL for trace links. Uses platform_url if set, else derives from base_url."""
    if platform_url:
        return platform_url.rstrip("/")
    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return f"{origin}/logs"


def resolve_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Resolve API key from explicit value or RESPAN_API_KEY environment variable."""
    if api_key:
        return api_key
    return os.getenv("RESPAN_API_KEY")


def resolve_base_url(base_url: Optional[str] = None, include_api_path: bool = False) -> str:
    """Resolve base URL from explicit value or RESPAN_BASE_URL environment variable.
    When include_api_path is True, the returned URL is normalized to end with /api (for API calls).
    When False, the trailing /api is stripped (e.g. for platform/logs URLs).
    """
    if base_url:
        resolved = base_url
    else:
        resolved = os.getenv("RESPAN_BASE_URL")
        if not resolved:
            resolved = (
                DEFAULT_RESPAN_API_BASE_URL if include_api_path else DEFAULT_RESPAN_BASE_URL
            )

    resolved = resolved.rstrip("/")
    if include_api_path:
        if not resolved.endswith("/api"):
            resolved = f"{resolved}/api"
    else:
        if resolved.endswith("/api"):
            resolved = resolved.removesuffix("/api")
    return resolved
