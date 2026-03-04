"""Unit tests for config and URL resolution utils."""

import pytest

from respan_sdk.constants import resolve_chat_completions_endpoint
from respan_sdk.constants.api_constants import build_api_endpoint

from respan_exporter_haystack.utils.config_utils import (
    DEFAULT_RESPAN_BASE_URL,
    resolve_api_key,
    resolve_base_url,
    resolve_platform_logs_url,
)


class TestResolveBaseUrl:
    """Unit tests for resolve_base_url."""

    def test_returns_explicit_base_url_normalized(self):
        # include_api_path=False (default): strip /api if present
        assert resolve_base_url(base_url="https://custom.respan.ai") == "https://custom.respan.ai"
        assert resolve_base_url(base_url="https://custom.respan.ai/api") == "https://custom.respan.ai"
        # include_api_path=True: ensure URL ends with /api
        assert resolve_base_url(base_url="https://custom.respan.ai", include_api_path=True) == "https://custom.respan.ai/api"
        assert resolve_base_url(base_url="https://custom.respan.ai/api", include_api_path=True) == "https://custom.respan.ai/api"

    def test_without_api_path_returns_base_without_api_suffix(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("RESPAN_BASE_URL", raising=False)
        url = resolve_base_url(base_url=None, include_api_path=False)
        assert url == DEFAULT_RESPAN_BASE_URL
        assert not url.endswith("/api")

    def test_with_api_path_returns_base_with_api(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("RESPAN_BASE_URL", raising=False)
        url = resolve_base_url(base_url=None, include_api_path=True)
        assert url.endswith("/api")

    def test_prefers_env_over_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RESPAN_BASE_URL", "https://env.respan.ai")
        assert resolve_base_url(base_url=None) == "https://env.respan.ai"
        monkeypatch.setenv("RESPAN_BASE_URL", "https://env.respan.ai/api")
        assert resolve_base_url(base_url=None, include_api_path=False) == "https://env.respan.ai"

    def test_explicit_overrides_env_normalized_by_include_api_path(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RESPAN_BASE_URL", "https://env.respan.ai")
        assert resolve_base_url(base_url="https://explicit.respan.ai") == "https://explicit.respan.ai"
        assert resolve_base_url(base_url="https://explicit.respan.ai/api", include_api_path=True) == "https://explicit.respan.ai/api"


class TestResolveApiKey:
    """Unit tests for resolve_api_key."""

    def test_returns_explicit_key(self):
        assert resolve_api_key(api_key="sk-explicit") == "sk-explicit"

    def test_returns_env_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RESPAN_API_KEY", "sk-from-env")
        assert resolve_api_key(api_key=None) == "sk-from-env"

    def test_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RESPAN_API_KEY", "sk-env")
        assert resolve_api_key(api_key="sk-explicit") == "sk-explicit"


class TestResolvePlatformLogsUrl:
    """Unit tests for resolve_platform_logs_url."""

    def test_uses_platform_url_when_given(self):
        assert (
            resolve_platform_logs_url(base_url="https://api.respan.ai", platform_url="https://app.respan.ai/logs")
            == "https://app.respan.ai/logs"
        )

    def test_derives_from_base_url_when_no_platform_url(self):
        url = resolve_platform_logs_url(base_url="https://api.respan.ai/api", platform_url=None)
        assert url == "https://api.respan.ai/logs"

    def test_strips_trailing_slash_from_platform_url(self):
        assert (
            resolve_platform_logs_url(base_url="https://api.respan.ai", platform_url="https://app.respan.ai/")
            == "https://app.respan.ai"
        )


class TestSdkEndpointResolution:
    """Unit tests for SDK endpoint resolution (build_api_endpoint / resolve_chat_completions_endpoint)."""

    def test_resolve_chat_completions_base_without_api(self):
        assert (
            resolve_chat_completions_endpoint(base_url="https://api.respan.ai")
            == "https://api.respan.ai/api/chat/completions"
        )

    def test_resolve_chat_completions_base_with_api(self):
        assert (
            resolve_chat_completions_endpoint(base_url="https://api.respan.ai/api")
            == "https://api.respan.ai/api/chat/completions"
        )

    def test_resolve_chat_completions_none_returns_default(self):
        url = resolve_chat_completions_endpoint(base_url=None)
        assert "/chat/completions" in url
        assert "api.respan.ai" in url

    def test_build_api_endpoint_normalizes_trailing_slash(self):
        assert (
            build_api_endpoint(base_url="https://api.respan.ai/", relative_path="chat/completions")
            == "https://api.respan.ai/api/chat/completions"
        )

    def test_build_api_endpoint_with_api_suffix_appends_path_only(self):
        assert (
            build_api_endpoint(base_url="https://api.respan.ai/api", relative_path="v1/traces/ingest")
            == "https://api.respan.ai/api/v1/traces/ingest"
        )
