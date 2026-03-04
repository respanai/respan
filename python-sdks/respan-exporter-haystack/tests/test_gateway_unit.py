"""Unit tests for gateway _build_payload logic."""

import pytest

from respan_exporter_haystack.gateway import RespanGenerator


class TestBuildPayload:
    """Unit tests for _BaseRespanGenerator._build_payload."""

    def test_build_payload_with_model_and_messages(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RESPAN_API_KEY", "test-key")
        gen = RespanGenerator(model="gpt-4o-mini")
        payload = gen._build_payload(messages=[{"role": "user", "content": "Hi"}])
        assert payload["model"] == "gpt-4o-mini"
        assert payload["messages"] == [{"role": "user", "content": "Hi"}]
        assert "prompt" not in payload

    def test_build_payload_with_prompt_id_and_variables(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RESPAN_API_KEY", "test-key")
        gen = RespanGenerator(prompt_id="abc123")
        payload = gen._build_payload(
            prompt_variables={"name": "Alice"},
        )
        assert payload["prompt"] == {
            "prompt_id": "abc123",
            "override": True,
            "variables": {"name": "Alice"},
        }
        assert "model" not in payload

    def test_build_payload_merges_generation_kwargs(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RESPAN_API_KEY", "test-key")
        gen = RespanGenerator(
            model="gpt-4",
            generation_kwargs={"temperature": 0.7, "max_tokens": 100},
        )
        payload = gen._build_payload(
            messages=[{"role": "user", "content": "Hi"}],
            generation_kwargs={"max_tokens": 200},
        )
        assert payload["model"] == "gpt-4"
        assert payload["messages"] == [{"role": "user", "content": "Hi"}]
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 200  # run-time overrides init

    def test_build_payload_omits_none_values(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RESPAN_API_KEY", "test-key")
        gen = RespanGenerator(model="gpt-4o-mini")
        payload = gen._build_payload(messages=[])
        assert "messages" in payload
        assert payload["model"] == "gpt-4o-mini"
        # No None entries
        assert None not in payload.values()

    def test_build_payload_without_messages_or_prompt_vars(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RESPAN_API_KEY", "test-key")
        gen = RespanGenerator(model="gpt-4o-mini")
        payload = gen._build_payload()
        assert payload["model"] == "gpt-4o-mini"
        assert "messages" not in payload
        assert "prompt" not in payload
