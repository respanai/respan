import pytest
from haystack.dataclasses import ChatMessage

from respan_exporter_haystack.connector import RespanConnector
from respan_exporter_haystack.gateway import RespanChatGenerator
from respan_exporter_haystack.logger import RespanLogger


class _DummyResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200
        self.text = "ok"

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_chat_generator_supports_latest_haystack_chat_message_api(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(name="RESPAN_API_KEY", value="test-api-key")

    def _mock_post(*args, **kwargs):
        return _DummyResponse(
            payload={
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello from respan"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            }
        )

    monkeypatch.setattr(target="respan_exporter_haystack.gateway.requests.post", name=_mock_post)

    generator = RespanChatGenerator(model="gpt-4o-mini")
    result = generator.run(messages=[ChatMessage.from_user(text="hello")])

    assert len(result["replies"]) == 1
    assert result["replies"][0].text == "hello from respan"
    assert result["meta"][0]["total_tokens"] == 5


def test_logger_resolves_tracing_endpoint_for_base_url_without_api_path():
    logger = RespanLogger(api_key="test-api-key", base_url="https://api.respan.ai")

    assert logger.traces_endpoint == "https://api.respan.ai/api/v1/traces/ingest"
