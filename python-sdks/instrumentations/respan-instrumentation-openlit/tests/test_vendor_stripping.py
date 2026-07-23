from types import SimpleNamespace

from respan_instrumentation_openlit._constants import (
    STANDARD_DB_ATTRIBUTES,
    STANDARD_GEN_AI_ATTRIBUTES,
)
from respan_instrumentation_openlit._processor import translate_openlit_span


class FakeSpan:
    def __init__(self, attributes: dict[str, object]) -> None:
        self._attributes = attributes
        self.name = "openai.chat"
        self.instrumentation_scope = SimpleNamespace(name="openlit")


def test_openlit_vendor_attributes_are_stripped_after_canonical_mapping() -> None:
    span = FakeSpan(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": "gpt-4.1-mini",
            "gen_ai.embeddings.dimension.count": 513,
            "gen_ai.sdk.version": "1.44.0",
            "gen_ai.environment": "default",
            "gen_ai.application_name": "default",
            "gen_ai.usage.cost": 0.01,
            "gen_ai.client.token.usage": 12,
            "gen_ai.client.operation.duration": 0.4,
            "gen_ai.server.request.duration": 0.3,
            "gen_ai.framework.tags": "vendor-only",
            "gen_ai.framework.error.message": "vendor-only",
            "gen_ai.serialized.signature": "vendor-only",
            "gen_ai.rag.strategy": "vendor-only",
            "gen_ai.agent.operation.duration": 0.2,
            "db.sdk.version": "vendor-only",
            "db.operation.cost": 0.5,
            "openlit.agent.version_hash": "vendor-only",
            "server.address": "api.openai.com",
            "http.response.status_code": 202,
            "openai.response.system_fingerprint": "fp_test",
        }
    )

    translate_openlit_span(span, capture_content=True)
    attrs = span._attributes
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4.1-mini"
    assert attrs["gen_ai.embeddings.dimension.count"] == 513
    assert attrs["server.address"] == "api.openai.com"
    assert attrs["http.response.status_code"] == 202
    assert attrs["openai.response.system_fingerprint"] == "fp_test"
    assert attrs["status_code"] == 202
    assert not any(key.startswith("openlit.") for key in attrs)
    assert all(
        not key.startswith("gen_ai.") or key in STANDARD_GEN_AI_ATTRIBUTES
        for key in attrs
    )
    assert all(
        not key.startswith("db.") or key in STANDARD_DB_ATTRIBUTES for key in attrs
    )
