from types import SimpleNamespace

from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_instrumentation_openlit._processor import translate_openlit_span


class FakeSpan:
    def __init__(self, attributes, status) -> None:
        self._attributes = attributes
        self.name = "openai.chat"
        self.instrumentation_scope = SimpleNamespace(name="openlit")
        self.status = status
        self.events = ()


def test_otel_error_sets_backend_visible_status_and_message() -> None:
    span = FakeSpan(
        {"gen_ai.operation.name": "chat"},
        Status(StatusCode.ERROR, "openlit provider failed"),
    )
    translate_openlit_span(span, capture_content=False)
    assert "status_code" not in span._attributes
    assert "error.message" not in span._attributes
    assert span._status.status_code is StatusCode.ERROR
    assert span._status.description is None
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span._attributes


def test_upstream_http_status_is_preserved() -> None:
    span = FakeSpan(
        {
            "gen_ai.operation.name": "chat",
            "http.response.status_code": 429,
        },
        Status(StatusCode.ERROR, "rate limited"),
    )
    translate_openlit_span(span, capture_content=False)
    assert "status_code" not in span._attributes
    assert span._attributes["http.response.status_code"] == 429
    assert "error.message" not in span._attributes
    assert span._status.status_code is StatusCode.ERROR
    assert span._status.description is None
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span._attributes


def test_success_sets_backend_visible_200() -> None:
    span = FakeSpan(
        {"gen_ai.operation.name": "chat"},
        Status(StatusCode.OK),
    )
    translate_openlit_span(span, capture_content=False)
    assert "status_code" not in span._attributes
    assert "error.message" not in span._attributes
