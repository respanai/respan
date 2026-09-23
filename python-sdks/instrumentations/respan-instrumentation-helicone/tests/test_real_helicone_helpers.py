from __future__ import annotations

import asyncio

import pytest
from helicone_helpers import HeliconeManualLogger
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_MESSAGE,
    ERROR_TYPE,
)
from opentelemetry.semconv.attributes.http_attributes import (
    HTTP_RESPONSE_STATUS_CODE,
)
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_helicone import HeliconeInstrumentor, _emitter
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


class Response:
    status_code = 200
    text = "ok"


@pytest.mark.integration
def test_real_121_log_request_and_direct_sink(monkeypatch):
    posts = []
    spans = []
    monkeypatch.setattr(
        "helicone_helpers.manual_logger.requests.post",
        lambda url, **kwargs: posts.append((url, kwargs)) or Response(),
    )
    monkeypatch.setattr(
        _emitter, "inject_span", lambda span: spans.append(span) or True
    )
    instrumentor = HeliconeInstrumentor()
    instrumentor.activate()
    logger = HeliconeManualLogger(api_key="local-helicone-key")
    try:

        def operation(recorder):
            recorder.append_results(
                {
                    "model": "manual-model",
                    "choices": [{"message": {"role": "assistant", "content": "done"}}],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                }
            )
            return "application-result"

        assert (
            logger.log_request(
                {
                    "model": "manual-model",
                    "messages": [{"role": "user", "content": "run"}],
                },
                operation,
                provider="openai",
            )
            == "application-result"
        )
        logger.send_log(
            None,
            {"_type": "data", "name": "direct"},
            {"status": "success"},
            {"start_time": 1.0, "end_time": 2.0},
        )
    finally:
        instrumentor.deactivate()

    assert len(posts) == 2
    assert posts[0][0].endswith("/oai/v1/log")
    assert posts[1][0].endswith("/custom/v1/log")
    assert [span.attributes[RESPAN_LOG_TYPE] for span in spans] == ["chat", "task"]


@pytest.mark.integration
def test_real_121_builder_stream_and_error(monkeypatch):
    posts = []
    spans = []
    monkeypatch.setattr(
        "helicone_helpers.manual_logger.requests.post",
        lambda url, **kwargs: posts.append((url, kwargs)) or Response(),
    )
    monkeypatch.setattr(
        _emitter, "inject_span", lambda span: spans.append(span) or True
    )
    instrumentor = HeliconeInstrumentor()
    instrumentor.activate()
    logger = HeliconeManualLogger(api_key="local-helicone-key")

    async def exercise():
        stream = logger.log_builder(
            {
                "model": "stream-model",
                "messages": [{"role": "user", "content": "stream"}],
            }
        )
        stream.add_chunk(
            {
                "model": "stream-model",
                "choices": [{"delta": {"content": "one "}}],
            }
        )
        stream.add_chunk(
            {
                "model": "stream-model",
                "choices": [{"delta": {"content": "two"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2},
            }
        )
        await stream.send_log()

        failed = logger.new_builder(
            {
                "model": "error-model",
                "messages": [{"role": "user", "content": "fail"}],
            }
        )
        failed.set_error(RuntimeError("builder exploded"))
        await failed.send_log()

    try:
        asyncio.run(exercise())
    finally:
        instrumentor.deactivate()

    assert len(posts) == 2
    assert len(spans) == 2
    assert spans[0].attributes[SpanAttributes.GEN_AI_IS_STREAMING] is True
    assert (
        spans[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "one two"
    )
    assert spans[1].status.is_ok is False
    assert spans[1].attributes[ERROR_MESSAGE] == "builder exploded"
    assert spans[1].attributes[ERROR_TYPE] == "RuntimeError"
    assert spans[1].attributes[HTTP_RESPONSE_STATUS_CODE] == 500
    assert "status_code" not in spans[1].attributes


@pytest.mark.integration
def test_real_121_log_request_error_uses_canonical_otel_status(monkeypatch):
    posts = []
    spans = []
    monkeypatch.setattr(
        "helicone_helpers.manual_logger.requests.post",
        lambda url, **kwargs: posts.append((url, kwargs)) or Response(),
    )
    monkeypatch.setattr(
        _emitter, "inject_span", lambda span: spans.append(span) or True
    )
    instrumentor = HeliconeInstrumentor()
    instrumentor.activate()
    logger = HeliconeManualLogger(api_key="local-helicone-key")

    def operation(_recorder):
        raise ValueError("outer operation failed")

    try:
        with pytest.raises(ValueError, match="outer operation failed"):
            logger.log_request(
                {"model": "failed-model", "messages": []},
                operation,
                provider="openai",
            )
    finally:
        instrumentor.deactivate()

    assert posts == []
    assert len(spans) == 1
    assert spans[0].status.is_ok is False
    assert spans[0].attributes[ERROR_MESSAGE] == "outer operation failed"
    assert spans[0].attributes[ERROR_TYPE] == "ValueError"
    assert spans[0].attributes[HTTP_RESPONSE_STATUS_CODE] == 500
    assert "status_code" not in spans[0].attributes


@pytest.mark.integration
def test_real_121_nested_direct_success_and_outer_failure(monkeypatch):
    posts = []
    spans = []
    monkeypatch.setattr(
        "helicone_helpers.manual_logger.requests.post",
        lambda url, **kwargs: posts.append((url, kwargs)) or Response(),
    )
    monkeypatch.setattr(
        _emitter, "inject_span", lambda span: spans.append(span) or True
    )
    instrumentor = HeliconeInstrumentor()
    instrumentor.activate()
    logger = HeliconeManualLogger(api_key="local-helicone-key")

    def operation(_recorder):
        logger.send_log(
            "openai",
            {"model": "nested-model", "messages": []},
            {"model": "nested-response", "choices": []},
            {"start_time": 1.0, "end_time": 1.5},
        )
        raise RuntimeError("outer failed after nested success")

    try:
        with pytest.raises(RuntimeError, match="outer failed after nested success"):
            logger.log_request(
                {"model": "outer-model", "messages": []},
                operation,
                provider="openai",
            )
    finally:
        instrumentor.deactivate()

    assert len(posts) == 1
    assert len(spans) == 2
    assert spans[0].status.is_ok is True
    assert spans[0].attributes[SpanAttributes.LLM_REQUEST_MODEL] == "nested-model"
    assert spans[1].status.is_ok is False
    assert spans[1].attributes[SpanAttributes.LLM_REQUEST_MODEL] == "outer-model"
