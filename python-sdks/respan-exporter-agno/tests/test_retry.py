"""Tests for retry behaviour in RespanAgnoExporter._send."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

# Import the exporter module directly to avoid __init__.py pulling in
# opentelemetry.instrumentation which may not be installed in the test env.
import respan_exporter_agno.exporter as _exporter_mod

RespanAgnoExporter = _exporter_mod.RespanAgnoExporter


def _make_exporter(**kwargs):
    return RespanAgnoExporter(api_key="test-key", **kwargs)


def _mock_response(status_code, text="error body"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    return resp


class TestSendRetryOn5xx:
    """5xx responses should raise, triggering RetryHandler retries."""

    def test_retries_on_500_then_succeeds(self):
        exporter = _make_exporter()
        responses = [
            _mock_response(500, "internal server error"),
            _mock_response(200),
        ]

        with patch("respan_exporter_agno.exporter.requests.post", side_effect=responses):
            # Should not raise: first call 500 -> retry -> second call 200
            exporter._send(payloads=[{"test": True}])

    def test_all_retries_exhausted_logs_giving_up(self):
        exporter = _make_exporter()
        responses = [_mock_response(500, "fail")] * 5  # more than max_retries

        with patch("respan_exporter_agno.exporter.requests.post", side_effect=responses):
            # Should not propagate, the outer except catches it
            exporter._send(payloads=[{"test": True}])


class TestNoRetryOn4xx:
    """4xx responses should NOT trigger retries (they return immediately)."""

    def test_400_returns_without_retry(self):
        exporter = _make_exporter()
        mock_post = MagicMock(return_value=_mock_response(400, "bad request"))

        with patch("respan_exporter_agno.exporter.requests.post", mock_post):
            exporter._send(payloads=[{"test": True}])

        # Only one call — no retry
        assert mock_post.call_count == 1

    def test_404_returns_without_retry(self):
        exporter = _make_exporter()
        mock_post = MagicMock(return_value=_mock_response(404, "not found"))

        with patch("respan_exporter_agno.exporter.requests.post", mock_post):
            exporter._send(payloads=[{"test": True}])

        assert mock_post.call_count == 1

    def test_3xx_returns_without_retry(self):
        exporter = _make_exporter()
        mock_post = MagicMock(return_value=_mock_response(301, "moved"))

        with patch("respan_exporter_agno.exporter.requests.post", mock_post):
            exporter._send(payloads=[{"test": True}])

        assert mock_post.call_count == 1


class TestBackoffSleep:
    """RetryHandler should sleep between retries with backoff."""

    def test_sleep_called_on_retry(self):
        exporter = _make_exporter()
        responses = [
            _mock_response(500, "err"),
            _mock_response(200),
        ]

        with patch("respan_exporter_agno.exporter.requests.post", side_effect=responses), \
             patch("time.sleep") as mock_sleep:
            exporter._send(payloads=[{"test": True}])

        assert mock_sleep.call_count >= 1

    def test_no_sleep_on_success(self):
        exporter = _make_exporter()

        with patch("respan_exporter_agno.exporter.requests.post",
                    return_value=_mock_response(200)), \
             patch("time.sleep") as mock_sleep:
            exporter._send(payloads=[{"test": True}])

        assert mock_sleep.call_count == 0


class TestResponseTextTruncation:
    """Log messages should truncate response.text to 200 chars."""

    def test_4xx_response_text_truncated_in_log(self):
        exporter = _make_exporter()
        long_text = "x" * 500

        with patch("respan_exporter_agno.exporter.requests.post",
                    return_value=_mock_response(400, long_text)), \
             patch("respan_exporter_agno.exporter.logger") as mock_logger:
            exporter._send(payloads=[{"test": True}])

        # The warning call should have truncated text (200 chars)
        call_args = mock_logger.warning.call_args
        logged_text = call_args[0][2]  # third positional arg is response.text[:200]
        assert len(logged_text) == 200
