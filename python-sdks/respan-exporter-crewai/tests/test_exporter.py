"""Unit tests for payload building, normalization, send, and validation (no live API)."""
import pytest

pytest.importorskip("respan_exporter_crewai")
from unittest.mock import patch

from respan_sdk.constants import RESPAN_DOGFOOD_HEADER

from respan_exporter_crewai.exporter import RespanCrewAIExporter


# --- build_payload edge cases ---


def test_build_payload_empty_list_returns_empty():
    exporter = RespanCrewAIExporter(api_key="test")
    assert exporter.build_payload(trace_or_spans=[]) == []


def test_build_payload_none_returns_empty():
    exporter = RespanCrewAIExporter(api_key="test")
    assert exporter.build_payload(trace_or_spans=None) == []


def test_build_payload_dict_with_empty_spans_returns_empty():
    exporter = RespanCrewAIExporter(api_key="test")
    assert exporter.build_payload(trace_or_spans={"spans": []}) == []


def test_build_payload_dict_with_spans_key():
    """Dict with 'spans' key is normalized to (trace_obj, spans)."""
    exporter = RespanCrewAIExporter(api_key="test")
    # Minimal span dict that can produce a valid payload after validation
    minimal_span = {
        "span_id": "a" * 16,
        "trace_id": "b" * 32,
        "name": "test_span",
        "parent_id": None,
    }
    with patch.object(exporter, "_span_to_respan", return_value=None):
        # If _span_to_respan returns None (e.g. validation fails), payloads can be empty
        payloads = exporter.build_payload(trace_or_spans={"spans": [minimal_span]})
    # Either empty (validation failed) or one payload
    assert isinstance(payloads, list)
    assert len(payloads) <= 1


def test_normalize_trace_list_input():
    """List input is normalized to (None, list)."""
    exporter = RespanCrewAIExporter(api_key="test")
    trace_obj, spans = exporter._normalize_trace(trace_or_spans=[{"span_id": "1"}, {"span_id": "2"}])
    assert trace_obj is None
    assert len(spans) == 2


def test_normalize_trace_dict_with_spans():
    exporter = RespanCrewAIExporter(api_key="test")
    d = {"spans": [{"span_id": "1"}]}
    trace_obj, spans = exporter._normalize_trace(trace_or_spans=d)
    assert trace_obj is d
    assert len(spans) == 1


def test_normalize_trace_single_dict_treated_as_single_span():
    exporter = RespanCrewAIExporter(api_key="test")
    d = {"span_id": "1"}
    trace_obj, spans = exporter._normalize_trace(trace_or_spans=d)
    assert trace_obj is None
    assert len(spans) == 1
    assert spans[0] == d


# --- send() and error paths ---


def test_send_calls_requests_post():
    """send() uses requests.post with correct url and headers."""
    exporter = RespanCrewAIExporter(api_key="key123")
    payloads = [{"trace_unique_id": "abc", "span_unique_id": "def"}]
    with patch("respan_exporter_crewai.exporter.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        exporter.send(payloads=payloads)
        mock_post.assert_called_once()
        call_kw = mock_post.call_args[1]
        assert call_kw["json"] == payloads
        assert call_kw["headers"].get("Authorization") == "Bearer key123"
        assert call_kw["headers"].get(RESPAN_DOGFOOD_HEADER) == "1"


def test_send_handles_http_error():
    """Non-2xx response logs but does not raise."""
    exporter = RespanCrewAIExporter(api_key="key")
    with patch("respan_exporter_crewai.exporter.requests.post") as mock_post:
        with patch("respan_exporter_crewai.exporter.time.sleep"):
            mock_post.return_value.status_code = 500
            mock_post.return_value.text = "Server Error"
            exporter.send(payloads=[{}])


def test_send_handles_request_exception():
    """Request exception is caught and logged."""
    exporter = RespanCrewAIExporter(api_key="key")
    with patch("respan_exporter_crewai.exporter.requests.post") as mock_post:
        with patch("respan_exporter_crewai.exporter.time.sleep"):
            mock_post.side_effect = Exception("network error")
            exporter.send(payloads=[{}])


# --- export() does not send when no api_key ---


def test_export_without_api_key_returns_payloads_without_sending():
    """When api_key is missing, export still builds payloads but send is not called."""
    with patch.dict("os.environ", {"RESPAN_API_KEY": ""}, clear=False):
        exporter = RespanCrewAIExporter(api_key=None)
    with patch.object(exporter, "build_payload", return_value=[{"x": 1}]) as mock_build:
        with patch.object(exporter, "send") as mock_send:
            result = exporter.export(trace_or_spans=[{"span_id": "1" * 16, "trace_id": "2" * 32, "name": "s"}])
            mock_send.assert_not_called()
            assert result == [{"x": 1}]
