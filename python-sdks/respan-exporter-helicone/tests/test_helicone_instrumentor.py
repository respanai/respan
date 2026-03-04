import json
import threading
from unittest.mock import MagicMock, patch

import pytest
from helicone_helpers import HeliconeManualLogger

from respan_sdk.constants.llm_logging import LOG_TYPE_GENERATION, LogMethodChoices
from respan_exporter_helicone.instrumentor import HeliconeInstrumentor


@pytest.fixture
def mock_requests_post():
    with patch("respan_exporter_helicone.instrumentor.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def instrumentor():
    instr = HeliconeInstrumentor()
    # use a dummy api key to ensure it enables
    instr.instrument(api_key="test-api-key", endpoint="https://test.endpoint/api")
    yield instr
    instr.uninstrument()


def test_helicone_instrumentor_patching(instrumentor, mock_requests_post):
    """Test that the instrumentor successfully intercepts send_log."""
    
    # Initialize Helicone logger
    helicone_logger = HeliconeManualLogger(api_key="helicone-test-key")
    
    # Setup test data
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    response_data = {
        "choices": [{"message": {"role": "assistant", "content": "Hi there!"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    options_data = {
        "start_time": 1000.0,
        "end_time": 1001.5,
        "additional_headers": {
            "Helicone-User-Id": "user-123",
            "Helicone-Session-Id": "session-456"
        }
    }

    # Since threading is used in _send_to_respan, we need to mock or wait for it.
    # In this test, we can patch threading.Thread.start to run synchronously for testing.
    def sync_start(self):
        self._target(*self._args, **self._kwargs)
    
    with patch.object(threading.Thread, "start", sync_start):
        helicone_logger.send_log(
            provider="openai",
            request=request_data,
            response=response_data,
            options=options_data
        )

    # Verify requests.post was called to send data to Respan
    assert mock_requests_post.called
    
    call_args = mock_requests_post.call_args
    assert call_args.kwargs["url"] == "https://test.endpoint/api"
    
    # Verify payload
    payload = call_args.kwargs["json"][0]
    assert payload["log_method"] == LogMethodChoices.TRACING_INTEGRATION.value
    assert payload["log_type"] == LOG_TYPE_GENERATION
    assert payload["provider"] == "openai"
    assert payload["provider_id"] == "openai"
    assert payload["model"] == "gpt-4"
    assert payload["prompt_tokens"] == 10
    assert payload["completion_tokens"] == 5
    assert payload["total_request_tokens"] == 15
    assert payload["customer_identifier"] == "user-123"
    assert payload["session_identifier"] == "session-456"
    assert "start_time" in payload
    assert "timestamp" in payload
    assert payload["latency"] == 1.5

    # Check input and output JSON encoding
    input_decoded = json.loads(payload["input"])
    assert input_decoded == request_data["messages"]
    
    output_decoded = json.loads(payload["output"])
    assert output_decoded == response_data["choices"]


def test_helicone_metadata_header_mapping_is_case_insensitive(instrumentor, mock_requests_post):
    instrumentor._send_to_respan(
        provider="openai",
        request={"model": "gpt-4", "prompt": "hello"},
        response="world",
        options={
            "start_time": 1000.0,
            "end_time": 1001.0,
            "additional_headers": {
                "helicone-user-id": "user-lower",
                "helicone-session-id": "session-lower",
            },
        },
    )

    payload = mock_requests_post.call_args.kwargs["json"][0]
    assert payload["provider"] == "openai"
    assert payload["provider_id"] == "openai"
    assert payload["customer_identifier"] == "user-lower"
    assert payload["session_identifier"] == "session-lower"

