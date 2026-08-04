"""Tests for the ``default_params`` merge on ``RespanSpanExporter.export``.

The exporter accepts a generic ``default_params`` mapping that is merged into
every exported span payload. Span-provided keys must win so real converted
data is never clobbered; the defaults only fill fields a span did not set.
"""

from unittest.mock import MagicMock

from respan_exporter_openai_agents import RespanSpanExporter


def _exporter_with_spans(spans, default_params):
    """Build an exporter whose conversion yields ``spans`` and capture the POST."""
    exporter = RespanSpanExporter(api_key="test-key", default_params=default_params)
    # Bypass real Trace/Span conversion — return the prepared dicts in order.
    iterator = iter(spans)
    exporter._respan_export = lambda item: next(iterator)  # type: ignore[method-assign]
    # Capture the HTTP call instead of hitting the network.
    mock_client = MagicMock()
    mock_client.post.return_value = MagicMock(status_code=200)
    exporter._client = mock_client
    # ``items`` length only needs to match ``spans`` — contents are ignored.
    exporter.export(items=[object() for _ in spans])
    return mock_client.post.call_args.kwargs["json"]["data"]


def test_default_params_fill_missing_keys():
    data = _exporter_with_spans(
        spans=[{"span_unique_id": "a"}],
        default_params={"is_event_dispatch_enabled": False},
    )
    assert data[0]["is_event_dispatch_enabled"] is False
    assert data[0]["span_unique_id"] == "a"


def test_span_keys_win_over_defaults():
    data = _exporter_with_spans(
        spans=[{"span_unique_id": "a", "environment": "prod"}],
        default_params={"environment": "default"},
    )
    assert data[0]["environment"] == "prod"


def test_defaults_applied_to_every_span():
    data = _exporter_with_spans(
        spans=[{"span_unique_id": "a"}, {"span_unique_id": "b"}],
        default_params={"is_event_dispatch_enabled": False},
    )
    assert all(row["is_event_dispatch_enabled"] is False for row in data)


def test_no_default_params_leaves_payload_untouched():
    data = _exporter_with_spans(
        spans=[{"span_unique_id": "a"}],
        default_params=None,
    )
    assert data[0] == {"span_unique_id": "a"}
