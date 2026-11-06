"""respan_params time_to_first_token / ttft land on the span as a bare attribute.

The backend reads a bare ``time_to_first_token`` span attribute (seconds) on any
span kind. Before this mapping existed the value was validated and then
silently dropped, because no span attribute was set for it.
"""

import math

import pytest

from respan_sdk.constants.span_attributes import (
    RESPAN_CUSTOMER_PARAMS_ID,
    RespanSpanAttributes,
)
from respan_tracing import RespanTelemetry, get_client
from respan_tracing.constants.tracing import TIME_TO_FIRST_TOKEN_ATTR
from respan_tracing.contexts.span import respan_span_attributes

VALID_SECONDS = 0.42
INVALID_VALUES = [
    True,
    False,
    -0.1,
    math.inf,
    -math.inf,
    math.nan,
    "abc",
    "0.42",
    None,
    [0.42],
]


@pytest.fixture(scope="module")
def client():
    RespanTelemetry(app_name="test-respan-params-ttft")
    return get_client()


def _attrs_after_update(client, respan_params):
    with client.start_span("ttft_span") as span:
        client.update_current_span(respan_params=respan_params)
        return dict(span.attributes)


def _attrs_after_context(client, respan_params):
    with client.start_span("ttft_span") as span:
        with respan_span_attributes(respan_params):
            pass
        return dict(span.attributes)


SETTERS = [_attrs_after_update, _attrs_after_context]


def test_local_key_mirrors_respan_sdk_constant():
    assert TIME_TO_FIRST_TOKEN_ATTR == RespanSpanAttributes.TIME_TO_FIRST_TOKEN.value


@pytest.mark.parametrize("setter", SETTERS)
@pytest.mark.parametrize("param_key", ["time_to_first_token", "ttft"])
def test_ttft_param_sets_bare_span_attribute(client, setter, param_key):
    attrs = setter(client, {param_key: VALID_SECONDS})
    assert attrs[TIME_TO_FIRST_TOKEN_ATTR] == VALID_SECONDS
    assert isinstance(attrs[TIME_TO_FIRST_TOKEN_ATTR], float)


@pytest.mark.parametrize("setter", SETTERS)
def test_integer_ttft_is_exported_as_float_seconds(client, setter):
    attrs = setter(client, {"ttft": 2})
    assert attrs[TIME_TO_FIRST_TOKEN_ATTR] == 2.0
    assert isinstance(attrs[TIME_TO_FIRST_TOKEN_ATTR], float)


@pytest.mark.parametrize("setter", SETTERS)
def test_zero_ttft_is_kept(client, setter):
    attrs = setter(client, {"time_to_first_token": 0})
    assert attrs[TIME_TO_FIRST_TOKEN_ATTR] == 0.0


@pytest.mark.parametrize("setter", SETTERS)
@pytest.mark.parametrize("value", INVALID_VALUES)
def test_invalid_ttft_is_not_set_and_other_params_survive(client, setter, value):
    attrs = setter(
        client,
        {"time_to_first_token": value, "customer_identifier": "user-123"},
    )
    assert TIME_TO_FIRST_TOKEN_ATTR not in attrs
    assert attrs[RESPAN_CUSTOMER_PARAMS_ID] == "user-123"
