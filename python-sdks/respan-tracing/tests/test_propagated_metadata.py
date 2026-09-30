import json

from respan_sdk.constants.span_attributes import RESPAN_METADATA
from respan_tracing.processors.base import RespanSpanProcessor
from respan_tracing.utils.span_factory import (
    build_readable_span,
    propagate_attributes,
    read_propagated_attributes,
)


class _Span:
    def __init__(self, metadata: dict[str, object]) -> None:
        self.name = "vendor.call"
        self.attributes = {RESPAN_METADATA: json.dumps(metadata)}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class _Processor:
    def on_start(self, _span, _parent_context=None) -> None:
        return None


def _assert_single_metadata_attribute(attributes: dict[str, object]) -> None:
    assert RESPAN_METADATA in attributes
    assert not any(key.startswith(f"{RESPAN_METADATA}.") for key in attributes)


def test_propagated_metadata_is_one_canonical_json_attribute() -> None:
    with (
        propagate_attributes(metadata={"run_id": "outer", "outer": True}),
        propagate_attributes(metadata={"run_id": "inner", "nested": {"ok": True}}),
    ):
        attributes = read_propagated_attributes()

    _assert_single_metadata_attribute(attributes)
    assert json.loads(attributes[RESPAN_METADATA]) == {
        "nested": {"ok": True},
        "outer": True,
        "run_id": "inner",
    }


def test_synthetic_span_merges_propagated_and_instrumentation_metadata() -> None:
    with propagate_attributes(metadata={"run_id": "run-1", "shared": "propagated"}):
        span = build_readable_span(
            "vendor.call",
            attributes={
                RESPAN_METADATA: json.dumps(
                    {"provider_request_id": "request-1", "shared": "instrumentation"}
                )
            },
        )

    _assert_single_metadata_attribute(span.attributes)
    assert json.loads(span.attributes[RESPAN_METADATA]) == {
        "provider_request_id": "request-1",
        "run_id": "run-1",
        "shared": "instrumentation",
    }


def test_live_span_processor_merges_propagated_metadata_without_aliases() -> None:
    span = _Span({"provider_request_id": "request-1", "shared": "instrumentation"})
    processor = RespanSpanProcessor(_Processor())

    with propagate_attributes(metadata={"run_id": "run-1", "shared": "propagated"}):
        processor.on_start(span)

    _assert_single_metadata_attribute(span.attributes)
    assert json.loads(span.attributes[RESPAN_METADATA]) == {
        "provider_request_id": "request-1",
        "run_id": "run-1",
        "shared": "instrumentation",
    }
