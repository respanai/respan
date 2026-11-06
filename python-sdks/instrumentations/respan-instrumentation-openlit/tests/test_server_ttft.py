from types import SimpleNamespace

import pytest

from respan_instrumentation_openlit._processor import translate_openlit_span

SERVER_TTFT = "gen_ai.server.time_to_first_token"
TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"


class FakeSpan:
    def __init__(self, attributes: dict[str, object]) -> None:
        self._attributes = attributes
        self.name = "openai.chat"
        self.instrumentation_scope = SimpleNamespace(name="openlit")


def _openlit_chat_attributes(**extra: object) -> dict[str, object]:
    return {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-4.1-mini",
        "gen_ai.request.stream": True,
        "gen_ai.sdk.version": "1.44.0",
        **extra,
    }


def _translate(attributes: dict[str, object]) -> dict[str, object]:
    span = FakeSpan(attributes)
    translate_openlit_span(span, capture_content=True)
    return span._attributes


def test_server_ttft_is_kept_as_time_to_first_chunk_in_seconds() -> None:
    attrs = _translate(_openlit_chat_attributes(**{SERVER_TTFT: 0.31}))

    assert attrs[TIME_TO_FIRST_CHUNK] == 0.31
    # The OpenLIT-only source attribute is still stripped.
    assert SERVER_TTFT not in attrs


def test_integer_server_ttft_is_kept() -> None:
    attrs = _translate(_openlit_chat_attributes(**{SERVER_TTFT: 2}))

    assert attrs[TIME_TO_FIRST_CHUNK] == 2


def test_sender_time_to_first_chunk_wins_over_server_ttft() -> None:
    attrs = _translate(
        _openlit_chat_attributes(**{SERVER_TTFT: 0.31, TIME_TO_FIRST_CHUNK: 0.2})
    )

    assert attrs[TIME_TO_FIRST_CHUNK] == 0.2
    assert SERVER_TTFT not in attrs


@pytest.mark.parametrize(
    "value",
    [True, False, 0, 0.0, -0.31, "0.31", "garbage", float("nan"), float("inf")],
)
def test_invalid_server_ttft_is_not_kept(value: object) -> None:
    attrs = _translate(_openlit_chat_attributes(**{SERVER_TTFT: value}))

    assert TIME_TO_FIRST_CHUNK not in attrs
    assert SERVER_TTFT not in attrs


def test_span_without_server_ttft_gets_no_time_to_first_chunk() -> None:
    attrs = _translate(_openlit_chat_attributes())

    assert TIME_TO_FIRST_CHUNK not in attrs
