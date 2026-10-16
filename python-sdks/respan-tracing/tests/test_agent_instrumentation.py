from __future__ import annotations

import base64
import io

import pytest

Image = pytest.importorskip("PIL.Image")

from respan_tracing.utils.agent_instrumentation import (  # noqa: E402 - after importorskip
    SCREENSHOT_MAX_WIDTH,
    AgentScope,
    MethodPatcher,
    compact_image_data_url,
    secret_replacements,
    set_error,
)


def _png_data_url(width: int, height: int) -> str:
    # Noise, like a real page screenshot: PNG compresses it poorly, JPEG well.
    image = Image.effect_noise((width, height), 64).convert("RGB")
    out = io.BytesIO()
    image.save(out, "PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def _decode(data_url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data_url.partition(",")[2])))


def test_screenshot_becomes_a_smaller_capped_jpeg():
    original = _png_data_url(1728, 1117)
    compact = compact_image_data_url(original)
    assert compact.startswith("data:image/jpeg;base64,")
    assert len(compact) < len(original) / 2
    image = _decode(compact)
    assert image.size == (SCREENSHOT_MAX_WIDTH, round(1117 * SCREENSHOT_MAX_WIDTH / 1728))


def test_small_image_is_not_upscaled():
    compact = compact_image_data_url(_png_data_url(400, 300))
    assert _decode(compact).size == (400, 300)


def test_image_that_would_grow_is_kept():
    tiny = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    assert compact_image_data_url(tiny) == tiny


def test_anything_unreadable_is_returned_unchanged():
    for value in (
        "data:image/png;base64,not-base64!!",
        "data:image/png;base64,aGVsbG8=",  # valid base64, not an image
        "data:text/plain,hello",
        "https://example.com/a.png",
    ):
        assert compact_image_data_url(value) == value


def test_secret_replacements_cover_nested_and_url_encoded_forms():
    pairs = dict(secret_replacements({"site.com": {"pw": "a b&c"}, "token": "t0k"}))
    assert pairs["a b&c"] == "<secret>pw</secret>"
    assert pairs["a%20b%26c"] == "<secret>pw</secret>"  # quote
    assert pairs["a+b%26c"] == "<secret>pw</secret>"  # quote_plus
    assert pairs["t0k"] == "<secret>token</secret>"


def test_scope_masks_and_groups_only_inside_a_run():
    scope = AgentScope("test")
    assert scope.mask("pw=hunter2") == "pw=hunter2"
    with scope.run("session-1", {"pw": "hunter2"}):
        assert scope.mask("pw=hunter2") == "pw=<secret>pw</secret>"

        class Span:
            attributes = {"respan.threads.thread_identifier": "customer-thread"}

            def set_attribute(self, key, value):
                self.attributes[key] = value

        span = Span()
        scope.apply_grouping(span)
        # The customer's own thread wins; the rest default to the session.
        assert span.attributes["respan.threads.thread_identifier"] == "customer-thread"
        assert span.attributes["respan.sessions.session_identifier"] == "session-1"
    assert scope.mask("pw=hunter2") == "pw=hunter2"


def test_set_error_message_is_masked_and_optional():
    class Span:
        status = None

        def set_status(self, status):
            self.status = status

    span = Span()
    set_error(span, "step failed", "bad pw hunter2", with_message=True, mask=lambda t: t.replace("hunter2", "***"))
    assert span.status.description == "step failed: bad pw ***"
    set_error(span, "step failed", "bad pw hunter2", with_message=False)
    assert span.status.description == "step failed"


def test_method_patcher_patches_once_and_restores():
    class Target:
        def run(self):
            return "original"

    patcher = MethodPatcher()
    patcher.patch(Target, "run", lambda original: lambda self: "patched:" + original(self))
    patcher.patch(Target, "run", lambda original: lambda self: "twice")  # ignored
    assert Target().run() == "patched:original"
    patcher.restore()
    assert Target().run() == "original"


def test_jsonable_handles_framework_values_without_dumping_unknown_objects():
    import dataclasses
    import datetime
    import enum

    from respan_tracing.utils.agent_instrumentation import jsonable

    class Color(enum.Enum):
        RED = "red"

    @dataclasses.dataclass
    class Step:
        n: int
        note: object = None

    class Session:  # e.g. a browser session holding credentials
        token = "do-not-export"

    assert jsonable([Color.RED, Step(1), datetime.date(2026, 1, 2), b"ab", float("inf")]) == [
        "red",
        {"n": 1},
        "2026-01-02",
        {"type": "bytes", "length": 2},
        "inf",
    ]
    assert jsonable(Session()) == {"type": "Session"}


def test_suppress_provider_llm_spans_sets_the_otel_suppression_key():
    from opentelemetry import context
    from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY

    from respan_tracing.utils.agent_instrumentation import suppress_provider_llm_spans

    with suppress_provider_llm_spans():
        assert context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY) is True
    assert context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY) is None
