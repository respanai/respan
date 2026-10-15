from __future__ import annotations

import base64
import io

from PIL import Image

from respan_instrumentation_browser_use._constants import SCREENSHOT_MAX_WIDTH
from respan_instrumentation_browser_use._serialization import compact_image


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
    compact = compact_image(original)
    assert compact.startswith("data:image/jpeg;base64,")
    assert len(compact) < len(original) / 2
    image = _decode(compact)
    assert image.size == (SCREENSHOT_MAX_WIDTH, round(1117 * SCREENSHOT_MAX_WIDTH / 1728))


def test_small_image_is_not_upscaled():
    compact = compact_image(_png_data_url(400, 300))
    assert _decode(compact).size == (400, 300)


def test_image_that_would_grow_is_kept():
    tiny = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    assert compact_image(tiny) == tiny


def test_anything_unreadable_is_returned_unchanged():
    for value in (
        "data:image/png;base64,not-base64!!",
        "data:image/png;base64,aGVsbG8=",  # valid base64, not an image
        "data:text/plain,hello",
        "https://example.com/a.png",
    ):
        assert compact_image(value) == value
