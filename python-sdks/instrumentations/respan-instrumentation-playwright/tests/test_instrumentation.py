from __future__ import annotations

import json
from typing import Any

import pytest
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from playwright.async_api import async_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import expect, sync_playwright
from respan_instrumentation_playwright import PlaywrightInstrumentor
from respan_instrumentation_playwright._constants import ATTR_PAGE_TITLE, ATTR_PAGE_URL

from respan import workflow

from .conftest import chrome_path, launch_options

pytestmark = pytest.mark.skipif(chrome_path() is None, reason="needs a local Chrome/Chromium")


def output(span: Any) -> Any:
    return json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])


def named(spans: list[Any], name: str) -> list[Any]:
    return [s for s in spans if s.name == name]


def drive(site: str, steps) -> None:
    @workflow(name="drive")
    def run() -> None:
        with sync_playwright() as p:
            browser = p.chromium.launch(**launch_options())
            try:
                steps(browser.new_page())
            finally:
                browser.close()

    run()


def test_screenshot_output_is_a_compact_inline_image(pipeline, site):
    _, spans = pipeline()
    drive(site, lambda page: (page.goto(f"{site}/index.html"), page.screenshot()))

    (shot,) = named(spans, "Page.screenshot")
    url = output(shot)["screenshot"]["image_url"]["url"]
    assert url.startswith("data:image/")
    (goto,) = named(spans, "Page.goto")
    assert goto.attributes[ATTR_PAGE_TITLE] == "Sign in"
    assert goto.attributes[ATTR_PAGE_URL] == f"{site}/index.html"
    assert output(goto)["status"] == 200


def test_capture_screenshots_off_drops_images(pipeline, site):
    _, spans = pipeline(instrumentor=PlaywrightInstrumentor(capture_screenshots=False))
    drive(site, lambda page: (page.goto(f"{site}/index.html"), page.screenshot()))

    for span in spans:
        assert "image_url" not in json.dumps(dict(span.attributes), default=str), span.name
    (shot,) = named(spans, "Page.screenshot")
    assert output(shot)["type"] == "bytes"


@pytest.mark.parametrize(
    ("mode", "with_screenshot"),
    [("navigation", {"Page.goto"}), ("action", {"Page.goto", "Page.fill", "Locator.click"})],
)
def test_screenshot_on_adds_a_page_screenshot_after_matching_calls(
    pipeline, site, mode, with_screenshot
):
    _, spans = pipeline(instrumentor=PlaywrightInstrumentor(screenshot_on=mode))

    def steps(page: Any) -> None:
        page.goto(f"{site}/index.html")
        page.fill("#user", "someone")
        page.title()
        page.locator("#submit").click()

    drive(site, steps)
    has_image = {
        s.name
        for s in spans
        if "image_url" in s.attributes.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, "")
    }
    assert has_image == with_screenshot
    # The probes are internal: no extra user-visible calls were traced.
    assert not named(spans, "Page.screenshot")


def test_invalid_screenshot_on_is_rejected():
    with pytest.raises(ValueError):
        PlaywrightInstrumentor(screenshot_on="always")


def test_error_span_carries_the_playwright_error(pipeline, site):
    _, spans = pipeline()

    def steps(page: Any) -> None:
        page.goto(f"{site}/index.html")
        with pytest.raises(PlaywrightTimeoutError):
            page.locator("#missing").click(timeout=200)

    drive(site, steps)
    (click,) = named(spans, "Locator.click")
    assert click.status.status_code == StatusCode.ERROR
    assert click.status.description.startswith("TimeoutError: Locator.click: Timeout 200ms")


def test_assertions_and_event_handlers(pipeline, site):
    _, spans = pipeline()
    seen: list[str] = []

    def steps(page: Any) -> None:
        page.on("console", lambda message: (seen.append(message.text), message.page.title()))
        page.goto(f"{site}/index.html")
        expect(page.locator("#submit")).to_have_text("Sign in")
        page.evaluate("console.log('hello')")
        page.wait_for_timeout(100)

    drive(site, steps)
    assert "hello" in seen
    (root,) = [s for s in spans if s.parent is None]
    assertion = named(spans, "LocatorAssertions.to_have_text")
    assert assertion and assertion[0].parent.span_id == root.context.span_id
    # The console handler ran in Playwright's greenlet; its call still nests.
    titles = named(spans, "Page.title")
    assert titles and all(s.parent.span_id == root.context.span_id for s in titles)


async def test_async_evaluate_output_and_locator_input(pipeline, site):
    _, spans = pipeline()

    @workflow(name="async")
    async def run() -> None:
        async with async_playwright() as p:
            browser = await p.chromium.launch(**launch_options())
            page = await browser.new_page()
            await page.goto(f"{site}/index.html")
            assert await page.evaluate("() => 1 + 1") == 2
            await page.get_by_role("button", name="Sign in").is_visible()
            await browser.close()

    await run()
    (evaluate,) = named(spans, "Page.evaluate")
    assert output(evaluate) == 2
    (visible,) = named(spans, "Locator.is_visible")
    locator = json.loads(visible.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])["locator"]
    assert "Sign in" in locator
    # Local, no-I/O calls (page.locator, get_by_role) are not spans.
    assert not named(spans, "Page.get_by_role")


def test_keyboard_and_element_handle_text_into_password_fields_is_masked(pipeline, site):
    _, spans = pipeline()

    def steps(page: Any) -> None:
        page.goto(f"{site}/index.html")
        page.click("#password")
        page.keyboard.type("typed-secret-1")
        page.query_selector("#password").fill("handle-secret-2")
        page.click("#user")
        page.keyboard.type("visible-name")

    drive(site, steps)
    inputs = {
        s.name: s.attributes.get(SpanAttributes.TRACELOOP_ENTITY_INPUT, "")
        for s in spans
        if s.name in ("Keyboard.type", "ElementHandle.fill")
    }
    assert "typed-secret-1" not in json.dumps(inputs)
    assert "handle-secret-2" not in inputs["ElementHandle.fill"]
    typed = [s for s in spans if s.name == "Keyboard.type"]
    assert ["<masked>", "visible-name"] == [
        json.loads(s.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])["text"] for s in typed
    ]


def test_deactivate_restores_playwright():
    from playwright._impl._sync_base import SyncBase
    from playwright.async_api._generated import Page as AsyncPage
    from playwright.sync_api._generated import Page as SyncPage

    originals = (AsyncPage.goto, SyncPage.goto, SyncBase._wrap_handler)
    first, second = PlaywrightInstrumentor(), PlaywrightInstrumentor()
    first.activate()
    second.activate()
    assert SyncPage.goto is not originals[1]
    with pytest.raises(ValueError):
        PlaywrightInstrumentor(capture_content=False).activate()
    first.deactivate()
    assert SyncPage.goto is not originals[1]  # still used by `second`
    second.deactivate()
    assert (AsyncPage.goto, SyncPage.goto, SyncBase._wrap_handler) == originals
