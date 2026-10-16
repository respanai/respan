"""The same quickstart on the async API - mirrors the async example on
Playwright for Python's "Library" page.

    python scenarios/02_async_quickstart.py
"""

from playwright.async_api import async_playwright

from _harness import ARTIFACTS, Expect, Scenario, launch_options, main


async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(**launch_options())
        page = await browser.new_page()
        await page.goto("https://playwright.dev/python")
        print(await page.title())
        await page.screenshot(path=ARTIFACTS / "playwright-dev-async.png")
        await browser.close()


SCENARIO = Scenario(
    name="async_quickstart",
    title="Async API: goto, title, screenshot",
    source="docs/library (async)",
    look_for="the same tree as library_quickstart, from the async API",
    run=run,
    expect=Expect(
        calls_include=("BrowserType.launch", "Page.goto", "Page.title", "Page.screenshot"),
        screenshots=("Page.screenshot",),
    ),
)

if __name__ == "__main__":
    main(SCENARIO)
