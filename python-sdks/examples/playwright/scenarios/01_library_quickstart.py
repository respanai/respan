"""Launch, open a page, read its title, take a screenshot - mirrors the sync
example on Playwright for Python's "Library" page.

    python scenarios/01_library_quickstart.py
"""

from playwright.sync_api import sync_playwright

from _harness import ARTIFACTS, Expect, Scenario, launch_options, main


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_options())
        page = browser.new_page()
        page.goto("https://playwright.dev/python")
        print(page.title())
        page.screenshot(path=ARTIFACTS / "playwright-dev.png")
        browser.close()


SCENARIO = Scenario(
    name="library_quickstart",
    title="Sync API: goto, title, screenshot",
    source="docs/library (sync)",
    look_for="workflow -> BrowserType.launch, Page.goto (URL + title), Page.title, Page.screenshot with the image",
    run=run,
    expect=Expect(
        calls_include=("BrowserType.launch", "Page.goto", "Page.title", "Page.screenshot"),
        screenshots=("Page.screenshot",),
    ),
)

if __name__ == "__main__":
    main(SCENARIO)
