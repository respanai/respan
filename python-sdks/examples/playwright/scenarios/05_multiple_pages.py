"""Several pages in one context, and a link that opens a new tab - mirrors
Playwright for Python's "Pages" page (multiple pages, handling new pages).

    python scenarios/05_multiple_pages.py
"""

from playwright.sync_api import sync_playwright

from _harness import SITE, Expect, Scenario, launch_options, main


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_options())
        context = browser.new_context()
        first = context.new_page()
        second = context.new_page()
        first.goto(SITE)
        second.goto(f"{SITE}/form.html")
        with context.expect_page() as new_page_info:
            first.get_by_text("Open a new tab").click()
        popup = new_page_info.value
        popup.wait_for_load_state()
        print([page.url for page in context.pages], popup.title())
        first.bring_to_front()
        browser.close()


SCENARIO = Scenario(
    name="multiple_pages",
    title="Two pages and a new tab",
    source="docs/pages",
    look_for="each call's playwright.page.url says which page it acted on; the popup's calls show popup.html",
    run=run,
    expect=Expect(
        calls_include=("Browser.new_context", "BrowserContext.new_page", "Page.goto", "Page.wait_for_load_state", "Page.bring_to_front"),
    ),
)

if __name__ == "__main__":
    main(SCENARIO)
