"""Web-first assertions, one of which fails - mirrors Playwright for Python's
"Assertions" page.

    python scenarios/04_assertions.py
"""

from playwright.sync_api import expect, sync_playwright

from _harness import SITE, Expect, Scenario, launch_options, main


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_options())
        page = browser.new_page()
        page.goto(SITE)
        expect(page).to_have_title("Respan Playwright demo")
        expect(page.get_by_role("heading")).to_be_visible()
        try:
            expect(page.get_by_role("heading")).to_have_text("Something else", timeout=1000)
        except AssertionError as error:
            print("assertion failed as expected:", str(error).splitlines()[0])
        browser.close()


SCENARIO = Scenario(
    name="assertions",
    title="expect(...) assertions, one failing",
    source="docs/test-assertions",
    look_for="LocatorAssertions.to_have_text is an error span with Playwright's assertion message",
    run=run,
    expect=Expect(
        calls_include=("PageAssertions.to_have_title", "LocatorAssertions.to_be_visible"),
        error_calls=("LocatorAssertions.to_have_text",),
    ),
)

if __name__ == "__main__":
    main(SCENARIO)
