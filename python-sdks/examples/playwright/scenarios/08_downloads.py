"""Download a file - mirrors Playwright for Python's "Downloads" page.

    python scenarios/08_downloads.py
"""

from playwright.sync_api import sync_playwright

from _harness import ARTIFACTS, SITE, Expect, Scenario, launch_options, main


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_options())
        page = browser.new_page()
        page.goto(SITE)
        with page.expect_download() as download_info:
            page.get_by_text("Download the report").click()
        download = download_info.value
        download.save_as(ARTIFACTS / download.suggested_filename)
        browser.close()


SCENARIO = Scenario(
    name="downloads",
    title="Download a file",
    source="docs/downloads",
    look_for="Download.save_as with the target path",
    run=run,
    expect=Expect(calls_include=("Locator.click", "Download.save_as")),
)

if __name__ == "__main__":
    main(SCENARIO)
