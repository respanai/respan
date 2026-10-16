"""Record a Playwright trace.zip alongside the Respan trace - mirrors Playwright
for Python's "Trace viewer" page.

    python scenarios/10_tracing_zip.py
    playwright show-trace artifacts/trace.zip
"""

from playwright.sync_api import sync_playwright

from _harness import ARTIFACTS, SITE, Expect, Scenario, launch_options, main


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_options())
        context = browser.new_context()
        context.tracing.start(screenshots=True, snapshots=True, sources=False)
        page = context.new_page()
        page.goto(f"{SITE}/form.html")
        page.get_by_label("Email").fill("trace@example.com")
        context.tracing.stop(path=ARTIFACTS / "trace.zip")
        browser.close()


SCENARIO = Scenario(
    name="tracing_zip",
    title="Playwright tracing alongside",
    source="docs/trace-viewer",
    look_for="Tracing.start / Tracing.stop spans; the trace.zip shows the same calls (Respan's internal probes do not appear in it)",
    run=run,
    expect=Expect(calls_include=("Tracing.start", "Tracing.stop", "Locator.fill")),
)

if __name__ == "__main__":
    main(SCENARIO)
