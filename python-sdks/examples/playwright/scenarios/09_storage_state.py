"""Sign in once, save the storage state, reuse it in a new context - mirrors
Playwright for Python's "Authentication" page.

    python scenarios/09_storage_state.py
"""

from playwright.sync_api import sync_playwright

from _harness import ARTIFACTS, SITE, Expect, Scenario, launch_options, main


def run():
    state = ARTIFACTS / "state.json"
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_options())
        context = browser.new_context()
        page = context.new_page()
        page.goto(f"{SITE}/welcome.html")  # "signs in": stores a session in localStorage
        context.storage_state(path=state)
        context.close()

        reused = browser.new_context(storage_state=state)
        page = reused.new_page()
        page.goto(SITE)
        print("session:", page.evaluate("() => localStorage.getItem('session')"))
        browser.close()


SCENARIO = Scenario(
    name="storage_state",
    title="Save and reuse storage state",
    source="docs/auth",
    look_for="BrowserContext.storage_state, then Browser.new_context with storage_state; the evaluate returns the reused session",
    run=run,
    expect=Expect(calls_include=("BrowserContext.storage_state", "Browser.new_context", "BrowserContext.close")),
)

if __name__ == "__main__":
    main(SCENARIO)
