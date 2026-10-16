"""Run JavaScript in the page - mirrors Playwright for Python's "Evaluating
JavaScript" page.

    python scenarios/07_evaluate.py
"""

from playwright.sync_api import sync_playwright

from _harness import SITE, Expect, Scenario, launch_options, main


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_options())
        page = browser.new_page()
        page.goto(SITE)
        print(page.evaluate("() => document.links.length"))
        print(page.evaluate("([a, b]) => a + b", [2, 3]))
        print(page.eval_on_selector_all("a", "links => links.map(l => l.textContent)"))
        handle = page.evaluate_handle("() => document.body")
        print(handle.evaluate("body => body.children.length"))
        browser.close()


SCENARIO = Scenario(
    name="evaluate",
    title="page.evaluate and handles",
    source="docs/evaluating",
    look_for="Page.evaluate spans carry the expression and its JSON result",
    run=run,
    expect=Expect(calls_include=("Page.evaluate", "Page.eval_on_selector_all", "Page.evaluate_handle", "JSHandle.evaluate")),
)

if __name__ == "__main__":
    main(SCENARIO)
