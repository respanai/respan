"""Two browser sessions at once, each in its own workflow - the async
concurrency pattern from Playwright for Python's "Browser contexts" page.

    python scenarios/13_parallel_contexts.py
"""

import asyncio

from playwright.async_api import async_playwright
from respan import workflow

from _harness import SITE, Expect, Scenario, init, launch_options, main


async def session(browser, path: str, name: str) -> None:
    @workflow(name=f"parallel_{name}")
    async def browse() -> None:
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(f"{SITE}{path}")
        print(name, await page.title())
        await context.close()

    with init().propagate_attributes(thread_identifier=f"parallel-{name}"):
        await browse()


async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(**launch_options())
        await asyncio.gather(
            session(browser, "/form.html", "a"), session(browser, "/products.html", "b")
        )
        await browser.close()


SCENARIO = Scenario(
    name="parallel_contexts",
    title="Parallel sessions, separate traces",
    source="docs/browser-contexts",
    look_for="2 traces, each with its own thread id and only its own page's calls",
    run=run,
    expect=Expect(traces=2, calls_include=("Browser.new_context", "Page.goto")),
    own_roots=True,
)

if __name__ == "__main__":
    main(SCENARIO)
