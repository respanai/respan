"""Kernel's `sample-app` template (Python): get the title of a page with Playwright
over CDP on a Kernel browser.

    python scenarios/01_sample_app.py        # KERNEL_FAKE=1 for the local fake Kernel
"""

from typing import TypedDict
from urllib.parse import urlparse

import kernel
from _harness import TARGET_URL, Expect, Scenario, main
from kernel import Kernel
from playwright.async_api import async_playwright

client = Kernel()

app = kernel.App("python-basic")


class PageTitleInput(TypedDict):
    url: str


class PageTitleOutput(TypedDict):
    title: str


@app.action("get-page-title")
async def get_page_title(ctx: kernel.KernelContext, input_data: PageTitleInput) -> PageTitleOutput:
    url = input_data.get("url")
    if not url or not isinstance(url, str):
        raise ValueError("URL is required and must be a string")
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    urlparse(url)

    kernel_browser = client.browsers.create(invocation_id=ctx.invocation_id)
    print("Kernel browser live view url: ", kernel_browser.browser_live_view_url)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.connect_over_cdp(kernel_browser.cdp_ws_url)
        context = await browser.new_context()
        page = await context.new_page()
        try:
            await page.goto(url)
            title = await page.title()
            return {"title": title}
        finally:
            await browser.close()
            client.browsers.delete_by_id(kernel_browser.session_id)


SCENARIO = Scenario(
    name="sample_app",
    title="sample-app: page title via Playwright over CDP",
    source="kernel/cli pkg/templates/python/sample-app",
    look_for="agent kernel.action.get-page-title -> browsers.create (session id, live view) / browsers.delete_by_id; thread = the Kernel session",
    app=app,
    action="get-page-title",
    payload={"url": TARGET_URL},
    expect=Expect(calls_include=("browsers.create", "browsers.delete_by_id")),
)

if __name__ == "__main__":
    main(SCENARIO)
