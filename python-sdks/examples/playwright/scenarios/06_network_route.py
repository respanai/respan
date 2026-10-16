"""Mock an API response and block images with page.route - mirrors Playwright
for Python's "Network" / "Mock APIs" pages. Uses the async API.

    python scenarios/06_network_route.py
"""

import json

from playwright.async_api import async_playwright

from _harness import SITE, Expect, Scenario, launch_options, main


async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(**launch_options())
        page = await browser.new_page()

        async def mock_products(route):
            await route.fulfill(json=[{"name": "Mocked widget", "price": 1}])

        await page.route("**/api/products.json", mock_products)
        await page.route("**/*.{png,jpg,jpeg}", lambda route: route.abort())
        await page.goto(f"{SITE}/products.html")
        await page.locator(".product").first.wait_for()
        print(json.dumps(await page.locator(".product").all_inner_texts()))
        await browser.close()


SCENARIO = Scenario(
    name="network_route",
    title="Mock an API with page.route",
    source="docs/network, docs/mock",
    look_for="Route.fulfill runs in the route handler and still nests under the workflow, with the mocked body as input",
    run=run,
    expect=Expect(calls_include=("Page.route", "Route.fulfill", "Locator.all_inner_texts")),
)

if __name__ == "__main__":
    main(SCENARIO)
