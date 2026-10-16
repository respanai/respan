"""HTTP calls with Playwright's APIRequestContext - mirrors Playwright for
Python's "API testing" page.

    python scenarios/11_api_request.py
"""

from playwright.sync_api import sync_playwright

from _harness import SITE, Expect, Scenario, main


def run():
    with sync_playwright() as p:
        request = p.request.new_context(base_url=SITE)
        response = request.get("/api/products.json")
        assert response.ok
        print(response.json())
        request.dispose()


SCENARIO = Scenario(
    name="api_request",
    title="APIRequestContext without a browser",
    source="docs/api-testing",
    look_for="APIRequest.new_context, APIRequestContext.get (status + URL), APIResponse.json",
    run=run,
    expect=Expect(calls_include=("APIRequest.new_context", "APIRequestContext.get", "APIResponse.json")),
)

if __name__ == "__main__":
    main(SCENARIO)
