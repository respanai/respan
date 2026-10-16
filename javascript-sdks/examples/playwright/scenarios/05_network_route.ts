/** Mock an API response and block images with page.route - mirrors Playwright's "Network" / "Mock APIs" pages. */
import { chromium } from "playwright";
import { launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "network_route",
  title: "Mock an API with page.route",
  source: "docs/network, docs/mock",
  lookFor: "route.fulfill runs in the route handler and still nests under the workflow, with the mocked body as input",
  async run(site) {
    const browser = await chromium.launch(launchOptions());
    try {
      const page = await browser.newPage();
      await page.route("**/api/products.json", (route) => route.fulfill({ json: [{ name: "Mocked widget", price: 1 }] }));
      await page.route("**/*.{png,jpg,jpeg}", (route) => route.abort());
      await page.goto(`${site}/products.html`);
      await page.locator(".product").first().waitFor();
      console.log(await page.locator(".product").allInnerTexts());
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["page.route", "route.fulfill", "locator.allInnerTexts"] },
};
