/** Mock an API and block images with request interception - mirrors pptr.dev's "Request interception" guide. */
import puppeteer from "puppeteer-core";
import { launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "interception",
  title: "Request interception: mock an API",
  source: "guides/network-interception",
  lookFor: "request.respond / request.continue run in the request listener and still nest under the workflow",
  async run(site) {
    const browser = await puppeteer.launch(launchOptions());
    try {
      const page = await browser.newPage();
      await page.setRequestInterception(true);
      page.on("request", (request) => {
        if (request.url().endsWith("/api/products.json")) {
          void request.respond({ contentType: "application/json", body: JSON.stringify([{ name: "Mocked widget", price: 1 }]) });
        } else if (request.resourceType() === "image") {
          void request.abort();
        } else {
          void request.continue();
        }
      });
      await page.goto(`${site}/products.html`);
      await page.waitForSelector(".product");
      console.log(await page.$$eval(".product", (items) => items.map((item) => item.textContent)));
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["page.setRequestInterception", "request.respond", "request.continue", "page.waitForSelector", "page.$$eval"] },
};
