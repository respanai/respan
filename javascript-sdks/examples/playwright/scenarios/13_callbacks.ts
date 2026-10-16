/**
 * Event listeners, an exposed function and a route handler on a browser that
 * was launched outside the workflow - mirrors Playwright's "Events" and
 * "Dialogs" pages. Playwright fires these callbacks from its connection; their
 * calls still nest under the workflow that registered them.
 */
import { chromium } from "playwright";
import { withWorkflow } from "@respan/tracing";
import { launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "callbacks",
  title: "Listeners, exposeFunction, route handler",
  source: "docs/events, docs/dialogs",
  lookFor: "dialog.accept, route.fulfill and the page.title an exposed function calls all nest under the workflow",
  ownRoots: true,
  async run(site) {
    const browser = await chromium.launch(launchOptions()); // no span active: not traced
    try {
      await withWorkflow({ name: "callbacks" }, async () => {
        const page = await browser.newPage();
        page.on("dialog", (dialog) => dialog.accept());
        page.on("console", (message) => console.log("page says:", message.text()));
        await page.exposeFunction("currentTitle", () => page.title());
        await page.route("**/api/products.json", (route) => route.fulfill({ json: [{ name: "From the handler", price: 3 }] }));
        await page.goto(`${site}/products.html`);
        await page.evaluate(async () => {
          console.log("title via exposed function:", await (window as any).currentTitle());
          alert("done");
        });
        await page.locator(".product").first().waitFor();
        await page.close();
      });
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["dialog.accept", "route.fulfill", "page.title", "page.exposeFunction"] },
};
