/**
 * Event listeners and an exposed function on a browser that was launched
 * outside the workflow - mirrors pptr.dev's "Events" and "Dialogs" docs.
 * Puppeteer fires these from its connection; their calls still nest under the
 * workflow that registered them.
 */
import puppeteer from "puppeteer-core";
import { withWorkflow } from "@respan/tracing";
import { launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "callbacks",
  title: "Listeners, exposeFunction, dialogs",
  source: "api/puppeteer.page.on, api/puppeteer.page.exposefunction",
  lookFor: "dialog.accept and the page.title an exposed function calls nest under the workflow",
  ownRoots: true,
  async run(site) {
    const browser = await puppeteer.launch(launchOptions()); // no span active: not traced
    try {
      await withWorkflow({ name: "callbacks" }, async () => {
        const page = await browser.newPage();
        page.on("dialog", (dialog) => dialog.accept());
        page.on("console", (message) => console.log("page says:", message.text()));
        await page.exposeFunction("currentTitle", () => page.title());
        await page.goto(`${site}/products.html`);
        await page.evaluate(async () => {
          console.log("title via exposed function:", await (window as any).currentTitle());
          alert("done");
        });
        await page.close();
      });
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["dialog.accept", "page.title", "page.exposeFunction", "page.evaluate"] },
};
