/** Run JavaScript in the page - mirrors Playwright's "Evaluating JavaScript" page. */
import { chromium } from "playwright";
import { launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "evaluate",
  title: "page.evaluate and handles",
  source: "docs/evaluating",
  lookFor: "page.evaluate spans carry the expression, its argument and its JSON result",
  async run(site) {
    const browser = await chromium.launch(launchOptions());
    try {
      const page = await browser.newPage();
      await page.goto(site);
      console.log(await page.evaluate(() => document.links.length));
      console.log(await page.evaluate(([a, b]) => a + b, [2, 3]));
      console.log(await page.$$eval("a", (links) => links.map((l) => l.textContent)));
      const handle = await page.evaluateHandle(() => document.body);
      console.log(await handle.evaluate((body) => body.children.length));
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["page.evaluate", "page.$$eval", "page.evaluateHandle", "elementHandle.evaluate"] },
};
