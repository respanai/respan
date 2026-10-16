/** Run JavaScript in the page - mirrors pptr.dev's "JavaScript execution" guide. */
import puppeteer from "puppeteer-core";
import { launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "evaluate",
  title: "evaluate, $eval, evaluateHandle",
  source: "guides/javascript-execution",
  lookFor: "page functions as source text with their arguments in; JSON results out",
  async run(site) {
    const browser = await puppeteer.launch(launchOptions());
    try {
      const page = await browser.newPage();
      await page.goto(site);
      console.log(await page.evaluate(() => ({ links: document.links.length, title: document.title })));
      console.log(await page.evaluate((a: number, b: number) => a * b, 6, 7));
      console.log(await page.$eval("h1", (h1) => h1.textContent));
      const body = await page.evaluateHandle(() => document.body);
      console.log(await body.evaluate((element) => element.children.length));
      await body.dispose();
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["page.evaluate", "page.$eval", "page.evaluateHandle"] },
};
