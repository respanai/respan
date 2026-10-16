/** Launch, open a page, take a screenshot - mirrors pptr.dev's "Screenshots" guide. */
import { join } from "node:path";
import puppeteer from "puppeteer-core";
import { ARTIFACTS, launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "screenshot",
  title: "Launch, goto, screenshot",
  source: "guides/screenshots",
  lookFor: "workflow -> puppeteer.launch, page.goto with URL + title, page.screenshot with the image",
  async run(site) {
    const browser = await puppeteer.launch(launchOptions());
    try {
      const page = await browser.newPage();
      await page.goto(site);
      console.log(await page.title());
      await page.screenshot({ path: join(ARTIFACTS, "screenshot.png") });
    } finally {
      await browser.close();
    }
  },
  expect: {
    callsInclude: ["puppeteer.launch", "browser.newPage", "page.goto", "page.title", "page.screenshot", "browser.close"],
    screenshots: ["page.screenshot"],
  },
};
