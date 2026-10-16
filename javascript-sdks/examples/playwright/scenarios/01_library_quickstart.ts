/** Launch, open a page, take a screenshot - mirrors Playwright's "Library" quickstart. */
import { join } from "node:path";
import { chromium } from "playwright";
import { ARTIFACTS, launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "library_quickstart",
  title: "Launch, goto, screenshot",
  source: "docs/library",
  lookFor: "workflow -> browserType.launch, page.goto with URL + title, page.screenshot with the image",
  async run(site) {
    const browser = await chromium.launch(launchOptions());
    try {
      const page = await browser.newPage();
      await page.goto(site);
      console.log(await page.title());
      await page.screenshot({ path: join(ARTIFACTS, "quickstart.png") });
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["browserType.launch", "page.goto", "page.title", "page.screenshot"], screenshots: ["page.screenshot"] },
};
