/** The same calls over WebDriver BiDi instead of CDP - mirrors pptr.dev's "WebDriver BiDi" guide. */
import puppeteer from "puppeteer-core";
import { launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "webdriver_bidi",
  title: "WebDriver BiDi protocol",
  source: "webdriver-bidi",
  lookFor: "the same page.goto / page.type / page.screenshot spans as over CDP",
  async run(site) {
    const browser = await puppeteer.launch(launchOptions({ protocol: "webDriverBiDi" }));
    try {
      const page = await browser.newPage();
      await page.goto(`${site}/form.html`);
      await page.type("input[name=email]", "bidi@example.com");
      await page.screenshot();
      console.log(await page.title());
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["puppeteer.launch", "page.goto", "page.type", "page.screenshot"], screenshots: ["page.screenshot"] },
};
