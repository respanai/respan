/** Timeouts and a missing element - mirrors pptr.dev's "Timeouts" / error handling docs. */
import puppeteer from "puppeteer-core";
import { launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "errors",
  title: "A timeout and a missing element",
  source: "api/puppeteer.timeouterror",
  lookFor: "error spans whose message is Puppeteer's own (TimeoutError: Waiting for selector `#missing` failed ...)",
  async run(site) {
    const browser = await puppeteer.launch(launchOptions());
    try {
      const page = await browser.newPage();
      await page.goto(site);
      for (const attempt of [
        () => page.waitForSelector("#missing", { timeout: 500 }),
        () => page.locator("#missing").setTimeout(500).click(),
        () => page.click("#also-missing"),
      ]) {
        try {
          await attempt();
        } catch (error) {
          console.log(`${(error as Error).name}: ${(error as Error).message}`);
        }
      }
    } finally {
      await browser.close();
    }
  },
  expect: { errorCalls: ["page.waitForSelector", "locator.click", "page.click"] },
};
