/** Locators: text and ARIA selectors, fill, click, a function locator - mirrors pptr.dev's "Locators" guide. */
import puppeteer from "puppeteer-core";
import { launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "locators",
  title: "Locators: click, fill, wait",
  source: "guides/page-interactions#locators",
  lookFor: "locator.click / locator.fill show the selector they were created from; locator.wait returns the heading",
  async run(site) {
    const browser = await puppeteer.launch(launchOptions());
    try {
      const page = await browser.newPage();
      await page.goto(site);
      await Promise.all([page.waitForNavigation(), page.locator("::-p-text(Sign-up form)").click()]);
      await page.locator("::-p-aria(Email)").fill("ada@example.com");
      await page.locator("select[name=plan]").fill("pro");
      await page.locator("input[name=terms]").setTimeout(5000).click();
      const heading = await page.locator(() => document.querySelector("h1")?.textContent).wait();
      console.log(heading);
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["locator.click", "locator.fill", "locator.wait"] },
};
