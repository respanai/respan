/** Web-first assertions, one of which fails - mirrors Playwright's "Assertions" page. */
import { expect } from "@playwright/test";
import { chromium } from "playwright";
import { launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "assertions",
  title: "expect(...) assertions, one failing",
  source: "docs/test-assertions",
  lookFor: "expect.toHaveText is an AssertionError span with what the page showed",
  async run(site) {
    const browser = await chromium.launch(launchOptions());
    try {
      const page = await browser.newPage();
      await page.goto(site);
      await expect(page).toHaveTitle("Respan Playwright demo");
      await expect(page.getByRole("heading")).toBeVisible();
      try {
        await expect(page.getByRole("heading")).toHaveText("Something else", { timeout: 1000 });
      } catch (error) {
        console.log("assertion failed as expected:", String((error as Error).message).split("\n")[0]);
      }
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["expect.toHaveTitle", "expect.toBeVisible"], errorCalls: ["expect.toHaveText"] },
};
