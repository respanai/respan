/** Several pages in one context, and a link that opens a new tab - mirrors Playwright's "Pages" page. */
import { chromium } from "playwright";
import { launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "multiple_pages",
  title: "Two pages and a new tab",
  source: "docs/pages",
  lookFor: "each call's playwright.page.url says which page it acted on; the popup's calls show popup.html",
  async run(site) {
    const browser = await chromium.launch(launchOptions());
    try {
      const context = await browser.newContext();
      const first = await context.newPage();
      const second = await context.newPage();
      await first.goto(site);
      await second.goto(`${site}/form.html`);
      const [popup] = await Promise.all([context.waitForEvent("page"), first.getByText("Open a new tab").click()]);
      await popup.waitForLoadState();
      console.log(context.pages().map((p) => p.url()), await popup.title());
      await first.bringToFront();
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["browser.newContext", "browserContext.newPage", "page.goto", "page.waitForLoadState", "page.bringToFront"] },
};
