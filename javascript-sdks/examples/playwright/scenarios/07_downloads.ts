/** Download a file - mirrors Playwright's "Downloads" page. */
import { join } from "node:path";
import { chromium } from "playwright";
import { ARTIFACTS, launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "downloads",
  title: "Download a file",
  source: "docs/downloads",
  lookFor: "page.waitForEvent for the download and the click that started it (Playwright treats download.saveAs as internal, so it is not a call of its own)",
  async run(site) {
    const browser = await chromium.launch(launchOptions());
    try {
      const page = await browser.newPage();
      await page.goto(site);
      const [download] = await Promise.all([page.waitForEvent("download"), page.getByText("Download the report").click()]);
      await download.saveAs(join(ARTIFACTS, download.suggestedFilename()));
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["page.waitForEvent", "locator.click"] },
};
