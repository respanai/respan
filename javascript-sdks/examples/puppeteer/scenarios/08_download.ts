/** Download a file into a folder - mirrors pptr.dev's "Downloads" guide (downloadBehavior). */
import { existsSync, rmSync } from "node:fs";
import { join } from "node:path";
import puppeteer from "puppeteer-core";
import { ARTIFACTS, launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "download",
  title: "Download a file",
  source: "guides/downloads",
  lookFor: "browser.createBrowserContext with the download folder, then the click that started the download",
  async run(site) {
    const file = join(ARTIFACTS, "report.csv");
    rmSync(file, { force: true });
    const browser = await puppeteer.launch(launchOptions());
    try {
      const context = await browser.createBrowserContext({ downloadBehavior: { policy: "allow", downloadPath: ARTIFACTS } });
      const page = await context.newPage();
      await page.goto(site);
      await page.click("#download");
      for (let i = 0; i < 50 && !existsSync(file); i++) await new Promise((resolve) => setTimeout(resolve, 100));
      if (!existsSync(file)) throw new Error("the download did not arrive");
      console.log("downloaded", file);
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["browser.createBrowserContext", "browserContext.newPage", "page.click"] },
};
