/** Sign in once, save the storage state, reuse it in a new context - mirrors Playwright's "Authentication" page. */
import { join } from "node:path";
import { chromium } from "playwright";
import { ARTIFACTS, launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "storage_state",
  title: "Save and reuse storage state",
  source: "docs/auth",
  lookFor: "browserContext.storageState, then browser.newContext with storageState; the evaluate returns the reused session",
  async run(site) {
    const state = join(ARTIFACTS, "state.json");
    const browser = await chromium.launch(launchOptions());
    try {
      const context = await browser.newContext();
      const page = await context.newPage();
      await page.goto(`${site}/welcome.html`); // "signs in": stores a session in localStorage
      await context.storageState({ path: state });
      await context.close();
      const reused = await browser.newContext({ storageState: state });
      const again = await reused.newPage();
      await again.goto(site);
      console.log("session:", await again.evaluate(() => localStorage.getItem("session")));
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["browserContext.storageState", "browser.newContext", "browserContext.close"] },
};
