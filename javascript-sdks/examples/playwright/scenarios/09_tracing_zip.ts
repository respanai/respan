/**
 * Record a Playwright trace.zip alongside the Respan trace - mirrors Playwright's "Trace viewer" page.
 *
 *   npx playwright show-trace artifacts/trace.zip
 */
import { join } from "node:path";
import { chromium } from "playwright";
import { ARTIFACTS, launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "tracing_zip",
  title: "Playwright tracing alongside",
  source: "docs/trace-viewer",
  lookFor: "tracing.start / tracing.stop spans; the trace.zip shows the same calls (Respan's internal probes do not appear in it)",
  async run(site) {
    const browser = await chromium.launch(launchOptions());
    try {
      const context = await browser.newContext();
      await context.tracing.start({ screenshots: true, snapshots: true });
      const page = await context.newPage();
      await page.goto(`${site}/form.html`);
      await page.getByLabel("Email").fill("trace@example.com");
      await context.tracing.stop({ path: join(ARTIFACTS, "trace.zip") });
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["tracing.start", "tracing.stop", "locator.fill"] },
};
