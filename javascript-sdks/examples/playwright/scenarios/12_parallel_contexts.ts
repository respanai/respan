/** Two browser sessions at once on one browser, each in its own workflow - mirrors Playwright's "Browser contexts" page. */
import { chromium, type Browser } from "playwright";
import { propagateAttributes, withWorkflow } from "@respan/tracing";
import { launchOptions, type Scenario } from "../harness.js";

async function session(browser: Browser, url: string, name: string): Promise<void> {
  await propagateAttributes({ thread_identifier: `parallel-${name}-${process.pid}` }, () =>
    withWorkflow({ name: `parallel_${name}` }, async () => {
      const context = await browser.newContext();
      const page = await context.newPage();
      await page.goto(url);
      console.log(name, await page.title());
      await context.close();
    }),
  );
}

export const scenario: Scenario = {
  name: "parallel_contexts",
  title: "Parallel sessions, separate traces",
  source: "docs/browser-contexts",
  lookFor: "2 traces, each with its own thread id and only its own page's calls",
  ownRoots: true,
  async run(site) {
    // Launched outside both workflows and shared: neither session owns it.
    const browser = await chromium.launch(launchOptions());
    try {
      await Promise.all([session(browser, `${site}/form.html`, "a"), session(browser, `${site}/products.html`, "b")]);
    } finally {
      await browser.close();
    }
  },
  expect: { traces: 2, callsInclude: ["browser.newContext", "page.goto"] },
};
