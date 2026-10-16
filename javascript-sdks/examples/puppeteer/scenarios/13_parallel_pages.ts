/** Two sessions at once on one browser, each in its own workflow - mirrors pptr.dev's "Browser contexts" guide. */
import puppeteer, { type Browser } from "puppeteer-core";
import { propagateAttributes, withWorkflow } from "@respan/tracing";
import { launchOptions, type Scenario } from "../harness.js";

async function session(browser: Browser, url: string, name: string): Promise<void> {
  await propagateAttributes({ thread_identifier: `parallel-${name}-${process.pid}` }, () =>
    withWorkflow({ name: `parallel_${name}` }, async () => {
      const context = await browser.createBrowserContext();
      const page = await context.newPage();
      await page.goto(url);
      console.log(name, await page.title());
      await context.close();
    }),
  );
}

export const scenario: Scenario = {
  name: "parallel_pages",
  title: "Parallel sessions, separate traces",
  source: "guides/browser-management#browser-contexts",
  lookFor: "2 traces, each with its own thread id and only its own page's calls",
  ownRoots: true,
  async run(site) {
    // Launched outside both workflows and shared: neither session owns it.
    const browser = await puppeteer.launch(launchOptions());
    try {
      await Promise.all([session(browser, `${site}/form.html`, "a"), session(browser, `${site}/products.html`, "b")]);
    } finally {
      await browser.close();
    }
  },
  expect: { traces: 2, callsInclude: ["browser.createBrowserContext", "page.goto"] },
};
