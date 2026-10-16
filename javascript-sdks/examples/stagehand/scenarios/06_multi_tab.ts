/** Two tabs, extracted concurrently: each LLM call stays under its own operation. */
import { z } from "zod";
import { withStagehand, type Scenario } from "../harness.js";

const Heading = z.object({ heading: z.string() });

export const scenario: Scenario = {
  name: "multi_tab",
  title: "newPage + concurrent extracts on two tabs",
  source: "docs: multiple pages (page option)",
  lookFor: "context.newPage; two extract spans running in parallel, each with its own LLM calls",
  run: (site) =>
    withStagehand(async (stagehand, page, browser) => {
      await page.goto(`${site}/login.html`);
      const second = await browser.context.newPage(`${site}/catalog-1.html`);
      return Promise.all([
        stagehand.extract("extract the main heading of the sign-in page", Heading, { page }),
        stagehand.extract("extract the main heading of the catalog page", Heading, { page: second }),
      ]);
    }),
  expect: { spansInclude: ["context.newPage", "stagehand.extract"] },
};
