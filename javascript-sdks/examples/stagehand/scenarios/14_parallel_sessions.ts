/** Two Stagehand sessions at once: two traces, nothing crosses over. */
import { z } from "zod";
import { withStagehand, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "parallel_sessions",
  title: "two sessions in parallel",
  source: "docs: running multiple Stagehand instances",
  lookFor: "2 traces, each with its own thread id",
  run: (site) =>
    Promise.all(
      ["catalog-1.html", "catalog-2.html"].map((path) =>
        withStagehand(async (stagehand, page) => {
          await page.goto(`${site}/${path}`);
          return (await stagehand.extract("extract the page heading", z.object({ heading: z.string() }))).data;
        }),
      ),
    ),
  expect: { traces: 2, spansInclude: ["stagehand.extract"] },
};
