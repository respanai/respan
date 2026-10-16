/** showcase/saas-pricing-monitor: extract a pricing table from a live SaaS page. */
import { z } from "zod";
import { withStagehand, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "pricing_monitor",
  title: "extract pricing tiers from a live pricing page",
  source: "showcase/saas-pricing-monitor",
  lookFor: "extract output lists the plans with prices",
  run: () =>
    withStagehand(async (stagehand, page) => {
      await page.goto("https://linear.app/pricing");
      return (
        await stagehand.extract(
          "extract every pricing plan with its name and monthly price as shown",
          z.object({ plans: z.array(z.object({ name: z.string(), price: z.string() })) }),
        )
      ).data;
    }),
  expect: { spansInclude: ["stagehand.extract"] },
};
