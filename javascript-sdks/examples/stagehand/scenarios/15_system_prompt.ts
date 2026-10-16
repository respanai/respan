/** Stagehand.create({ systemPrompt }): the session records it, every prompt carries it. */
import { z } from "zod";
import { withStagehand, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "system_prompt",
  title: "a custom system prompt",
  source: "docs: systemPrompt",
  lookFor: "the session input shows the system prompt; the LLM system messages contain it",
  run: (site) =>
    withStagehand(
      async (stagehand, page) => {
        await page.goto(`${site}/catalog-1.html`);
        return (await stagehand.extract("extract the cheapest product", z.object({ name: z.string(), price: z.number() }))).data;
      },
      { systemPrompt: "Always answer as a careful shopping assistant. Prices are in US dollars." },
    ),
  expect: { spansInclude: ["stagehand.extract"] },
};
