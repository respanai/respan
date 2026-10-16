/** Two sessions on two models: each LLM span reports its own model and cost. */
import { z } from "zod";
import { SMALL_MODEL, withStagehand, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "second_model",
  title: "a session on a second model",
  source: "docs: model configuration",
  lookFor: "the chat spans name the small model",
  run: (site) =>
    withStagehand(
      async (stagehand, page) => {
        await page.goto(`${site}/login.html`);
        return (await stagehand.extract("extract the page heading", z.object({ heading: z.string() }))).data;
      },
      { model: SMALL_MODEL },
    ),
  expect: { spansInclude: ["stagehand.extract"], models: [SMALL_MODEL] },
};
