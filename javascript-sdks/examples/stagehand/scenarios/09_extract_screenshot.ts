/** extract({ screenshot: true }): the viewport image goes into the prompt. */
import { z } from "zod";
import { withStagehand, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "extract_screenshot",
  title: "extract with a screenshot in the prompt",
  source: "docs: extract screenshot option",
  lookFor: "the extract's LLM prompt shows the page image",
  run: (site) =>
    withStagehand(async (stagehand, page) => {
      await page.goto(`${site}/catalog-3.html`);
      return (await stagehand.extract("which products are sold out", z.object({ soldOut: z.array(z.string()) }), { screenshot: true })).data;
    }),
  expect: { spansInclude: ["stagehand.extract"], promptImages: true },
};
