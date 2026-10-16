/** Stagehand's quickstart: open a page, act, extract. */
import { z } from "zod";
import { withStagehand, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "hello",
  title: "act + extract on a live page",
  source: "README quickstart",
  lookFor: "session -> act / extract, each with its LLM calls; a screenshot on the act",
  run: () =>
    withStagehand(async (stagehand, page) => {
      await page.goto("https://example.com");
      const { data } = await stagehand.extract("extract the page heading and the text of the link", z.object({ heading: z.string(), link: z.string() }));
      await stagehand.act("click the link on the page");
      return data;
    }),
  expect: { spansInclude: ["stagehand.act", "stagehand.extract", "page.goto"] },
};
