/** A typed extract with a nested zod schema on a live site. */
import { z } from "zod";
import { withStagehand, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "extract_schema",
  title: "extract a typed list from Hacker News",
  source: "showcase/yc-companies-by-industry (extract with a schema)",
  lookFor: "extract input shows the JSON schema; output.data is the typed list",
  run: () =>
    withStagehand(async (stagehand, page) => {
      await page.goto("https://news.ycombinator.com");
      return (
        await stagehand.extract(
          "extract the top 5 stories with title, points and comment count",
          z.object({ stories: z.array(z.object({ title: z.string(), points: z.number(), comments: z.number() })).max(5) }),
        )
      ).data;
    }),
  expect: { spansInclude: ["stagehand.extract"] },
};
