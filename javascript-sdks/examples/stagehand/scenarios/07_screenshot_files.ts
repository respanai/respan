/** cookbooks/files-to-bucket (local): extract data, screenshot, write both to disk. */
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { withStagehand, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "screenshot_files",
  title: "screenshot + JSON written to files",
  source: "cookbooks/files-to-bucket",
  lookFor: "page.screenshot output renders the image; extract carries the JSON",
  run: (site) =>
    withStagehand(async (stagehand, page) => {
      const out = mkdtempSync(join(tmpdir(), "stagehand-files-"));
      try {
        await page.goto(`${site}/catalog-2.html`);
        const { data } = await stagehand.extract("extract the product names on this page", z.object({ names: z.array(z.string()) }));
        writeFileSync(join(out, "products.json"), JSON.stringify(data));
        writeFileSync(join(out, "page.png"), await page.screenshot());
      } finally {
        rmSync(out, { recursive: true, force: true });
      }
    }),
  expect: { spansInclude: ["page.screenshot", "stagehand.extract"] },
};
