/** Stagehand + Respan in one file: one session, one act, one extract. */
import "dotenv/config";
import { localBrowser, Stagehand } from "@browserbasehq/stagehand";
import { Respan } from "@respan/respan";
import { StagehandInstrumentor } from "@respan/instrumentation-stagehand";
import { z } from "zod";
import { respanModel } from "./harness.js";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL,
  instrumentations: [new StagehandInstrumentor()],
});
await respan.initialize();

const browser = await localBrowser.launch({ headless: process.env.HEADLESS === "1" });
// A client model: Stagehand's inferences come back to this process and go to
// the Respan gateway. Built-in models (`{ modelName, apiKey }`) work too.
const stagehand = await Stagehand.create({ browser, model: respanModel() });
try {
  const [page] = await browser.context.pages();
  await page.goto("https://news.ycombinator.com");
  await stagehand.act("click the 'new' link in the top bar");
  const { data } = await stagehand.extract("the titles of the first 3 stories", z.object({ titles: z.array(z.string()) }));
  console.log(data.titles);
} finally {
  await stagehand.close();
  await browser.close();
  await respan.shutdown(); // short scripts: send spans before exiting
}
