/** Record the page as a video while acting on it - mirrors pptr.dev's "Screen recording" guide (page.record). */
import { existsSync, rmSync, statSync } from "node:fs";
import { join } from "node:path";
import puppeteer from "puppeteer-core";
import { ARTIFACTS, launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "screen_recording",
  title: "Record a video of the page",
  source: "guides/screen-recording",
  lookFor: "page.record carries the video's path (puppeteer.recording.path); screenRecorder.stop ends it",
  async run(site) {
    const path = join(ARTIFACTS, "recording.webm");
    rmSync(path, { force: true });
    const browser = await puppeteer.launch(launchOptions());
    try {
      const page = await browser.newPage();
      await page.goto(`${site}/form.html`);
      const recording = await page.record({ path });
      await page.type("input[name=email]", "recorded@example.com");
      await page.select("select[name=plan]", "pro");
      await recording.stop();
      console.log("recording:", path, existsSync(path) ? `${statSync(path).size} bytes` : "missing");
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["page.record", "page.type", "screenRecorder.stop"] },
};
