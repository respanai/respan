/** A link that opens a new tab, and a second page - mirrors pptr.dev's "Pages" / "Targets" guides. */
import puppeteer from "puppeteer-core";
import { launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "new_tab",
  title: "Two pages and a new tab",
  source: "guides/page-interactions, api/puppeteer.browser.waitfortarget",
  lookFor: "each call's puppeteer.page.url says which page it acted on; the new tab's calls show popup.html",
  async run(site) {
    const browser = await puppeteer.launch(launchOptions());
    try {
      const first = await browser.newPage();
      const second = await browser.newPage();
      await first.goto(site);
      await second.goto(`${site}/form.html`);
      // A background tab does not render, so clicks in it wait forever: bring it to the front first.
      await first.bringToFront();
      await first.click("#open-popup");
      const target = await browser.waitForTarget((t) => t.url().endsWith("/popup.html"));
      const popup = (await target.page())!;
      await popup.waitForSelector("h1");
      console.log((await browser.pages()).map((p) => p.url()), await popup.title());
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["browser.newPage", "page.goto", "page.click", "page.waitForSelector", "browser.pages", "page.bringToFront"] },
};
