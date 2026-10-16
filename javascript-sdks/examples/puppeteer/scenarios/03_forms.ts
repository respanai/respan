/** Type into a sign-up form, select, check, submit - mirrors pptr.dev's "Page interactions" guide. */
import puppeteer from "puppeteer-core";
import { launchOptions, type Scenario } from "../harness.js";

const PASSWORD = "correct-horse-battery-staple-42";

export const scenario: Scenario = {
  name: "forms",
  title: "Fill and submit a form",
  source: "guides/page-interactions",
  lookFor: "page.type shows the email; the password type shows <masked>; the URL after submit shows <secret>password</secret>",
  async run(site) {
    const browser = await puppeteer.launch(launchOptions());
    try {
      const page = await browser.newPage();
      await page.goto(`${site}/form.html`);
      await page.type("input[name=email]", "ada@example.com");
      await page.type("input[name=password]", PASSWORD);
      await page.select("select[name=plan]", "pro");
      await page.click("input[name=terms]");
      // The form submits by GET: the password ends up in the page URL too.
      await Promise.all([page.waitForNavigation(), page.click("button[type=submit]")]);
      console.log(await page.title());
    } finally {
      await browser.close();
    }
  },
  expect: {
    callsInclude: ["page.type", "page.select", "page.click", "page.waitForNavigation", "page.title"],
    forbidden: [PASSWORD],
  },
};
