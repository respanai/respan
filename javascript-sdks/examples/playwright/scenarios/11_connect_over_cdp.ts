/**
 * Attach to an already-running Chrome over CDP - mirrors BrowserType.connectOverCDP
 * in Playwright's API docs. The "remote" browser is a local Chrome started with a
 * debugging port.
 */
import net from "node:net";
import { chromium } from "playwright";
import { launchOptions, type Scenario } from "../harness.js";

async function freePort(): Promise<number> {
  const server = net.createServer();
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as net.AddressInfo;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

export const scenario: Scenario = {
  name: "connect_over_cdp",
  title: "connectOverCDP to a running Chrome",
  source: "docs/api/class-browsertype#browser-type-connect-over-cdp",
  lookFor: "browserType.connectOverCDP, then the attached page's calls under the same workflow",
  async run(site) {
    const port = await freePort();
    const remote = await chromium.launch(launchOptions({ args: [`--remote-debugging-port=${port}`] }));
    try {
      const browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`);
      const context = browser.contexts()[0] ?? (await browser.newContext());
      const page = await context.newPage();
      await page.goto(site);
      console.log(await page.title());
      await browser.close();
    } finally {
      await remote.close();
    }
  },
  expect: { callsInclude: ["browserType.connectOverCDP", "page.goto"] },
};
