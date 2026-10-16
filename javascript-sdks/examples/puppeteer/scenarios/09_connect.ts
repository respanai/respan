/** Connect to an already-running Chrome - mirrors pptr.dev's "Connecting to an existing browser" guide. */
import net from "node:net";
import puppeteer from "puppeteer-core";
import { launchOptions, type Scenario } from "../harness.js";

async function freePort(): Promise<number> {
  const server = net.createServer();
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as net.AddressInfo;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

export const scenario: Scenario = {
  name: "connect",
  title: "puppeteer.connect to a running Chrome",
  source: "guides/browser-management#connecting-to-a-running-browser",
  lookFor: "puppeteer.connect, then the attached browser's calls under the same workflow, then browser.disconnect",
  async run(site) {
    const port = await freePort();
    const remote = await puppeteer.launch(launchOptions({ args: [`--remote-debugging-port=${port}`] }));
    try {
      const browser = await puppeteer.connect({ browserURL: `http://127.0.0.1:${port}` });
      const page = await browser.newPage();
      await page.goto(site);
      console.log(await page.title());
      await browser.disconnect();
    } finally {
      await remote.close();
    }
  },
  expect: { callsInclude: ["puppeteer.connect", "page.goto", "browser.disconnect"] },
};
