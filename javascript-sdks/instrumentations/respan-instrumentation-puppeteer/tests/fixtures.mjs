// Shared test fixtures: a local sign-in site, a headless Chrome launcher, and a
// scripted mini-agent. Nothing here touches the network.
import { existsSync } from "node:fs";
import http from "node:http";
import { execFileSync } from "node:child_process";

export const LOGIN_PAGE = `<!doctype html><title>Sign in</title>
<form action="/done.html">
  <label>User <input id="user" name="user"></label>
  <label>Password <input id="password" name="password" type="password"></label>
  <button id="submit">Sign in</button>
</form>`;
export const DONE_PAGE = "<!doctype html><title>Signed in</title><h1>Welcome</h1>";

/** Serve the sign-in site on 127.0.0.1; resolves to { url, close }. */
export async function serveSite() {
  const pages = { "/index.html": LOGIN_PAGE, "/done.html": DONE_PAGE };
  const server = http.createServer((req, res) => {
    const body = pages[(req.url ?? "/").split("?")[0]];
    res.statusCode = body ? 200 : 404;
    res.setHeader("content-type", "text/html");
    res.end(body ?? "not found");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${server.address().port}`,
    close: () => new Promise((resolve) => server.close(resolve)),
  };
}

/** A local Chrome / Chromium (the CI runner's google-chrome), or undefined (the tests are then skipped). */
export function chromePath() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  const mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  if (existsSync(mac)) return mac;
  for (const name of ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"]) {
    try {
      return execFileSync("which", [name], { encoding: "utf8" }).trim() || undefined;
    } catch {
      // not on PATH
    }
  }
  return undefined;
}

export function launchOptions(extra = {}) {
  // --no-sandbox: CI runners (Ubuntu 24.04) block the unprivileged user namespaces Chrome's sandbox needs.
  return { executablePath: chromePath(), headless: true, args: ["--no-sandbox"], ...extra };
}

// The scripted model: each turn is one tool call the "agent" maps to Puppeteer.
export const USERNAME = "ada.lovelace";
export const PASSWORD = "hunter2-correct-horse";
export const SCRIPT = [
  ["goto", { path: "/index.html" }],
  ["fill", { selector: "#user", text: USERNAME }],
  ["type", { selector: "#password", text: PASSWORD }],
  ["submit", { selector: "#submit" }],
];

/** Run the scripted agent's tool calls on a page. */
export async function runAgent(page, site) {
  for (const [tool, args] of SCRIPT) {
    if (tool === "goto") await page.goto(site + args.path);
    else if (tool === "fill") await page.locator(args.selector).fill(args.text);
    else if (tool === "type") await page.type(args.selector, args.text);
    else if (tool === "submit") await Promise.all([page.waitForNavigation(), page.click(args.selector)]);
  }
}
