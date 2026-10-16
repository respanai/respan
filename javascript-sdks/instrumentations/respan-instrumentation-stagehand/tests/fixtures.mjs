// Shared test fixtures: a local HTTP page, a scripted Stagehand client model,
// and a headless Chrome launcher. Nothing here touches the network.
import http from "node:http";
import { localBrowser } from "@browserbasehq/stagehand";

export const PAGE = `<!doctype html><title>Respan Test</title>
<h1>Hello browser agent</h1>
<form onsubmit="event.preventDefault(); document.title = 'Signed in';">
  <label>Email <input id="email" type="email"></label>
  <label>Password <input id="password" type="password"></label>
  <button id="go" type="submit">Sign in</button>
</form>`;

/** Serve PAGE on 127.0.0.1; resolves to { url, close }. */
export async function servePage(html = PAGE) {
  const server = http.createServer((_, res) => {
    res.setHeader("content-type", "text/html");
    res.end(html);
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${server.address().port}/`,
    close: () => new Promise((resolve) => server.close(resolve)),
  };
}

export const USAGE = { inputTokens: 120, outputTokens: 30, totalTokens: 150, cachedInputTokens: 20 };

/** Accessibility-tree id of the first line matching `pattern`, e.g. /button: Sign in/. */
function elementId(params, pattern) {
  const text = JSON.stringify(params.messages);
  const line = text.split("\\n").find((l) => pattern.test(l));
  const match = line?.match(/\[(\d+-\d+)\]/);
  if (!match) throw new Error(`scripted model: no element for ${pattern}`);
  return match[1];
}

/**
 * A Stagehand v4 client model (`model: { generate }`) that answers every
 * structured inference deterministically. `act` and `observe` target the
 * element whose accessibility-tree line matches `target`.
 */
export function scriptedModel({ target = /button: Sign in/, method = "click", args = [], extraction = "Hello browser agent" } = {}) {
  const calls = [];
  const generate = async (params) => {
    calls.push(params);
    const name = params.responseFormat?.name;
    const structured = (structuredContent) => ({
      role: "assistant",
      content: [{ type: "text", text: JSON.stringify(structuredContent) }],
      outputFormat: "json_schema",
      structuredContent,
      usage: USAGE,
    });
    switch (name) {
      case "Extraction":
        return structured(typeof extraction === "function" ? extraction(params) : { extraction });
      case "Metadata":
        return structured({ progress: "done", completed: true });
      case "Observation":
        return structured({
          elements: [{ elementId: elementId(params, target), description: "the target", method, arguments: args }],
        });
      case "Act":
        return structured({
          action: { elementId: elementId(params, target), description: "the target", method, arguments: args },
          twoStep: false,
        });
      default:
        return { role: "assistant", content: [{ type: "text", text: "ok" }], outputFormat: "text", usage: USAGE };
    }
  };
  return { generate, calls };
}

export function launchBrowser() {
  return localBrowser.launch({ headless: true });
}
