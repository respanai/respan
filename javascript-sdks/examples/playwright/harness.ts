/**
 * Shared setup for the scenarios: env, Respan, the browser, the local demo
 * site, and the run record the live verifier reads.
 *
 * Every scenario runs inside a Respan workflow span (Playwright calls are
 * traced only under an active span) under customer `playwright-js-scenarios`
 * with `metadata.scenario=<name>` and `metadata.run_id=<run>`. After a run,
 * `verify_live.py` reads each trace back from the Respan logs API and checks
 * it against the scenario's `expect` - the live half of the package's
 * baseline test (`tests/playwright_baseline.test.mjs`).
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import http from "node:http";
import { dirname, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { config } from "dotenv";
import type { LaunchOptions } from "playwright";
import { Respan } from "@respan/respan";
import { propagateAttributes, registerSpanTransformer, withWorkflow } from "@respan/tracing";
import { OpenAIInstrumentor } from "@respan/instrumentation-openai";
import { PlaywrightInstrumentor, type PlaywrightInstrumentorOptions } from "@respan/instrumentation-playwright";

export const HERE = dirname(fileURLToPath(import.meta.url));
config({ path: join(HERE, ".env"), override: true, quiet: true });

export const API_KEY = process.env.RESPAN_API_KEY ?? "";
export const BASE_URL = (process.env.RESPAN_BASE_URL ?? "https://api.respan.ai/api").replace(/\/$/, "");
export const MODEL = process.env.RESPAN_MODEL ?? "gpt-4.1-mini";
export const CUSTOMER = "playwright-js-scenarios";
export const RUN_ID = process.env.SCENARIO_RUN_ID ?? new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
/** Screenshots, downloads, storage state and trace.zip (gitignored). */
export const ARTIFACTS = join(HERE, "artifacts");
mkdirSync(ARTIFACTS, { recursive: true });
if (!API_KEY) throw new Error("Set RESPAN_API_KEY (copy .env.example to .env).");

/** "chrome" drives the installed Google Chrome; empty = Playwright's own Chromium. */
export function launchOptions(extra: LaunchOptions = {}): LaunchOptions {
  const channel = process.env.PLAYWRIGHT_CHANNEL ?? "chrome";
  return { headless: process.env.HEADED !== "1", ...(channel ? { channel } : {}), ...extra };
}

// -- Respan ---------------------------------------------------------------------------

const SCENARIO_ATTR = "respan.metadata.scenario";
const API_NAME_ATTR = "playwright.api.name";
/** scenario -> trace id -> { spans the SDK exported, Playwright calls by name }. */
const traces = new Map<string, Map<string, { spans: number; calls: Record<string, number> }>>();
let respan: Respan | undefined;

/** Respan exports a span only when it has a kind, an entity path, or LLM attributes. */
function isExported(attributes: Record<string, unknown>): boolean {
  return ["traceloop.span.kind", "respan.entity.log_type", "traceloop.entity.path", "gen_ai.system", "llm.request.type"].some(
    (key) => attributes[key] !== undefined && attributes[key] !== "",
  );
}

function scenarioOf(attributes: Record<string, unknown>): string | undefined {
  if (typeof attributes[SCENARIO_ATTR] === "string") return attributes[SCENARIO_ATTR] as string;
  try {
    return JSON.parse(String(attributes["respan.metadata"] ?? "{}")).scenario;
  } catch {
    return undefined;
  }
}

export async function init(options: PlaywrightInstrumentorOptions = {}): Promise<Respan> {
  if (respan) return respan;
  respan = new Respan({
    apiKey: API_KEY,
    baseURL: BASE_URL,
    appName: "playwright-js-scenarios",
    silenceInitializationMessage: true,
    instrumentations: [new PlaywrightInstrumentor(options), new OpenAIInstrumentor()],
  });
  await respan.initialize();
  // Record which traces each scenario produced, how many spans, and which
  // Playwright calls, so the verifier can compare against what Respan stored.
  registerSpanTransformer("playwright-js-scenarios.recorder", {
    onEnd(span) {
      const attributes = span.attributes as Record<string, unknown>;
      const scenario = scenarioOf(attributes);
      if (!scenario || !isExported(attributes)) return;
      const byTrace = traces.get(scenario) ?? new Map();
      const id = span.spanContext().traceId;
      const entry = byTrace.get(id) ?? { spans: 0, calls: {} };
      entry.spans += 1;
      const api = attributes[API_NAME_ATTR];
      if (typeof api === "string") entry.calls[api] = (entry.calls[api] ?? 0) + 1;
      byTrace.set(id, entry);
      traces.set(scenario, byTrace);
    },
  });
  return respan;
}

// -- the local demo site -----------------------------------------------------------------

const TYPES: Record<string, string> = { ".html": "text/html", ".json": "application/json", ".csv": "text/csv" };

/** Serve ./site on 127.0.0.1; resolves to the base URL. */
export async function serveSite(): Promise<{ url: string; close: () => Promise<void> }> {
  const root = join(HERE, "site");
  const server = http.createServer((request, response) => {
    let path = (request.url ?? "/").split("?")[0];
    if (path.endsWith("/")) path += "index.html";
    const file = normalize(join(root, path));
    try {
      if (!file.startsWith(root)) throw new Error("outside the site");
      const body = readFileSync(file);
      response.setHeader("content-type", TYPES[extname(file)] ?? "application/octet-stream");
      response.end(body);
    } catch {
      response.statusCode = 404;
      response.end("not found");
    }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as { port: number };
  return { url: `http://127.0.0.1:${port}`, close: () => new Promise((resolve) => server.close(() => resolve())) };
}

// -- scenarios ----------------------------------------------------------------------------

export interface Expect {
  /** Traces the scenario produces. Default 1. */
  traces?: number;
  /** Playwright calls (API names, e.g. "page.goto") that must appear. */
  callsInclude?: string[];
  /** Calls that must be error spans. */
  errorCalls?: string[];
  /** Calls whose output must carry an image. */
  screenshots?: string[];
  /** Strings no stored span may contain (secrets). */
  forbidden?: string[];
  /** Chat spans (with tokens and cost) in the trace. */
  hasLlm?: boolean;
}

export interface Scenario {
  name: string;
  title: string;
  /** The Playwright docs page it mirrors. */
  source: string;
  /** What to look for in Respan. */
  lookFor: string;
  run: (site: string) => Promise<unknown>;
  expect?: Expect;
  /** The scenario opens its own workflow spans (e.g. one per parallel session). */
  ownRoots?: boolean;
}

export interface Outcome {
  scenario: Scenario;
  error?: string;
}

export async function execute(scenario: Scenario, site: string): Promise<Outcome> {
  await init();
  try {
    await propagateAttributes({ customer_identifier: CUSTOMER, metadata: { scenario: scenario.name, run_id: RUN_ID } }, () =>
      scenario.ownRoots ? scenario.run(site) : withWorkflow({ name: scenario.name }, () => scenario.run(site)),
    );
    return { scenario };
  } catch (error) {
    return { scenario, error: `${(error as Error)?.name}: ${(error as Error)?.message}` };
  }
}

/** Flush spans and write runs/<run id>.json for verify_live.py. */
export async function writeRun(outcomes: Outcome[], settings: Record<string, unknown>): Promise<string> {
  await respan?.flush();
  const record = {
    run_id: RUN_ID,
    customer: CUSTOMER,
    model: MODEL,
    settings,
    scenarios: outcomes.map(({ scenario, error }) => ({
      name: scenario.name,
      title: scenario.title,
      source: scenario.source,
      look_for: scenario.lookFor,
      expect: scenario.expect ?? {},
      error,
      traces: Object.fromEntries(traces.get(scenario.name) ?? []),
    })),
  };
  mkdirSync(join(HERE, "runs"), { recursive: true });
  const path = join(HERE, "runs", `${RUN_ID}.json`);
  writeFileSync(path, JSON.stringify(record, null, 2) + "\n");
  return path;
}

export async function shutdown(): Promise<void> {
  await respan?.shutdown();
}
