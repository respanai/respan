/**
 * Shared setup for the scenarios: env, Respan, the model, the browser, local
 * fixture pages, and the run record the live verifier reads.
 *
 * Every scenario runs under customer `stagehand-scenarios` with
 * `metadata.scenario=<name>` and `metadata.run_id=<run>`. After a run,
 * `verify_live.py` reads each trace back from the Respan logs API and checks it
 * against the scenario's `expect` - the live half of the package's baseline
 * test (`tests/stagehand_baseline.test.mjs`).
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import http from "node:http";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { config } from "dotenv";
import OpenAI from "openai";
import { localBrowser, Stagehand } from "@browserbasehq/stagehand";
import { Respan } from "@respan/respan";
import { propagateAttributes, registerSpanTransformer } from "@respan/tracing";
import { StagehandInstrumentor, withModelName } from "@respan/instrumentation-stagehand";

export const HERE = dirname(fileURLToPath(import.meta.url));
config({ path: join(HERE, ".env"), override: true, quiet: true });

export const API_KEY = process.env.RESPAN_API_KEY ?? "";
export const BASE_URL = (process.env.RESPAN_BASE_URL ?? "https://api.respan.ai/api").replace(/\/$/, "");
export const MODEL = process.env.RESPAN_MODEL ?? "gpt-4.1-mini";
export const SMALL_MODEL = process.env.RESPAN_SMALL_MODEL ?? "gpt-4o-mini";
export const CUSTOMER = "stagehand-scenarios";
export const RUN_ID = process.env.SCENARIO_RUN_ID ?? new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
if (!API_KEY) throw new Error("Set RESPAN_API_KEY (copy .env.example to .env).");

// -- the model: Stagehand v4's client-model callback, through the Respan gateway --

type Block = Record<string, any>;

function openAIContent(content: Block | Block[]): any {
  const blocks = Array.isArray(content) ? content : [content];
  return blocks.flatMap((block): any[] => {
    if (block.type === "text") return [{ type: "text", text: block.text }];
    if (block.type === "image") return [{ type: "image_url", image_url: { url: `data:${block.mimeType};base64,${block.data}` } }];
    return [];
  });
}

/**
 * `Stagehand.create({ model: respanModel() })`: every inference Stagehand's
 * browser extension needs comes back to this process as an `llm.generate`
 * call. Here it goes to the Respan gateway (OpenAI-compatible), so one Respan
 * key covers the traces and the model calls.
 */
export function respanModel(model = MODEL) {
  const client = new OpenAI({ apiKey: API_KEY, baseURL: BASE_URL });
  // withModelName: Stagehand knows nothing about the model behind `generate`;
  // this tells Respan, so the LLM spans name it and get priced.
  return { generate: withModelName(generate, model) };

  async function generate(params: any): Promise<any> {
      const messages: any[] = [];
      if (params.systemPrompt) messages.push({ role: "system", content: params.systemPrompt });
      for (const message of params.messages) messages.push({ role: message.role, content: openAIContent(message.content) });
      const structured = params.responseFormat?.type === "json_schema";
      const response = await client.chat.completions.create({
        model,
        messages,
        temperature: params.temperature,
        stop: params.stopSequences,
        ...(structured
          ? { response_format: { type: "json_schema", json_schema: { name: params.responseFormat.name, schema: params.responseFormat.schema } } }
          : {}),
        ...(params.tools?.length
          ? {
              tools: params.tools.map((tool: Block) => ({
                type: "function",
                function: { name: tool.name, description: tool.description, parameters: tool.inputSchema },
              })),
            }
          : {}),
      } as any);
      const choice = response.choices[0];
      const text = choice?.message?.content ?? "";
      const usage = response.usage
        ? {
            inputTokens: response.usage.prompt_tokens,
            outputTokens: response.usage.completion_tokens,
            totalTokens: response.usage.total_tokens,
            cachedInputTokens: response.usage.prompt_tokens_details?.cached_tokens ?? 0,
          }
        : undefined;
      const content: Block[] = text ? [{ type: "text", text }] : [];
      for (const call of choice?.message?.tool_calls ?? []) {
        if (call.type === "function")
          content.push({ type: "tool_use", id: call.id, name: call.function.name, input: JSON.parse(call.function.arguments || "{}") });
      }
      return structured
        ? { role: "assistant", content, outputFormat: "json_schema", structuredContent: JSON.parse(text), stopReason: choice?.finish_reason, usage }
        : { role: "assistant", content, outputFormat: "text", stopReason: choice?.finish_reason, usage };
  }
}

// -- Respan ---------------------------------------------------------------------------

const traces = new Map<string, Map<string, number>>(); // scenario -> trace id -> exported spans
let respan: Respan | undefined;

export async function init(options: { captureContent?: boolean; captureScreenshots?: boolean } = {}): Promise<Respan> {
  if (respan) return respan;
  respan = new Respan({
    apiKey: API_KEY,
    baseURL: BASE_URL,
    appName: "stagehand-scenarios",
    silenceInitializationMessage: true,
    instrumentations: [new StagehandInstrumentor(options)],
  });
  await respan.initialize();
  // Record which traces each scenario produced, and how many spans it
  // exported, so the verifier can compare against what Respan stored.
  registerSpanTransformer("stagehand-scenarios.recorder", {
    onEnd(span) {
      if (span.attributes["respan.entity.log_type"] === undefined) return; // not exported
      const scenario = scenarioOf(span.attributes);
      if (!scenario) return;
      const byTrace = traces.get(scenario) ?? new Map<string, number>();
      const id = span.spanContext().traceId;
      byTrace.set(id, (byTrace.get(id) ?? 0) + 1);
      traces.set(scenario, byTrace);
    },
  });
  return respan;
}

function scenarioOf(attributes: Record<string, unknown>): string | undefined {
  if (typeof attributes["respan.metadata.scenario"] === "string") return attributes["respan.metadata.scenario"];
  try {
    return JSON.parse(String(attributes["respan.metadata"] ?? "{}")).scenario;
  } catch {
    return undefined;
  }
}

// -- browser, fixtures ------------------------------------------------------------------

const TYPES: Record<string, string> = { ".html": "text/html", ".json": "application/json" };

/** Serve ./fixtures on 127.0.0.1; resolves to the base URL. */
export async function serveFixtures(): Promise<{ url: string; close: () => Promise<void> }> {
  const server = http.createServer((request, response) => {
    const path = join(HERE, "fixtures", (request.url ?? "/").split("?")[0].replace(/\.\.+/g, ""));
    try {
      const body = readFileSync(path);
      response.setHeader("content-type", TYPES[extname(path)] ?? "application/octet-stream");
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

export interface SessionOptions {
  model?: string;
  userDataDir?: string;
  systemPrompt?: string;
}

/** Launch local Chrome, attach Stagehand with the Respan model, run, close. */
export async function withStagehand<T>(
  run: (stagehand: Stagehand, page: any, browser: any) => Promise<T>,
  options: SessionOptions = {},
): Promise<T> {
  const browser = await localBrowser.launch({
    headless: process.env.HEADLESS === "1",
    ...(options.userDataDir ? { userDataDir: options.userDataDir, preserveUserDataDir: true } : {}),
  });
  try {
    const stagehand = await Stagehand.create({
      browser,
      model: respanModel(options.model),
      logging: { level: "error" },
      ...(options.systemPrompt ? { systemPrompt: options.systemPrompt } : {}),
    });
    try {
      const [page] = await browser.context.pages();
      return await run(stagehand, page, browser);
    } finally {
      await stagehand.close();
    }
  } finally {
    await browser.close();
  }
}

// -- scenarios ---------------------------------------------------------------------------

export interface Expect {
  /** Traces (= Stagehand sessions) the scenario produces. Default 1. */
  traces?: number;
  /** Spans that must appear, by instrumentation name, e.g. "page.screenshot". */
  spansInclude?: string[];
  /** Spans (by instrumentation name) that must be errors. */
  errorSpans?: string[];
  /** Strings no stored span may contain (secrets). */
  forbidden?: string[];
  /** true: some LLM prompt carried an image; false: none did. */
  promptImages?: boolean;
  /** Models that must have been called. */
  models?: string[];
}

export interface Scenario {
  name: string;
  title: string;
  /** Stagehand's own example this mirrors. */
  source: string;
  /** What to look for in Respan. */
  lookFor: string;
  run: (site: string) => Promise<unknown>;
  expect?: Expect;
}

export interface Outcome {
  scenario: Scenario;
  error?: string;
}

export async function execute(scenario: Scenario, site: string): Promise<Outcome> {
  await init();
  try {
    await propagateAttributes({ customer_identifier: CUSTOMER, metadata: { scenario: scenario.name, run_id: RUN_ID } }, () =>
      scenario.run(site),
    );
    return { scenario };
  } catch (error) {
    return { scenario, error: `${(error as Error)?.name}: ${(error as Error)?.message}` };
  }
}

/** Flush spans and write runs/<run id>.json for verify_live.py. */
export async function writeRun(outcomes: Outcome[]): Promise<string> {
  await respan?.flush();
  const record = {
    run_id: RUN_ID,
    customer: CUSTOMER,
    model: MODEL,
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
