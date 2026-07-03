import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";
import assert from "node:assert/strict";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sourcePath = resolve(root, "src/index.ts");

test("exports an OpenRouter instrumentor", async () => {
  const mod = await import(pathToFileURL(resolve(root, "dist/index.js")));
  assert.equal(typeof mod.OpenRouterInstrumentor, "function");
  assert.equal(typeof mod.instrumentOpenRouter, "function");
});

test("uses canonical constants and avoids banned package-owned aliases", async () => {
  const source = await readFile(sourcePath, "utf8");
  assert.match(source, /@opentelemetry\/semantic-conventions\/incubating/);
  assert.match(source, /@traceloop\/ai-semantic-conventions/);
  assert.match(source, /@respan\/respan-sdk/);
  assert.doesNotMatch(source, /respan\.span\.tools/);
  assert.doesNotMatch(source, /respan\.span\.tool_calls/);
  assert.doesNotMatch(source, /\["tools"\]/);
  assert.doesNotMatch(source, /\["tool_calls"\]/);
  assert.doesNotMatch(source, /\["model"\]/);
  assert.doesNotMatch(source, /\["prompt_tokens"\]/);
  assert.doesNotMatch(source, /\["completion_tokens"\]/);
  assert.doesNotMatch(source, /\["total_tokens"\]/);
});
