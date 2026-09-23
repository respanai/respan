import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { N8nTransformingExporter } from "../dist/_exporter.js";
import {
  buildN8nNodeSdkConfiguration,
  isN8nNodeSdkConfiguration,
} from "../dist/_native_instrumentation.js";
import { N8nSpanProcessor } from "../dist/_processor.js";
import { N8nInstrumentor } from "../dist/index.js";
import * as publicApi from "../dist/index.js";

const require = createRequire(import.meta.url);

test("public surface exposes only the standard instrumentor", () => {
  assert.deepEqual(Object.keys(publicApi), ["N8nInstrumentor"]);
});

function n8nResource() {
  return {
    attributes: {
      "service.name": "n8n",
      "service.version": "2.37.7",
      "n8n.instance.id": "instance-1",
      "n8n.instance.role": "main",
    },
  };
}

test("strictly scopes NodeSDK configuration replacement to n8n resources", () => {
  const exporter = new InMemorySpanExporter();
  const n8nConfig = { resource: n8nResource(), traceExporter: exporter, sampler: "sentinel" };
  assert.equal(isN8nNodeSdkConfiguration(n8nConfig), true);

  const replaced = buildN8nNodeSdkConfiguration(n8nConfig);
  assert.notStrictEqual(replaced, n8nConfig);
  assert.strictEqual(replaced.traceExporter, exporter);
  assert.equal(replaced.sampler, "sentinel");
  assert.equal(replaced.spanProcessors.length, 2);
  assert.ok(replaced.spanProcessors[0] instanceof N8nSpanProcessor);
  assert.ok(
    replaced.spanProcessors[1]._exporter instanceof N8nTransformingExporter ||
      replaced.spanProcessors[1].constructor.name === "BatchSpanProcessor",
  );

  const nonN8n = {
    resource: { attributes: { "service.name": "another-service" } },
    traceExporter: exporter,
  };
  assert.equal(isN8nNodeSdkConfiguration(nonN8n), false);
  assert.strictEqual(buildN8nNodeSdkConfiguration(nonN8n), nonN8n);

  const missingRole = {
    resource: { attributes: { "n8n.instance.id": "instance-1" } },
    traceExporter: exporter,
  };
  assert.strictEqual(buildN8nNodeSdkConfiguration(missingRole), missingRole);

  const customProcessors = {
    resource: n8nResource(),
    traceExporter: exporter,
    spanProcessors: [],
  };
  assert.strictEqual(buildN8nNodeSdkConfiguration(customProcessors), customProcessors);

  const missingExporter = { resource: n8nResource() };
  assert.strictEqual(buildN8nNodeSdkConfiguration(missingExporter), missingExporter);
});

test("activate/deactivate is idempotent and reference-counted", () => {
  const first = new N8nInstrumentor();
  const second = new N8nInstrumentor();

  first.activate();
  first.activate();
  const patched = require("@opentelemetry/sdk-node").NodeSDK;
  assert.equal(first.isActive(), true);

  second.activate();
  assert.strictEqual(require("@opentelemetry/sdk-node").NodeSDK, patched);

  first.deactivate();
  first.deactivate();
  assert.equal(first.isActive(), false);
  assert.equal(second.isActive(), true);
  assert.strictEqual(require("@opentelemetry/sdk-node").NodeSDK, patched);

  second.deactivate();
  assert.equal(second.isActive(), false);
  assert.notStrictEqual(require("@opentelemetry/sdk-node").NodeSDK, patched);
});

test("preload hook patches the public NodeSDK export and restores it on deactivate", () => {
  const instrumentor = new N8nInstrumentor();
  instrumentor.activate();
  const loaded = require("@opentelemetry/sdk-node");
  const patched = loaded.NodeSDK;

  instrumentor.deactivate();
  const restored = require("@opentelemetry/sdk-node").NodeSDK;
  assert.notStrictEqual(patched, restored);
});

test("deactivate does not overwrite a later foreign NodeSDK wrapper", () => {
  const instrumentor = new N8nInstrumentor();
  instrumentor.activate();
  const loaded = require("@opentelemetry/sdk-node");
  const patched = loaded.NodeSDK;
  const original = Object.getPrototypeOf(patched);
  class ForeignNodeSDK extends patched {}
  loaded.NodeSDK = ForeignNodeSDK;

  instrumentor.deactivate();
  assert.strictEqual(require("@opentelemetry/sdk-node").NodeSDK, ForeignNodeSDK);

  // Restore only the test's foreign mutation so later tests see the host SDK.
  loaded.NodeSDK = original;
});
