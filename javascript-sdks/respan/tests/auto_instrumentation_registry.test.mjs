import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { Respan } from "../dist/_core.js";
import {
  AUTO_INSTRUMENTATION_REGISTRY,
  DIRECT_LLM_AUTO_INSTRUMENTATIONS,
} from "../dist/_auto_instrumentation_registry.js";

const registryById = new Map(
  AUTO_INSTRUMENTATION_REGISTRY.map((entry) => [entry.id, entry]),
);

test("clean committed direct LLM packages are enabled for onboarding", () => {
  const expected = {
    "aws-bedrock": [
      "@aws-sdk/client-bedrock-runtime",
      "@respan/instrumentation-aws-bedrock",
      "AWSBedrockInstrumentor",
    ],
    cohere: [
      "cohere-ai",
      "@respan/instrumentation-cohere",
      "CohereInstrumentor",
    ],
    "together-ai": [
      "together-ai",
      "@respan/instrumentation-together-ai",
      "TogetherAIInstrumentor",
    ],
    writer: [
      "writer-sdk",
      "@respan/instrumentation-writer",
      "WriterInstrumentor",
    ],
  };

  for (const [
    id,
    [sdkPackage, instrumentationPackage, instrumentorClass],
  ] of Object.entries(expected)) {
    const entry = registryById.get(id);
    assert.ok(entry, id + " must be present in the onboarding registry");
    assert.equal(entry.category, "direct-llm");
    assert.equal(entry.enabledByDefault, true);
    assert.equal(entry.sdkPackage, sdkPackage);
    assert.equal(entry.instrumentationPackage, instrumentationPackage);
    assert.equal(entry.instrumentorClass, instrumentorClass);
  }

  const directIds = new Set(
    DIRECT_LLM_AUTO_INSTRUMENTATIONS.map((entry) => entry.id),
  );
  for (const id of Object.keys(expected)) {
    assert.ok(directIds.has(id), id + " must be auto-discoverable");
  }
});

test("clean committed framework packages stay explicit-only", () => {
  const expected = {
    "codex-sdk": "agent-framework",
    "cursor-sdk": "agent-framework",
    pi: "agent-framework",
    dify: "app-framework",
    livekit: "agent-framework",
    flue: "app-framework",
  };

  for (const [id, category] of Object.entries(expected)) {
    const entry = registryById.get(id);
    assert.ok(entry, id + " must be present in the onboarding registry");
    assert.equal(entry.category, category);
    assert.equal(entry.enabledByDefault, false);
    assert.ok(entry.autoDisabledReason);
  }
});

test("registry instrumentation packages are unique", () => {
  const packages = AUTO_INSTRUMENTATION_REGISTRY.map(
    (entry) => entry.instrumentationPackage,
  );
  assert.equal(new Set(packages).size, packages.length);
});

test("only default-enabled direct LLM entries enter the auto pool", () => {
  const expected = AUTO_INSTRUMENTATION_REGISTRY.filter(
    (entry) => entry.category === "direct-llm" && entry.enabledByDefault,
  );

  assert.deepEqual(DIRECT_LLM_AUTO_INSTRUMENTATIONS, expected);
  for (const entry of AUTO_INSTRUMENTATION_REGISTRY) {
    if (entry.enabledByDefault) {
      assert.equal(entry.category, "direct-llm", entry.id);
    } else if (entry.category !== "direct-llm") {
      assert.ok(entry.autoDisabledReason, entry.id);
    }
  }
});

test("optional instrumentation dependencies match the auto pool", async () => {
  const packageJson = JSON.parse(
    await readFile(new URL("../package.json", import.meta.url), "utf8"),
  );
  const bundled = Object.keys(packageJson.optionalDependencies ?? {})
    .filter((name) => name.startsWith("@respan/instrumentation-"))
    .sort();
  const enabled = DIRECT_LLM_AUTO_INSTRUMENTATIONS.map(
    (entry) => entry.instrumentationPackage,
  ).sort();

  assert.deepEqual(bundled, enabled);
});

test("auto activation requires confirmation without changing explicit plugin behavior", async () => {
  const respan = Object.create(Respan.prototype);
  respan._instrumentations = new Map();

  const explicitWithPrivateState = {
    name: "explicit-private-state",
    _isInstrumented: false,
    async activate() {},
    deactivate() {},
  };
  assert.equal(await respan._activate(explicitWithPrivateState), true);
  assert.equal(
    respan._instrumentations.has(explicitWithPrivateState.name),
    true,
  );

  const confirmedNoOp = {
    name: "confirmed-no-op",
    _isInstrumented: false,
    async activate() {},
    deactivate() {},
  };
  assert.equal(await respan._activate(confirmedNoOp, true), false);
  assert.equal(respan._instrumentations.has(confirmedNoOp.name), false);

  const unconfirmedAutoPlugin = {
    name: "unconfirmed-auto-plugin",
    activate() {},
    deactivate() {},
  };
  assert.equal(await respan._activate(unconfirmedAutoPlugin, true), false);
  assert.equal(respan._instrumentations.has(unconfirmedAutoPlugin.name), false);

  const legacy = {
    name: "legacy",
    activate() {},
    deactivate() {},
  };
  assert.equal(await respan._activate(legacy), true);
  assert.equal(respan._instrumentations.has(legacy.name), true);
});
