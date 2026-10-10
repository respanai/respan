import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
const worker = fileURLToPath(new URL("./native-worker.mjs", import.meta.url));
for (const scenario of [
  "metadata-getter",
  "reactivate",
  "inputs-disabled",
  "outputs-disabled",
  "basic",
  "chunks",
  "constructor",
  "context",
  "env",
  "suppression",
  "sampler",
  "record-only",
  "ancestor",
  "late-parent",
  "late-veto",
  "error",
  "missing-key",
  "usage",
  "full",
  "redact",
  "promise",
  "cancel-activation",
  "drain",
  "owners",
  "foreign",
  "metadata",
  "fallback",
  "scan-success",
  "scan-validation",
]) {
  test("native SafetyClient " + scenario, (t) => {
    const result = spawnSync(process.execPath, [worker, scenario], {
      encoding: "utf8",
      env: { ...process.env, OTEL_TRACES_SAMPLER: "always_on" },
      timeout: 30000,
    });
    assert.equal(result.status, 0, result.stderr + "\n" + result.stdout);
    if (result.stdout.includes('"skipped":'))
      t.skip("minimum SDK lacks fallbackModel");
    else assert.ok(result.stdout.includes('"passed":true'));
  });
}
