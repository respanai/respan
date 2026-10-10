import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
for (const mode of [
  "stable",
  "latest",
  "constructor",
  "env-respan",
  "env-traceloop",
  "env-flip",
  "context",
  "ancestor",
  "suppressed",
  "owners",
  "drop",
  "record-only",
  "late",
  "late-env",
  "drain",
  "stream",
  "scalar",
  "large",
  "error",
  "callback-error",
  "hostile",
  "hostile-enabled",
  "multimodal",
  "recording-veto",
  "readable-veto",
  "event-veto",
  "module-owners",
  "cancel",
  "tool-string",
  "tool-empty",
]) {
  test(`released SDK native provider: ${mode}`, () => {
    const run = spawnSync(
      process.execPath,
      [fileURLToPath(new URL("./fixtures/native.mjs", import.meta.url)), mode],
      {
        env: {
          ...process.env,
          RESPAN_TRACE_CONTENT: undefined,
          TRACELOOP_TRACE_CONTENT: undefined,
        },
        encoding: "utf8",
      },
    );
    assert.equal(run.status, 0, run.stderr + run.stdout);
  });
}
