import assert from "node:assert/strict";
import test from "node:test";
import {
  buildSuperagentSpanAttributes,
  buildSuperagentModelSpanAttributes,
  responseAttributes,
} from "../dist/_span_attributes.js";
import {
  normalizeCallInput,
  safeJsonStringify,
} from "../dist/_serialization.js";
test("operation bodies retain scalars without inventing model attributes", () => {
  const attrs = buildSuperagentSpanAttributes({
    methodName: "scan",
    args: [{ repo: "https://example.invalid/repo", branch: "main" }],
    result: { result: false, vector: [0, 0], tail: { empty: "", nil: null } },
  });
  assert.equal(attrs["respan.entity.log_type"], "tool");
  assert.equal(attrs["gen_ai.request.model"], undefined);
  assert.equal(attrs["gen_ai.usage.input_tokens"], undefined);
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), {
    result: false,
    vector: [0, 0],
    tail: { empty: "", nil: null },
  });
  assert.deepEqual(
    buildSuperagentModelSpanAttributes({
      methodName: "scan",
      args: [],
      result: { usage: { totalTokens: 99 } },
    }),
    {},
  );
});
test("usage originates in provider data and totals are never inferred", () => {
  const attrs = responseAttributes(
    { usage: { prompt_tokens: 0, completion_tokens: 0 } },
    { choices: [], usage: { totalTokens: 99 } },
    false,
  );
  assert.equal(attrs["gen_ai.usage.input_tokens"], 0);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 0);
  assert.equal(attrs["llm.usage.total_tokens"], undefined);
  assert.equal(attrs["traceloop.entity.output"], undefined);
});
test("serializer observes data properties only and never customer coercions", () => {
  let calls = 0;
  const value = {
    nil: null,
    zero: 0,
    empty: "",
    off: false,
    toJSON() {
      calls++;
      return "wrong";
    },
  };
  Object.defineProperty(value, "secret", {
    enumerable: true,
    get() {
      calls++;
      return "wrong";
    },
  });
  assert.deepEqual(JSON.parse(safeJsonStringify(value)), {
    nil: null,
    zero: 0,
    empty: "",
    off: false,
  });
  assert.equal(calls, 0);
  const proxy = new Proxy(
    {},
    {
      ownKeys() {
        calls++;
        throw new Error("trap");
      },
    },
  );
  assert.equal(safeJsonStringify(proxy), "null");
  assert.equal(calls, 0);
});
test("known SDK options preserve false and zero without extra properties", () => {
  let calls = 0;
  const options = {
    input: "controlled",
    chunkSize: 0,
    unknown() {
      calls++;
    },
  };
  Object.defineProperty(options, "ignored", {
    enumerable: true,
    get() {
      calls++;
      return "private";
    },
  });
  assert.deepEqual(normalizeCallInput("guard", [options]), {
    method: "guard",
    arguments: { input: "controlled", chunkSize: 0 },
  });
  assert.equal(calls, 0);
});
