import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";
import test from "node:test";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const adkVersion = JSON.parse(
  readFileSync(
    join(
      dirname(dirname(dirname(require.resolve("@google/adk")))),
      "package.json",
    ),
    "utf8",
  ),
).version;
const worker = fileURLToPath(
  new URL("./native-sdk-worker.mjs", import.meta.url),
);

function run(scenario, options = {}) {
  const result = spawnSync(process.execPath, [worker, scenario], {
    encoding: "utf8",
    env: { ...process.env, ...options },
  });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  const marker = result.stdout
    .split("\n")
    .find((line) => line.startsWith("NATIVE_RESULT="));
  assert.ok(marker, result.stdout);
  return JSON.parse(marker.slice("NATIVE_RESULT=".length));
}

test("released Gemini tool execution retains model/tool/result call correlation", () => {
  const result = run("tool");
  assert.equal(result.error, undefined);
  assert.equal(result.requests.length, 2);
  assert.equal(result.methodIdentityPreserved, true);
  const tool = result.spans.find(
    (span) => span.name === "execute_tool weather",
  );
  assert.equal(tool.attributes["gen_ai.tool.call.id"], "audit_call_1");
  assert.deepEqual(JSON.parse(tool.attributes["traceloop.entity.output"]), {
    city: "Tokyo",
    forecast: "sunny",
  });
  const call = result.spans.find(
    (span) => span.attributes["gen_ai.completion.0.tool_calls"],
  );
  assert.equal(
    JSON.parse(call.attributes["gen_ai.completion.0.tool_calls"])[0].id,
    tool.attributes["gen_ai.tool.call.id"],
  );
  assert.equal(
    result.requests[1].contents
      .flatMap((content) => content.parts ?? [])
      .find((part) => part.functionResponse)?.functionResponse.id,
    tool.attributes["gen_ai.tool.call.id"],
  );
  assert.ok(
    result.spans.every((span) =>
      Object.keys(span.attributes).every(
        (key) => !key.startsWith("gcp.vertex.agent."),
      ),
    ),
  );
});

test(
  "native 2.x graph workflows retain workflow type, node identity and parent links",
  { skip: Number(adkVersion.split(".")[0]) < 2 },
  () => {
    const result = run("workflow");
    assert.equal(result.error, undefined);
    assert.equal(result.requests.length, 0);
    assert.deepEqual(result.events.at(-1).output, { value: "graph-result" });
    const workflow = result.spans.find(
      (span) => span.name === "invoke_workflow audit_workflow",
    );
    assert.equal(workflow.attributes["respan.entity.log_type"], "workflow");
    assert.equal(
      workflow.attributes["traceloop.entity.name"],
      "audit_workflow",
    );
    const node = result.spans.find(
      (span) => span.name === "execute_node finish",
    );
    assert.equal(node.attributes["respan.entity.log_type"], "task");
    assert.equal(node.attributes["traceloop.entity.name"], "finish");
    assert.equal(
      node.attributes["traceloop.entity.path"],
      "audit_workflow.finish",
    );
    assert.equal(
      JSON.parse(node.attributes["respan.metadata"]).google_adk_node_status,
      "completed",
    );
    assert.equal(node.parentSpanId, workflow.spanId);
  },
);

test("SDK content opt-out retains usage and excludes prompt/response/tool values", () => {
  const result = run("tool", { ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS: "false" });
  assert.equal(result.error, undefined);
  const serialized = JSON.stringify(result.spans);
  assert.equal(serialized.includes("Native SDK fixture prompt."), false);
  assert.equal(serialized.includes("Native ADK answer."), false);
  assert.equal(serialized.includes("Tokyo"), false);
  const chat = result.spans.find((span) => span.name === "call_llm");
  assert.equal(chat.attributes["gen_ai.usage.input_tokens"], 11);
});

test("always-off sampler creates no recording spans and preserves native events", () => {
  const result = run("tool", { ADK_TEST_DROP: "true" });
  assert.equal(result.error, undefined);
  assert.equal(result.requests.length, 2);
  assert.equal(result.events.length, 3);
  assert.equal(result.spans.length, 0);
});

test("provider failure preserves native error and unsuccessful response shape", () => {
  const bare = run("error", { ADK_TEST_BARE: "true" });
  const result = run("error");
  assert.deepEqual(result.error, bare.error);
  assert.ok(result.error || result.events.some((event) => event.errorCode));
  assert.deepEqual(
    result.events
      .filter((event) => event.errorCode)
      .map(({ errorCode, errorMessage }) => ({ errorCode, errorMessage })),
    bare.events
      .filter((event) => event.errorCode)
      .map(({ errorCode, errorMessage }) => ({ errorCode, errorMessage })),
  );
  assert.equal(result.methodIdentityPreserved, true);
  for (const span of result.spans.filter((span) => span.name === "call_llm")) {
    assert.equal(span.attributes["gen_ai.completion.0.content"], undefined);
    assert.equal(span.attributes["llm.usage.total_tokens"], undefined);
  }
});

test("native Gemini stream is assembled once with real token usage", () => {
  const result = run("stream");
  assert.equal(result.error, undefined);
  const chat = result.spans.find((span) => span.name === "call_llm");
  assert.equal(
    chat.attributes["gen_ai.completion.0.content"],
    "Native ADK answer.",
  );
  assert.equal(chat.attributes["gen_ai.usage.output_tokens"], 6);
  assert.equal(chat.attributes["llm.usage.total_tokens"], 17);
});

for (const [name, options] of [
  ["constructor", { ADK_TEST_CONTENT: "false" }],
  ["another active SDK module copy", { ADK_TEST_SECOND_OWNER: "true" }],
  ["environment", { RESPAN_TRACE_CONTENT: "false" }],
  ["canonical context", { ADK_TEST_CONTEXT_VETO: "true" }],
  [
    "observed false then true on the actual recording span",
    { ADK_TEST_LATE_VETO: "true" },
  ],
  ["late actual ReadableSpan", { ADK_TEST_READABLE_VETO: "true" }],
]) {
  test(`${name} privacy veto removes real SDK payloads`, () => {
    const result = run("tool", options);
    assert.equal(result.error, undefined);
    const payload = JSON.stringify(result.spans);
    for (const value of [
      "Native SDK fixture prompt.",
      "Native ADK answer.",
      "Tokyo",
      "sunny",
    ])
      assert.equal(payload.includes(value), false);
    assert.ok(result.spans.length > 0);
  });
}

test("deactivation drains the actual in-flight SDK span and restores capture methods", () => {
  const result = run("stream", { ADK_TEST_DEACTIVATE: "true" });
  assert.equal(result.error, undefined);
  const chat = result.spans.find((span) => span.name === "call_llm");
  assert.equal(
    chat.attributes["gen_ai.completion.0.content"],
    "Native ADK answer.",
  );
  assert.equal(result.methodIdentityPreserved, true);
  assert.equal(result.captureMethodsRestored, true);
});

test("OTel suppression preserves native events without admitting ADK spans", () => {
  const result = run("tool", { ADK_TEST_SUPPRESS: "true" });
  assert.equal(result.error, undefined);
  assert.equal(result.events.length, 3);
  assert.equal(result.spans.length, 0);
});

test("a real RECORD_ONLY sampler prevents adapter capture and admission", () => {
  const result = run("tool", {
    ADK_TEST_NATIVE_PROVIDER: "true",
    ADK_TEST_RECORD_ONLY: "true",
  });
  assert.equal(result.error, undefined);
  assert.ok(result.samplerCalls > 0);
  assert.equal(result.spans.length, 0);
  assert.ok(result.originals.length > 0);
  assert.ok(result.originals.every((span) => span.traceFlags === 0));
});

test("a processor after Respan receives the original native ReadableSpan attributes", () => {
  const result = run("tool", { ADK_TEST_NATIVE_PROVIDER: "true" });
  assert.equal(result.error, undefined);
  assert.ok(
    result.spans.some(
      (span) => span.attributes["respan.entity.log_type"] === "chat",
    ),
  );
  const original = result.originals.find((span) => span.name === "call_llm");
  assert.ok(original.attributes["gcp.vertex.agent.llm_response"]);
  assert.equal(original.attributes["respan.entity.log_type"], undefined);
  assert.equal(original.attributes["gen_ai.completion.0.content"], undefined);
});

test("native parallel tool results retain every payload and call ID", () => {
  const result = run("parallel");
  assert.equal(result.error, undefined);
  assert.equal(
    result.spans.filter(
      (span) => span.attributes["respan.entity.log_type"] === "tool",
    ).length,
    2,
  );
  const merged = result.spans.find(
    (span) => span.name === "execute_tool (merged)",
  );
  assert.ok(merged);
  assert.equal(merged.attributes["respan.entity.log_type"], "task");
  assert.equal(merged.attributes["gen_ai.tool.call.id"], undefined);
  const toolResults = result.requests[1].contents
    .flatMap((content) => content.parts ?? [])
    .filter((part) => part.functionResponse);
  assert.equal(toolResults.length, 2);
  const call = result.spans
    .filter((span) => span.name === "call_llm")
    .find((span) => Object.values(span.attributes).includes("audit_call_2"));
  assert.ok(call);
  const messages = Object.keys(call.attributes)
    .filter(
      (key) =>
        /^gen_ai.prompt.\d+.role$/.test(key) && call.attributes[key] === "tool",
    )
    .map((key) => key.slice(0, -5));
  assert.equal(messages.length, 2);
  assert.deepEqual(
    messages.map((prefix) => call.attributes[prefix + ".tool_call_id"]).sort(),
    ["audit_call_1", "audit_call_2"],
  );
  assert.deepEqual(
    messages
      .map(
        (prefix) => JSON.parse(call.attributes[prefix + ".content"]).forecast,
      )
      .sort(),
    ["cloudy", "sunny"],
  );
});

test("content veto excludes arbitrary actual span attributes and canonical metadata", () => {
  const result = run("tool", {
    ADK_TEST_CONTENT: "false",
    ADK_TEST_PRIVATE_METADATA: "true",
  });
  assert.equal(result.error, undefined);
  assert.equal(
    JSON.stringify(result.spans).includes("controlled-private-metadata"),
    false,
  );
});
