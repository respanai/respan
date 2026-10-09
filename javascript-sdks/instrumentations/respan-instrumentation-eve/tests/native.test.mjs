import assert from "node:assert/strict";
import test from "node:test";
import { ROOT_CONTEXT, context, trace } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import { generateText, streamText } from "ai";
import { mockModel } from "eve/evals";
import { EveSpanProcessor } from "../dist/_processor.js";
import { withEveLineage } from "../dist/lineage.js";
import { safeJsonStr } from "../dist/_translator/shared.js";

function harness(options) {
  const translator = new EveSpanProcessor(options),
    spans = [];
  const sink = {
    onStart() {},
    onEnd(span) {
      spans.push(span);
    },
    forceFlush: async () => {},
    shutdown: async () => {},
  };
  const provider = new BasicTracerProvider({
    spanProcessors: [translator, sink],
  });
  const tracer = provider.getTracer("eve.agent");
  const exported = [];
  const exporter = translator.wrapExporter({
    export(batch, cb) {
      exported.push(...batch);
      cb({ code: 0 });
    },
    shutdown: async () => {},
  });
  return {
    translator,
    provider,
    tracer,
    spans,
    exported,
    exporter,
    send(batch = spans) {
      let calls = 0;
      exporter.export(batch, (result) => {
        calls++;
        assert.equal(result.code, 0);
      });
      assert.equal(calls, 1);
    },
  };
}
const chatAttrs = {
  "gen_ai.operation.name": "chat",
  "gen_ai.request.model": "native-fixture",
  "gen_ai.input.messages": JSON.stringify([
    { role: "user", parts: [{ type: "text", content: "synthetic-input" }] },
  ]),
  "gen_ai.output.messages": JSON.stringify([
    {
      role: "assistant",
      parts: [{ type: "text", content: "synthetic-output" }],
    },
  ]),
};

test("native Eve models preserve generation and streaming callback contracts", async () => {
  let calls = 0;
  const model = mockModel({
    modelId: "eve-native-fixture",
    respond() {
      calls++;
      return {
        text: "native response",
        usage: { inputTokens: 0, outputTokens: 2 },
      };
    },
  });
  const generated = await generateText({ model, prompt: "native request" });
  assert.equal(generated.text, "native response");
  assert.equal(generated.usage.inputTokens, 0);
  assert.equal(calls, 1);
  const streamed = streamText({ model, prompt: "native stream" });
  let text = "";
  for await (const part of streamed.textStream) text += part;
  assert.equal(text, "native response");
  assert.equal(await streamed.text, "native response");
  assert.equal(calls, 2);
});

test("current native OTel integration runtimeContext keeps authored callbacks and SDK brands", async () => {
  let native;
  try {
    native = await import("eve/instrumentation/otel");
  } catch (error) {
    if (error.code === "ERR_PACKAGE_PATH_NOT_EXPORTED") return;
    throw error;
  }
  let calls = 0;
  const payload = { owned: false };
  const original = native.otelIntegration({
    runtimeContext() {
      calls++;
      return payload;
    },
  });
  const wrapped = withEveLineage(original);
  const result = wrapped.runtimeContext({ session: { id: "native-session" } });
  assert.equal(calls, 1);
  assert.equal(result.owned, false);
  assert.equal(result.__respan_eve.lineage.rootSessionId, "native-session");
  assert.deepEqual(
    Object.getOwnPropertySymbols(wrapped),
    Object.getOwnPropertySymbols(original),
  );
  assert.deepEqual(payload, { owned: false });
  const failure = new Error("authored failure");
  const bad = withEveLineage(
    native.otelIntegration({
      runtimeContext() {
        throw failure;
      },
    }),
  );
  assert.throws(
    () => bad.runtimeContext({ session: { id: "native-session" } }),
    (error) => error === failure,
  );
  const promise = Promise.resolve({ owned: true });
  const asyncResult = withEveLineage(
    native.otelIntegration({ runtimeContext: () => promise }),
  );
  assert.equal(
    asyncResult.runtimeContext({ session: { id: "native-session" } }),
    promise,
  );
});

test("real native spans map current identities, system history, schemas and tool call IDs", () => {
  const h = harness();
  const span = h.tracer.startSpan("chat native-fixture", {
    attributes: {
      ...chatAttrs,
      "agent.run.id": "session-real",
      "gen_ai.conversation.id": "conversation-real",
      "gen_ai.agent.name": "native-agent",
      "gen_ai.system_instructions": JSON.stringify([
        { type: "text", content: "system instruction" },
      ]),
      "gen_ai.tool.definitions": JSON.stringify([
        {
          name: "echo",
          description: "echo tool",
          parameters: {
            type: "object",
            properties: {
              enabled: { type: "boolean", default: false },
              count: { type: "number", default: 0 },
            },
            additionalProperties: false,
          },
        },
      ]),
      "gen_ai.usage.input_tokens": 0,
      "gen_ai.usage.output_tokens": 2,
    },
  });
  span.end();
  const tool = h.tracer.startSpan("execute_tool echo", {
    attributes: {
      "gen_ai.operation.name": "execute_tool",
      "gen_ai.tool.name": "echo",
      "gen_ai.tool.call.id": "native-call-0",
      "gen_ai.tool.call.arguments": "0",
      "gen_ai.tool.call.result": "false",
    },
  });
  tool.end();
  h.send();
  const llm = h.exported.find((x) => x.name === "llm.native-fixture"),
    executed = h.exported.find((x) => x.name === "tool.echo");
  assert.equal(
    llm.attributes["respan.threads.thread_identifier"],
    "session-real",
  );
  assert.equal(
    llm.attributes["respan.trace.trace_group_identifier"],
    "conversation-real",
  );
  assert.equal(llm.attributes["gen_ai.prompt.0.role"], "system");
  assert.equal(llm.attributes["gen_ai.prompt.0.content"], "system instruction");
  const tools = JSON.parse(llm.attributes["llm.request.functions"]);
  assert.equal(tools[0].function.parameters.properties.enabled.default, false);
  assert.equal(tools[0].function.parameters.properties.count.default, 0);
  assert.equal(llm.attributes["gen_ai.usage.input_tokens"], 0);
  assert.equal(llm.attributes["llm.usage.total_tokens"], undefined);
  assert.equal(executed.attributes["gen_ai.tool.call.id"], "native-call-0");
  assert.deepEqual(JSON.parse(executed.attributes["traceloop.entity.input"]), {
    name: "echo",
    arguments: 0,
  });
  assert.equal(executed.attributes["traceloop.entity.output"], "false");
});

test("ambient and supplied content veto remains immutable across descendants and actual export", () => {
  const h = harness();
  const veto = ROOT_CONTEXT.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false);
  const parent = h.tracer.startSpan(
    "invoke_agent parent",
    {
      attributes: {
        "gen_ai.operation.name": "invoke_agent",
        "agent.trace.content.input": false,
      },
    },
    veto,
  );
  parent.setAttribute("agent.trace.content.input", true);
  const child = h.tracer.startSpan(
    "chat native-fixture",
    { attributes: chatAttrs },
    trace.setSpan(ROOT_CONTEXT, parent),
  );
  child.end();
  parent.end();
  const native = h.spans.find((x) => x.name === "chat native-fixture");
  native.attributes["traceloop.entity.input"] = "late-private-input";
  native.attributes["ai.response.text"] = "late-private-output";
  native.attributes["respan.metadata"] = '{"private":"late-private"}';
  native.events.push({
    name: "exception",
    time: [0, 0],
    attributes: {
      "exception.type": "Error",
      "exception.message": "late-private-error",
      "exception.stacktrace": "late-private-stack",
    },
  });
  native.status.message = "late-private-description";
  h.send();
  assert.equal(h.exported.length, 2);
  assert.ok(
    !JSON.stringify(
      h.exported.map((x) => ({
        attrs: x.attributes,
        events: x.events,
        status: x.status,
      })),
    ).includes("private"),
  );
  assert.equal(native.attributes["traceloop.entity.input"], undefined);
  assert.equal(native.events.length, 0);
  assert.equal(native.status.message, undefined);
});

test("real suppression and unsampled parents remove spans through the destination exporter", () => {
  for (const parentContext of [
    suppressTracing(ROOT_CONTEXT),
    trace.setSpanContext(ROOT_CONTEXT, {
      traceId: "11111111111111111111111111111111",
      spanId: "2222222222222222",
      traceFlags: 0,
      isRemote: true,
    }),
  ]) {
    const h = harness();
    h.tracer
      .startSpan(
        "chat native-fixture",
        { attributes: chatAttrs },
        parentContext,
      )
      .end();
    h.send();
    assert.equal(h.exported.length, 0);
  }
});

test("metadata-only and directional constructor flags survive late native ReadableSpan injection", () => {
  for (const options of [
    { traceContent: false },
    { recordInputs: false },
    { recordOutputs: false },
  ]) {
    const h = harness(options);
    h.tracer.startSpan("chat native-fixture", { attributes: chatAttrs }).end();
    const native = h.spans[0];
    native.attributes["traceloop.entity.input"] = "late-input";
    native.attributes["traceloop.entity.output"] = "late-output";
    h.send();
    const attrs = h.exported[0].attributes;
    if (options.traceContent === false || options.recordInputs === false)
      assert.equal(attrs["traceloop.entity.input"], undefined);
    if (options.traceContent === false || options.recordOutputs === false)
      assert.equal(attrs["traceloop.entity.output"], undefined);
  }
});

test("telemetry serialization never invokes unknown Proxy traps, getters, toJSON or iterators", () => {
  let calls = 0;
  const hostile = new Proxy(
    {},
    new Proxy(
      {},
      {
        get() {
          return () => {
            calls++;
            throw Error("trap");
          };
        },
      },
    ),
  );
  assert.equal(JSON.parse(safeJsonStr(hostile)), "[unavailable]");
  assert.equal(withEveLineage(hostile), hostile);
  const input = {
    safe: false,
    zero: 0,
    empty: "",
    get secret() {
      calls++;
      return "secret";
    },
    toJSON() {
      calls++;
      return "secret";
    },
    [Symbol.iterator]() {
      calls++;
      throw Error("iterator");
    },
  };
  assert.deepEqual(JSON.parse(safeJsonStr(input)), {
    safe: false,
    zero: 0,
    empty: "",
  });
  assert.equal(calls, 0);
  assert.deepEqual(
    JSON.parse(safeJsonStr(new Float64Array([0, 0.25, -1]))),
    [0, 0.25, -1],
  );
  assert.deepEqual(JSON.parse(safeJsonStr(Buffer.from([0, 1, 255]))), {
    type: "Buffer",
    data: [0, 1, 255],
  });
  assert.equal(safeJsonStr(null), "null");
});

test("unknown local parents deny capture while unknown sampled remote parents preserve it", () => {
  for (const isRemote of [false, true]) {
    const h = harness();
    const supplied = trace.setSpanContext(ROOT_CONTEXT, {
      traceId: "3".repeat(32),
      spanId: "4".repeat(16),
      traceFlags: 1,
      isRemote,
    });
    h.tracer
      .startSpan("chat native-fixture", { attributes: chatAttrs }, supplied)
      .end();
    h.send();
    assert.equal(
      h.exported[0].attributes["traceloop.entity.input"] !== undefined,
      isRemote,
    );
  }
});

test("empty strings and explicit null tool results survive native export", () => {
  for (const value of ['""', "null"]) {
    const h = harness();
    h.tracer
      .startSpan("execute_tool echo", {
        attributes: {
          "gen_ai.operation.name": "execute_tool",
          "gen_ai.tool.name": "echo",
          "gen_ai.tool.call.id": "empty-call",
          "gen_ai.tool.call.arguments": '""',
          "gen_ai.tool.call.result": value,
        },
      })
      .end();
    h.send();
    assert.equal(h.exported[0].attributes["traceloop.entity.output"], value);
    assert.deepEqual(
      JSON.parse(h.exported[0].attributes["traceloop.entity.input"]),
      { name: "echo", arguments: "" },
    );
  }
});

test("privacy peer regression scrubs error messages, tool descriptions and event names at actual export", () => {
  const h = harness({ traceContent: false });
  h.tracer.startSpan("chat native-fixture", { attributes: chatAttrs }).end();
  const span = h.spans[0];
  span.attributes["error.message"] = "late-private";
  span.attributes["gen_ai.tool.description"] = "late-private";
  span.events.push({
    name: "late-private",
    time: [0, 0],
    attributes: { content: "late-private" },
  });
  h.send();
  assert.ok(
    !JSON.stringify(
      h.exported.map((x) => ({
        attrs: x.attributes,
        events: x.events,
        status: x.status,
      })),
    ).includes("late-private"),
  );
  assert.equal(span.events.length, 0);
});

test("capture does not invoke a caller attributes getter on a real unrelated native parent", () => {
  const h = harness();
  let reads = 0;
  const parent = h.provider.getTracer("foreign").startSpan("foreign parent");
  Object.defineProperty(parent, "attributes", {
    configurable: true,
    get() {
      reads++;
      return { allow_trace_content: false };
    },
  });
  const supplied = trace.setSpan(ROOT_CONTEXT, parent);
  h.tracer
    .startSpan("chat native-fixture", { attributes: chatAttrs }, supplied)
    .end();
  h.send();
  assert.equal(reads, 0);
  assert.equal(h.exported[0].attributes["traceloop.entity.input"], undefined);
});

test("exporter transport suppression never changes a completed native execution decision", () => {
  const h = harness();
  h.tracer.startSpan("chat native-fixture", { attributes: chatAttrs }).end();
  context.with(suppressTracing(ROOT_CONTEXT), () => h.send());
  assert.equal(h.exported.length, 1);
  assert.ok(
    h.exported[0].attributes["traceloop.entity.output"].includes(
      "synthetic-output",
    ),
  );
});

test("current memory completed before model setup receives the same trace workflow at actual export", () => {
  const h = harness();
  const root = h.tracer.startSpan("invoke_agent native", {
      attributes: { "gen_ai.operation.name": "invoke_agent" },
    }),
    ctx = trace.setSpan(ROOT_CONTEXT, root);
  h.tracer
    .startSpan(
      "search_memory",
      {
        attributes: {
          "gen_ai.operation.name": "search_memory",
          "agent.run.id": "native-session",
          "gen_ai.memory.records": "[]",
        },
      },
      ctx,
    )
    .end();
  h.tracer
    .startSpan(
      "agent.step",
      {
        attributes: {
          "agent.name": "native-workflow",
          "agent.run.id": "native-session",
        },
      },
      ctx,
    )
    .end();
  root.end();
  h.send();
  assert.ok(
    h.exported.every(
      (span) =>
        span.attributes["traceloop.workflow.name"] === "native-workflow",
    ),
  );
});

test("canonical and legacy environment vetoes are frozen before conversion and cannot reenable queued content", () => {
  for (const key of ["RESPAN_TRACE_CONTENT", "TRACELOOP_TRACE_CONTENT"]) {
    const old = process.env[key];
    process.env[key] = "false";
    try {
      const h = harness();
      h.tracer
        .startSpan("chat native-fixture", { attributes: chatAttrs })
        .end();
      process.env[key] = "true";
      h.send();
      assert.equal(
        h.exported[0].attributes["traceloop.entity.input"],
        undefined,
      );
      assert.equal(
        h.exported[0].attributes["traceloop.entity.output"],
        undefined,
      );
    } finally {
      if (old === undefined) delete process.env[key];
      else process.env[key] = old;
    }
  }
});
