import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { context, trace } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import {
  BasicTracerProvider,
  AlwaysOffSampler,
  AlwaysOnSampler,
  SimpleSpanProcessor,
  InMemorySpanExporter,
} from "@opentelemetry/sdk-trace-base";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { RespanTelemetry } from "@respan/tracing";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import * as sdk from "safety-agent";
import { SuperagentInstrumentor } from "../dist/index.js";

const scenario = process.argv[2];
const supportsFallback = Function.prototype.toString
  .call(sdk.SafetyClient.prototype.guard)
  .includes("fallbackModel");
let skippedReason;
const exporter = new InMemorySpanExporter();
let runtime, provider, manager;
if (["sampler", "record-only", "late-veto"].includes(scenario)) {
  provider = new BasicTracerProvider({
    sampler:
      scenario === "sampler"
        ? new AlwaysOffSampler()
        : scenario === "record-only"
          ? {
              shouldSample() {
                return { decision: 1 };
              },
              toString() {
                return "RecordOnly";
              },
            }
          : new AlwaysOnSampler(),
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  trace.setGlobalTracerProvider(provider);
  manager = new AsyncLocalStorageContextManager().enable();
  context.setGlobalContextManager(manager);
} else {
  runtime = new RespanTelemetry({
    apiKey: "controlled-local-only",
    exporter,
    disableBatch: true,
    silenceInitializationMessage: true,
    disabledInstrumentations: [
      "openAI",
      "anthropic",
      "azureOpenAI",
      "cohere",
      "bedrock",
      "googleVertexAI",
      "googleAIPlatform",
      "pinecone",
      "together",
      "langChain",
      "llamaIndex",
      "chromaDB",
      "qdrant",
    ],
  });
  await runtime.initialize();
}
const nativeFetch = globalThis.fetch;
const requests = [];
const responses = [];
let plan = {};
process.env.OPENAI_COMPATIBLE_API_KEY = "synthetic";
sdk.providers["openai-compatible"].baseUrl = "http://fixture.invalid/v1";
sdk.providers.openai.baseUrl = "http://fallback.invalid/v1";
process.env.OPENAI_API_KEY = "synthetic";
globalThis.fetch = async (url, options) => {
  if (String(url).includes("/billing/usage"))
    return new Response(null, { status: 204 });
  const body = JSON.parse(options.body);
  requests.push(body);
  if (plan.wait) await plan.wait;
  if (plan.error) throw plan.error;
  if (plan.status && (!plan.once || requests.length === 1))
    return Response.json(
      { error: { message: "controlled provider failure" } },
      { status: plan.status },
    );
  const result = (body.messages ?? body.input ?? []).some((m) =>
    String(m.content).includes("redact"),
  )
    ? { redacted: plan.redacted ?? "controlled redacted", findings: [] }
    : {
        classification: "pass",
        reasoning: plan.reasoning ?? "controlled explanation",
        violation_types: [],
        cwe_codes: [],
        ...(plan.extra ?? {}),
      };
  const raw = {
    id: "controlled-response-" + requests.length,
    model: "controlled-resolved",
    choices: [
      { message: { role: "assistant", content: JSON.stringify(result) } },
    ],
    usage: plan.usage ?? {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
    },
  };
  if (body.input) {
    raw.output = [
      {
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: JSON.stringify(result) }],
      },
    ];
    raw.usage = { input_tokens: 0, output_tokens: 0, total_tokens: 0 };
  }
  responses.push(raw);
  return Response.json(raw);
};
const options = {
  safetyAgentModule: sdk,
  ...(scenario === "constructor" ? { traceContent: false } : {}),
  ...(scenario === "inputs-disabled" ? { recordInputs: false } : {}),
  ...(scenario === "outputs-disabled" ? { recordOutputs: false } : {}),
  ...(scenario === "metadata"
    ? { metadata: { run_id: "controlled-root", keep: 0 } }
    : {}),
};
let metadataReads = 0;
if (scenario === "metadata-getter") {
  const metadata = { keep: 0 };
  Object.defineProperty(metadata, "ignored", {
    enumerable: true,
    get() {
      metadataReads++;
      throw new Error("telemetry-only getter");
    },
  });
  options.metadata = metadata;
}
const instrumentation = new SuperagentInstrumentor(options);
await instrumentation.activate();
const client = sdk.createClient({
  apiKey: "controlled-client",
  enableFallback: false,
});
const guard = (more = {}) =>
  client.guard({
    input: "private alpha beta gamma",
    model: "openai-compatible/controlled",
    chunkSize: 0,
    ...more,
  });
const all = () =>
  exporter
    .getFinishedSpans()
    .filter((s) => s.instrumentationScope.name === "superagent");
const chats = () =>
  all().filter((s) => s.attributes["respan.entity.log_type"] === "chat");
const payloadKeys = (a) =>
  Object.keys(a).filter((k) =>
    /entity\.(input|output)|prompt\.|completion\.|metadata|error|exception|late\./.test(
      k,
    ),
  );
try {
  if (scenario === "metadata-getter") {
    const result = await guard();
    assert.equal(result.classification, "pass");
    assert.equal(metadataReads, 0);
    assert.equal(requests.length, 1);
    assert.equal(all().length, 2);
  } else if (scenario === "reactivate") {
    instrumentation.deactivate();
    const next = new SuperagentInstrumentor({ safetyAgentModule: sdk });
    const first = next.activate();
    next.deactivate();
    const second = next.activate();
    assert.notEqual(first, second);
    await Promise.all([first, second]);
    assert.equal(next.isActive(), true);
    await guard();
    assert.equal(all().length, 2);
    next.deactivate();
  } else if (
    scenario === "inputs-disabled" ||
    scenario === "outputs-disabled"
  ) {
    const result = await guard();
    assert.equal(result.classification, "pass");
    assert.equal(all().length, 2);
    for (const span of all()) {
      assert.equal(
        span.attributes[
          scenario === "inputs-disabled"
            ? "traceloop.entity.input"
            : "traceloop.entity.output"
        ],
        undefined,
      );
      assert.notEqual(
        span.attributes[
          scenario === "inputs-disabled"
            ? "traceloop.entity.output"
            : "traceloop.entity.input"
        ],
        undefined,
      );
    }
  } else if (scenario === "basic") {
    const result = await guard();
    assert.equal(result.classification, "pass");
    assert.equal(all().length, 2);
    assert.equal(chats().length, 1);
    const chat = chats()[0],
      parent = all().find(
        (s) => s.attributes["respan.entity.log_type"] === "guardrail",
      );
    assert.equal(chat.parentSpanContext.spanId, parent.spanContext().spanId);
    assert.deepEqual(
      JSON.parse(chat.attributes["traceloop.entity.input"]),
      requests[0].messages,
    );
    assert.equal(chat.attributes["gen_ai.usage.input_tokens"], 0);
    assert.equal(chat.attributes["llm.usage.total_tokens"], 0);
    assert.equal(chat.attributes["gen_ai.request.model"], "controlled");
    assert.equal(
      chat.attributes["gen_ai.response.model"],
      "controlled-resolved",
    );
  } else if (scenario === "chunks") {
    await guard({ chunkSize: 8 });
    assert.ok(requests.length > 1);
    assert.equal(chats().length, requests.length);
    assert.equal(all().length, requests.length + 1);
    const parent = all().find(
      (s) => s.attributes["respan.entity.log_type"] === "guardrail",
    );
    for (const span of chats())
      assert.equal(span.parentSpanContext.spanId, parent.spanContext().spanId);
  } else if (
    [
      "constructor",
      "context",
      "env",
      "suppression",
      "sampler",
      "record-only",
      "ancestor",
      "late-parent",
    ].includes(scenario)
  ) {
    let reads = 0;
    const extra = {
      input: "private alpha beta gamma",
      model: "openai-compatible/controlled",
      chunkSize: 0,
    };
    Object.defineProperty(extra, "unused", {
      enumerable: true,
      get() {
        reads++;
        return "private getter";
      },
    });
    const invoke = () => client.guard(extra);
    if (scenario === "context")
      await context.with(
        context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
        invoke,
      );
    else if (scenario === "env") {
      process.env.RESPAN_TRACE_CONTENT = "false";
      await invoke();
      process.env.RESPAN_TRACE_CONTENT = "true";
    } else if (scenario === "suppression")
      await context.with(suppressTracing(context.active()), invoke);
    else if (scenario === "ancestor") {
      const parent = context.with(
        context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
        () => trace.getTracer("controlled-parent").startSpan("parent"),
      );
      await context.with(
        trace
          .setSpan(context.active(), parent)
          .setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, true),
        invoke,
      );
      parent.end();
    } else if (scenario === "late-parent") {
      const parent = trace.getTracer("controlled-parent").startSpan("parent");
      parent.setAttribute("allow_trace_content", false);
      await context.with(trace.setSpan(context.active(), parent), invoke);
      parent.end();
    } else await invoke();
    assert.equal(reads, 0);
    if (["suppression", "sampler", "record-only"].includes(scenario))
      assert.equal(all().length, 0);
    else {
      assert.equal(all().length, 2);
      for (const s of all()) assert.deepEqual(payloadKeys(s.attributes), []);
    }
  } else if (scenario === "late-veto") {
    await guard();
    for (const span of all()) {
      span.attributes.allow_trace_content = false;
      span.attributes["late.secret"] = "private late";
      span.attributes["gen_ai.usage.secret"] = "private usage";
      span.attributes["gen_ai.request.model"] = { secret: "private model" };
      assert.equal(span.attributes["gen_ai.usage.secret"], undefined);
      assert.equal(span.attributes["gen_ai.request.model"], undefined);
      span.events = [
        {
          name: "private event",
          attributes: { secret: "private" },
          time: [0, 1],
        },
      ];
      span.status = { code: 2, message: "private error" };
      assert.deepEqual(payloadKeys(span.attributes), []);
      assert.equal(span.events.length, 0);
      assert.equal(span.status.message, undefined);
      assert.throws(() =>
        Object.defineProperty(span, "attributes", {
          value: { "traceloop.entity.output": "private" },
        }),
      );
      span.attributes.allow_trace_content = true;
      assert.deepEqual(payloadKeys(span.attributes), []);
    }
  } else if (scenario === "error") {
    const original = new Error("controlled native identity");
    plan.error = original;
    await assert.rejects(guard(), (e) => e === original);
    assert.equal(all().length, 2);
    for (const span of all()) {
      assert.equal(span.status.code, 2);
      assert.equal(span.attributes["traceloop.entity.output"], undefined);
    }
  } else if (scenario === "missing-key") {
    delete process.env.OPENAI_COMPATIBLE_API_KEY;
    await assert.rejects(guard(), /Missing API key/);
    assert.equal(chats().length, 0);
    assert.equal(all().length, 1);
  } else if (scenario === "usage") {
    plan.usage = { prompt_tokens: 7, completion_tokens: 2 };
    await guard();
    assert.equal(chats()[0].attributes["gen_ai.usage.input_tokens"], 7);
    assert.equal(chats()[0].attributes["llm.usage.total_tokens"], undefined);
  } else if (scenario === "full") {
    const vector = Array.from({ length: 5001 }, (_, i) => i / 5001);
    plan.extra = {
      vector,
      tail: { false: false, zero: 0, empty: "", nil: null },
    };
    await guard({ input: "full input ".repeat(10000), chunkSize: 0 });
    const output = JSON.parse(
      JSON.parse(chats()[0].attributes["traceloop.entity.output"])[0].content,
    );
    assert.deepEqual(output.vector, vector);
    assert.deepEqual(output.tail, plan.extra.tail);
    assert.equal(
      JSON.parse(chats()[0].attributes["traceloop.entity.input"])[1].content
        .length,
      requests[0].messages[1].content.length,
    );
  } else if (scenario === "redact") {
    plan.redacted = "controlled false 0 empty null";
    const result = await client.redact({
      input: "please redact controlled text",
      model: "openai-compatible/controlled",
      rewrite: false,
      entities: [],
    });
    assert.equal(result.redacted, plan.redacted);
    assert.equal(chats().length, 1);
    const tools = all().filter(
      (s) => s.attributes["respan.entity.log_type"] === "tool",
    );
    assert.equal(tools.length, 1);
    assert.equal(
      JSON.parse(tools[0].attributes["traceloop.entity.input"]).name,
      "superagent.redact",
    );
  } else if (scenario === "promise") {
    instrumentation.deactivate();
    const descriptor = Object.getOwnPropertyDescriptor(
      sdk.SafetyClient.prototype,
      "guard",
    );
    let originalPromise;
    Object.defineProperty(sdk.SafetyClient.prototype, "guard", {
      ...descriptor,
      value: function (...args) {
        originalPromise = descriptor.value.apply(this, args);
        return originalPromise;
      },
    });
    await instrumentation.activate();
    const result = guard();
    assert.equal(result, originalPromise);
    await result;
    instrumentation.deactivate();
    Object.defineProperty(sdk.SafetyClient.prototype, "guard", descriptor);
  } else if (scenario === "cancel-activation") {
    instrumentation.deactivate();
    const next = new SuperagentInstrumentor({ safetyAgentModule: sdk });
    const pending = next.activate();
    next.deactivate();
    await pending;
    assert.equal(next.isActive(), false);
    await guard();
    assert.equal(all().length, 0);
  } else if (scenario === "drain") {
    let resolve;
    plan.wait = new Promise((r) => (resolve = r));
    const pending = guard();
    await new Promise((r) => setTimeout(r, 0));
    instrumentation.deactivate();
    resolve();
    await pending;
    assert.equal(all().length, 2);
    await guard();
    assert.equal(all().length, 2);
  } else if (scenario === "owners") {
    const other = new SuperagentInstrumentor({
      safetyAgentModule: sdk,
      methods: ["redact"],
    });
    await other.activate();
    instrumentation.deactivate();
    await guard();
    assert.equal(all().length, 0);
    await client.redact({
      input: "redact controlled",
      model: "openai-compatible/controlled",
    });
    assert.equal(all().length, 2);
    other.deactivate();
  } else if (scenario === "foreign") {
    const foreign = function () {
      return Promise.resolve("foreign");
    };
    sdk.SafetyClient.prototype.guard = foreign;
    instrumentation.deactivate();
    assert.equal(sdk.SafetyClient.prototype.guard, foreign);
  } else if (scenario === "metadata") {
    await guard();
    const root = all().find(
      (s) => s.attributes["respan.entity.log_type"] === "guardrail",
    );
    const meta = JSON.parse(root.attributes["respan.metadata"]);
    assert.equal(meta.run_id, "controlled-root");
    assert.equal(meta.keep, 0);
    assert.equal(meta.triggered, false);
  } else if (scenario === "fallback") {
    if (!supportsFallback)
      skippedReason = "safety-agent0.1.6 predates fallbackModel";
    else {
      plan.status = 429;
      plan.once = true;
      await guard({ fallbackModel: "openai/controlled-fallback" });
      assert.equal(requests.length, 2);
      assert.equal(chats().length, 2);
      assert.equal(chats()[0].attributes["traceloop.entity.output"], undefined);
      assert.equal(chats()[0].status.code, 2);
      assert.equal(chats()[1].attributes["gen_ai.system"], "openai");
    }
  } else if (scenario === "scan-success") {
    const sdkRequire = createRequire(import.meta.resolve("safety-agent"));
    const nativeRequire = createRequire(sdkRequire.resolve("@daytonaio/sdk"));
    const axios = nativeRequire("axios");
    const priorAdapter = axios.defaults.adapter;
    const calls = [];
    process.env.DAYTONA_API_KEY = "controlled";
    process.env.DAYTONA_API_URL = "http://fixture.invalid/api";
    process.env.DAYTONA_TARGET = "controlled";
    const reportLines = [
      JSON.stringify({
        type: "text",
        part: { text: "controlled native scan report" },
      }),
      JSON.stringify({
        type: "step_finish",
        part: { tokens: { input: 0, output: 0, reasoning: 0 }, cost: 0 },
      }),
    ].join("\n");
    axios.defaults.adapter = async (config) => {
      const path = config.url;
      calls.push({ path, method: config.method });
      let response = {};
      if (path.includes("toolbox-proxy-url"))
        response = { url: "http://fixture.invalid/toolbox" };
      else if (path.includes("/process/execute"))
        response = { result: reportLines, exitCode: 0 };
      else if (path.includes("/sandbox"))
        response = {
          id: "controlled-sandbox",
          name: "controlled",
          state: "started",
          regionId: "controlled",
          labels: { "code-toolbox-language": "typescript" },
        };
      return {
        data: response,
        status: 200,
        statusText: "OK",
        headers: {},
        config,
      };
    };
    try {
      const result = await client.scan({
        repo: "https://example.invalid/controlled-repo",
        branch: "main",
      });
      assert.equal(
        result.result,
        "controlled native scan report",
        JSON.stringify({ result, calls }),
      );
      assert.equal(chats().length, 0);
      assert.equal(all().length, 1);
      assert.deepEqual(
        JSON.parse(all()[0].attributes["traceloop.entity.output"]),
        result,
      );
      assert.ok(calls.some((c) => c.method === "delete"));
    } finally {
      axios.defaults.adapter = priorAdapter;
    }
  } else if (scenario === "scan-validation") {
    await assert.rejects(client.scan({ repo: "invalid" }), /Repository URL/);
    assert.equal(chats().length, 0);
    assert.equal(all().length, 1);
    assert.equal(all()[0].status.code, 2);
  } else throw new Error("Unknown scenario " + scenario);
  console.log(
    JSON.stringify({
      scenario,
      passed: !skippedReason,
      ...(skippedReason ? { skipped: skippedReason } : {}),
      spans: all().length,
      requests: requests.length,
    }),
  );
} finally {
  instrumentation.deactivate();
  globalThis.fetch = nativeFetch;
  if (runtime) await runtime.shutdown();
  if (provider) {
    await provider.shutdown();
    manager.disable();
    trace.disable();
    context.disable();
  }
}
