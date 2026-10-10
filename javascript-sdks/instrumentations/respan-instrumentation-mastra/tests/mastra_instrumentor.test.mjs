import assert from "node:assert/strict";
import test from "node:test";
import {
  context,
  createContextKey,
  trace,
  SpanStatusCode,
} from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SamplingDecision,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { Observability, SamplingStrategyType } from "@mastra/observability";
import { SpanType } from "@mastra/core/observability";
import { Agent } from "@mastra/core/agent";
import { Mastra } from "@mastra/core/mastra";
import { createMockModel } from "@mastra/core/test-utils/llm-mock";
import { contentAttributes } from "../dist/_translator.js";
import { MastraInstrumentor, RespanMastraExporter } from "../dist/index.js";

let decision = SamplingDecision.RECORD_AND_SAMPLED;
let started = [];
let onStart;
const exporter = new InMemorySpanExporter();
const provider = new BasicTracerProvider({
  spanLimits: { attributeCountLimit: 100000 },
  sampler: {
    shouldSample() {
      return { decision };
    },
    toString() {
      return "NativeFixtureSampler";
    },
  },
  spanProcessors: [
    {
      onStart(span) {
        started.push(span);
        onStart?.(span);
      },
      onEnd() {},
      async forceFlush() {},
      async shutdown() {},
    },
    new SimpleSpanProcessor(exporter),
  ],
});
trace.setGlobalTracerProvider(provider);
context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
test.beforeEach(() => {
  exporter.reset();
  started = [];
  onStart = undefined;
  decision = SamplingDecision.RECORD_AND_SAMPLED;
  delete process.env.RESPAN_TRACE_CONTENT;
  delete process.env.TRACELOOP_TRACE_CONTENT;
});
test.after(async () => {
  await provider.shutdown();
  trace.disable();
  context.disable();
});
function setup(options = {}, extra = {}) {
  const plugin = new MastraInstrumentor(options);
  const observability = new Observability({
    configs: {
      default: {
        serviceName: "native-fixtures",
        sampling: { type: SamplingStrategyType.ALWAYS },
        exporters: [plugin],
        serializationOptions: {
          maxArrayLength: 100000,
          maxObjectKeys: 100000,
          maxDepth: 100,
          maxStringLength: 10000000,
        },
        ...extra,
      },
    },
    sensitiveDataFilter: false,
  });
  return {
    plugin,
    observability,
    instance: observability.getDefaultInstance(),
  };
}
function native(
  instance,
  name = "native",
  input = { secret: "input" },
  output = { secret: "output" },
) {
  const span = instance.startSpan({ name, type: SpanType.GENERIC, input });
  span.end({ output });
  return span;
}
function spans() {
  return exporter.getFinishedSpans();
}
function attrs(name) {
  return spans().find((s) => s.name === name)?.attributes;
}
function noContent(span) {
  for (const key of Object.keys(span.attributes))
    assert.ok(
      !key.includes(".input") &&
        !key.includes(".output") &&
        !key.startsWith("gen_ai.prompt.") &&
        !key.startsWith("gen_ai.completion.") &&
        !key.startsWith("respan.metadata") &&
        !key.startsWith("exception.") &&
        !key.startsWith("error."),
      key,
    );
  assert.equal(span.status.message, undefined);
  assert.deepEqual(span.events, []);
}

test("public alias and native hierarchy retain real sampler-created contexts", async () => {
  assert.equal(MastraInstrumentor, RespanMastraExporter);
  const { instance, observability } = setup();
  const root = instance.startSpan({
    name: "root",
    type: SpanType.AGENT_RUN,
    attributes: { agentId: "native" },
  });
  const generation = root.createChildSpan({
    name: "generation",
    type: SpanType.MODEL_GENERATION,
  });
  const step = generation.createChildSpan({
    name: "step",
    type: SpanType.MODEL_STEP,
  });
  const inference = step.createChildSpan({
    name: "inference",
    type: SpanType.MODEL_INFERENCE,
  });
  inference.end();
  step.end();
  generation.end();
  root.end();
  assert.equal(spans().length, 4);
  for (const [child, parent] of [
    ["generation", "root"],
    ["step", "generation"],
    ["inference", "step"],
  ]) {
    const c = spans().find((s) => s.name === child),
      p = spans().find((s) => s.name === parent);
    assert.equal(c.parentSpanContext.spanId, p.spanContext().spanId);
    assert.equal(c.spanContext().traceId, p.spanContext().traceId);
  }
  assert.equal(attrs("root")[SpanAttributes.TRACELOOP_SPAN_KIND], undefined);
  await observability.shutdown();
});

test("DROP prevents content conversion with genuine native SDK spans", async () => {
  decision = SamplingDecision.NOT_RECORD;
  let touched = 0;
  const poison = {
    secret: "private",
    toJSON() {
      touched++;
      throw Error("toJSON");
    },
  };
  const { instance, observability } = setup();
  native(instance, "dropped", poison, poison);
  assert.equal(touched, 0);
  assert.equal(spans().length, 0);
  await observability.shutdown();
});

test("RECORD_ONLY is not admitted to payload conversion", async () => {
  decision = SamplingDecision.RECORD;
  const { instance, observability } = setup();
  native(instance);
  // SimpleSpanProcessor intentionally exports only sampled spans, but the actual SDK span records.
  assert.equal(started.length, 1);
  assert.equal(
    started[0].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT],
    undefined,
  );
  await observability.shutdown();
});

for (const gate of [
  "constructor",
  "context",
  "respan-env",
  "traceloop-env",
  "ancestor",
  "language-model",
  "suppression",
]) {
  test(`immutable ${gate} veto before native body traversal`, async () => {
    let touched = 0;
    const poison = {
      secret: "private",
      toJSON() {
        touched++;
        throw Error("toJSON");
      },
    };
    const { instance, observability } = setup(
      gate === "constructor" ? { traceContent: false } : {},
    );
    let ctx = context.active(),
      parent;
    if (gate === "context")
      ctx = ctx.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false);
    if (gate === "suppression") ctx = suppressTracing(ctx);
    if (gate === "language-model")
      ctx = ctx.setValue(
        createContextKey("suppress_language_model_instrumentation"),
        true,
      );
    if (gate === "respan-env") process.env.RESPAN_TRACE_CONTENT = "false";
    if (gate === "traceloop-env") process.env.TRACELOOP_TRACE_CONTENT = "false";
    if (gate === "ancestor") {
      parent = trace.getTracer("fixture").startSpan("external-parent");
      parent.setAttribute("allow_trace_content", false);
      ctx = trace.setSpan(ctx, parent);
    }
    const live = context.with(ctx, () =>
      instance.startSpan({ name: gate, type: SpanType.GENERIC, input: poison }),
    );
    delete process.env.RESPAN_TRACE_CONTENT;
    delete process.env.TRACELOOP_TRACE_CONTENT;
    parent?.setAttribute("allow_trace_content", true);
    live.end({ output: poison });
    assert.equal(touched, 0);
    if (["suppression", "language-model"].includes(gate))
      assert.equal(spans().length, 0);
    else {
      assert.equal(spans().length, 1);
      noContent(spans()[0]);
    }
    await observability.shutdown();
    parent?.end();
  });
}

test("late actual span veto is immutable and guards queued attrs events and status", async () => {
  const { instance, observability } = setup();
  const nativeSpan = instance.startSpan({
    name: "late",
    type: SpanType.GENERIC,
    input: { secret: "early" },
  });
  const actual = started[0];
  actual.setAttribute("allow_trace_content", false);
  // Observe false and then attempt to re-enable it before export.
  void actual.attributes;
  actual.setAttribute("allow_trace_content", true);
  nativeSpan.end({ output: { secret: "late" } });
  const readable = spans()[0];
  assert.equal(readable, actual);
  noContent(readable);
  readable.attributes = {
    ...readable.attributes,
    [SpanAttributes.TRACELOOP_ENTITY_OUTPUT]: "secret",
  };
  readable.events = [{ name: "secret", attributes: { secret: "secret" } }];
  readable.status = { code: SpanStatusCode.ERROR, message: "secret" };
  noContent(readable);
  assert.equal(readable.status.code, SpanStatusCode.ERROR);
  for (const key of ["attributes", "events", "status"])
    assert.throws(() =>
      Object.defineProperty(readable, key, { value: "unsafe" }),
    );
  await observability.shutdown();
});

test("observed native ancestor veto remains in force for its descendants", async () => {
  const { instance, observability } = setup();
  const root = instance.startSpan({
    name: "ancestor-root",
    type: SpanType.GENERIC,
  });
  const parent = started[0];
  parent.setAttribute("allow_trace_content", false);
  void parent.attributes;
  parent.setAttribute("allow_trace_content", true);
  const child = root.createChildSpan({
    name: "private-child",
    type: SpanType.GENERIC,
    input: { secret: true },
  });
  child.end({ output: { secret: true } });
  root.end();
  for (const span of spans()) noContent(span);
  await observability.shutdown();
});

test("full native messages, history, schemas, outputs and scalars are preserved", async () => {
  const { instance, observability } = setup();
  const history = Array.from({ length: 80 }, (_, i) => ({
    role: "assistant",
    content: [
      { type: "text", text: `text${i}` },
      {
        type: "tool-call",
        toolCallId: `history${i}`,
        toolName: "weather",
        input: { zero: 0, no: false, empty: "", nil: null },
      },
    ],
  }));
  const schema = {
    type: "object",
    properties: Object.fromEntries(
      Array.from({ length: 80 }, (_, i) => [
        `field${i}`,
        { type: "string", description: `description${i}` },
      ]),
    ),
  };
  const input = {
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "look" },
          { type: "image", image: "https://example.test/image.png" },
        ],
      },
      ...history,
    ],
  };
  const output = {
    text: "",
    reasoning: [{ type: "reasoning", text: "reason" }],
    files: [{ mediaType: "image/png", base64: "native" }],
    toolCalls: [
      {
        toolCallId: "call1",
        toolName: "weather",
        input: { zero: 0, no: false, empty: "", nil: null },
      },
    ],
    object: { zero: 0, no: false, empty: "", nil: null },
  };
  const span = instance.startSpan({
    name: "full",
    type: SpanType.MODEL_GENERATION,
    input,
    attributes: {
      model: "provider/model",
      provider: "native",
      parameters: { temperature: 0, topP: 0, maxOutputTokens: 0 },
      tools: [
        {
          type: "function",
          name: "weather",
          description: "schema",
          parameters: schema,
        },
      ],
    },
  });
  span.end({
    output,
    attributes: {
      usage: {
        inputTokens: 0,
        outputTokens: 0,
        inputDetails: { cacheRead: 0 },
      },
      finishReason: "tool-calls",
    },
  });
  const a = attrs("full");
  assert.deepEqual(JSON.parse(a[SpanAttributes.TRACELOOP_ENTITY_INPUT]), input);
  assert.deepEqual(
    JSON.parse(a[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]),
    output,
  );
  assert.equal(
    JSON.parse(a[`${SpanAttributes.LLM_PROMPTS}.80.tool_calls`])[0].id,
    "history79",
  );
  assert.equal(
    JSON.parse(a[`${SpanAttributes.LLM_PROMPTS}.0.content`])[1].type,
    "image",
  );
  assert.deepEqual(
    JSON.parse(a[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0].function.parameters,
    schema,
  );
  assert.equal(a[SpanAttributes.LLM_REQUEST_TEMPERATURE], 0);
  assert.equal(a[SpanAttributes.LLM_USAGE_TOTAL_TOKENS], undefined);
  assert.equal(a[SpanAttributes.LLM_REQUEST_MODEL], "provider/model");
  assert.equal(a[`${SpanAttributes.LLM_COMPLETIONS}.0.content`], "");
  assert.equal(
    JSON.parse(a[`${SpanAttributes.LLM_COMPLETIONS}.0.tool_calls`])[0].id,
    "call1",
  );
  for (const k of [
    "tools",
    "tool_calls",
    "model",
    "respan.span.tools",
    "respan.span.tool_calls",
  ])
    assert.equal(a[k], undefined);
  await observability.shutdown();
});

test("native embedding vectors and null/false/zero/empty outputs have no truncation", async () => {
  const { instance, observability } = setup();
  const vector = Array.from({ length: 5003 }, (_, i) => i / 5003);
  const embedding = instance.startSpan({
    name: "vectors",
    type: SpanType.RAG_EMBEDDING,
    input: ["first", "second"],
    attributes: { model: "native-embedding", dimensions: 5003 },
  });
  embedding.end({ output: { vectors: [vector, vector] } });
  assert.deepEqual(
    JSON.parse(attrs("vectors")[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
      .vectors[1],
    vector,
  );
  for (const [i, value] of [null, false, 0, ""].entries()) {
    native(instance, `scalar${i}`, value, value);
    assert.equal(
      JSON.parse(attrs(`scalar${i}`)[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]),
      value,
    );
  }
  await observability.shutdown();
});

test("serializer skips getters toJSON proxies and preserves cycles without touching SDK results", async () => {
  const { instance, observability } = setup();
  let touched = 0;
  const input = {
    safe: null,
    secret: "private",
    toJSON() {
      touched++;
      throw Error("toJSON");
    },
  };
  Object.defineProperty(input, "getter", {
    enumerable: true,
    get() {
      touched++;
      throw Error("native getter");
    },
  });
  const proxy = new Proxy(
    {},
    {
      ownKeys() {
        touched++;
        throw Error("proxy");
      },
    },
  );
  input.proxy = proxy;
  input.self = input;
  const output = { same: input };
  const span = native(instance, "safe-copy", input, output);
  assert.deepEqual(
    JSON.parse(attrs("safe-copy")[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]),
    span.output,
  );
  assert.equal(touched, 4);
  assert.deepEqual(
    JSON.parse(attrs("safe-copy")[SpanAttributes.TRACELOOP_ENTITY_INPUT]),
    span.input,
  );
  const before = touched;
  contentAttributes({ type: "generic", input, output });
  assert.equal(touched, before);
  await observability.shutdown();
});

test("native error status does not invent HTTP status completions or child errors", async () => {
  const { instance, observability } = setup();
  const root = instance.startSpan({
    name: "error-root",
    type: SpanType.GENERIC,
  });
  const child = root.createChildSpan({
    name: "successful-child",
    type: SpanType.GENERIC,
  });
  child.end({ output: false });
  const error = new Error("native failure");
  root.error({ error, endSpan: true });
  assert.equal(error.message, "native failure");
  const failed = spans().find((s) => s.name === "error-root");
  assert.equal(failed.status.code, SpanStatusCode.ERROR);
  assert.equal(failed.status.message, "native failure");
  assert.equal(failed.attributes["http.status_code"], undefined);
  assert.equal(
    failed.attributes[`${SpanAttributes.LLM_COMPLETIONS}.0.content`],
    undefined,
  );
  assert.equal(
    spans().find((s) => s.name === "successful-child").status.code,
    SpanStatusCode.UNSET,
  );
  await observability.shutdown();
});

test("native exclusions, duplicate exporter notifications and deactivate drain only once", async () => {
  const { plugin, instance, observability } = setup({
    excludeSpanTypes: ["model_step"],
  });
  const root = instance.startSpan({
    name: "lifecycle-root",
    type: SpanType.GENERIC,
  });
  const excluded = root.createChildSpan({
    name: "excluded",
    type: SpanType.MODEL_STEP,
  });
  const child = excluded.createChildSpan({
    name: "through-excluded",
    type: SpanType.MODEL_INFERENCE,
  });
  child.end();
  excluded.end();
  root.end();
  assert.equal(spans().length, 2);
  assert.equal(
    spans().find((s) => s.name === "through-excluded").parentSpanContext.spanId,
    spans()
      .find((s) => s.name === "lifecycle-root")
      .spanContext().spanId,
  );
  const pending = instance.startSpan({
    name: "inflight",
    type: SpanType.GENERIC,
    input: "private pending",
  });
  plugin.deactivate();
  pending.end({ output: "late" });
  assert.equal(spans().filter((s) => s.name === "inflight").length, 1);
  assert.equal(
    attrs("inflight")[SpanAttributes.TRACELOOP_ENTITY_INPUT],
    undefined,
  );
  native(instance, "disabled");
  assert.equal(attrs("disabled"), undefined);
  plugin.activate();
  native(instance, "reactivated");
  assert.ok(attrs("reactivated"));
  await observability.shutdown();
});

test("released native Agent generate and stream preserve results and callbacks", async () => {
  const { plugin, observability } = setup();
  const agent = new Agent({
    id: "fixture-agent",
    name: "Fixture Agent",
    instructions: "Reply",
    model: createMockModel({ mockText: "native released result" }),
  });
  const mastra = new Mastra({ agents: { agent }, observability });
  let finish;
  const result = await mastra.getAgent("agent").generate("Generate", {
    onFinish: (r) => {
      finish = r;
    },
  });
  assert.equal(result.text, "native released result");
  assert.ok(finish);
  const stream = await mastra.getAgent("agent").stream("Stream");
  const streamObject = stream;
  let text = "";
  for await (const chunk of stream.textStream) text += chunk;
  assert.equal(text, "native released result");
  assert.equal(stream, streamObject);
  assert.ok(
    spans().some(
      (s) => s.attributes[RespanSpanAttributes.RESPAN_LOG_TYPE] === "chat",
    ),
  );
  assert.equal(
    spans().some((s) =>
      s.attributes[RespanSpanAttributes.RESPAN_METADATA]?.includes(
        "model_chunk",
      ),
    ),
    false,
  );
  await mastra.shutdown();
  assert.equal(plugin.name, "mastra");
});

test("multiple exporters and concurrent ambient workflows retain independent parents", async () => {
  const first = setup(),
    second = setup();
  await Promise.all(
    [first, second].map(async ({ instance }, i) => {
      const parent = trace.getTracer("fixture").startSpan(`owner${i}`);
      await context.with(trace.setSpan(context.active(), parent), async () => {
        await Promise.resolve();
        native(instance, `owned${i}`, { owner: i }, { owner: i });
      });
      const child = spans().find((s) => s.name === `owned${i}`);
      assert.equal(child.parentSpanContext.spanId, parent.spanContext().spanId);
      parent.end();
    }),
  );
  await first.observability.shutdown();
  await second.observability.shutdown();
});

test("queued readable spans recheck environment and retain an observed false ceiling", async () => {
  const { instance, observability } = setup();
  native(instance, "queued");
  const queued = spans()[0];
  assert.ok(queued.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]);
  process.env.RESPAN_TRACE_CONTENT = "false";
  noContent(queued);
  delete process.env.RESPAN_TRACE_CONTENT;
  queued.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = "late payload";
  noContent(queued);
  await observability.shutdown();
});

test("native tool executions retain canonical arguments and invocation identity", async () => {
  const { instance, observability } = setup();
  const tool = instance.startSpan({
    name: "tool: weather",
    entityName: "weather",
    type: SpanType.TOOL_CALL,
    input: { city: "Tokyo", empty: "", nil: null },
    attributes: { toolCallId: "native-tool-id" },
  });
  tool.end({ output: { temperature: 0 } });
  const a = attrs("tool: weather");
  assert.equal(a["gen_ai.tool.call.id"], "native-tool-id");
  assert.deepEqual(JSON.parse(a[SpanAttributes.TRACELOOP_ENTITY_INPUT]), {
    name: "weather",
    arguments: { city: "Tokyo", empty: "", nil: null },
  });
  assert.deepEqual(JSON.parse(a[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]), {
    temperature: 0,
  });
  assert.equal(a[`${SpanAttributes.LLM_COMPLETIONS}.0.tool_calls`], undefined);
  await observability.shutdown();
});

test("late owned parent veto is observed before child conversion", async () => {
  const { instance, observability } = setup();
  const root = instance.startSpan({
    name: "late-owner",
    type: SpanType.GENERIC,
  });
  const parent = started[0];
  const child = root.createChildSpan({
    name: "late-owned-child",
    type: SpanType.GENERIC,
    input: { secret: true },
  });
  parent.setAttribute("allow_trace_content", false);
  child.end({ output: { secret: true } });
  parent.setAttribute("allow_trace_content", true);
  root.end();
  for (const span of spans()) noContent(span);
  await observability.shutdown();
});

test("denied actual span rejects arbitrary late payload attrs before traversal", async () => {
  const { instance, observability } = setup({ traceContent: false });
  native(instance, "deny-arbitrary");
  const readable = spans()[0];
  let touched = 0;
  const poison = new Proxy(
    {},
    {
      ownKeys() {
        touched++;
        throw Error("proxy traversal");
      },
    },
  );
  readable.attributes = {
    ...readable.attributes,
    "late.secret": poison,
    custom_payload: { nested: poison },
  };
  readable.events = [poison];
  readable.status = { code: SpanStatusCode.ERROR, message: poison };
  assert.equal(readable.attributes["late.secret"], undefined);
  assert.equal(readable.attributes.custom_payload, undefined);
  noContent(readable);
  assert.equal(touched, 0);
  assert.equal(readable.status.code, SpanStatusCode.ERROR);
  await observability.shutdown();
});

test("agent workflow and tool attrs do not become synthetic model work", async () => {
  const { instance, observability } = setup();
  for (const type of [
    SpanType.AGENT_RUN,
    SpanType.WORKFLOW_RUN,
    SpanType.TOOL_CALL,
  ]) {
    const span = instance.startSpan({
      name: `common-${type}`,
      type,
      attributes: {
        model: "must-not-map",
        provider: "must-not-map",
        usage: { inputTokens: 100 },
        parameters: { temperature: 1 },
      },
    });
    span.end();
    const a = attrs(`common-${type}`);
    assert.equal(a[SpanAttributes.LLM_REQUEST_MODEL], undefined);
    assert.equal(a[SpanAttributes.LLM_SYSTEM], undefined);
    assert.equal(a["gen_ai.usage.input_tokens"], undefined);
    assert.equal(a[SpanAttributes.LLM_REQUEST_TEMPERATURE], undefined);
  }
  await observability.shutdown();
});

test("unrelated exporter suppression cannot veto an already admitted native span", async () => {
  const { instance, observability } = setup();
  const input = { native: "input" },
    output = { native: "output" };
  const live = instance.startSpan({
    name: "admitted-before-exporter",
    type: SpanType.GENERIC,
    input,
  });
  const actual = started[0];
  context.with(suppressTracing(context.active()), () => live.end({ output }));
  const exported = spans().find((s) => s.name === "admitted-before-exporter");
  assert.equal(exported, actual);
  assert.deepEqual(
    JSON.parse(exported.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]),
    input,
  );
  assert.deepEqual(
    JSON.parse(exported.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]),
    output,
  );
  const failing = instance.startSpan({
    name: "admitted-error",
    type: SpanType.GENERIC,
  });
  const error = new Error("native admitted failure");
  context.with(suppressTracing(context.active()), () =>
    failing.error({ error, endSpan: true }),
  );
  assert.equal(
    spans().find((s) => s.name === "admitted-error").status.message,
    error.message,
  );
  await observability.shutdown();
});
