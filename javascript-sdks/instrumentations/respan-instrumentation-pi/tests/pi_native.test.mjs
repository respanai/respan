import assert from "node:assert/strict";
import test from "node:test";
import { context, ROOT_CONTEXT, trace } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  BasicTracerProvider,
  AlwaysOffSampler,
} from "@opentelemetry/sdk-trace-base";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import { PiInstrumentor, PiSessionTracer } from "../dist/index.js";
import { nativeFixture } from "./native_runtime.mjs";

const defaultProvider = new BasicTracerProvider();
trace.setGlobalTracerProvider(defaultProvider);
const manager = new AsyncLocalStorageContextManager().enable();
context.setGlobalContextManager(manager);
test.after(async () => {
  context.disable();
  manager.disable();
  trace.disable();
  await defaultProvider.shutdown();
});
const metadata = (span) =>
  JSON.parse(span.attributes["respan.metadata"] ?? "{}");
const byType = (spans, type) =>
  spans.filter((span) => span.attributes["respan.entity.log_type"] === type);
const noContent = (span) => {
  const attrs = span.attributes;
  assert.equal(attrs["traceloop.entity.input"], undefined);
  assert.equal(attrs["traceloop.entity.output"], undefined);
  assert.equal(attrs["llm.request.functions"], undefined);
  assert.equal(attrs["respan.metadata"], undefined);
  assert.equal(attrs["error.message"], undefined);
  assert.ok(
    !Object.keys(attrs).some((key) =>
      /gen_ai\.(prompt|completion)\./.test(key),
    ),
  );
  assert.deepEqual(span.events, []);
  assert.equal(span.status.message, undefined);
};
async function run(plan, options = {}, callback) {
  const spans = [];
  const instrumentor = new PiInstrumentor({
    traceScope: "run",
    emit: (span) => spans.push(span),
    ...options,
  });
  instrumentor.activate();
  const fixture = await nativeFixture({ instrumentor, ...plan });
  try {
    if (callback) await callback(fixture, spans, instrumentor);
    else await fixture.session.prompt("native prompt");
    return {
      spans,
      requests: fixture.requests,
      messages: fixture.session.messages,
      output: fixture.session.getLastAssistantText(),
    };
  } finally {
    await fixture.close();
    instrumentor.deactivate();
  }
}

for (const extension of [false, true])
  test(`native ${extension ? "extension" : "attach"}: streaming output, usage, IDs, parentage and system context`, async () => {
    const result = await run({
      extension,
      responses: [
        {
          text: "native streamed final",
          usage: { prompt_tokens: 19, completion_tokens: 7, total_tokens: 26 },
        },
      ],
    });
    assert.equal(result.requests.length, 1);
    assert.equal(result.output, "native streamed final");
    const [chat] = byType(result.spans, "chat");
    const [agent] = byType(result.spans, "agent");
    assert.equal(chat.parentSpanContext.spanId, agent.spanContext().spanId);
    assert.equal(chat.spanContext().traceId, agent.spanContext().traceId);
    assert.equal(chat.attributes["gen_ai.request.model"], "native-audit");
    assert.equal(chat.attributes["gen_ai.system"], "audit");
    assert.equal(chat.attributes["gen_ai.usage.input_tokens"], 19);
    assert.equal(chat.attributes["gen_ai.usage.output_tokens"], 7);
    assert.equal(chat.attributes["llm.usage.total_tokens"], 26);
    assert.equal(metadata(chat).response_id, "native-response-1");
    const prompts = JSON.parse(chat.attributes["traceloop.entity.input"]);
    assert.ok(
      prompts.some(
        (item) =>
          item.role === "system" &&
          item.content.includes("native system instructions"),
      ),
    );
    assert.equal(
      JSON.parse(chat.attributes["traceloop.entity.output"]).content,
      result.output,
    );
    assert.ok(
      Object.keys(chat.attributes).every(
        (key) => !key.startsWith("respan.metadata."),
      ),
    );
  });

test("native tool execution retains full schema, 75-message history, current calls and full 5001-vector structured result", async () => {
  const vector = Array.from({ length: 5001 }, (_, i) => i / 5001);
  const details = {
    vector,
    falseValue: false,
    zero: 0,
    empty: "",
    nullable: null,
  };
  const resultObject = {
    content: [{ type: "text", text: "native tool output" }],
    details,
    structuredContent: { accepted: false, count: 0, nullable: null },
  };
  let nativeCallback;
  let nativeResult;
  const tool = {
    name: "inspect",
    label: "Inspect",
    description: "d".repeat(25000),
    parameters: {
      type: "object",
      properties: {
        value: { type: "integer" },
        huge: { type: "string", description: "schema".repeat(5000) },
      },
      required: ["value"],
    },
    async execute(_id, _args, _signal, onUpdate) {
      nativeCallback = onUpdate;
      nativeResult = resultObject;
      onUpdate?.({
        content: [{ type: "text", text: "native partial" }],
        details: { progress: 0 },
      });
      return resultObject;
    },
  };
  const history = Array.from({ length: 75 }, (_, i) => ({
    role: "user",
    content: `native history ${i}`,
    timestamp: i + 1,
  }));
  const result = await run({
    tools: [tool],
    history,
    responses: [
      {
        tools: [
          { id: "native-call-1", name: "inspect", arguments: { value: 0 } },
        ],
      },
      { text: "native tool final" },
    ],
  });
  assert.equal(typeof nativeCallback, "function");
  assert.equal(nativeResult, resultObject);
  assert.equal(result.requests.length, 2);
  const [first, second] = byType(result.spans, "chat");
  const [execution] = byType(result.spans, "tool");
  const prompts = JSON.parse(first.attributes["traceloop.entity.input"]);
  assert.equal(
    prompts.filter((item) => item.content.startsWith("native history ")).length,
    75,
  );
  const schemas = JSON.parse(first.attributes["llm.request.functions"]);
  assert.equal(
    schemas.find((item) => item.name === "inspect").parameters.properties.huge
      .description,
    tool.parameters.properties.huge.description,
  );
  assert.equal(
    JSON.parse(first.attributes["gen_ai.completion.0.tool_calls"])[0].id,
    "native-call-1",
  );
  assert.equal(second.attributes["gen_ai.completion.0.tool_calls"], undefined);
  const secondPrompts = JSON.parse(second.attributes["traceloop.entity.input"]);
  assert.ok(
    secondPrompts.some((item) => item.tool_calls?.[0]?.id === "native-call-1"),
  );
  assert.equal(execution.attributes["gen_ai.tool.call.id"], "native-call-1");
  assert.deepEqual(
    JSON.parse(execution.attributes["traceloop.entity.output"]),
    resultObject,
  );
});

for (const gate of [
  "constructor",
  "context",
  "respan-env",
  "traceloop-env",
  "general-suppression",
  "unsampled",
])
  test(`native capture admission: ${gate}`, async () => {
    let parent;
    let ctx = ROOT_CONTEXT;
    if (gate === "context")
      ctx = ctx.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false);
    if (gate === "general-suppression") ctx = suppressTracing(ctx);
    if (gate === "unsampled") {
      parent = new BasicTracerProvider({ sampler: new AlwaysOffSampler() })
        .getTracer("native-test")
        .startSpan("parent");
      ctx = trace.setSpan(ctx, parent);
    }
    const envName =
      gate === "respan-env"
        ? "RESPAN_TRACE_CONTENT"
        : gate === "traceloop-env"
          ? "TRACELOOP_TRACE_CONTENT"
          : undefined;
    const previous = envName && process.env[envName];
    if (envName) process.env[envName] = "false";
    try {
      const result = await context.with(ctx, () =>
        run({}, { traceContent: gate !== "constructor" }),
      );
      assert.equal(result.output, "native final");
      if (["general-suppression", "unsampled"].includes(gate))
        assert.equal(result.spans.length, 0);
      else {
        assert.equal(result.spans.length, 2);
        result.spans.forEach(noContent);
      }
    } finally {
      if (envName) {
        if (previous === undefined) delete process.env[envName];
        else process.env[envName] = previous;
      }
      parent?.end();
    }
  });

test("native readable veto survives replacements, queued export, deactivation, foreign activation and parent false then true", async () => {
  const provider = new BasicTracerProvider();
  const parent = provider
    .getTracer("native-test")
    .startSpan("parent", { attributes: { allow_trace_content: true } });
  const result = await context.with(trace.setSpan(ROOT_CONTEXT, parent), () =>
    run({}, {}, async (fixture, spans, instrumentor) => {
      await fixture.session.prompt("native secret");
      instrumentor.deactivate();
      parent.setAttribute("allow_trace_content", false);
      for (const span of spans) {
        span.attributes = {
          ...span.attributes,
          "custom.payload": "late secret",
          "openinference.secret": "late secret",
          "traceloop.entity.input": "late secret",
          "respan.metadata": JSON.stringify({ late: "secret" }),
          "error.message": "secret",
        };
        span.events = [
          {
            name: "exception",
            attributes: { "exception.message": "secret" },
            time: [0, 0],
          },
        ];
        span.status = { code: 2, message: "secret" };
        noContent(span);
        assert.equal(span.attributes["custom.payload"], undefined);
        assert.equal(span.attributes["openinference.secret"], undefined);
      }
      parent.setAttribute("allow_trace_content", true);
      instrumentor.activate();
      const foreign = new PiInstrumentor();
      foreign.activate();
      foreign.deactivate();
      await context.with(suppressTracing(ROOT_CONTEXT), async () => {
        spans.forEach(noContent);
      });
      spans.forEach(noContent);
    }),
  );
  assert.equal(result.output, "native final");
  result.spans.forEach(noContent);
  parent.end();
});

test("passive capture never invokes serializers, getters, iterators or proxy traps", () => {
  let reads = 0;
  const hostile = {
    get secret() {
      reads++;
      throw new Error("getter");
    },
    toJSON() {
      reads++;
      throw new Error("serializer");
    },
    toString() {
      reads++;
      throw new Error("stringifier");
    },
  };
  const proxy = new Proxy(
    {},
    {
      ownKeys() {
        reads++;
        throw new Error("proxy");
      },
      get() {
        reads++;
        throw new Error("proxy");
      },
    },
  );
  const array = [hostile, proxy];
  array[Symbol.iterator] = () => {
    reads++;
    throw new Error("iterator");
  };
  for (const traceContent of [true, false]) {
    const spans = [];
    const tracer = new PiSessionTracer({
      traceContent,
      emit: (span) => spans.push(span),
    });
    tracer.setToolDefinitions([{ name: "hostile", parameters: hostile }]);
    tracer.onBeforeAgentStart({ prompt: hostile });
    tracer.onContext([{ role: "user", content: array }]);
    tracer.onToolExecutionStart({
      toolCallId: "hostile-id",
      toolName: "hostile",
      args: hostile,
    });
    tracer.onToolExecutionEnd({
      toolCallId: "hostile-id",
      toolName: "hostile",
      result: hostile,
    });
    tracer.onAgentEnd();
    if (!traceContent) spans.forEach(noContent);
  }
  assert.equal(reads, 0);
});

test("native error and abort outcomes match bare SDK behavior", async () => {
  const plan = {
    responses: [{ status: 400, error: "controlled native error" }],
  };
  const bare = await nativeFixture(plan);
  let bareMessage;
  try {
    await bare.session.prompt("native prompt");
    bareMessage = bare.session.messages.findLast(
      (item) => item.role === "assistant",
    );
  } finally {
    await bare.close();
  }
  const result = await run(plan);
  const message = result.messages.findLast((item) => item.role === "assistant");
  assert.equal(message.stopReason, bareMessage.stopReason);
  assert.equal(message.errorMessage, bareMessage.errorMessage);
  assert.equal(byType(result.spans, "chat")[0].status.code, 2);
  const aborted = await run(
    { responses: [{ text: "partial", hold: true }] },
    {},
    async (fixture) => {
      const promise = fixture.session.prompt("abort native");
      await new Promise((resolve) => {
        const unsubscribe = fixture.session.subscribe((event) => {
          if (event.type === "message_update") {
            unsubscribe();
            resolve();
          }
        });
      });
      await fixture.session.abort();
      await promise;
    },
  );
  assert.equal(
    aborted.messages.findLast((item) => item.role === "assistant").stopReason,
    "aborted",
  );
  assert.equal(byType(aborted.spans, "chat")[0].status.code, 2);
});

for (const extension of [false, true])
  test(`native ${extension ? "extension" : "attach"}: manual compaction lifecycle`, async () => {
    const history = Array.from({ length: 75 }, (_, i) => ({
      role: "user",
      content: `compaction history ${i} ` + "context ".repeat(80),
      timestamp: i + 1,
    }));
    await run(
      {
        extension,
        history,
        responses: [
          { text: "before compaction" },
          { text: "native compaction summary" },
        ],
      },
      {},
      async (fixture, spans) => {
        await fixture.session.prompt("compact this native context");
        const result = await fixture.session.compact();
        assert.ok(result.summary.includes("native compaction summary"));
        const task = byType(spans, "task").find(
          (span) => span.name === "pi.compaction",
        );
        assert.ok(task);
        assert.equal(
          JSON.parse(task.attributes["traceloop.entity.output"]).summary,
          result.summary,
        );
      },
    );
  });

test("native extension: branch summary, resumed prompts and session trace scope", async () => {
  await run(
    {
      extension: true,
      responses: [
        { text: "first branch" },
        { text: "second branch" },
        { text: "native branch summary" },
        { text: "resumed branch" },
      ],
    },
    { traceScope: "session" },
    async (fixture, spans) => {
      await fixture.session.prompt("first native branch");
      await fixture.session.prompt("second native branch");
      const target = fixture.session.getUserMessagesForForking()[0].entryId;
      const result = await fixture.session.navigateTree(target, {
        summarize: true,
      });
      assert.equal(result.cancelled, false);
      const task = byType(spans, "task").find(
        (span) => span.name === "pi.branch_summary",
      );
      assert.ok(task);
      assert.ok(
        JSON.parse(task.attributes["traceloop.entity.output"]).summary.includes(
          "native branch summary",
        ),
      );
      await fixture.session.prompt("resume native branch");
      assert.equal(
        new Set(spans.map((span) => span.spanContext().traceId)).size,
        1,
      );
    },
  );
});

test("native observers do not read customer-defined session or context accessors", () => {
  let reads = 0;
  const session = {
    subscribe() {
      return () => {};
    },
    get messages() {
      reads++;
      return [];
    },
    get model() {
      reads++;
      return {};
    },
    get sessionId() {
      reads++;
      return "secret";
    },
    get sessionFile() {
      reads++;
      return "secret";
    },
    get sessionManager() {
      reads++;
      return {};
    },
    get agent() {
      reads++;
      return {};
    },
  };
  const instrumentor = new PiInstrumentor({ traceContent: false });
  instrumentor.activate();
  instrumentor.attach(session)();
  instrumentor.deactivate();
  assert.equal(reads, 0);
});

test("native actual root sampler rejects before content conversion without a parent span", async () => {
  const provider = new BasicTracerProvider({ sampler: new AlwaysOffSampler() });
  trace.disable();
  trace.setGlobalTracerProvider(provider);
  try {
    const result = await run({});
    assert.equal(result.output, "native final");
    assert.equal(result.spans.length, 0);
  } finally {
    trace.disable();
    trace.setGlobalTracerProvider(defaultProvider);
    await provider.shutdown();
  }
});

test("native custom sampler sees each exact owned span once, with deterministic trace and parent IDs", async () => {
  const calls = [];
  const sampler = {
    shouldSample(ctx, traceId, name, kind, attributes) {
      calls.push({
        traceId,
        name,
        kind,
        attributes,
        parent: trace.getSpanContext(ctx),
      });
      return { decision: name === "pi.chat" ? 0 : 2 };
    },
    toString() {
      return "controlled sampler";
    },
  };
  const provider = new BasicTracerProvider({ sampler });
  trace.disable();
  trace.setGlobalTracerProvider(provider);
  try {
    const result = await run({}, { traceScope: "session" });
    assert.equal(result.output, "native final");
    assert.equal(result.spans.length, 1);
    assert.equal(calls.length, 2);
    const [agent] = result.spans;
    assert.equal(calls[0].traceId, agent.spanContext().traceId);
    assert.equal(calls[0].name, agent.name);
    assert.equal(calls[0].kind, 0);
    assert.equal(calls[0].parent, undefined);
    assert.equal(calls[1].name, "pi.chat");
    assert.equal(calls[1].traceId, agent.spanContext().traceId);
    assert.equal(calls[1].parent.spanId, agent.spanContext().spanId);
    assert.equal(calls[1].parent.traceFlags, 1);
  } finally {
    trace.disable();
    trace.setGlobalTracerProvider(defaultProvider);
    await provider.shutdown();
  }
});

test("native root sampler rejects before catalog work; unknown providers fail closed", async () => {
  const provider = new BasicTracerProvider({ sampler: new AlwaysOffSampler() });
  trace.disable();
  trace.setGlobalTracerProvider(provider);
  const fixture = await nativeFixture();
  let catalogs = 0;
  const original = fixture.session.getAllTools;
  fixture.session.getAllTools = function () {
    catalogs++;
    return original.call(this);
  };
  const spans = [];
  const instrumentor = new PiInstrumentor({ emit: (span) => spans.push(span) });
  instrumentor.activate();
  const detach = instrumentor.attach(fixture.session);
  try {
    await fixture.session.prompt("native no catalog");
    assert.equal(catalogs, 0);
    assert.equal(spans.length, 0);
    trace.disable();
    trace.setGlobalTracerProvider({
      getTracer() {
        return {
          startSpan() {
            throw new Error("unexpected native span probe");
          },
        };
      },
    });
    await fixture.session.prompt("native unknown provider");
    assert.equal(spans.length, 0);
    assert.equal(catalogs, 0);
  } finally {
    detach();
    instrumentor.deactivate();
    await fixture.close();
    trace.disable();
    trace.setGlobalTracerProvider(defaultProvider);
    await provider.shutdown();
  }
});
