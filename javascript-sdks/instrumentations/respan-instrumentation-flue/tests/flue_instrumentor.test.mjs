import assert from "node:assert/strict";
import test from "node:test";
import { context, createContextKey, trace } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  AlwaysOffSampler,
  BasicTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import { FlueInstrumentor } from "../dist/index.js";
import {
  BODY,
  VECTOR,
  current,
  nativeRuntime,
  runtime,
} from "./native_runtime.mjs";

const exported = [];
let dropped = false;
const provider = new BasicTracerProvider({
  sampler: {
    shouldSample(...args) {
      return dropped
        ? new AlwaysOffSampler().shouldSample(...args)
        : { decision: 2 };
    },
  },
  spanLimits: {
    attributeValueLengthLimit: Infinity,
    attributeCountLimit: 4096,
  },
  spanProcessors: [
    {
      onStart() {},
      onEnd(span) {
        exported.push(span);
      },
      async forceFlush() {},
      async shutdown() {},
    },
  ],
});
trace.setGlobalTracerProvider(provider);
context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
test.after(async () => {
  await provider.shutdown();
  context.disable();
  trace.disable();
});
const contentEntries = (span) =>
  Object.entries(span.attributes).filter(
    ([key]) =>
      /(?:prompt|completion|entity\.(?:input|output)|functions|exception|error\.message|description|arguments|result)/.test(
        key,
      ) && !/usage|tokens/.test(key),
  );
const ownSpans = () =>
  exported.filter(
    (span) => span.instrumentationScope.name === "@respan/instrumentation-flue",
  );
async function scenario(options, run, instrumentorOptions = {}) {
  exported.length = 0;
  const instrumentor = new FlueInstrumentor({
    runtimeModule: runtime,
    ...instrumentorOptions,
  });
  await instrumentor.activate();
  let native;
  try {
    native = await nativeRuntime(options);
    await run(native, instrumentor);
  } finally {
    await native?.close();
    await instrumentor.deactivate();
  }
  return ownSpans();
}

test("real runtime preserves complete model history, schema, native tool IDs, vector5001 and zero usage", async () => {
  const spans = await scenario({ tool: true, large: true }, async (native) => {
    const handle = native.prompt();
    assert.equal(typeof handle.abort, "function");
    assert.ok(handle.signal instanceof AbortSignal);
    assert.equal(typeof handle.catch, "function");
    assert.equal(typeof handle.finally, "function");
    const answer = await handle;
    assert.equal(answer.text, BODY);
    assert.equal(native.calls.length, 1);
    assert.deepEqual(native.calls[0].data, {
      flag: false,
      count: 0,
      empty: "",
    });
    assert.ok(native.calls[0].call.signal instanceof AbortSignal);
    assert.equal(
      (await native.prompt("continue the actual session")).text,
      BODY,
    );
    assert.equal(native.requests.length, 3);
    assert.ok(
      native.requests[2].messages.some((message) => message.role === "tool"),
    );
  });
  const chats = spans.filter(
    (span) => span.attributes["respan.entity.log_type"] === "chat",
  );
  const tools = spans.filter(
    (span) => span.attributes["respan.entity.log_type"] === "tool",
  );
  assert.equal(chats.length, 3);
  assert.equal(tools.length, 1);
  assert.equal(chats.at(-1).attributes["gen_ai.completion.0.content"], BODY);
  assert.ok(chats[0].attributes["llm.request.functions"].includes(BODY));
  assert.equal(chats[0].attributes["gen_ai.usage.input_tokens"], 0);
  assert.equal(chats[0].attributes["gen_ai.usage.output_tokens"], 0);
  assert.equal(chats[0].attributes["llm.usage.total_tokens"], 0);
  assert.equal(tools[0].attributes["gen_ai.tool.call.id"], "native-tool-call");
  const output = JSON.parse(tools[0].attributes["traceloop.entity.output"]);
  const parsed =
    typeof output === "string"
      ? JSON.parse(output)
      : Array.isArray(output) && output[0]?.text
        ? JSON.parse(output[0].text)
        : output.content?.[0]?.text
          ? JSON.parse(output.content[0].text)
          : output;
  assert.equal(parsed.vector?.length, 5001);
  assert.ok(parsed.vector.every((value, index) => value === VECTOR[index]));
  assert.deepEqual([parsed.flag, parsed.count, parsed.empty], [false, 0, ""]);
  for (const span of spans) {
    assert.equal(span.attributes["traceloop.span.kind"], undefined);
    for (const key of [
      "tools",
      "tool_calls",
      "model",
      "prompt_tokens",
      "respan.span.tools",
    ])
      assert.equal(span.attributes[key], undefined);
    assert.ok(
      !Object.keys(span.attributes).some((key) => key.startsWith("flue.")),
    );
    if (span.parentSpanContext)
      assert.ok(
        spans.some(
          (parent) =>
            parent.spanContext().spanId === span.parentSpanContext.spanId,
        ),
      );
  }
});

test("content context, constructor, canonical environment and legacy environment veto before capture", async () => {
  for (const gate of ["context", "option", "respan-env", "legacy-env"]) {
    const key =
      gate === "respan-env"
        ? "RESPAN_TRACE_CONTENT"
        : "TRACELOOP_TRACE_CONTENT";
    const previous = process.env[key];
    if (gate.endsWith("env")) process.env[key] = "false";
    try {
      const spans = await context.with(
        gate === "context"
          ? context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false)
          : context.active(),
        () =>
          scenario(
            { tool: true },
            async (native) =>
              assert.equal((await native.prompt()).text, "native answer"),
            { traceContent: gate !== "option" },
          ),
      );
      assert.ok(spans.length >= 4);
      for (const span of spans)
        assert.equal(
          contentEntries(span).length,
          0,
          span.attributes["respan.entity.log_type"],
        );
    } finally {
      if (previous === undefined) delete process.env[key];
      else process.env[key] = previous;
    }
  }
});

test("sampling and general/LM suppression preserve native execution", async () => {
  for (const gate of ["sampling", "general", "lm"]) {
    dropped = gate === "sampling";
    const ctx =
      gate === "general"
        ? suppressTracing(context.active())
        : gate === "lm"
          ? context
              .active()
              .setValue(
                createContextKey("suppress_language_model_instrumentation"),
                true,
              )
          : context.active();
    try {
      const spans = await context.with(ctx, () =>
        scenario({}, async (native) =>
          assert.equal((await native.prompt()).text, "native answer"),
        ),
      );
      assert.equal(spans.length, 0);
    } finally {
      dropped = false;
    }
  }
});

test("local unknown parent fails closed and a remote parent allows unless content context vetoes", async () => {
  for (const remote of [false, true]) {
    const parent = {
      traceId: "1".repeat(32),
      spanId: "2".repeat(16),
      traceFlags: 1,
      isRemote: remote,
    };
    const spans = await context.with(
      trace.setSpanContext(context.active(), parent),
      () =>
        scenario({}, async (native) =>
          assert.equal((await native.prompt()).text, "native answer"),
        ),
    );
    assert.ok(spans.length >= 2);
    assert.equal(
      contentEntries(
        spans.find(
          (span) => span.attributes["respan.entity.log_type"] === "chat",
        ),
      ).length > 0,
      remote,
    );
  }
});

test("ancestor false is immutable across later true, and late native attrs/events/status are scrubbed", async () => {
  const root = trace.getTracer("test").startSpan("parent");
  const spans = await context.with(trace.setSpan(context.active(), root), () =>
    scenario(
      {
        tool: true,
        onTool() {
          root.setAttribute("allow_trace_content", false);
        },
        onEvent(event) {
          if (event.type === "turn_messages")
            root.setAttribute("allow_trace_content", true);
        },
      },
      async (native) =>
        assert.equal((await native.prompt()).text, "native answer"),
    ),
  );
  root.end();
  for (const span of spans.filter((span, index) => index > 0)) {
    assert.equal(
      contentEntries(span).length,
      0,
      span.attributes["respan.entity.log_type"],
    );
    assert.deepEqual(span.events, []);
    assert.equal(span.status.message, undefined);
  }
});

test("real provider errors remain SDK failures, without invented HTTP statuses", async () => {
  let failure;
  const spans = await scenario({ providerError: true }, async (native) => {
    try {
      await native.prompt();
    } catch (error) {
      failure = error;
    }
  });
  assert.ok(failure instanceof runtime.OperationFailedError);
  const chat = spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "chat",
  );
  assert.equal(chat.status.code, 2);
  assert.equal(chat.attributes.status_code, undefined);
  assert.equal(chat.attributes["http.response.status_code"], undefined);
});

test("deactivation and reactivation preserve native ownership without duplicate subtrees", async () => {
  const instrumentor = new FlueInstrumentor({ runtimeModule: runtime });
  for (let index = 0; index < 2; index++) {
    exported.length = 0;
    await Promise.all([instrumentor.activate(), instrumentor.activate()]);
    const native = await nativeRuntime();
    try {
      await native.prompt();
    } finally {
      await native.close();
      await instrumentor.deactivate();
      await instrumentor.deactivate();
    }
    assert.equal(
      ownSpans().filter(
        (span) => span.attributes["respan.entity.log_type"] === "chat",
      ).length,
      1,
    );
    assert.equal(instrumentor.isActive(), false);
  }
});

test("telemetry never adds caller getter or toJSON calls beyond bare runtime behavior", async () => {
  async function run(instrumented) {
    let getter = 0,
      serialization = 0;
    const output = {
      toJSON() {
        serialization++;
        return { flag: false, count: 0, empty: "" };
      },
    };
    Object.defineProperty(output, "lazy", {
      enumerable: true,
      get() {
        getter++;
        return "native getter";
      },
    });
    const instrumentor = new FlueInstrumentor({ runtimeModule: runtime });
    if (instrumented) await instrumentor.activate();
    const native = await nativeRuntime({ tool: true, toolOutput: output });
    try {
      assert.equal((await native.prompt()).text, "native answer");
    } finally {
      await native.close();
      await instrumentor.deactivate();
    }
    return { getter, serialization };
  }
  assert.deepEqual(await run(true), await run(false));
});

test("a native tool callback late veto removes prior and later attributes, exception events and status descriptions at readable export", async () => {
  const spans = await scenario(
    {
      tool: true,
      onTool() {
        const active = trace.getActiveSpan();
        if (!active) return; // Minimum observer has no execution interceptor.
        active.setAttribute("allow_trace_content", false);
        active.setAttribute("allow_trace_content", true);
        active.setAttribute(
          "gen_ai.prompt.99.content",
          "late controlled content",
        );
        active.setAttribute("llm.request.functions", "late controlled schema");
        active.addEvent("late controlled event", {
          content: "late controlled event data",
        });
        active.recordException(new Error("late controlled exception"));
        active.setStatus({
          code: 2,
          message: "late controlled status description",
        });
      },
    },
    async (native) =>
      assert.equal((await native.prompt()).text, "native answer"),
  );
  const tool = spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "tool",
  );
  if (current) {
    assert.equal(contentEntries(tool).length, 0);
    assert.equal(tool.events.length, 0);
    assert.equal(tool.status.message, undefined);
  } else assert.ok(contentEntries(tool).length > 0);
});

test("a foreign observation subscriber survives activation, disposal and reactivation", async () => {
  let calls = 0;
  const stop = runtime.observe(() => calls++);
  const instrumentor = new FlueInstrumentor({ runtimeModule: runtime });
  try {
    for (let index = 0; index < 2; index++) {
      await instrumentor.activate();
      await instrumentor.deactivate();
      const native = await nativeRuntime();
      try {
        await native.prompt();
      } finally {
        await native.close();
      }
    }
    assert.ok(calls > 0);
  } finally {
    stop();
    await instrumentor.deactivate();
  }
});

test("native delegated tasks and explicit compaction retain connected SDK parentage", async () => {
  const delegated = await scenario({ delegate: true }, async (native) => {
    assert.equal((await native.prompt()).text, "native answer");
    assert.equal(native.requests.length, 3);
    assert.ok(native.events.some((event) => event.type === "task_start"));
  });
  assert.ok(
    delegated.filter(
      (span) => span.attributes["respan.entity.log_type"] === "agent",
    ).length >= 2,
  );
  for (const span of delegated)
    if (span.parentSpanContext)
      assert.ok(
        delegated.some(
          (parent) =>
            parent.spanContext().spanId === span.parentSpanContext.spanId,
        ),
      );
  const compacted = await scenario({ compaction: true }, async (native) => {
    await native.prompt("Create actual native session history.");
    await native.prompt("Continue actual native session history.");
    await native.session.compact();
    assert.ok(native.events.some((event) => event.type === "compaction_start"));
  });
  assert.ok(compacted.some((span) => span.name === "flue.compaction"));
});

test("original content veto survives onEnd injection and actual queued exporter reads after deactivate/reactivate", async () => {
  const spans = await scenario(
    { tool: true },
    async (native) => await native.prompt(),
    { traceContent: false },
  );
  const source = spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "tool",
  );
  const oldAttributes = source.attributes;
  const oldEvents = source.events;
  const oldStatus = source.status;
  source.attributes["traceloop.entity.output"] = "controlled-after-end-private";
  source.attributes["respan.metadata"] = "controlled-after-end-private";
  source.attributes["respan.metadata.private"] = "controlled-after-end-private";
  source.events.push({
    name: "exception",
    attributes: { "exception.message": "controlled-after-end-private" },
  });
  source.status.message = "controlled-after-end-private";
  source.attributes = {
    ...source.attributes,
    "gen_ai.completion.0.content": "controlled-after-end-private",
  };
  source.events = [
    {
      name: "exception",
      attributes: { "exception.message": "controlled-after-end-private" },
    },
  ];
  source.status = { code: 2, message: "controlled-after-end-private" };
  const next = new FlueInstrumentor({ runtimeModule: runtime });
  await next.activate();
  await next.deactivate();
  let serialized;
  const processor = new SimpleSpanProcessor({
    export(batch, done) {
      serialized = JSON.stringify(
        batch.map((span) => ({
          attributes: span.attributes,
          events: span.events,
          status: span.status,
        })),
      );
      done({ code: 0 });
    },
    async shutdown() {},
  });
  processor.onEnd(source);
  await processor.forceFlush();
  await processor.shutdown();
  assert.ok(serialized && !serialized.includes("controlled-after-end-private"));
  assert.equal(contentEntries(source).length, 0);
  assert.equal(source.events.length, 0);
  assert.equal(source.status.message, undefined);
  assert.equal(contentEntries({ attributes: oldAttributes }).length, 0);
  assert.equal(oldEvents.length, 0);
  assert.equal(oldStatus.message, undefined);
});

test("a late suppressed real terminal event closes upstream lifecycle and emits no captured content", async () => {
  exported.length = 0;
  const instrumentor = new FlueInstrumentor({ runtimeModule: runtime });
  const stop = runtime.observe((event) => {
    if (["turn", "operation"].includes(event.type))
      context.with(suppressTracing(context.active()), () =>
        instrumentor.handleEvent(event),
      );
  });
  await instrumentor.activate();
  const native = await nativeRuntime();
  try {
    await native.prompt();
    assert.ok(ownSpans().length >= 2);
    for (const span of ownSpans()) assert.equal(contentEntries(span).length, 0);
  } finally {
    await native.close();
    await instrumentor.deactivate();
    stop();
  }
});

test("compatible owners share one official registration across original-owner deactivation and reactivation", async () => {
  exported.length = 0;
  const first = new FlueInstrumentor({ runtimeModule: runtime });
  const second = new FlueInstrumentor({
    runtimeModule: runtime,
    traceContent: true,
  });
  await Promise.all([first.activate(), second.activate()]);
  await first.deactivate();
  assert.equal(first.isActive(), false);
  assert.equal(second.isActive(), true);
  await first.activate();
  await first.deactivate();
  const parent = trace.getTracer("owner-parent").startSpan("owner-parent");
  const native = await nativeRuntime();
  try {
    await context.with(trace.setSpan(context.active(), parent), () =>
      native.prompt(),
    );
    for (const span of ownSpans())
      assert.equal(span.spanContext().traceId, parent.spanContext().traceId);
  } finally {
    parent.end();
    await native.close();
    await second.deactivate();
    await first.deactivate();
  }
  assert.equal(
    ownSpans().filter(
      (span) => span.attributes["respan.entity.log_type"] === "chat",
    ).length,
    1,
  );
});
