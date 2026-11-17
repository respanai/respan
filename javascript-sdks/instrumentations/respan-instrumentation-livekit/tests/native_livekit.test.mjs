import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { context, trace, SpanStatusCode } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SamplingDecision,
} from "@opentelemetry/sdk-trace-base";
import {
  MultiProcessorManager,
  RespanCompositeProcessor,
} from "@respan/tracing/dist/processor/index.js";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import * as LiveKit from "@livekit/agents";
import * as LiveKitOpenAI from "@livekit/agents-plugin-openai";
import { OpenAI } from "openai";
import { z } from "zod";
import { LiveKitInstrumentor } from "../dist/index.js";

LiveKit.initializeLogger({ pretty: false, level: "silent" });
context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());

function transport({
  tool,
  error = false,
  content = "Native transport response.",
} = {}) {
  const requests = [];
  const fetch = async (_url, init) => {
    const body = init?.body ? JSON.parse(init.body) : {};
    requests.push(body);
    if (!body.messages)
      return new Response(JSON.stringify({ object: "list", data: [] }), {
        headers: { "content-type": "application/json" },
      });
    if (error)
      return new Response(
        JSON.stringify({
          error: {
            message: "controlled native transport error",
            type: "server_error",
          },
        }),
        { status: 400, headers: { "content-type": "application/json" } },
      );
    const call =
      tool && !body.messages.some((message) => message.role === "tool");
    const base = {
      id: "native-response-id",
      object: "chat.completion.chunk",
      created: 1,
      model: "native-model",
    };
    const events = [
      {
        ...base,
        choices: [
          {
            index: 0,
            delta: call
              ? {
                  role: "assistant",
                  tool_calls: [
                    {
                      index: 0,
                      id: "native-provider-call",
                      type: "function",
                      function: { name: tool, arguments: '{"city":"Tokyo"}' },
                    },
                  ],
                }
              : { role: "assistant", content },
            finish_reason: null,
          },
        ],
      },
      {
        ...base,
        choices: [
          { index: 0, delta: {}, finish_reason: call ? "tool_calls" : "stop" },
        ],
      },
      {
        ...base,
        choices: [],
        usage: {
          prompt_tokens: 7,
          completion_tokens: 5,
          total_tokens: 12,
          prompt_tokens_details: { cached_tokens: 0 },
        },
      },
    ];
    return new Response(
      events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("") +
        "data: [DONE]\n\n",
      { headers: { "content-type": "text/event-stream" } },
    );
  };
  const client = new OpenAI({
    apiKey: "fixture-only",
    baseURL: "http://fixture.invalid/v1",
    fetch,
    maxRetries: 0,
  });
  const model = new LiveKitOpenAI.LLM({ model: "native-model", client });
  return { model, requests };
}

function harness(options = {}, sampler) {
  const exporter = new InMemorySpanExporter();
  const raw = [];
  const originals = [];
  const manager = new MultiProcessorManager({ disableBatch: true });
  manager.addProcessor({ exporter, name: "default", disableBatch: true });
  const provider = new BasicTracerProvider({
    sampler,
    spanProcessors: [
      {
        onStart() {},
        onEnd(span) {
          raw.push({
            name: span.name,
            attrs: { ...span.attributes },
            status: { ...span.status },
          });
          originals.push(span);
        },
        async forceFlush() {},
        async shutdown() {},
      },
      new RespanCompositeProcessor(manager),
    ],
  });
  LiveKit.telemetry.tracer.setProvider(provider);
  const instrumentor = new LiveKitInstrumentor({
    livekitModule: LiveKit,
    syncTracerProvider: false,
    ...options,
  });
  return { provider, exporter, raw, originals, instrumentor };
}

async function sessionRun({
  options,
  sampler,
  fixture,
  tools,
  chatCtx,
  scope = context.active(),
} = {}) {
  const h = harness(options, sampler);
  await h.instrumentor.activate();
  const native = transport(fixture);
  const session = new LiveKit.voice.AgentSession({ llm: native.model });
  let run;
  try {
    await context.with(scope, async () => {
      await session.start({
        agent: new LiveKit.voice.Agent({
          id: "native-agent",
          instructions: "Use the native transport.",
          tools,
          chatCtx,
        }),
        record: false,
      });
      run = session.run({ userInput: "Native prompt." });
      assert.equal(
        await run.wait(),
        run,
        "native RunResult.wait identity is preserved",
      );
      await session.close();
    });
    await h.provider.forceFlush();
    return {
      ...h,
      requests: native.requests,
      events: run.events,
      spans: h.exporter.getFinishedSpans(),
    };
  } finally {
    await session.close().catch(() => {});
    h.instrumentor.deactivate();
    await h.provider.shutdown();
  }
}

const chat = (result) =>
  result.spans.filter(
    (span) => span.attributes["respan.entity.log_type"] === "chat",
  );
const tool = (execute) => ({
  native_weather: LiveKit.llm.tool({
    description: "Native schema description",
    parameters: z.object({ city: z.string() }),
    execute,
  }),
});
function assertNoPayload(span) {
  assert.equal(span.attributes["traceloop.entity.input"], undefined);
  assert.equal(span.attributes["traceloop.entity.output"], undefined);
  assert.equal(span.attributes["gen_ai.completion.0.content"], undefined);
  assert.equal(span.attributes["gen_ai.prompt.0.content"], undefined);
  assert.equal(span.events.length, 0);
  assert.equal(span.status.message, undefined);
}

test("real AgentSession maps current/minimum input/output without mutating native raw attributes", async () => {
  const r = await sessionRun();
  const llm = chat(r)[0];
  assert.ok(llm);
  assert.equal(
    llm.attributes["gen_ai.completion.0.content"],
    "Native transport response.",
  );
  assert.match(llm.attributes["traceloop.entity.input"], /Native prompt/);
  assert.equal(llm.attributes["gen_ai.usage.input_tokens"], 7);
  assert.equal(llm.attributes["llm.usage.total_tokens"], 12);
  const rawNode = r.raw.find((span) => span.name === "llm_node");
  const originalNode = r.originals.find((span) => span.name === "llm_node");
  const contentKey = rawNode.attrs["lk.pii.chat_ctx"]
    ? "lk.pii.chat_ctx"
    : "lk.chat_ctx";
  assert.equal(originalNode.attributes[contentKey], rawNode.attrs[contentKey]);
  assert.equal(llm.attributes[contentKey], undefined);
  assert.notEqual(llm, originalNode);
  assert.equal(
    r.events.some((event) => event.type === "message"),
    true,
  );
});

test("real tool execution maps arguments/results/falsy values and only native available IDs", async () => {
  const r = await sessionRun({
    fixture: { tool: "native_weather" },
    tools: tool(async ({ city }) => ({
      city,
      zero: 0,
      disabled: false,
      empty: "",
      nil: null,
    })),
  });
  const execution = r.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "tool",
  );
  assert.ok(execution);
  assert.deepEqual(JSON.parse(execution.attributes["traceloop.entity.input"]), {
    name: "native_weather",
    arguments: { city: "Tokyo" },
  });
  assert.deepEqual(
    JSON.parse(execution.attributes["traceloop.entity.output"]).output,
    { city: "Tokyo", zero: 0, disabled: false, empty: "", nil: null },
  );
  for (const key of Object.keys(execution.attributes).filter((key) =>
    key.startsWith("gen_ai."),
  ))
    assert.equal(key, "gen_ai.tool.call.id");
  const nativeId = r.raw.find((span) => span.name === "function_tool").attrs[
    "gen_ai.tool.call.id"
  ];
  assert.equal(execution.attributes["gen_ai.tool.call.id"], nativeId);
  assert.equal(chat(r).length, 2);
  const resultId = r.events.find(
    (event) => event.type === "function_call_output",
  ).item.callId;
  const resultPrompt = Object.entries(chat(r)[1].attributes)
    .find(
      ([key, value]) =>
        key.startsWith("gen_ai.prompt.") &&
        key.endsWith(".role") &&
        value === "tool",
    )[0]
    .replace(/\.role$/, "");
  assert.equal(chat(r)[1].attributes[`${resultPrompt}.tool_call_id`], resultId);

  assert.equal(
    chat(r)[1].attributes["gen_ai.completion.0.tool_calls"],
    undefined,
    "history is not a new tool call",
  );
});

test("real AgentSession handoff remains native and connected", async () => {
  const next = new LiveKit.voice.Agent({
    id: "billing",
    instructions: "Answer billing questions.",
  });
  const r = await sessionRun({
    fixture: { tool: "native_weather" },
    tools: tool(async () =>
      LiveKit.llm.handoff({ agent: next, returns: "Transferred." }),
    ),
  });
  assert.ok(r.events.some((event) => event.type === "agent_handoff"));
  assert.ok(
    r.spans.some(
      (span) => span.attributes["respan.entity.log_type"] === "tool",
    ),
  );
  const agents = r.spans.filter(
    (span) => span.attributes["respan.entity.log_type"] === "agent",
  );
  const isTurn = (span) => span.attributes["traceloop.entity.path"] === "agent_turn";
  // Every agent span stays connected, except each turn: the root of its own
  // trace in the call's session (DEV-14483).
  assert.ok(agents.filter((span) => !isTurn(span)).every((span) => span.parentSpanContext));
  const turns = agents.filter(isTurn);
  assert.ok(turns.length > 0);
  for (const turn of turns) {
    assert.equal(turn.parentSpanContext?.spanId, undefined);
    assert.equal(typeof turn.attributes["respan.metadata.voice.turn_index"], "number");
  }
});

test("native long context and 75 tools preserve aggregate input/output and available definitions", async () => {
  const ctx = LiveKit.llm.ChatContext.empty();
  for (let index = 0; index < 80; index++)
    ctx.addMessage({
      role: index % 2 ? "assistant" : "user",
      content: `history-${index}`,
    });
  const tools = Object.fromEntries(
    Array.from({ length: 75 }, (_, index) => [
      `native_${index}`,
      LiveKit.llm.tool({
        description: `description-${index}`,
        parameters: z.object({ value: z.string() }),
        execute: async () => "ok",
      }),
    ]),
  );
  const r = await sessionRun({ chatCtx: ctx, tools });
  const llm = chat(r)[0];
  assert.match(llm.attributes["traceloop.entity.input"], /history-79/);
  assert.match(llm.attributes["traceloop.entity.input"], /native_74/);
  assert.equal(JSON.parse(llm.attributes["llm.request.functions"]).length, 75);
  assert.equal(
    llm.attributes["gen_ai.completion.0.content"],
    "Native transport response.",
  );
  const wire = r.requests.find((request) => request.messages);
  assert.equal(wire.tools.length, 75);
  assert.ok(wire.tools[0].function.parameters);
});

test("constructor/env/canonical context and ancestor vetoes are immutable capture ceilings", async () => {
  for (const options of [
    { traceContent: false },
    { recordInputs: false, recordOutputs: false },
  ]) {
    const r = await sessionRun({ options });
    for (const span of r.spans) assertNoPayload(span);
  }
  const previous = process.env.RESPAN_TRACE_CONTENT;
  process.env.RESPAN_TRACE_CONTENT = "false";
  try {
    const r = await sessionRun();
    for (const span of r.spans) assertNoPayload(span);
  } finally {
    if (previous === undefined) delete process.env.RESPAN_TRACE_CONTENT;
    else process.env.RESPAN_TRACE_CONTENT = previous;
  }
  const r = await sessionRun({
    scope: context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
  });
  for (const span of r.spans) assertNoPayload(span);
  const parent = harness();
  const ancestor = parent.provider.getTracer("external").startSpan("ancestor");
  ancestor.setAttribute("allow_trace_content", false);
  try {
    const r = await sessionRun({
      scope: trace.setSpan(context.active(), ancestor),
    });
    for (const span of r.spans) assertNoPayload(span);
  } finally {
    ancestor.end();
    await parent.provider.shutdown();
  }
});

test("independent content lanes and actual-span late veto skip unknown getters and remain sticky", async () => {
  for (const options of [{ recordInputs: false }, { recordOutputs: false }]) {
    const r = await sessionRun({ options });
    const llm = chat(r)[0];
    assert.equal(
      llm.attributes["traceloop.entity.input"] !== undefined,
      options.recordInputs !== false,
    );
    assert.equal(
      llm.attributes["gen_ai.completion.0.content"] !== undefined,
      options.recordOutputs !== false,
    );
  }
  let reads = 0;
  const r = await sessionRun({
    fixture: { tool: "native_weather" },
    tools: tool(async () => {
      const span = trace.getSpan(context.active());
      span.setAttribute("allow_trace_content", false);
      span.setAttribute("allow_trace_content", true);
      span.setAttribute("gen_ai.request.model", ["PRIVATE_ARRAY_FIXTURE"]);
      span.setAttribute("gen_ai.usage.input_tokens", [12, 34]);

      span.setAttributes(
        Object.defineProperty({}, "unknown.private", {
          enumerable: true,
          get() {
            reads++;
            throw new Error("getter must not run");
          },
        }),
      );
      span.addEvent("private", { content: "secret" });
      span.setStatus({ code: SpanStatusCode.ERROR, message: "private" });
      return "native result";
    }),
  });
  assert.equal(reads, 0);
  const guardedActual = r.originals.find(
    (span) => span.name === "function_tool",
  );
  assert.equal(guardedActual.attributes["gen_ai.request.model"], undefined);
  assert.equal(
    guardedActual.attributes["gen_ai.usage.input_tokens"],
    undefined,
  );
  const execution = r.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "tool",
  );
  assertNoPayload(execution);
  const readable = r.originals.find((span) => span.name === "function_tool");
  readable.attributes = {
    "unknown.private": {
      get secret() {
        reads++;
        return "secret";
      },
    },
    "gen_ai.usage.unknown": "secret",
    "gen_ai.usage.input_tokens": 0,
  };
  readable.events = [{ name: "private", attributes: { secret: "secret" } }];
  readable.status = { code: "2", message: "secret" };
  assert.equal(readable.attributes["unknown.private"], undefined);
  assert.equal(readable.attributes["gen_ai.usage.unknown"], undefined);
  assert.equal(readable.attributes["gen_ai.usage.input_tokens"], 0);
  assert.equal(readable.status.code, 0);
  assert.equal(reads, 0);
  assertNoPayload(readable);
  assert.throws(
    () =>
      Object.defineProperty(readable, "attributes", {
        value: { private: "secret" },
      }),
    TypeError,
  );
});

test("real native sampler DROP/RECORD_ONLY do not admit adapter payloads", async () => {
  for (const decision of [
    SamplingDecision.NOT_RECORD,
    SamplingDecision.RECORD,
  ]) {
    let samples = 0;
    const r = await sessionRun({
      sampler: {
        shouldSample() {
          samples++;
          return { decision };
        },
        toString() {
          return "native-test-sampler";
        },
      },
    });
    assert.ok(samples > 0);
    for (const span of r.originals) assertNoPayload(span);
    if (decision === SamplingDecision.NOT_RECORD)
      assert.equal(r.spans.length, 0);
  }
});

test("native provider HTTP error matches bare SDK, including minimum upstream rejection behavior", () => {
  const run = (mode) => {
    const child = spawnSync(
      process.execPath,
      [
        fileURLToPath(new URL("./native_error_probe.mjs", import.meta.url)),
        mode,
      ],
      { encoding: "utf8" },
    );
    assert.equal(child.status, 0, child.stderr);
    return JSON.parse(child.stdout.trim().split("\n").at(-1));
  };
  const bare = run("bare"),
    instrumented = run("instrumented");
  assert.deepEqual(instrumented, bare);
  assert.equal(instrumented.chunks, 0);
  assert.ok(instrumented.errors.length);
  assert.equal(instrumented.completions, 0);
});

test("exporter ambient suppression does not revoke an admitted native span", async () => {
  const h = harness();
  await h.instrumentor.activate();
  const fixture = transport();
  try {
    const stream = fixture.model.chat({
      chatCtx: LiveKit.llm.ChatContext.empty(),
    });
    for await (const _chunk of stream) {
      /* preserve original stream */
    }
    await context.with(suppressTracing(context.active()), () =>
      h.provider.forceFlush(),
    );
    assert.equal(
      h.exporter
        .getFinishedSpans()
        .find((span) => span.attributes["respan.entity.log_type"] === "chat")
        .attributes["gen_ai.completion.0.content"] !== undefined,
      typeof LiveKit.telemetry.genAI !== "undefined",
    );
  } finally {
    h.instrumentor.deactivate();
    await h.provider.shutdown();
  }
});

test("real OpenAI TTS provider PCM transport maps only native emitted model and metrics", async () => {
  const h = harness();
  await h.instrumentor.activate();
  let request;
  const client = new OpenAI({
    apiKey: "fixture-only",
    baseURL: "http://fixture.invalid/v1",
    fetch: async (_url, init) => {
      request = JSON.parse(init.body);
      return new Response(new Uint8Array(4800), {
        headers: { "content-type": "application/octet-stream" },
      });
    },
  });
  const tts = new LiveKitOpenAI.TTS({
    client,
    model: "native-tts",
    voice: "alloy",
  });
  try {
    const stream = tts.synthesize("Native audio transport.");
    assert.equal(stream[Symbol.asyncIterator](), stream);
    const frames = [];
    for await (const event of stream) frames.push(event.frame);
    assert.ok(frames.length);
    assert.equal(request.input, "Native audio transport.");
    await new Promise((resolve) => setImmediate(resolve));
    await h.provider.forceFlush();
    const span = h.exporter
      .getFinishedSpans()
      .find((span) => span.attributes["respan.entity.log_type"] === "text");
    assert.equal(span.attributes["gen_ai.request.model"], "native-tts");
    assert.equal(span.attributes["gen_ai.completion.0.content"], undefined);
  } finally {
    await tts.close();
    h.instrumentor.deactivate();
    await h.provider.shutdown();
  }
});

test("multiple module owners share capture ceilings and native in-flight spans drain after deactivation", async () => {
  const strict = harness({ traceContent: false });
  await strict.instrumentor.activate();
  const secondModule = await import("../dist/index.js?second-owner");
  const permissive = new secondModule.LiveKitInstrumentor({
    livekitModule: LiveKit,
    syncTracerProvider: false,
  });
  await permissive.activate();
  try {
    const fixture = transport();
    for await (const _chunk of fixture.model.chat({
      chatCtx: LiveKit.llm.ChatContext.empty(),
    })) {
      /* genuine native stream */
    }
    await strict.provider.forceFlush();
    for (const span of strict.exporter.getFinishedSpans())
      assertNoPayload(span);
  } finally {
    strict.instrumentor.deactivate();
    permissive.deactivate();
    await strict.provider.shutdown();
  }

  const h = harness();
  await h.instrumentor.activate();
  let release;
  const response = new Promise((resolve) => {
    release = resolve;
  });
  let started;
  const fetchStarted = new Promise((resolve) => {
    started = resolve;
  });
  const client = new OpenAI({
    apiKey: "fixture-only",
    fetch: async () => {
      started();
      return response;
    },
  });
  const model = new LiveKitOpenAI.LLM({ client, model: "native-model" });
  const stream = model.chat({ chatCtx: LiveKit.llm.ChatContext.empty() });
  await fetchStarted;
  h.instrumentor.deactivate();
  const frame = {
    id: "draining-native-response",
    choices: [
      { index: 0, delta: { content: "drained" }, finish_reason: "stop" },
    ],
  };
  release(
    new Response(`data: ${JSON.stringify(frame)}\n\ndata: [DONE]\n\n`, {
      headers: { "content-type": "text/event-stream" },
    }),
  );
  try {
    for await (const _chunk of stream) {
      /* original stream remains usable */
    }
    await h.provider.forceFlush();
    const span = h.exporter
      .getFinishedSpans()
      .find((span) => span.attributes["respan.entity.log_type"] === "chat");
    if (LiveKit.telemetry.genAI)
      assert.equal(span.attributes["gen_ai.completion.0.content"], "drained");
    assert.equal(h.instrumentor.isActive(), false);
  } finally {
    await h.provider.shutdown();
  }
});
