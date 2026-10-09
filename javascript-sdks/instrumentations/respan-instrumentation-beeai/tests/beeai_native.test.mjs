import assert from "node:assert/strict";
import test from "node:test";
import {
  context,
  createContextKey,
  ROOT_CONTEXT,
  trace,
  SpanStatusCode,
} from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  AlwaysOffSampler,
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import * as bee from "beeai-framework";
import { DummyChatModel } from "beeai-framework/adapters/dummy/backend/chat";
import { ChatModelOutput } from "beeai-framework/backend/chat";
import {
  AssistantMessage,
  SystemMessage,
  ToolMessage,
  UserMessage,
} from "beeai-framework/backend/message";
import { DynamicTool, JSONToolOutput } from "beeai-framework/tools/base";
import { OpenAIChatModel } from "beeai-framework/adapters/openai/backend/chat";
import { OpenAIEmbeddingModel } from "beeai-framework/adapters/openai/backend/embedding";
import { OpenAIClient } from "beeai-framework/adapters/openai/backend/client";

import { CalculatorTool } from "beeai-framework/tools/calculator";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { Run } from "beeai-framework/context";
import { z } from "zod";
import { BeeAIInstrumentor } from "../dist/index.js";

const modern = Number(bee.Version.split(".")[2]) >= 14;
const { RequirementAgent } = modern
  ? await import("beeai-framework/agents/requirement/agent")
  : {};
const nativeTest = (name, fn) => test(name, fn);
const kind = (span) => span.attributes["respan.entity.log_type"];
const input = (span) => JSON.parse(span.attributes["traceloop.entity.input"]);
const output = (span) => JSON.parse(span.attributes["traceloop.entity.output"]);
async function setup(options = {}) {
  trace.disable();
  context.disable();
  const manager = new AsyncLocalStorageContextManager().enable();
  context.setGlobalContextManager(manager);
  const exporter = new InMemorySpanExporter();
  const provider = new BasicTracerProvider({
    spanLimits: {
      attributeCountLimit: Infinity,
      attributeValueLengthLimit: Infinity,
    },
    spanProcessors: [new SimpleSpanProcessor(exporter)],
    ...options,
  });
  trace.setGlobalTracerProvider(provider);
  const instrumentor = new BeeAIInstrumentor({ sdkModule: bee });
  await instrumentor.activate();
  return {
    provider,
    exporter,
    instrumentor,
    async close() {
      instrumentor.deactivate();
      await provider.shutdown();
      manager.disable();
      trace.disable();
      context.disable();
    },
  };
}
const chatResponse = (
  message,
  usage = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
) => ({
  id: "controlled-response",
  object: "chat.completion",
  created: 1,
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: { role: "assistant", ...message },
      finish_reason: message.tool_calls ? "tool_calls" : "stop",
    },
  ],
  usage,
});
function providerClient(handler) {
  return new OpenAIClient({
    apiKey: "controlled-provider-key",
    fetch: async (url, options) => {
      const body = JSON.parse(options.body);
      const response = await handler(body, String(url));
      return response instanceof Response ? response : Response.json(response);
    },
  });
}
function dummy(answer = "answer") {
  const model = new DummyChatModel("dummy");
  model._create = async () =>
    new ChatModelOutput([new AssistantMessage(answer)]);
  return model;
}
function assertContract(spans) {
  for (const span of spans) {
    assert.match(span.spanContext().traceId, /^[0-9a-f]{32}$/);
    assert.match(span.spanContext().spanId, /^[0-9a-f]{16}$/);
    for (const alias of [
      "traceloop.span.kind",
      "model",
      "tools",
      "tool_calls",
      "input.value",
      "output.value",
      "target",
      "data",
      "respan.span.tools",
      "respan.span.tool_calls",
    ])
      assert.equal(span.attributes[alias], undefined, alias);
    for (const value of Object.values(span.attributes))
      assert.ok(
        ["string", "number", "boolean"].includes(typeof value) ||
          Array.isArray(value),
      );
  }
}

nativeTest(
  "released SDK history, schemas, tool calls, zero usage, callbacks and Run identity",
  async () => {
    const h = await setup();
    try {
      let schemaCalls = 0;
      let fetched;
      const tool = new DynamicTool({
        name: "lookup",
        description: "Look up every value",
        inputSchema: z.object({
          count: z.number(),
          enabled: z.boolean(),
          query: z.string(),
        }),
        handler: async () => new JSONToolOutput(false),
      });
      const original = tool.inputSchema;
      tool.inputSchema = function (...args) {
        schemaCalls++;
        return original.apply(this, args);
      };
      const model = new OpenAIChatModel(
        "gpt-4o-mini",
        {},
        providerClient((body) => {
          fetched = body;
          return chatResponse({
            content: "",
            tool_calls: [
              {
                id: "call-current",
                type: "function",
                function: {
                  name: "lookup",
                  arguments: '{"count":0,"enabled":false,"query":""}',
                },
              },
            ],
          });
        }),
      );
      const history = [
        new SystemMessage("all history"),
        new UserMessage("old"),
        new AssistantMessage({
          type: "tool-call",
          toolCallId: "call-history",
          toolName: "lookup",
          ...(modern
            ? { input: { count: 0, enabled: false, query: "" } }
            : { args: { count: 0, enabled: false, query: "" } }),
        }),
        new ToolMessage([
          {
            type: "tool-result",
            toolCallId: "call-history",
            toolName: "lookup",
            ...(modern
              ? { output: { type: "json", value: false } }
              : { result: false }),
          },
          {
            type: "tool-result",
            toolCallId: "call-history-two",
            toolName: "lookup",
            ...(modern
              ? { output: { type: "json", value: 0 } }
              : { result: 0 }),
          },
        ]),
        new UserMessage("next"),
      ];
      history.push(
        ...Array.from(
          { length: 70 },
          (_, i) => new UserMessage(`history-${i}`),
        ),
      );
      const run = model.create({ messages: history, tools: [tool] });
      assert.ok(run instanceof Run);
      let middleware = 0;
      let callbacks = 0;
      assert.equal(run.context({ audit: "native" }), run);
      assert.equal(
        run.middleware(() => {
          middleware++;
        }),
        run,
      );
      assert.equal(
        run.observe((emitter) =>
          emitter.on("success", () => {
            callbacks++;
          }),
        ),
        run,
      );
      const result = await run;
      assert.ok(result instanceof ChatModelOutput);
      assert.equal(middleware, 1);
      assert.equal(callbacks, 1);
      assert.equal(schemaCalls, 1);
      assert.equal(result.getToolCalls()[0].toolCallId, "call-current");
      assert.equal(
        fetched.tools[0].function.parameters.properties.enabled.type,
        "boolean",
      );
      const [span] = h.exporter.getFinishedSpans();
      assert.equal(kind(span), "chat");
      assert.equal(span.attributes["gen_ai.request.model"], "gpt-4o-mini");
      assert.equal(span.attributes["gen_ai.system"], "openai");
      assert.equal(input(span).length, 75);
      assert.equal(input(span)[74].content, "history-69");
      assert.deepEqual(
        input(span)
          .slice(0, 5)
          .map((m) => m.role),
        ["system", "user", "assistant", "tool", "user"],
      );
      assert.equal(input(span)[2].tool_calls[0].id, "call-history");
      assert.deepEqual(
        input(span)[3].tool_results.map((r) => r.content),
        [false, 0],
      );
      assert.equal(
        JSON.parse(span.attributes["gen_ai.completion.0.tool_calls"])[0].id,
        "call-current",
      );
      assert.equal(
        JSON.parse(span.attributes["llm.request.functions"])[0].parameters
          .properties.count.type,
        "number",
      );
      for (const field of [
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.prompt_tokens",
        "gen_ai.usage.completion_tokens",
        "llm.usage.total_tokens",
      ])
        assert.equal(span.attributes[field], 0);
      assertContract([span]);
    } finally {
      await h.close();
    }
  },
);

test(
  "real RequirementAgent creates one complete connected tree with native tool IDs",
  { skip: !modern },
  async () => {
    const h = await setup();
    try {
      let turn = 0;
      const model = new OpenAIChatModel(
        "gpt-4o-mini",
        {},
        providerClient(() =>
          chatResponse({
            content: "",
            tool_calls: [
              {
                id: ++turn === 1 ? "call-calculator" : "call-final",
                type: "function",
                function:
                  turn === 1
                    ? {
                        name: "Calculator",
                        arguments: '{"expression":"(19+23)*2"}',
                      }
                    : { name: "final_answer", arguments: '{"response":"84"}' },
              },
            ],
          }),
        ),
      );
      const agent = new RequirementAgent({
        llm: model,
        tools: [new CalculatorTool()],
        memory: new UnconstrainedMemory(),
      });
      let parent;
      const result = await trace
        .getTracer("native-test")
        .startActiveSpan("workflow", async (span) => {
          parent = span;
          try {
            return await agent.run({ prompt: "Compute (19+23)*2" });
          } finally {
            span.end();
          }
        });
      assert.equal(result.result.text, "84");
      assert.equal(turn, 2);
      const spans = h.exporter.getFinishedSpans();
      const agents = spans.filter((s) => kind(s) === "agent");
      assert.equal(agents.length, 1);
      assert.equal(
        agents[0].parentSpanContext.spanId,
        parent.spanContext().spanId,
      );
      assert.equal(
        agents[0].attributes["traceloop.entity.name"],
        "RequirementAgent",
      );
      const children = spans.filter((s) => ["tool", "chat"].includes(kind(s)));
      assert.equal(children.filter((s) => kind(s) === "chat").length, 2);
      assert.equal(children.filter((s) => kind(s) === "tool").length, 2);
      for (const child of children)
        assert.equal(
          child.parentSpanContext.spanId,
          agents[0].spanContext().spanId,
        );
      assert.deepEqual(
        children
          .filter((s) => kind(s) === "tool")
          .map((s) => s.attributes["gen_ai.tool.call.id"]),
        ["call-calculator", "call-final"],
      );
      assert.equal(output(agents[0]).content, "84");
      assertContract(spans);
    } finally {
      await h.close();
    }
  },
);

nativeTest(
  "stream tokens and large embedding vectors use real released adapters",
  async () => {
    const h = await setup();
    try {
      const streamFrames = [
        {
          id: "stream-id",
          object: "chat.completion.chunk",
          created: 1,
          model: "gpt-4o-mini",
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "stream " },
              finish_reason: null,
            },
          ],
        },
        {
          id: "stream-id",
          object: "chat.completion.chunk",
          created: 1,
          model: "gpt-4o-mini",
          choices: [
            { index: 0, delta: { content: "answer" }, finish_reason: null },
          ],
        },
        {
          id: "stream-id",
          object: "chat.completion.chunk",
          created: 1,
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        },
      ];
      const model = new OpenAIChatModel(
        "gpt-4o-mini",
        {},
        providerClient(
          () =>
            new Response(
              streamFrames
                .map((f) => `data: ${JSON.stringify(f)}\n\n`)
                .join("") + "data: [DONE]\n\n",
              { headers: { "content-type": "text/event-stream" } },
            ),
        ),
      );
      let tokens = 0;
      const result = await model
        .create({ messages: [new UserMessage("stream")], stream: true })
        .observe((emitter) =>
          emitter.on("newToken", () => {
            tokens++;
          }),
        );
      assert.equal(result.getTextContent(), "stream answer");
      assert.ok(tokens >= 2);
      const vector = Array.from({ length: 5001 }, (_, i) =>
        i === 0 ? 0 : i / 5001,
      );
      const embed = new OpenAIEmbeddingModel(
        "text-embedding-3-small",
        {},
        providerClient(() => ({
          object: "list",
          data: [{ object: "embedding", index: 0, embedding: vector }],
          model: "text-embedding-3-small",
          usage: { prompt_tokens: 0, total_tokens: 0 },
        })),
      );
      const embedded = await embed.create({ values: ["full vector"] });
      assert.deepEqual(embedded.embeddings, [vector]);
      const spans = h.exporter.getFinishedSpans();
      assert.equal(spans.filter((s) => kind(s) === "chat").length, 1);
      const embedding = spans.find((s) => kind(s) === "embedding");
      assert.deepEqual(output(embedding), [vector]);
      assert.deepEqual(input(embedding), ["full vector"]);
      assert.equal(embedding.attributes["gen_ai.usage.input_tokens"], 0);
      assertContract(spans);
    } finally {
      await h.close();
    }
  },
);

nativeTest(
  "tool outputs preserve false, zero, empty and never invoke caller telemetry hooks",
  async () => {
    const h = await setup();
    try {
      const values = [false, 0, "", []];
      let getters = 0;
      let stringCalls = 0;
      const hostile = {
        safe: 0,
        toJSON() {
          stringCalls++;
          throw Error("do not invoke");
        },
        toString() {
          stringCalls++;
          throw Error("do not invoke");
        },
        [Symbol.iterator]() {
          stringCalls++;
          throw Error("do not iterate");
        },
      };
      Object.defineProperty(hostile, "secret", {
        enumerable: true,
        get() {
          getters++;
          throw Error("do not invoke");
        },
      });
      let proxyTraps = 0;
      const proxy = new Proxy(
        { secret: "never inspect" },
        {
          ownKeys() {
            proxyTraps++;
            throw Error("proxy must stay opaque");
          },
          getOwnPropertyDescriptor() {
            proxyTraps++;
            throw Error("proxy must stay opaque");
          },
          get() {
            proxyTraps++;
            throw Error("proxy must stay opaque");
          },
        },
      );
      values.push(
        hostile,
        { proxy },
        new Date("2026-10-09T00:00:00Z"),
        new URL("https://example.com/native"),
        new Map([["zero", 0]]),
        new Set([false]),
        new Uint8Array([0, 1, 255]),
      );
      for (const value of values) {
        const expected = new JSONToolOutput(value);
        const tool = new DynamicTool({
          name: "values",
          description: "values",
          inputSchema: z.object({}),
          handler: async () => expected,
        });
        const result = await tool.run({});
        assert.equal(result, expected);
      }
      assert.deepEqual(h.exporter.getFinishedSpans().map(output), [
        false,
        0,
        "",
        [],
        { safe: 0 },
        { proxy: "[Proxy]" },
        "2026-10-09T00:00:00.000Z",
        "https://example.com/native",
        { entries: [["zero", 0]] },
        { values: [false] },
        [0, 1, 255],
      ]);
      assert.equal(proxyTraps, 0);
      assert.equal(getters, 0);
      assert.equal(stringCalls, 0);
      const observations = [];
      for (const instrumented of [false, true]) {
        if (instrumented) await h.instrumentor.activate();
        else h.instrumentor.deactivate();
        const counts = { get: 0, ownKeys: 0, descriptors: 0 };
        const backing = new Proxy([{ type: "text", text: "caller proxy" }], {
          get(target, key, receiver) {
            counts.get++;
            return Reflect.get(target, key, receiver);
          },
          ownKeys(target) {
            counts.ownKeys++;
            return Reflect.ownKeys(target);
          },
          getOwnPropertyDescriptor(target, key) {
            counts.descriptors++;
            return Reflect.getOwnPropertyDescriptor(target, key);
          },
        });
        const message = new UserMessage(backing);
        counts.get = counts.ownKeys = counts.descriptors = 0;
        assert.equal(
          (await dummy().create({ messages: [message] })).getTextContent(),
          "answer",
        );
        observations.push({ ...counts });
      }
      assert.deepEqual(
        observations[1],
        observations[0],
        "SDK-owned proxy wrapping must not inspect a caller proxy backing array",
      );
      assert.equal(
        input(h.exporter.getFinishedSpans().at(-1))[0].content,
        "[Proxy]",
      );
    } finally {
      await h.close();
    }
  },
);

nativeTest(
  "content, ancestor and late vetoes precede conversion; general/LM suppression and sampling",
  async () => {
    const h = await setup();
    const previous = process.env.RESPAN_TRACE_CONTENT;
    try {
      const model = dummy("private answer");
      const denied = ROOT_CONTEXT.setValue(
        CONTEXT_KEY_ALLOW_TRACE_CONTENT,
        false,
      );
      const lazy = context.with(denied, () =>
        model.create({ messages: [new UserMessage("private prompt")] }),
      );
      await context.with(
        ROOT_CONTEXT.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, true),
        () => lazy,
      );
      process.env.RESPAN_TRACE_CONTENT = "false";
      await model.create({ messages: [new UserMessage("env secret")] });
      delete process.env.RESPAN_TRACE_CONTENT;
      const ancestor = new DynamicTool({
        name: "ancestor",
        description: "veto",
        inputSchema: z.object({}),
        handler: async () => {
          await context.with(
            context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, true),
            () =>
              model.create({
                messages: [new UserMessage("observed parent secret")],
              }),
          );
          return new JSONToolOutput(false);
        },
      });
      await context.with(denied, () => ancestor.run({}));
      await trace
        .getTracer("native")
        .startActiveSpan(
          "attr-parent",
          { attributes: { allow_trace_content: false } },
          async (span) => {
            try {
              const run = model.create({
                messages: [new UserMessage("attribute secret")],
              });
              span.setAttribute("allow_trace_content", true);
              await run;
            } finally {
              span.end();
            }
          },
        );
      const late = dummy("late secret");
      late._create = async () => {
        trace.getActiveSpan().setAttribute("allow_trace_content", false);
        return new ChatModelOutput([new AssistantMessage("late secret")]);
      };
      await late.create({ messages: [new UserMessage("late prompt")] });
      const before = h.exporter.getFinishedSpans().length;
      await context.with(suppressTracing(ROOT_CONTEXT), () =>
        model.create({ messages: [new UserMessage("suppressed")] }),
      );
      await context.with(
        ROOT_CONTEXT.setValue(
          createContextKey("suppress_language_model_instrumentation"),
          true,
        ),
        () => model.create({ messages: [new UserMessage("lm suppressed")] }),
      );
      assert.equal(h.exporter.getFinishedSpans().length, before);
      const chats = h.exporter
        .getFinishedSpans()
        .filter((s) => kind(s) === "chat");
      for (const span of chats)
        for (const key of Object.keys(span.attributes))
          assert.ok(
            !key.startsWith("gen_ai.prompt") &&
              !key.startsWith("gen_ai.completion") &&
              !key.endsWith(".input") &&
              !key.endsWith(".output"),
            key,
          );
    } finally {
      if (previous === undefined) delete process.env.RESPAN_TRACE_CONTENT;
      else process.env.RESPAN_TRACE_CONTENT = previous;
      await h.close();
    }
    const off = await setup({ sampler: new AlwaysOffSampler() });
    try {
      assert.equal(
        (
          await dummy().create({ messages: [new UserMessage("not sampled")] })
        ).getTextContent(),
        "answer",
      );
      assert.equal(off.exporter.getFinishedSpans().length, 0);
    } finally {
      await off.close();
    }
  },
);

nativeTest(
  "errors, cancellation, ref counts, deactivation drain and foreign patches preserve outcomes",
  async () => {
    const bareCreate = bee.ChatModel.prototype.create;
    const h = await setup();
    const second = new BeeAIInstrumentor({ sdkModule: bee });
    try {
      await second.activate();
      h.instrumentor.deactivate();
      const model = dummy();
      const originalError = new Error("controlled failure");
      model._create = async () => {
        throw originalError;
      };
      let observed;
      const failing = model
        .create({ messages: [new UserMessage("fail")], maxRetries: 0 })
        .observe((emitter) =>
          emitter.match(/\.run\.error$/, (e) => {
            observed = e;
          }),
        );
      await assert.rejects(failing, (e) => {
        assert.equal(e, observed);
        return true;
      });
      assert.equal(
        h.exporter.getFinishedSpans().at(-1).status.code,
        SpanStatusCode.ERROR,
      );
      const pending = dummy("drained").create({
        messages: [new UserMessage("drain")],
      });
      pending.observe((emitter) =>
        emitter.on("start", () => second.deactivate()),
      );
      assert.equal((await pending).getTextContent(), "drained");
      assert.equal(
        h.exporter.getFinishedSpans().filter((s) => kind(s) === "chat").length,
        2,
      );
      await h.instrumentor.activate();
      const instrumented = bee.ChatModel.prototype.create;
      function foreign(...args) {
        return instrumented.apply(this, args);
      }
      bee.ChatModel.prototype.create = foreign;
      h.instrumentor.deactivate();
      assert.equal(bee.ChatModel.prototype.create, foreign);
      await h.instrumentor.activate();
      assert.equal(
        (
          await dummy().create({ messages: [new UserMessage("reactivate")] })
        ).getTextContent(),
        "answer",
      );
      h.instrumentor.deactivate();
      bee.ChatModel.prototype.create = bareCreate;
    } finally {
      second.deactivate();
      await h.close();
    }
  },
);

nativeTest(
  "structured output schema, cancellation, constructor/env privacy and telemetry faults",
  async () => {
    const h = await setup();
    const previous = process.env.TRACELOOP_TRACE_CONTENT;
    try {
      const schema = {
        type: "object",
        properties: {
          count: { type: "number" },
          enabled: { type: "boolean" },
          text: { type: "string" },
        },
        required: ["count", "enabled", "text"],
        additionalProperties: false,
      };
      const model = new OpenAIChatModel(
        "gpt-4o-mini",
        {},
        providerClient((body) =>
          chatResponse(
            !modern && body.tools?.length
              ? {
                  content: "",
                  tool_calls: [
                    {
                      id: "structured-call",
                      type: "function",
                      function: {
                        name: body.tools[0].function.name,
                        arguments: '{"count":0,"enabled":false,"text":""}',
                      },
                    },
                  ],
                }
              : { content: '{"count":0,"enabled":false,"text":""}' },
          ),
        ),
      );
      const structured = await model.createStructure({
        messages: [new UserMessage("structure")],
        schema: modern
          ? { type: "object-json", schema, name: "value_schema" }
          : z.object({
              count: z.number(),
              enabled: z.boolean(),
              text: z.string(),
            }),
      });
      assert.deepEqual(structured.object, {
        count: 0,
        enabled: false,
        text: "",
      });
      const task = h.exporter
        .getFinishedSpans()
        .find((s) => kind(s) === "task");
      assert.deepEqual(input(task).schema.schema.properties, schema.properties);
      assert.deepEqual(output(task).object, structured.object);
      const controller = new AbortController();
      const abortModel = dummy();
      let observed;
      abortModel._create = async (_input, run) =>
        new Promise((_, reject) =>
          run.signal.addEventListener(
            "abort",
            () => reject(run.signal.reason),
            { once: true },
          ),
        );
      const cancel = abortModel
        .create({
          messages: [new UserMessage("cancel")],
          abortSignal: controller.signal,
          maxRetries: 0,
        })
        .observe((emitter) => {
          emitter.match(/\.run\.error$/, (error) => {
            observed = error;
          });
          emitter.on("start", () =>
            controller.abort(new Error("controlled cancellation")),
          );
        });
      await assert.rejects(cancel, (error) => {
        assert.equal(error, observed);
        return true;
      });
      process.env.TRACELOOP_TRACE_CONTENT = "false";
      await dummy().create({
        messages: [new UserMessage("legacy environment secret")],
      });
      const hidden = h.exporter.getFinishedSpans().at(-1);
      assert.equal(hidden.attributes["traceloop.entity.input"], undefined);
      delete process.env.TRACELOOP_TRACE_CONTENT;
      h.instrumentor.deactivate();
      const privacy = new BeeAIInstrumentor({
        sdkModule: bee,
        traceContent: false,
      });
      await privacy.activate();
      try {
        await dummy().create({
          messages: [new UserMessage("constructor secret")],
        });
        assert.equal(
          h.exporter.getFinishedSpans().at(-1).attributes[
            "traceloop.entity.output"
          ],
          undefined,
        );
      } finally {
        privacy.deactivate();
      }
      await h.instrumentor.activate();
      const getTracer = h.provider.getTracer;
      h.provider.getTracer = () => {
        throw new Error("controlled telemetry fault");
      };
      try {
        assert.equal(
          (
            await dummy().create({ messages: [new UserMessage("fault")] })
          ).getTextContent(),
          "answer",
        );
      } finally {
        h.provider.getTracer = getTracer;
      }
    } finally {
      if (previous === undefined) delete process.env.TRACELOOP_TRACE_CONTENT;
      else process.env.TRACELOOP_TRACE_CONTENT = previous;
      await h.close();
    }
  },
);

nativeTest(
  "denied readable spans survive onEnd injection, replacement and queued export after reactivation",
  async () => {
    trace.disable();
    context.disable();
    const manager = new AsyncLocalStorageContextManager().enable();
    context.setGlobalContextManager(manager);
    const queued = [];
    const injector = {
      onStart() {},
      onEnd(span) {
        span.attributes["traceloop.entity.output"] = "late-private-output";
        span.attributes["exception.message"] = "late-private-error";
        span.status.message = "late-private-status";
        span.events.push({
          name: "exception",
          attributes: { "exception.message": "late-private-event" },
        });
        span.attributes = {
          ...span.attributes,
          "traceloop.entity.input": "replaced-private-input",
          "traceloop.entity.output": "replaced-private-output",
          "exception.message": "replaced-private-error",
        };
        span.status = { ...span.status, message: "replaced-private-status" };
        span.events = [
          {
            name: "exception",
            attributes: { "exception.message": "replaced-private-event" },
          },
        ];
        queued.push(span);
      },
      async forceFlush() {},
      async shutdown() {},
    };
    const provider = new BasicTracerProvider({ spanProcessors: [injector] });
    trace.setGlobalTracerProvider(provider);
    const instrumentor = new BeeAIInstrumentor({
      sdkModule: bee,
      traceContent: false,
    });
    const exporter = new InMemorySpanExporter();
    const simple = new SimpleSpanProcessor(exporter);
    try {
      await instrumentor.activate();
      assert.equal(
        (
          await dummy().create({
            messages: [new UserMessage("private native input")],
          })
        ).getTextContent(),
        "answer",
      );
      instrumentor.deactivate();
      await instrumentor.activate();
      simple.onEnd(queued[0]);
      await simple.forceFlush();
      const [span] = exporter.getFinishedSpans();
      assert.equal(span.attributes["traceloop.entity.input"], undefined);
      assert.equal(span.attributes["traceloop.entity.output"], undefined);
      assert.equal(span.attributes["exception.message"], undefined);
      assert.equal(span.status.message, undefined);
      assert.deepEqual(span.events, []);
      assert.equal(span.attributes["respan.entity.log_type"], "chat");
    } finally {
      instrumentor.deactivate();
      await provider.shutdown();
      await simple.shutdown();
      manager.disable();
      trace.disable();
      context.disable();
    }
  },
);
