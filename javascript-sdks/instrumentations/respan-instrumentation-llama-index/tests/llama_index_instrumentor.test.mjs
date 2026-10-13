import assert from "node:assert/strict";
import test from "node:test";
import { context, trace, TraceFlags } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  BasicTracerProvider,
  SamplingDecision,
} from "@opentelemetry/sdk-trace-base";
import { suppressTracing } from "@opentelemetry/core";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import * as Native from "llamaindex";
import * as Provider from "@llamaindex/openai";
import * as ModularGlobal from "@llamaindex/core/global";
import { LlamaIndexInstrumentor } from "../dist/index.js";

const spans = [];
let decision = SamplingDecision.RECORD_AND_SAMPLED;
let samplerOverride;
let onStart = () => {};
const samplerInputs = [];
context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
const provider = new BasicTracerProvider({
  sampler: {
    shouldSample(_ctx, _trace, name, _kind, attrs) {
      samplerInputs.push({ name, attrs });
      return { decision: samplerOverride ? samplerOverride(name) : decision };
    },
    toString() {
      return "controlled-native";
    },
  },
  spanProcessors: [
    {
      onStart(s) {
        onStart(s);
      },
      onEnd(s) {
        spans.push(s);
      },
      async shutdown() {},
      async forceFlush() {},
    },
  ],
});
trace.setGlobalTracerProvider(provider);
const vector = Array.from({ length: 5001 }, (_, i) => (i === 0 ? 0 : i / 5001));
const fixtureRequests = [];
function makeLLM({
  stream = false,
  error = false,
  tool = false,
  delay,
  toolsSeen,
} = {}) {
  let turn = 0;
  const fetch = async (_url, init) => {
    const request = JSON.parse(init.body);
    fixtureRequests.push(request);
    toolsSeen?.(request);
    if (delay) await delay;
    if (error)
      return new Response(
        JSON.stringify({
          error: {
            message: "controlled native failure",
            type: "invalid_request_error",
          },
        }),
        { status: 422, headers: { "content-type": "application/json" } },
      );
    const toolTurn = tool && turn++ === 0;
    if (request.stream) {
      const content = [
        {
          id: "native-stream",
          model: "gpt-4o-mini",
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "first " },
              finish_reason: null,
            },
          ],
        },
        {
          id: "native-stream",
          model: "gpt-4o-mini",
          choices: [
            { index: 0, delta: { content: "last" }, finish_reason: "stop" },
          ],
        },
        {
          id: "native-stream",
          model: "gpt-4o-mini",
          choices: [],
          usage: { prompt_tokens: 0, completion_tokens: 2, total_tokens: 2 },
        },
      ];
      return new Response(
        content.map((c) => `data: ${JSON.stringify(c)}\n\n`).join("") +
          "data: [DONE]\n\n",
        { headers: { "content-type": "text/event-stream" } },
      );
    }
    return new Response(
      JSON.stringify({
        id: "native-chat",
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: toolTurn
              ? {
                  role: "assistant",
                  content: null,
                  tool_calls: [
                    {
                      id: "actual-native-tool-id",
                      type: "function",
                      function: {
                        name: "lookup",
                        arguments: '{"query":"Respan"}',
                      },
                    },
                  ],
                }
              : { role: "assistant", content: "native answer" },
            finish_reason: toolTurn ? "tool_calls" : "stop",
          },
        ],
        usage: {
          prompt_tokens: 0,
          completion_tokens: 2,
          total_tokens: 2,
          prompt_tokens_details: { cached_tokens: 0 },
        },
      }),
      { headers: { "content-type": "application/json" } },
    );
  };
  return new Provider.OpenAI({
    model: "gpt-4o-mini",
    apiKey: "fixture-only",
    additionalSessionOptions: { fetch, maxRetries: 0 },
    additionalChatOptions: stream
      ? { stream_options: { include_usage: true } }
      : undefined,
  });
}
function makeEmbed() {
  return new Provider.OpenAIEmbedding({
    model: "text-embedding-3-small",
    apiKey: "fixture-only",
    additionalSessionOptions: {
      maxRetries: 0,
      fetch: async (_url, init) => {
        const req = JSON.parse(init.body);
        fixtureRequests.push(req);
        const embedding =
          req.encoding_format === "base64"
            ? Buffer.from(new Float32Array(vector).buffer).toString("base64")
            : vector;
        return new Response(
          JSON.stringify({
            data: req.input.map((_, index) => ({
              index,
              object: "embedding",
              embedding,
            })),
            usage: { prompt_tokens: 3, total_tokens: 3 },
          }),
          { headers: { "content-type": "application/json" } },
        );
      },
    },
  });
}
const attrs = (s) => s.attributes;
const type = (s) => attrs(s)["respan.entity.log_type"];
const tick = () => new Promise((r) => setImmediate(r));
async function run(options, fn) {
  const inst = new LlamaIndexInstrumentor(options);
  await inst.activate();
  try {
    return await fn(inst);
  } finally {
    inst.deactivate();
    await tick();
  }
}
function reset() {
  spans.length = 0;
  fixtureRequests.length = 0;
  samplerInputs.length = 0;
  decision = SamplingDecision.RECORD_AND_SAMPLED;
  onStart = () => {};
  samplerOverride = undefined;
  delete process.env.RESPAN_TRACE_CONTENT;
  delete process.env.TRACELOOP_TRACE_CONTENT;
}

test("released LlamaIndex native compatibility", async (t) => {
  await t.test(
    "direct modular chat: full history, multimodal, tools, native zero usage and output before indexed budget",
    async () => {
      reset();
      await run({}, async () => {
        const llm = makeLLM();
        const history = Array.from({ length: 80 }, (_, i) => ({
          role: "user",
          content: `message-${i}`,
        }));
        history.push({
          role: "user",
          content: [
            { type: "text", text: "image question" },
            {
              type: "image_url",
              image_url: { url: "data:image/png;base64,AA==" },
            },
          ],
        });
        const definitions = Array.from({ length: 80 }, (_, i) =>
          Native.FunctionTool.from(() => false, {
            name: `tool_${i}`,
            description: "native tool",
            parameters: {
              type: "object",
              properties: { enabled: { type: "boolean", default: false } },
            },
          }),
        );
        const response = await llm.chat({
          messages: history,
          tools: definitions,
          responseFormat: { type: "json_object" },
        });
        assert.equal(response.message.content, "native answer");
        await tick();
        assert.equal(spans.length, 1);
        const a = attrs(spans[0]);
        assert.equal(type(spans[0]), "chat");
        assert.equal(JSON.parse(a["traceloop.entity.input"]).length, 81);
        assert.equal(
          JSON.parse(a["traceloop.entity.input"])[80].content[1].type,
          "image_url",
        );
        assert.equal(JSON.parse(a["llm.request.functions"]).length, 80);
        assert.equal(
          JSON.parse(a["traceloop.entity.output"])[0].content,
          "native answer",
        );
        assert.equal(a["gen_ai.usage.input_tokens"], 0);
        assert.equal(a["gen_ai.usage.cache_read.input_tokens"], 0);
        assert.equal(a["gen_ai.system"], undefined);
        assert.equal(a.status_code, undefined);
        assert.equal(a["traceloop.span.kind"], undefined);
        assert.equal(
          samplerInputs[0].attrs["traceloop.entity.input"],
          undefined,
        );
        assert.equal(fixtureRequests[0].tools.length, 80);
      });
    },
  );
  await t.test(
    "generic native callback fallback survives modular Settings and pre-existing provider clients",
    async () => {
      reset();
      const llm = makeLLM();
      await run({ llamaIndexModule: ModularGlobal }, async () => {
        await llm.chat({
          messages: [{ role: "user", content: "native generic callback" }],
        });
        await tick();
        assert.equal(spans.length, 1);
        assert.equal(
          JSON.parse(attrs(spans[0])["traceloop.entity.input"])[0].content,
          "native generic callback",
        );
        assert.equal(
          JSON.parse(attrs(spans[0])["traceloop.entity.output"])[0].content,
          "native answer",
        );
      });
    },
  );
  await t.test(
    "native embeddings retain every vector coordinate and no synthetic usage",
    async () => {
      reset();
      await run({}, async () => {
        const result = await makeEmbed().getTextEmbeddings(["one", "two"]);
        await tick();
        assert.equal(result[0].length, 5001);
        assert.equal(spans.length, 1);
        const a = attrs(spans[0]);
        assert.equal(type(spans[0]), "embedding");
        const output = JSON.parse(a["traceloop.entity.output"]);
        assert.equal(output.length, 2);
        assert.equal(output[0].length, 5001);
        assert.equal(output[0][0], 0);
        assert.equal(a["gen_ai.usage.input_tokens"], undefined);
      });
    },
  );
  await t.test(
    "native complete and native streaming preserve returned object identity",
    async () => {
      reset();
      await run({}, async () => {
        const llm = makeLLM({ stream: true });
        const result = await llm.complete({ prompt: "complete" });
        assert.equal(result.text, "native answer");
        assert.equal(
          attrs(spans.find((s) => type(s) === "text"))["llm.request.type"],
          "chat",
        );
        const stream = await llm.chat({
          messages: [{ role: "user", content: "stream" }],
          stream: true,
        });
        const iterator = stream[Symbol.asyncIterator]();
        assert.equal(iterator[Symbol.asyncIterator](), iterator);
        const chunks = [];
        for await (const chunk of iterator) chunks.push(chunk);
        await tick();
        assert.equal(chunks.map((c) => c.delta).join(""), "first last");
        assert.equal(spans.filter((s) => type(s) === "text").length, 1);
        const chat = spans.find((s) => type(s) === "chat");
        assert.equal(
          JSON.parse(attrs(chat)["traceloop.entity.output"])[0].content,
          "first last",
        );
        assert.equal(attrs(chat)["gen_ai.usage.output_tokens"], 2);
      });
    },
  );
  await t.test(
    "queued native stream reads preserve every native chunk",
    async () => {
      reset();
      await run({}, async () => {
        const stream = await makeLLM({ stream: true }).chat({
          messages: [{ role: "user", content: "queued" }],
          stream: true,
        });
        const iterator = stream[Symbol.asyncIterator]();
        const [first, last] = await Promise.all([
          iterator.next(),
          iterator.next(),
        ]);
        assert.equal(first.value.delta + last.value.delta, "first last");
        while (!(await iterator.next()).done) {}
        await tick();
        assert.equal(
          JSON.parse(
            attrs(spans.find((s) => type(s) === "chat"))[
              "traceloop.entity.output"
            ],
          )[0].content,
          "first last",
        );
      });
    },
  );
  await t.test(
    "real provider failure preserves native error, omits completion and records real HTTP status",
    async () => {
      reset();
      await run({}, async () => {
        let caught;
        try {
          await makeLLM({ error: true }).chat({
            messages: [{ role: "user", content: "fail" }],
          });
        } catch (e) {
          caught = e;
        }
        await tick();
        assert.equal(caught.status, 422);
        assert.equal(spans.length, 1);
        assert.equal(spans[0].status.code, 2);
        assert.equal(attrs(spans[0])["http.response.status_code"], 422);
        assert.equal(attrs(spans[0])["traceloop.entity.output"], undefined);
        assert.equal(attrs(spans[0])["gen_ai.completion.0.content"], undefined);
      });
    },
  );
  await t.test(
    "native agent and tool execution correlate actual model tool ID",
    async () => {
      reset();
      await run({}, async () => {
        const tool = Native.FunctionTool.from(
          ({ query }) => ({ query, result: false }),
          {
            name: "lookup",
            description: "lookup docs",
            parameters: {
              type: "object",
              properties: { query: { type: "string" } },
              required: ["query"],
            },
          },
        );
        const agent = new Provider.OpenAIAgent({
          llm: makeLLM({ tool: true }),
          tools: [tool],
          verbose: false,
        });
        const result = await agent.chat({ message: "use lookup" });
        assert.equal(result.message.content, "native answer");
        await tick();
        const execution = spans.find((s) => type(s) === "tool");
        assert.ok(execution);
        assert.equal(
          attrs(execution)["gen_ai.tool.call.id"],
          "actual-native-tool-id",
        );
        assert.deepEqual(
          JSON.parse(attrs(execution)["traceloop.entity.input"]),
          { name: "lookup", arguments: { query: "Respan" } },
        );
        assert.deepEqual(
          JSON.parse(attrs(execution)["traceloop.entity.output"]),
          { query: "Respan", result: false },
        );
        const parent = spans.find((s) => type(s) === "agent");
        assert.ok(parent);
        assert.equal(
          execution.parentSpanContext.spanId,
          parent.spanContext().spanId,
        );
        for (const s of spans.filter((s) =>
          ["agent", "task", "workflow"].includes(type(s)),
        ))
          assert.equal(attrs(s)["llm.request.functions"], undefined);
      });
    },
  );
  await t.test(
    "native RAG creates workflow, retrieval, synthesis and embedding spans",
    async () => {
      reset();
      await run({}, async () => {
        const llm = makeLLM();
        const embedModel = makeEmbed();
        Native.Settings.embedModel = embedModel;
        Native.Settings.llm = llm;
        const index = await Native.VectorStoreIndex.fromDocuments(
          [new Native.Document({ text: "Respan traces native applications." })],
          { embedModel },
        );
        const engine = index.asQueryEngine({ llm });
        await engine.query({ query: "What does Respan trace?" });
        await tick();
        assert.ok(spans.some((s) => type(s) === "workflow"));
        assert.ok(spans.some((s) => type(s) === "embedding"));
        assert.ok(spans.some((s) => type(s) === "chat" || type(s) === "text"));
        assert.ok(spans.some((s) => type(s) === "task"));
      });
    },
  );
  await t.test(
    "DROP and RECORD_ONLY run actual sampler before inspecting tool metadata",
    async () => {
      for (const sampling of [
        SamplingDecision.NOT_RECORD,
        SamplingDecision.RECORD,
      ]) {
        reset();
        decision = sampling;
        let reads = 0;
        const tool = Native.FunctionTool.from(() => false, {
          name: "native",
          parameters: { type: "object" },
        });
        const descriptor = Object.getOwnPropertyDescriptor(
          Native.FunctionTool.prototype,
          "metadata",
        );
        Object.defineProperty(Native.FunctionTool.prototype, "metadata", {
          ...descriptor,
          get() {
            reads++;
            return descriptor.get.call(this);
          },
        });
        try {
          await run({}, async () => {
            await makeLLM().chat({
              messages: [{ role: "user", content: "private" }],
              tools: [tool],
            });
            await tick();
          });
          assert.equal(reads, 4);
          assert.equal(samplerInputs.length, 1);
          for (const s of spans)
            assert.equal(attrs(s)["traceloop.entity.input"], undefined);
        } finally {
          Object.defineProperty(
            Native.FunctionTool.prototype,
            "metadata",
            descriptor,
          );
        }
      }
    },
  );
  await t.test(
    "native sampled child remains admitted under DROP and RECORD_ONLY agent parents",
    async () => {
      for (const parentDecision of [
        SamplingDecision.NOT_RECORD,
        SamplingDecision.RECORD,
      ]) {
        reset();
        samplerOverride = (name) =>
          name.includes("AgentRunner.chat")
            ? parentDecision
            : SamplingDecision.RECORD_AND_SAMPLED;
        await run({}, async () => {
          const agent = new Provider.OpenAIAgent({ llm: makeLLM(), tools: [] });
          await agent.chat({ message: "independent actual child sampler" });
          await tick();
          const chat = spans.find((s) => type(s) === "chat");
          assert.equal(chat.spanContext().traceFlags, TraceFlags.SAMPLED);
          assert.equal(
            JSON.parse(attrs(chat)["traceloop.entity.input"])[0].content,
            "independent actual child sampler",
          );
          assert.equal(
            JSON.parse(attrs(chat)["traceloop.entity.output"])[0].content,
            "native answer",
          );
          for (const parent of spans.filter((s) => type(s) === "agent"))
            assert.equal(attrs(parent)["traceloop.entity.input"], undefined);
        });
      }
    },
  );
  await t.test(
    "constructor, environment, canonical context, suppression and observed ancestor vetoes",
    async () => {
      for (const mode of [
        "constructor",
        "environment",
        "context",
        "suppression",
        "ancestor",
      ]) {
        reset();
        let parent;
        if (mode === "environment") process.env.RESPAN_TRACE_CONTENT = "false";
        if (mode === "ancestor") {
          parent = trace.getTracer("native-test").startSpan("parent");
          parent.setAttribute("allow_trace_content", false);
        }
        const ctx =
          mode === "context"
            ? context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false)
            : mode === "suppression"
              ? suppressTracing(context.active())
              : parent
                ? trace.setSpan(context.active(), parent)
                : context.active();
        await context.with(ctx, () =>
          run(
            mode === "constructor" ? { traceContent: false } : {},
            async () => {
              await makeLLM().chat({
                messages: [{ role: "user", content: "must stay private" }],
              });
              await tick();
            },
          ),
        );
        parent?.end();
        for (const s of spans.filter((s) => type(s) === "chat")) {
          assert.equal(attrs(s)["traceloop.entity.input"], undefined);
          assert.equal(attrs(s)["traceloop.entity.output"], undefined);
        }
        if (mode === "suppression") assert.equal(spans.length, 0);
      }
    },
  );
  await t.test("independent input and output lanes", async () => {
    for (const inputOnly of [true, false]) {
      reset();
      await run(
        inputOnly ? { recordOutputs: false } : { recordInputs: false },
        async () => {
          await makeLLM().chat({
            messages: [{ role: "user", content: "lane" }],
          });
          await tick();
        },
      );
      const a = attrs(spans[0]);
      assert.equal(a["traceloop.entity.input"] !== undefined, inputOnly);
      assert.equal(a["traceloop.entity.output"] !== undefined, !inputOnly);
    }
  });
  await t.test(
    "late writes and replacement cannot restore private content or coerce safe values",
    async () => {
      reset();
      let actual;
      onStart = (s) => {
        actual = s;
      };
      await run({ traceContent: false }, async () => {
        await makeLLM().chat({
          messages: [{ role: "user", content: "private" }],
        });
        await tick();
      });
      let reads = 0;
      const poison = {
        get secret() {
          reads++;
          return "leak";
        },
        toJSON() {
          reads++;
          return "leak";
        },
      };
      actual.attributes = {
        allow_trace_content: true,
        "unknown.payload": poison,
        "gen_ai.usage.input_tokens": poison,
      };
      actual.events = [poison];
      actual.status = { code: "2", message: poison };
      assert.throws(() =>
        Object.defineProperty(actual, "attributes", {
          value: { "traceloop.entity.input": "leak" },
        }),
      );
      assert.deepEqual(actual.attributes, { allow_trace_content: true });
      assert.deepEqual(actual.events, []);
      assert.deepEqual(actual.status, { code: 0 });
      assert.equal(reads, 0);
    },
  );
  await t.test(
    "late ancestor veto stays sticky when true is later set",
    async () => {
      reset();
      let release;
      const wait = new Promise((r) => (release = r));
      const parent = trace.getTracer("native-test").startSpan("parent");
      await context.with(trace.setSpan(context.active(), parent), () =>
        run({}, async () => {
          const request = makeLLM({ delay: wait }).chat({
            messages: [{ role: "user", content: "initial" }],
          });
          parent.setAttribute("allow_trace_content", false);
          const activeChild = spans.find((s) => type(s) === "chat");
          void activeChild;
          release();
          await request;
          await tick();
          parent.setAttribute("allow_trace_content", true);
          const chat = spans.find((s) => type(s) === "chat");
          assert.equal(attrs(chat)["traceloop.entity.input"], undefined);
          assert.equal(attrs(chat)["traceloop.entity.output"], undefined);
        }),
      );
      parent.end();
    },
  );
  await t.test(
    "coalesced activation, multi-owner restoration and in-flight deactivation",
    async () => {
      reset();
      const a = new LlamaIndexInstrumentor(),
        b = new LlamaIndexInstrumentor();
      await Promise.all([a.activate(), a.activate(), b.activate()]);
      assert.equal(a.isActive(), true);
      a.deactivate();
      await makeLLM().chat({ messages: [{ role: "user", content: "owner" }] });
      let release;
      const delay = new Promise((r) => (release = r));
      const request = makeLLM({ delay }).chat({
        messages: [{ role: "user", content: "inflight" }],
      });
      b.deactivate();
      release();
      await request;
      await tick();
      assert.equal(spans.length, 2);
      await makeLLM().chat({
        messages: [{ role: "user", content: "inactive" }],
      });
      assert.equal(spans.length, 2);
      await a.activate();
      await makeLLM().chat({
        messages: [{ role: "user", content: "reactivate" }],
      });
      a.deactivate();
      assert.equal(spans.length, 3);
    },
  );
  await t.test(
    "activation cancellation, immutable constructor and multi-owner content ceilings",
    async () => {
      reset();
      const options = { traceContent: false };
      const privateOwner = new LlamaIndexInstrumentor(options);
      const publicOwner = new LlamaIndexInstrumentor();
      const pending = privateOwner.activate();
      const coalesced = privateOwner.activate();
      privateOwner.deactivate();
      await Promise.all([pending, coalesced]);
      assert.equal(privateOwner.isActive(), false);
      options.traceContent = true;
      await privateOwner.activate();
      await publicOwner.activate();
      await makeLLM().chat({
        messages: [{ role: "user", content: "ceiling" }],
      });
      assert.equal(attrs(spans[0])["traceloop.entity.input"], undefined);
      privateOwner.deactivate();
      publicOwner.deactivate();
    },
  );
  await t.test(
    "native promises and foreign patch restoration retain identity",
    async () => {
      reset();
      const target = Provider.OpenAI.prototype;
      const descriptor = Object.getOwnPropertyDescriptor(target, "chat");
      let nativePromise;
      const original = descriptor.value;
      target.chat = function (...args) {
        nativePromise = original.apply(this, args);
        return nativePromise;
      };
      const inst = new LlamaIndexInstrumentor();
      await inst.activate();
      const llm = makeLLM();
      const result = llm.chat({
        messages: [{ role: "user", content: "promise" }],
      });
      assert.equal(result, nativePromise);
      await result;
      const foreign = function (...args) {
        return original.apply(this, args);
      };
      target.chat = foreign;
      inst.deactivate();
      assert.equal(target.chat, foreign);
      Object.defineProperty(target, "chat", descriptor);
    },
  );
  reset();
  await provider.shutdown();
  context.disable();
  trace.disable();
});
