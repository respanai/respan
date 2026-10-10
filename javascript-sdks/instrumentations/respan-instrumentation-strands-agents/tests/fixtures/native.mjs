import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { readFileSync, writeFileSync } from "node:fs";
import {
  context,
  trace,
  ROOT_CONTEXT,
  SpanStatusCode,
} from "@opentelemetry/api";
import {
  BasicTracerProvider,
  SamplingDecision,
} from "@opentelemetry/sdk-trace-base";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { suppressTracing } from "@opentelemetry/core";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import { RespanCompositeProcessor } from "@respan/tracing/dist/processor/composite.js";
import { registerSpanTransformer } from "@respan/tracing";
const requireSDK = createRequire(
  process.env.STRANDS_SDK_ROOT
    ? `${process.env.STRANDS_SDK_ROOT}/package.json`
    : import.meta.url,
);
const { Agent, tool, AfterModelCallEvent, AfterToolCallEvent } = await import(
  pathToFileURL(requireSDK.resolve("@strands-agents/sdk"))
);
const { OpenAIModel } = await import(
  pathToFileURL(requireSDK.resolve("@strands-agents/sdk/models/openai"))
);
const pluginModule = await import(
  process.env.RESPAN_STRANDS_TEST_MODULE ?? "../../dist/index.js"
);
const { StrandsAgentsInstrumentor } = pluginModule;
const mode = process.argv[2] ?? "latest";
const stable = mode === "stable";
process.env.OTEL_SEMCONV_STABILITY_OPT_IN = stable
  ? ""
  : "gen_ai_latest_experimental";
if (mode === "env-respan" || mode === "env-flip")
  process.env.RESPAN_TRACE_CONTENT = "false";
if (mode === "env-traceloop") process.env.TRACELOOP_TRACE_CONTENT = "false";
console.debug = () => {};
const native = [],
  originalAfter = [],
  exported = [],
  requests = [],
  live = [];
const snapshot = (s) => ({
  name: s.name,
  attributes: { ...s.attributes },
  events: structuredClone(s.events),
  status: { ...s.status },
  spanContext: s.spanContext(),
  parentSpanContext: s.parentSpanContext,
});
const raw = {
  onStart(s) {
    live.push(s);
  },
  onEnd(s) {
    native.push(snapshot(s));
    if (mode.startsWith("hostile")) {
      Object.defineProperty(s.attributes, "private-getter", {
        get() {
          accessed++;
          return "private-hostile";
        },
        enumerable: true,
        configurable: true,
      });
      s.attributes["private-proxy"] = new Proxy(
        {},
        {
          ownKeys() {
            accessed++;
            throw new Error("traversed proxy");
          },
        },
      );
      s.attributes["private-toJSON"] = {
        toJSON() {
          accessed++;
          return "private-hostile";
        },
      };
      if (mode === "hostile") {
        Object.defineProperty(s.attributes, "gen_ai.input.messages", {
          get() {
            accessed++;
            return "private-hostile";
          },
          enumerable: true,
          configurable: true,
        });
        Object.defineProperty(s.status, "message", {
          get() {
            accessed++;
            return "private-hostile";
          },
          enumerable: true,
          configurable: true,
        });
      }
    }
  },
  async shutdown() {},
  async forceFlush() {},
};
const manager = {
  onStart() {},
  onEnd(s) {
    exported.push(snapshot(s));
  },
  async shutdown() {},
  async forceFlush() {},
};
const composite = new RespanCompositeProcessor(manager);
const observer = {
  ...raw,
  onStart() {},
  onEnd(s) {
    if (!mode.startsWith("hostile")) originalAfter.push(snapshot(s));
  },
};
const sampler = {
  shouldSample() {
    return {
      decision:
        mode === "drop"
          ? SamplingDecision.NOT_RECORD
          : mode === "record-only"
            ? SamplingDecision.RECORD
            : SamplingDecision.RECORD_AND_SAMPLED,
    };
  },
  toString() {
    return "fixture-sampler";
  },
};
const provider = new BasicTracerProvider({
  sampler,
  spanLimits: {
    attributeCountLimit: 1000,
    eventCountLimit: 1000,
    attributeValueLengthLimit: Infinity,
  },
  spanProcessors: [raw, composite, observer],
});
trace.setGlobalTracerProvider(provider);
context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
const originalInvoke = Agent.prototype.invoke,
  originalStream = Agent.prototype.stream;
const plugin = new StrandsAgentsInstrumentor({
  traceContent: !["constructor", "hostile"].includes(mode),
});
plugin.activate();
let second,
  late,
  accessed = 0;
if (mode === "owners") {
  second = new StrandsAgentsInstrumentor({ traceContent: false });
  second.activate();
}
if (mode === "module-owners") {
  const copied = await import(
    `${process.env.RESPAN_STRANDS_TEST_MODULE ?? new URL("../../dist/index.js", import.meta.url).href}?module-copy=1`
  );
  second = new copied.StrandsAgentsInstrumentor({ traceContent: false });
  second.activate();
}
if (["late", "late-env", "readable-veto", "event-veto"].includes(mode))
  late = registerSpanTransformer("zz-strands-late-fixture", {
    onEnd(s) {
      if (mode === "late-env") process.env.RESPAN_TRACE_CONTENT = "false";
      if (mode === "readable-veto") s.attributes.allow_trace_content = false;
      if (mode === "event-veto")
        s.events.push({
          name: "veto",
          time: [0, 0],
          attributes: { allow_trace_content: false },
        });
      s.attributes["gen_ai.prompt.999.content"] = "private-late-attribute";
      s.events.push({
        name: "private-late-event",
        time: [0, 0],
        attributes: { content: "private-late-event-content" },
      });
      s.status = { code: SpanStatusCode.ERROR, message: "private-late-status" };
    },
  });
const frames = (objects) =>
  objects.map((o) => `data: ${JSON.stringify(o)}\n\n`).join("") +
  "data: [DONE]\n\n";
let callbackCount = 0,
  toolMode = ![
    "error",
    "stream",
    "scalar",
    "large",
    "multimodal",
    "cancel",
  ].includes(mode),
  callbackError = new Error("private-callback-error");
const nativeError = mode === "error";
const scalarPayload = {
  zero: 0,
  falsy: false,
  empty: "",
  nothing: null,
  vector: Array.from({ length: 5001 }, (_, i) => i / 5001),
};
const model = new OpenAIModel({
  api: "chat",
  modelId: "gpt-4.1-nano",
  apiKey: "controlled-fixture",
  clientConfig: {
    maxRetries: 0,
    fetch: async (_url, init) => {
      const request = JSON.parse(init.body);
      requests.push(request);
      if (mode === "env-flip") process.env.RESPAN_TRACE_CONTENT = "true";
      if (mode === "recording-veto") {
        const root = live.find((s) => s.name.startsWith("invoke_agent"));
        root.setAttribute("allow_trace_content", false);
        root.setAttribute("allow_trace_content", true);
      }
      if (mode === "drain") plugin.deactivate();
      if (nativeError)
        return new Response(
          JSON.stringify({
            error: { message: "private-native-error", type: "fixture_error" },
          }),
          { status: 400, headers: { "content-type": "application/json" } },
        );
      const hasResult = request.messages.some((m) => m.role === "tool");
      const finalText =
        mode === "scalar"
          ? "false"
          : mode === "large"
            ? JSON.stringify(scalarPayload)
            : "finished";
      const delta =
        toolMode && !hasResult
          ? {
              role: "assistant",
              tool_calls: [
                {
                  index: 0,
                  id: "call-native-1",
                  type: "function",
                  function: {
                    name: "lookup",
                    arguments: JSON.stringify({ query: "private-query" }),
                  },
                },
              ],
            }
          : { role: "assistant", content: finalText };
      return new Response(
        frames([
          {
            id: "fixture",
            object: "chat.completion.chunk",
            created: 1,
            model: "gpt-4.1-nano",
            choices: [{ index: 0, delta, finish_reason: null }],
          },
          {
            id: "fixture",
            object: "chat.completion.chunk",
            created: 1,
            model: "gpt-4.1-nano",
            choices: [
              {
                index: 0,
                delta: {},
                finish_reason: toolMode && !hasResult ? "tool_calls" : "stop",
              },
            ],
          },
          {
            id: "fixture",
            object: "chat.completion.chunk",
            created: 1,
            model: "gpt-4.1-nano",
            choices: [],
            usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
          },
        ]),
        { headers: { "content-type": "text/event-stream" } },
      );
    },
  },
});
const schema = {
  type: "object",
  properties: Object.fromEntries(
    Array.from({ length: 75 }, (_, i) => [
      `p${i}`,
      { type: "string", description: `field ${i}` },
    ]),
  ),
};
schema.properties.query = { type: "string" };
schema.required = ["query"];
const lookup = tool({
  name: "lookup",
  description: "private-tool-description",
  inputSchema: schema,
  callback: ({ query }) => {
    callbackCount++;
    assert.equal(query, "private-query");
    if (mode === "callback-error") throw callbackError;
    return mode === "tool-string"
      ? "false"
      : mode === "tool-empty"
        ? ""
        : scalarPayload;
  },
});
const agent = new Agent({
  name: "NativeFixture",
  traceAttributes: { "respan.metadata": JSON.stringify({ inherited: "kept" }) },
  model,
  tools: toolMode ? [lookup] : [],
  systemPrompt: "private-system-instructions",
  printer: false,
});
assert.equal(Agent.prototype.invoke, originalInvoke);
assert.equal(Agent.prototype.stream, originalStream);
let result,
  error,
  streamEvents = 0,
  observedError,
  observedToolError;
agent.addHook(AfterModelCallEvent, (event) => {
  observedError = event.error;
});
agent.addHook(AfterToolCallEvent, (event) => {
  observedToolError = event.error;
});
let parent = ROOT_CONTEXT;
if (mode === "context" || mode === "ancestor")
  parent = parent.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false);
if (mode === "suppressed") parent = suppressTracing(parent);
const invoke = async () => {
  try {
    if (mode === "cancel") {
      const stream = agent.stream("private-user-input");
      await stream.next();
      const token = { cancel: "native" };
      let stopped = await stream.return(token);
      while (!stopped.done) stopped = await stream.next();
      assert.equal(stopped.value, token);
      result = { toString: () => "cancelled" };
    } else if (mode === "stream") {
      const stream = agent.stream("private-user-input");
      assert.equal(stream[Symbol.asyncIterator](), stream);
      for (;;) {
        const next = await stream.next();
        if (next.done) {
          result = next.value;
          break;
        }
        streamEvents++;
      }
    } else
      result = await agent.invoke(
        mode === "large"
          ? Array.from({ length: 75 }, (_, i) => ({
              role: i % 2 ? "assistant" : "user",
              content: [{ text: `history-${i}` }],
            }))
          : mode === "multimodal"
            ? [
                {
                  role: "user",
                  content: [
                    { text: "private-user-input" },
                    {
                      image: {
                        format: "png",
                        source: { bytes: new Uint8Array([0, 1, 2]) },
                      },
                    },
                  ],
                },
              ]
            : "private-user-input",
      );
  } catch (e) {
    error = e;
  }
};
if (mode === "ancestor") {
  const outer = trace
    .getTracer("fixture-parent")
    .startSpan("outer", {}, parent);
  const child = trace.setSpan(
    parent.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, true),
    outer,
  );
  await context.with(child, invoke);
  outer.end();
} else await context.with(parent, invoke);
late?.unregister();
second?.deactivate();
plugin.deactivate();
assert.equal(Agent.prototype.invoke, originalInvoke);
assert.equal(Agent.prototype.stream, originalStream);
const chats = exported.filter(
  (s) => s.attributes["respan.entity.log_type"] === "chat",
);
const tools = exported.filter(
  (s) => s.attributes["respan.entity.log_type"] === "tool",
);
const privateModes = [
  "constructor",
  "env-respan",
  "env-traceloop",
  "env-flip",
  "context",
  "ancestor",
  "suppressed",
  "owners",
  "record-only",
  "late-env",
  "hostile",
  "recording-veto",
  "readable-veto",
  "event-veto",
  "module-owners",
];
if (privateModes.includes(mode))
  assert.equal(
    JSON.stringify(exported).includes("private-"),
    false,
    JSON.stringify(exported),
  );
if (mode === "drop") assert.equal(exported.length, 0);
else if (mode === "tool-empty") {
  assert.ok(error);
  assert.equal(observedError, error);
  assert.match(error.message, /empty content/);
  assert.equal(JSON.parse(tools[0].attributes["traceloop.entity.output"]), "");
} else if (mode === "error") {
  assert.ok(error);
  assert.equal(observedError, error);
  assert.ok(
    chats.every((s) => s.attributes["traceloop.entity.output"] === undefined),
  );
} else {
  assert.ok(result, error?.stack);
  assert.equal(
    result.toString(),
    mode === "cancel"
      ? "cancelled"
      : mode === "scalar"
        ? "false"
        : mode === "large"
          ? JSON.stringify(scalarPayload)
          : "finished",
  );
  if (!privateModes.includes(mode) && mode !== "cancel") {
    assert.ok(
      chats[0].attributes["traceloop.entity.input"].includes(
        "private-system-instructions",
      ),
    );
    if (toolMode && mode !== "drain") {
      assert.equal(callbackCount, 1);
      assert.equal(tools[0].attributes["gen_ai.tool.call.id"], "call-native-1");
      assert.ok(
        tools[0].attributes["traceloop.entity.input"].includes("private-query"),
      );
      if (!["callback-error", "tool-string", "tool-empty"].includes(mode))
        assert.ok(
          tools[0].attributes["traceloop.entity.output"].includes(
            '"nothing":null',
          ),
        );
      assert.equal(
        requests[0].tools[0].function.parameters.properties.p74.type,
        "string",
      );
    }
  }
  if (!privateModes.includes(mode) && !["cancel", "drain"].includes(mode)) {
    const parentAgent = exported.find(
      (span) => span.attributes["respan.entity.log_type"] === "agent",
    );
    if (parentAgent) {
      assert.equal(parentAgent.attributes["llm.request.functions"], undefined);
      assert.equal(parentAgent.attributes["gen_ai.request.model"], undefined);
      assert.equal(
        parentAgent.attributes["gen_ai.usage.input_tokens"],
        undefined,
      );
      const metadata = JSON.parse(parentAgent.attributes["respan.metadata"]);
      assert.equal(metadata.inherited, "kept");
      if (toolMode) assert.deepEqual(metadata.strands_agent_tools, ["lookup"]);
    }
  }
  if (mode === "tool-string" || mode === "tool-empty")
    assert.equal(
      JSON.parse(tools[0].attributes["traceloop.entity.output"]),
      mode === "tool-string" ? "false" : "",
    );
  if (mode === "callback-error") assert.equal(observedToolError, callbackError);
  if (mode === "drain") assert.ok(chats.length >= 1);
  if (mode === "scalar")
    assert.equal(
      JSON.parse(chats[0].attributes["traceloop.entity.output"])[0].content,
      "false",
    );
  if (mode === "large") {
    const input = JSON.parse(chats[0].attributes["traceloop.entity.input"]);
    assert.ok(input.length >= 76);
    const payload = JSON.parse(
      JSON.parse(chats[0].attributes["traceloop.entity.output"])[0].content,
    );
    assert.equal(payload.vector.length, 5001);
    assert.equal(payload.falsy, false);
    assert.equal(payload.zero, 0);
    assert.equal(payload.nothing, null);
  }
  if (mode === "multimodal")
    assert.ok(chats[0].attributes["traceloop.entity.input"].includes("image"));
  if (mode === "stream") assert.ok(streamEvents > 0);
  if (mode === "late")
    assert.equal(JSON.stringify(exported).includes("private-late"), false);
}
if (
  ![
    "late",
    "late-env",
    "readable-veto",
    "event-veto",
    "hostile",
    "hostile-enabled",
  ].includes(mode)
)
  assert.deepEqual(
    originalAfter,
    native,
    "adapter mutated the actual native ReadableSpan",
  );
assert.equal(accessed, 0);
const report = {
  mode,
  requests: requests.length,
  callbackCount,
  streamEvents,
  exported,
  native,
  error: error ? { name: error.name, status: error.status } : undefined,
};
if (process.env.STRANDS_CAPTURE_FILE)
  writeFileSync(
    process.env.STRANDS_CAPTURE_FILE,
    JSON.stringify(report, null, 2),
  );
console.log(
  JSON.stringify({
    mode,
    requests: requests.length,
    callbackCount,
    streamEvents,
    spans: exported.length,
    error: error?.name,
  }),
);
await provider.shutdown();
context.disable();
trace.disable();
