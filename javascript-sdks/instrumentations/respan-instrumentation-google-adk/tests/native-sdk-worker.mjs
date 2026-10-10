import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import { createRequire } from "node:module";
import { context, trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  SamplingDecision,
  SimpleSpanProcessor,
  InMemorySpanExporter,
} from "@opentelemetry/sdk-trace-base";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { RespanTelemetry } from "@respan/tracing";
import { suppressTracing } from "@opentelemetry/core";
import { registerSpanTransformer } from "@respan/tracing";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import { GoogleADKInstrumentor } from "../dist/index.js";

process.env.OTEL_METRICS_EXPORTER = "none";
const scenario = process.argv[2];
const drop = process.env.ADK_TEST_DROP === "true";
if (drop) process.env.OTEL_TRACES_SAMPLER = "always_off";
const exporter = new InMemorySpanExporter();
let samplerCalls = 0;
const originalSpans = [];
let telemetry;
if (process.env.ADK_TEST_NATIVE_PROVIDER === "true") {
  const packageRequire = createRequire(import.meta.url);
  const tracingDist = dirname(packageRequire.resolve("@respan/tracing"));
  const { RespanCompositeProcessor } = await import(
    pathToFileURL(join(tracingDist, "processor/composite.js")).href
  );
  const { MultiProcessorManager } = await import(
    pathToFileURL(join(tracingDist, "processor/manager.js")).href
  );
  const manager = new MultiProcessorManager({
    spanNameStyle: "legacy",
    disableBatch: true,
  });
  manager.addProcessor({ name: "default", exporter, disableBatch: true });
  const provider = new BasicTracerProvider({
    ...(process.env.ADK_TEST_RECORD_ONLY === "true"
      ? {
          sampler: {
            shouldSample() {
              samplerCalls += 1;
              return { decision: SamplingDecision.RECORD };
            },
            toString() {
              return "RecordOnlySampler";
            },
          },
        }
      : {}),
    spanProcessors: [
      new RespanCompositeProcessor(manager),
      {
        onStart() {},
        onEnd(span) {
          originalSpans.push(span);
        },
        async forceFlush() {},
        async shutdown() {},
      },
    ],
  });
  trace.setGlobalTracerProvider(provider);
  context.setGlobalContextManager(
    new AsyncLocalStorageContextManager().enable(),
  );
  telemetry = {
    flush: () => provider.forceFlush(),
    shutdown: () => provider.shutdown(),
  };
} else {
  telemetry = new RespanTelemetry({
    apiKey: "local-fixture-only",
    exporter,
    disableBatch: true,
    spanNameStyle: "legacy",
    silenceInitializationMessage: true,
    disabledInstrumentations: [
      "http",
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
  await telemetry.initialize();
}
const instrumentor = new GoogleADKInstrumentor({
  traceContent: process.env.ADK_TEST_CONTENT !== "false",
});
if (process.env.ADK_TEST_BARE !== "true") instrumentor.activate();
const lateRegistration =
  process.env.ADK_TEST_READABLE_VETO === "true"
    ? registerSpanTransformer("test.google-adk.late-veto", {
        onEnd(span) {
          if (span.instrumentationScope.name === "gcp.vertex.agent")
            span.attributes.allow_trace_content = false;
        },
      })
    : undefined;
let secondOwner;
if (process.env.ADK_TEST_SECOND_OWNER === "true") {
  const secondModule = await import("../dist/index.js?second-owner");
  secondOwner = new secondModule.GoogleADKInstrumentor({ traceContent: false });
  secondOwner.activate();
}
const adk = await import("@google/adk");
const nativeRun = adk.InMemoryRunner.prototype.runAsync;
const nativeGenerate = adk.Gemini.prototype.generateContentAsync;
const requests = [];
const originalFetch = globalThis.fetch;
globalThis.fetch = async (url, init) => {
  const target = String(url);
  if (!target.includes("generativelanguage.googleapis.com"))
    throw new Error(`Unexpected fixture transport: ${target}`);
  if (process.env.ADK_TEST_PRIVATE_METADATA === "true") {
    trace
      .getActiveSpan()
      ?.setAttribute("private_payload", "controlled-private-metadata");
    trace
      .getActiveSpan()
      ?.setAttribute(
        "respan.metadata",
        JSON.stringify({ private: "controlled-private-metadata" }),
      );
  }
  const body = JSON.parse(init?.body ?? "{}");
  requests.push(body);
  if (process.env.ADK_TEST_LATE_VETO === "true") {
    const active = trace.getActiveSpan();
    active?.setAttribute("allow_trace_content", false);
    active?.setAttribute("allow_trace_content", true);
  }
  if (process.env.ADK_TEST_DEACTIVATE === "true") instrumentor.deactivate();
  const toolResult = body.contents?.some((content) =>
    content.parts?.some((part) => part.functionResponse),
  );
  const toolCall = ["tool", "parallel"].includes(scenario) && !toolResult;
  const content = {
    role: "model",
    parts: toolCall
      ? [
          {
            functionCall: {
              id: "audit_call_1",
              name: "weather",
              args: { city: "Tokyo" },
            },
          },
        ]
      : [{ text: "Native ADK answer." }],
  };
  if (scenario === "parallel" && toolCall)
    content.parts.push({
      functionCall: {
        id: "audit_call_2",
        name: "weather",
        args: { city: "Paris" },
      },
    });
  const response = {
    candidates: [{ content, index: 0, finishReason: "STOP" }],
    usageMetadata: {
      promptTokenCount: 11,
      candidatesTokenCount: 4,
      thoughtsTokenCount: 2,
      totalTokenCount: 17,
    },
    modelVersion: "gemini-fixture-served",
  };
  if (scenario === "error")
    return new Response(
      JSON.stringify({
        error: {
          code: 429,
          status: "RESOURCE_EXHAUSTED",
          message: "controlled fixture failure",
        },
      }),
      { status: 429, headers: { "content-type": "application/json" } },
    );
  if (scenario === "stream") {
    const frames = [
      {
        candidates: [
          {
            content: { role: "model", parts: [{ text: "Native " }] },
            index: 0,
          },
        ],
      },
      {
        candidates: [
          {
            content: { role: "model", parts: [{ text: "ADK answer." }] },
            index: 0,
            finishReason: "STOP",
          },
        ],
      },
      { usageMetadata: response.usageMetadata },
    ];
    return new Response(
      frames.map((frame) => "data: " + JSON.stringify(frame) + "\n\n").join(""),
      { headers: { "content-type": "text/event-stream" } },
    );
  }
  return new Response(JSON.stringify(response), {
    headers: { "content-type": "application/json" },
  });
};
const events = [];
let error;
try {
  let agent;
  if (scenario === "workflow") {
    agent = new adk.Workflow({
      name: "audit_workflow",
      edges: [
        [
          "START",
          new adk.FunctionNode("finish", () => ({ value: "graph-result" })),
        ],
      ],
    });
  } else {
    const tool = new adk.FunctionTool({
      name: "weather",
      description: "Fixture weather",
      execute: ({ city }) => ({
        city,
        forecast: city === "Paris" ? "cloudy" : "sunny",
      }),
    });
    agent = new adk.LlmAgent({
      name: "audit_agent",
      model: new adk.Gemini({
        model: "gemini-2.5-flash",
        apiKey: "local-fixture-key",
      }),
      instruction: "Answer concisely.",
      tools: ["tool", "parallel"].includes(scenario) ? [tool] : [],
    });
  }
  const runner = new adk.InMemoryRunner({
    appName: "google-adk-native",
    agent,
  });
  const run = async () => {
    const iterator = runner.runEphemeral({
      userId: "fixture-user",
      newMessage: {
        role: "user",
        parts: [{ text: "Native SDK fixture prompt." }],
      },
      runConfig:
        scenario === "stream"
          ? { streamingMode: adk.StreamingMode.SSE }
          : undefined,
    });
    for await (const event of iterator) events.push(event);
  };
  if (process.env.ADK_TEST_CONTEXT_VETO === "true") {
    await context.with(
      context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
      run,
    );
  } else if (process.env.ADK_TEST_SUPPRESS === "true") {
    await context.with(suppressTracing(context.active()), run);
  } else await run();
} catch (caught) {
  error = { name: caught.name, message: caught.message };
} finally {
  globalThis.fetch = originalFetch;
  await telemetry.flush();
  instrumentor.deactivate();
  secondOwner?.deactivate();
  lateRegistration?.unregister();
}
const captureMethodsRestored = exporter
  .getFinishedSpans()
  .every(
    (span) =>
      typeof span.setAttribute !== "function" ||
      span.setAttribute === Object.getPrototypeOf(span).setAttribute,
  );
const spans = exporter.getFinishedSpans().map((span) => ({
  name: span.name,
  traceId: span.spanContext().traceId,
  spanId: span.spanContext().spanId,
  parentSpanId: span.parentSpanContext?.spanId,
  attributes: span.attributes,
  status: span.status,
}));
console.log(
  "NATIVE_RESULT=" +
    JSON.stringify({
      captureMethodsRestored,
      samplerCalls,
      originals: originalSpans.map((span) => ({
        name: span.name,
        attributes: span.attributes,
        traceFlags: span.spanContext().traceFlags,
      })),
      scenario,
      requests,
      events,
      error,
      methodIdentityPreserved:
        nativeRun === adk.InMemoryRunner.prototype.runAsync &&
        nativeGenerate === adk.Gemini.prototype.generateContentAsync,
      spans,
    }),
);
await telemetry.shutdown();
context.disable();
trace.disable();
