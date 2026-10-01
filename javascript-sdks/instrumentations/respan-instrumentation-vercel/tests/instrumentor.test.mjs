import test from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { context, trace } from "@opentelemetry/api";

import {
  BasicTracerProvider,
  InMemorySpanExporter,
} from "@opentelemetry/sdk-trace-base";
import { getRegisteredSpanTransformerKeys } from "@respan/tracing";
import {
  MultiProcessorManager,
  RespanCompositeProcessor,
} from "@respan/tracing/dist/processor/index.js";
import {
  ensureAISDKTelemetry,
  releaseOwnedAISDKTelemetry,
  resolveRuntimeModuleURL,
} from "../dist/_ai_sdk_telemetry.js";
import {
  VercelAIInstrumentor,
} from "../dist/index.js";

const telemetryMarker = Symbol.for(
  "@respan/instrumentation-vercel.ai-sdk-telemetry-registered",
);

async function withCleanTelemetryGlobals(run) {
  const originalRegistry = globalThis.AI_SDK_TELEMETRY_INTEGRATIONS;
  const originalMarker = globalThis[telemetryMarker];

  try {
    globalThis.AI_SDK_TELEMETRY_INTEGRATIONS = [];
    delete globalThis[telemetryMarker];
    await run();
  } finally {
    if (originalRegistry === undefined) {
      delete globalThis.AI_SDK_TELEMETRY_INTEGRATIONS;
    } else {
      globalThis.AI_SDK_TELEMETRY_INTEGRATIONS = originalRegistry;
    }

    if (originalMarker === undefined) {
      delete globalThis[telemetryMarker];
    } else {
      globalThis[telemetryMarker] = originalMarker;
    }
  }
}

function fakeAISDK7Modules(imports) {
  class OpenTelemetry {}
  return {
    OpenTelemetry,
    async importModule(specifier) {
      imports.push(specifier);
      if (specifier === "ai") {
        return {
          registerTelemetry(...integrations) {
            globalThis.AI_SDK_TELEMETRY_INTEGRATIONS.push(...integrations);
          },
        };
      }
      return { OpenTelemetry };
    },
  };
}

function createOTel2Harness(spanNameStyle = "legacy") {
  const exporter = new InMemorySpanExporter();
  const manager = new MultiProcessorManager({
    disableBatch: true,
    spanNameStyle,
  });
  manager.addProcessor({
    exporter,
    name: "default",
    disableBatch: true,
  });
  const composite = new RespanCompositeProcessor(manager);
  const provider = new BasicTracerProvider({ spanProcessors: [composite] });
  return {
    composite,
    exporter,
    provider,
    tracer: provider.getTracer("gen_ai"),
  };
}

async function exportNestedModernEmbedding(spanNameStyle) {
  const harness = createOTel2Harness(spanNameStyle);
  const instrumentor = new VercelAIInstrumentor({
    autoRegisterAISDKTelemetry: false,
  });

  try {
    await instrumentor.activate();
    const root = harness.tracer.startSpan("embeddings probe-embedding", {
      attributes: {
        "gen_ai.operation.name": "embeddings",
        "gen_ai.request.model": "probe-embedding",
        "ai.value": "otel2 embedding input",
      },
    });
    const rootContext = trace.setSpan(context.active(), root);
    const child = harness.tracer.startSpan(
      "embeddings probe-embedding",
      {
        attributes: {
          "gen_ai.operation.name": "embeddings",
          "gen_ai.request.model": "probe-embedding",
          "ai.values": ["otel2 embedding input"],
          // OTEL attributes cannot contain nested arrays; the AI SDK encodes
          // each vector as JSON inside a string array.
          "ai.embeddings": [JSON.stringify([0.11, 0.22, 0.33])],
        },
      },
      rootContext,
    );

    child.end();
    root.end();
    await harness.provider.forceFlush();
    return {
      spans: harness.exporter.getFinishedSpans(),
      rootSpanId: root.spanContext().spanId,
    };
  } finally {
    instrumentor.deactivate();
    await harness.provider.shutdown();
  }
}

test("semantic OTel 2 export collapses a nested AI SDK 7 embedding to one call", async () => {
  const { spans } = await exportNestedModernEmbedding("semantic");

  assert.equal(spans.length, 1);
  const [embedding] = spans;
  assert.equal(embedding.name, "embedding");
  assert.equal(embedding.attributes["respan.entity.log_type"], "embedding");
  assert.equal(embedding.parentSpanContext, undefined);
  assert.deepEqual(
    JSON.parse(embedding.attributes["traceloop.entity.input"]),
    ["otel2 embedding input"],
  );
  assert.deepEqual(
    JSON.parse(embedding.attributes["traceloop.entity.output"]),
    [[0.11, 0.22, 0.33]],
  );
  assert.equal(embedding.attributes["respan.internal.drop_span"], undefined);
  assert.equal(
    embedding.attributes["respan.internal.export_parent_span_id"],
    undefined,
  );
});

test("legacy OTel 2 export preserves the nested AI SDK 7 embedding tree", async () => {
  const { spans, rootSpanId } = await exportNestedModernEmbedding("legacy");

  assert.equal(spans.length, 2);
  const child = spans.find(span => span.parentSpanContext?.spanId === rootSpanId);
  const root = spans.find(span => span.spanContext().spanId === rootSpanId);
  assert.ok(child);
  assert.ok(root);
  assert.equal(child.name, "embeddings probe-embedding");
  assert.equal(root.name, "embeddings probe-embedding");
  assert.equal(child.attributes["respan.entity.log_type"], "embedding");
  assert.equal(root.attributes["respan.entity.log_type"], "embedding");
  for (const span of spans) {
    assert.equal(span.attributes["respan.internal.drop_span"], undefined);
    assert.equal(
      span.attributes["respan.internal.export_parent_span_id"],
      undefined,
    );
  }
});

test("nested embedding correlation drains after final deactivate and reactivates cleanly", async () => {
  const harness = createOTel2Harness("semantic");
  const instrumentor = new VercelAIInstrumentor({
    autoRegisterAISDKTelemetry: false,
  });

  try {
    await instrumentor.activate();
    const root = harness.tracer.startSpan("embeddings drain-model", {
      attributes: { "gen_ai.operation.name": "embeddings", "ai.value": "drain" },
    });
    const child = harness.tracer.startSpan(
      "embeddings drain-model",
      {
        attributes: {
          "gen_ai.operation.name": "embeddings",
          "ai.values": ["drain"],
          "ai.embeddings": [JSON.stringify([0.9, 0.8])],
        },
      },
      trace.setSpan(context.active(), root),
    );

    instrumentor.deactivate();
    assert.deepEqual(getRegisteredSpanTransformerKeys(), []);
    child.end();
    root.end();
    await harness.provider.forceFlush();
    assert.equal(harness.exporter.getFinishedSpans().length, 1);

    await instrumentor.activate();
    const standalone = harness.tracer.startSpan("embeddings drain-model", {
      attributes: {
        "gen_ai.operation.name": "embeddings",
        "ai.value": "fresh",
        "ai.embedding": [0.7, 0.6],
      },
    });
    standalone.end();
    await harness.provider.forceFlush();

    const exported = harness.exporter.getFinishedSpans();
    assert.equal(exported.length, 2);
    assert.equal(exported[1].attributes["traceloop.entity.input"], "fresh");
  } finally {
    instrumentor.deactivate();
    await harness.provider.shutdown();
  }
});

test("two instrumentors share one owned AI SDK 7 adapter until the final release", async () => {
  await withCleanTelemetryGlobals(async () => {
    const imports = [];
    const { OpenTelemetry, importModule } = fakeAISDK7Modules(imports);

    const first = await ensureAISDKTelemetry({ importModule });
    const second = await ensureAISDKTelemetry({ importModule });

    assert.equal(first.status, "registered");
    assert.equal(second.status, "already-registered");
    assert.ok(first.lease);
    assert.equal(second.lease, first.lease);
    assert.equal(first.lease.leases, 2);
    assert.equal(globalThis.AI_SDK_TELEMETRY_INTEGRATIONS.length, 1);
    assert.ok(globalThis.AI_SDK_TELEMETRY_INTEGRATIONS[0] instanceof OpenTelemetry);

    assert.equal(releaseOwnedAISDKTelemetry(first.lease), false);
    assert.equal(first.lease.leases, 1);
    assert.equal(globalThis.AI_SDK_TELEMETRY_INTEGRATIONS.length, 1);

    assert.equal(releaseOwnedAISDKTelemetry(second.lease), true);
    assert.equal(globalThis.AI_SDK_TELEMETRY_INTEGRATIONS.length, 0);
    assert.equal(globalThis[telemetryMarker], undefined);
    assert.deepEqual(imports, ["ai", "@ai-sdk/otel", "ai"]);
  });
});

test("concurrent AI SDK 7 activations share the adapter registered after import", async () => {
  await withCleanTelemetryGlobals(async () => {
    class OpenTelemetry {}
    const imports = [];
    let adapterImports = 0;
    let resolveAdapter;
    const adapterModule = new Promise(resolve => {
      resolveAdapter = resolve;
    });
    const aiModule = {
      registerTelemetry(...integrations) {
        globalThis.AI_SDK_TELEMETRY_INTEGRATIONS.push(...integrations);
      },
    };

    const importModule = async specifier => {
      imports.push(specifier);
      if (specifier === "ai") return aiModule;

      adapterImports += 1;
      if (adapterImports === 2) resolveAdapter({ OpenTelemetry });
      return adapterModule;
    };

    const [first, second] = await Promise.all([
      ensureAISDKTelemetry({ importModule }),
      ensureAISDKTelemetry({ importModule }),
    ]);

    assert.deepEqual(
      [first.status, second.status].sort(),
      ["already-registered", "registered"],
    );
    assert.ok(first.lease);
    assert.equal(second.lease, first.lease);
    assert.equal(first.lease.leases, 2);
    assert.equal(globalThis.AI_SDK_TELEMETRY_INTEGRATIONS.length, 1);
    assert.equal(adapterImports, 2);
    assert.deepEqual(imports.sort(), ["@ai-sdk/otel", "@ai-sdk/otel", "ai", "ai"]);

    assert.equal(releaseOwnedAISDKTelemetry(first.lease), false);
    assert.equal(globalThis.AI_SDK_TELEMETRY_INTEGRATIONS.length, 1);
    assert.equal(releaseOwnedAISDKTelemetry(second.lease), true);
    assert.equal(globalThis.AI_SDK_TELEMETRY_INTEGRATIONS.length, 0);
  });
});

test("a second lifecycle gets a fresh adapter and preserves user integrations", async () => {
  await withCleanTelemetryGlobals(async () => {
    class UserTelemetry {}
    const userIntegration = new UserTelemetry();
    globalThis.AI_SDK_TELEMETRY_INTEGRATIONS.push(userIntegration);

    const imports = [];
    const { importModule } = fakeAISDK7Modules(imports);

    const firstCycle = await ensureAISDKTelemetry({ importModule });
    assert.ok(firstCycle.lease);
    const firstAdapter = firstCycle.lease.integration;
    assert.equal(releaseOwnedAISDKTelemetry(firstCycle.lease), true);
    assert.deepEqual(globalThis.AI_SDK_TELEMETRY_INTEGRATIONS, [userIntegration]);

    const secondCycle = await ensureAISDKTelemetry({ importModule });
    assert.ok(secondCycle.lease);
    assert.notEqual(secondCycle.lease.integration, firstAdapter);
    assert.deepEqual(globalThis.AI_SDK_TELEMETRY_INTEGRATIONS, [
      userIntegration,
      secondCycle.lease.integration,
    ]);

    assert.equal(releaseOwnedAISDKTelemetry(secondCycle.lease), true);
    assert.deepEqual(globalThis.AI_SDK_TELEMETRY_INTEGRATIONS, [userIntegration]);
  });
});

test("final release removes only the exact Respan-owned integration", async () => {
  await withCleanTelemetryGlobals(async () => {
    const imports = [];
    const { OpenTelemetry, importModule } = fakeAISDK7Modules(imports);
    const registration = await ensureAISDKTelemetry({ importModule });
    assert.ok(registration.lease);

    const userIntegration = new OpenTelemetry();
    globalThis.AI_SDK_TELEMETRY_INTEGRATIONS.push(userIntegration);

    assert.equal(releaseOwnedAISDKTelemetry(registration.lease), true);
    assert.deepEqual(globalThis.AI_SDK_TELEMETRY_INTEGRATIONS, [userIntegration]);
  });
});

test("activation fails clearly when no compatible transformer host is active", async () => {
  const instrumentor = new VercelAIInstrumentor({
    autoRegisterAISDKTelemetry: false,
  });

  await assert.rejects(
    instrumentor.activate(),
    /No compatible Respan span-transformer host is active/,
  );
  assert.equal(instrumentor.isActive(), false);
  assert.deepEqual(getRegisteredSpanTransformerKeys(), []);
});

test("real OTel 2 provider translates once, drains in-flight spans, and reactivates", async () => {
  const harness = createOTel2Harness();
  assert.equal(
    typeof harness.provider.addSpanProcessor,
    "undefined",
    "the regression harness must exercise the OTel 2 provider API",
  );
  try {
    const first = new VercelAIInstrumentor({ autoRegisterAISDKTelemetry: false });
    const second = new VercelAIInstrumentor({ autoRegisterAISDKTelemetry: false });

    await first.activate();
    await second.activate();
    assert.deepEqual(getRegisteredSpanTransformerKeys(), [
      "@respan/instrumentation-vercel",
    ]);

    const firstSpan = harness.tracer.startSpan("chat gpt-4o-mini", {
      attributes: {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.input.messages": JSON.stringify([
          { role: "user", parts: [{ type: "text", content: "hello" }] },
        ]),
        "gen_ai.output.messages": JSON.stringify([
          { role: "assistant", parts: [{ type: "text", content: "world" }] },
        ]),
      },
    });
    firstSpan.end();
    assert.equal(harness.exporter.getFinishedSpans().length, 1);
    assert.equal(
      harness.exporter.getFinishedSpans()[0].attributes["respan.entity.log_type"],
      "text",
    );

    first.deactivate();
    const inFlightSpan = harness.tracer.startSpan("chat gpt-4o-mini", {
      attributes: {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.input.messages": JSON.stringify([
          { role: "user", parts: [{ type: "text", content: "in flight" }] },
        ]),
        "gen_ai.output.messages": JSON.stringify([
          { role: "assistant", parts: [{ type: "text", content: "drained" }] },
        ]),
      },
    });
    second.deactivate();
    inFlightSpan.end();
    assert.equal(
      harness.exporter.getFinishedSpans()[1].attributes["respan.entity.log_type"],
      "text",
      "a span started while active is fully translated after final deactivate",
    );
    assert.deepEqual(getRegisteredSpanTransformerKeys(), []);

    const inactiveSpan = harness.tracer.startSpan("chat gpt-4o-mini", {
      attributes: { "gen_ai.operation.name": "chat" },
    });
    inactiveSpan.end();
    assert.equal(harness.exporter.getFinishedSpans().length, 2);

    await first.activate();
    const reactivatedSpan = harness.tracer.startSpan("chat gpt-4o-mini", {
      attributes: {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "gpt-4o-mini",
      },
    });
    reactivatedSpan.end();
    assert.equal(harness.exporter.getFinishedSpans().length, 3);
    first.deactivate();
  } finally {
    await harness.provider.shutdown();
  }
});

test("failed adapter activation rolls back translator ownership and remains retryable", async () => {
  const harness = createOTel2Harness();

  class RetryableInstrumentor extends VercelAIInstrumentor {
    attempts = 0;

    async _ensureAISDKTelemetry() {
      this.attempts += 1;
      if (this.attempts === 1) {
        throw new Error("adapter registration failed");
      }
      return { status: "legacy", lease: undefined };
    }
  }

  try {
    const instrumentor = new RetryableInstrumentor();
    await assert.rejects(
      instrumentor.activate(),
      /adapter registration failed/,
    );

    assert.equal(instrumentor.isActive(), false);
    assert.deepEqual(getRegisteredSpanTransformerKeys(), []);

    await instrumentor.activate();
    assert.equal(instrumentor.isActive(), true);
    assert.deepEqual(getRegisteredSpanTransformerKeys(), [
      "@respan/instrumentation-vercel",
    ]);
    instrumentor.deactivate();
  } finally {
    await harness.provider.shutdown();
  }
});

test("runtime modules resolve from the host application", () => {
  const resolvedPath = "/host/app/node_modules/ai/dist/index.js";
  const url = resolveRuntimeModuleURL("ai", {
    hostResolve(specifier) {
      assert.equal(specifier, "ai");
      return resolvedPath;
    },
  });

  assert.equal(fileURLToPath(url), resolvedPath);
});

test("AI SDK 4-6 keep their native experimental telemetry path", async () => {
  await withCleanTelemetryGlobals(async () => {
    const imports = [];
    const result = await ensureAISDKTelemetry({
      importModule: async specifier => {
        imports.push(specifier);
        return {};
      },
    });

    assert.equal(result.status, "legacy");
    assert.equal(result.lease, undefined);
    assert.deepEqual(imports, ["ai"]);
  });
});

test("AI SDK 7 warns when its optional OpenTelemetry adapter is absent", async () => {
  await withCleanTelemetryGlobals(async () => {
    const warnings = [];
    const result = await ensureAISDKTelemetry({
      importModule: async specifier => {
        if (specifier === "ai") return { registerTelemetry() {} };
        const error = new Error("missing adapter");
        error.code = "ERR_MODULE_NOT_FOUND";
        throw error;
      },
      warn: message => warnings.push(message),
    });

    assert.equal(result.status, "missing-adapter");
    assert.equal(result.lease, undefined);
    assert.equal(warnings.length, 1);
    assert.match(warnings[0], /npm install @ai-sdk\/otel/);
  });
});

test("a user-registered OpenTelemetry integration is never leased", async () => {
  await withCleanTelemetryGlobals(async () => {
    class OpenTelemetry {}
    const userIntegration = new OpenTelemetry();
    globalThis.AI_SDK_TELEMETRY_INTEGRATIONS.push(userIntegration);
    const imports = [];
    const result = await ensureAISDKTelemetry({
      importModule: async specifier => {
        imports.push(specifier);
        return { registerTelemetry() {} };
      },
    });

    assert.equal(result.status, "already-registered");
    assert.equal(result.lease, undefined);
    assert.deepEqual(globalThis.AI_SDK_TELEMETRY_INTEGRATIONS, [userIntegration]);
    assert.deepEqual(imports, ["ai"]);
  });
});

for (const style of ["semantic", "legacy"]) {
  test(`${style} audio spans keep their content and exported names`, async () => {
    const harness = createOTel2Harness(style);
    const instrumentor = new VercelAIInstrumentor({ autoRegisterAISDKTelemetry: false });
    try {
      await instrumentor.activate();
      for (const [operation, semantic] of [["ai.generateSpeech", "speech"], ["ai.transcribe", "transcribe"], ["ai.streamTranscribe", "transcribe"]]) {
        const name = `${operation} audio-model`;
        const span = harness.tracer.startSpan(name, { attributes: {
          "gen_ai.operation.name": operation,
          "gen_ai.request.model": "audio-model",
          "gen_ai.provider.name": "test",
          "gen_ai.usage.input_tokens": 3,
          "ai.request.audio.size": 12,
          "ai.response.audio.size": 24,
          "gen_ai.output.messages": JSON.stringify([{ role: "assistant", parts: [{ type: "text", content: "hello" }] }]),
        } });
        span.end();
        await harness.provider.forceFlush();
        const exported = harness.exporter.getFinishedSpans().at(-1);
        assert.equal(exported.name, style === "semantic" ? semantic : name);
        assert.equal(exported.attributes["respan.entity.log_type"], "task");
        const metadata = JSON.parse(exported.attributes["respan.metadata"]);
        assert.equal(metadata.model, "audio-model");
        assert.equal(metadata.provider, "test");
        assert.deepEqual(metadata.usage, { input_tokens: 3 });
        assert.ok(exported.attributes["traceloop.entity.output"]);
        assert.ok(!Object.keys(exported.attributes).some(key => key.startsWith("respan.internal.")));
      }
    } finally {
      instrumentor.deactivate();
      await harness.provider.shutdown();
    }
  });
}
