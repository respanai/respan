import test from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";

import { trace } from "@opentelemetry/api";
import {
  ensureAISDKTelemetry,
  releaseOwnedAISDKTelemetry,
  resolveRuntimeModuleURL,
} from "../dist/_ai_sdk_telemetry.js";
import {
  VercelAIInstrumentor,
  VercelAITranslator,
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

function createTranslatorSpan() {
  const name = "ai.generateText.doGenerate";
  return {
    name,
    instrumentationScope: { name: "ai" },
    attributes: {
      "ai.model.id": "gpt-4o-mini",
      "ai.prompt": "hello",
      "ai.response.text": "world",
    },
    setAttribute(key, value) {
      this.attributes[key] = value;
    },
  };
}

function runTranslatorProcessor(processor) {
  const span = createTranslatorSpan();
  processor.onStart(span, undefined);
  processor.onEnd(span);
  return span.attributes;
}

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

test("translator ownership stops after final deactivate and resumes without duplicates", async () => {
  const originalDescriptor = Object.getOwnPropertyDescriptor(trace, "getTracerProvider");
  const firstProcessors = [];
  const secondProcessors = [];
  const firstProvider = {
    addSpanProcessor(processor) {
      firstProcessors.push(processor);
    },
  };
  const secondProvider = {
    addSpanProcessor(processor) {
      secondProcessors.push(processor);
    },
  };
  let currentProvider = firstProvider;

  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    value: () => currentProvider,
  });

  try {
    const first = new VercelAIInstrumentor({ autoRegisterAISDKTelemetry: false });
    const second = new VercelAIInstrumentor({ autoRegisterAISDKTelemetry: false });

    await first.activate();
    assert.equal(firstProcessors.length, 1);
    assert.ok(firstProcessors[0] instanceof VercelAITranslator);
    assert.equal(runTranslatorProcessor(firstProcessors[0])["respan.entity.log_type"], "text");

    await second.activate();
    assert.equal(firstProcessors.length, 1);

    first.deactivate();
    assert.equal(
      runTranslatorProcessor(firstProcessors[0])["respan.entity.log_type"],
      "text",
      "the remaining owner keeps translation active",
    );

    const inFlightSpan = createTranslatorSpan();
    firstProcessors[0].onStart(inFlightSpan, undefined);
    second.deactivate();
    firstProcessors[0].onEnd(inFlightSpan);
    assert.equal(
      inFlightSpan.attributes["respan.entity.log_type"],
      "text",
      "a span started while active is fully translated after final deactivate",
    );

    const inactiveAttrs = runTranslatorProcessor(firstProcessors[0]);
    assert.equal(inactiveAttrs["respan.entity.log_type"], undefined);
    assert.equal(inactiveAttrs["ai.model.id"], "gpt-4o-mini");

    await first.activate();
    assert.equal(firstProcessors.length, 1);
    const reactivatedAttrs = runTranslatorProcessor(firstProcessors[0]);
    assert.equal(reactivatedAttrs["respan.entity.log_type"], "text");
    assert.equal(reactivatedAttrs["traceloop.entity.path"], "");
    first.deactivate();

    currentProvider = secondProvider;
    const newProvider = new VercelAIInstrumentor({
      autoRegisterAISDKTelemetry: false,
    });
    await newProvider.activate();
    assert.equal(secondProcessors.length, 1);
    assert.ok(secondProcessors[0] instanceof VercelAITranslator);
    newProvider.deactivate();
  } finally {
    if (originalDescriptor) {
      Object.defineProperty(trace, "getTracerProvider", originalDescriptor);
    } else {
      delete trace.getTracerProvider;
    }
  }
});

test("failed adapter activation rolls back translator ownership and remains retryable", async () => {
  const originalDescriptor = Object.getOwnPropertyDescriptor(trace, "getTracerProvider");
  const processors = [];
  const provider = {
    addSpanProcessor(processor) {
      processors.push(processor);
    },
  };

  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    value: () => provider,
  });

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

    assert.equal(processors.length, 1);
    assert.equal(runTranslatorProcessor(processors[0])["respan.entity.log_type"], undefined);

    await instrumentor.activate();
    assert.equal(processors.length, 1);
    assert.equal(runTranslatorProcessor(processors[0])["respan.entity.log_type"], "text");
    instrumentor.deactivate();
  } finally {
    if (originalDescriptor) {
      Object.defineProperty(trace, "getTracerProvider", originalDescriptor);
    } else {
      delete trace.getTracerProvider;
    }
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
