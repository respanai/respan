import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";
import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import respanPiExtension, {
  SHUTDOWN_FLUSH_TIMEOUT_MS,
  createRespanPiExtension,
} from "../dist/extension.js";

function createFakePi() {
  const handlers = new Map();
  return {
    handlers,
    on(event, handler) {
      const list = handlers.get(event) ?? [];
      list.push(handler);
      handlers.set(event, list);
    },
    getAllTools() {
      return [];
    },
    async emit(event, payload = {}, ctx = undefined) {
      const results = [];
      for (const handler of handlers.get(event) ?? []) {
        results.push(await handler({ type: event, ...payload }, ctx));
      }
      return results;
    },
  };
}

function createFakeCtx() {
  const statuses = [];
  return {
    statuses,
    cwd: "/tmp/pi-demo",
    hasUI: true,
    mode: "tui",
    model: { id: "claude-sonnet-4-5", provider: "anthropic" },
    sessionManager: {
      getSessionId: () => "sess-ext",
      getSessionFile: () => undefined,
    },
    ui: {
      setStatus(key, text) {
        statuses.push([key, text]);
      },
    },
  };
}

function createFakeRespanFactory({
  rejectInitialize = false,
  rejectFlush = false,
  rejectShutdown = false,
  // Like the real Respan: activate the plugins on initialize, deactivate on shutdown.
  activate = false,
} = {}) {
  const calls = { initialize: 0, flush: 0, shutdown: 0, options: [] };
  const factory = (options) => {
    calls.options.push(options);
    return {
      async initialize() {
        calls.initialize += 1;
        if (rejectInitialize) {
          throw new Error("no network");
        }
        if (activate) {
          for (const instrumentation of options.instrumentations ?? []) {
            instrumentation.activate();
          }
        }
      },
      async flush() {
        calls.flush += 1;
        if (rejectFlush) {
          throw new Error("flush failed");
        }
      },
      async shutdown() {
        calls.shutdown += 1;
        if (activate) {
          for (const instrumentation of options.instrumentations ?? []) {
            instrumentation.deactivate();
          }
        }
        if (rejectShutdown) {
          throw new Error("shutdown failed");
        }
      },
    };
  };
  return { factory, calls };
}

/** A Respan whose flush()/shutdown() never resolve until `settle()` is called. */
function createDeferredRespanFactory() {
  const calls = { initialize: 0, flush: 0, shutdown: 0 };
  const pending = [];
  const factory = () => ({
    async initialize() {
      calls.initialize += 1;
    },
    flush() {
      calls.flush += 1;
      return new Promise((resolve) => pending.push(resolve));
    },
    shutdown() {
      calls.shutdown += 1;
      return new Promise((resolve) => pending.push(resolve));
    },
  });
  return {
    factory,
    calls,
    pending,
    settle() {
      for (const resolve of pending.splice(0)) {
        resolve();
      }
    },
  };
}

function withStubbedTracerProvider(onEnd, fn) {
  const original = trace.getTracerProvider.bind(trace);
  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value() {
      return {
        getTracer: (...args) => new BasicTracerProvider().getTracer(...args),
        activeSpanProcessor: { onEnd },
      };
    },
  });
  return Promise.resolve()
    .then(fn)
    .finally(() => {
      Object.defineProperty(trace, "getTracerProvider", {
        configurable: true,
        writable: true,
        value: original,
      });
    });
}

test("default export is a pi extension factory", () => {
  assert.equal(typeof respanPiExtension, "function");
  assert.equal(typeof createRespanPiExtension, "function");
});

test("enabled config initializes once, flushes after runs and shuts down on quit", async () => {
  const { factory, calls } = createFakeRespanFactory();
  const logs = [];
  const extension = createRespanPiExtension({
    config: {
      enabled: true,
      apiKey: "sk-test",
      baseURL: "https://api.respan.ai/api",
      workflowName: "pi-test",
      customerIdentifier: "cust-9",
      metadata: { team: "qa" },
      debug: true,
    },
    createRespan: factory,
    log: (message) => logs.push(message),
  });
  const pi = createFakePi();
  extension(pi);

  assert.equal(calls.options.length, 1);
  const options = calls.options[0];
  assert.equal(options.apiKey, "sk-test");
  assert.equal(options.baseURL, "https://api.respan.ai/api");
  assert.equal(options.appName, "pi-test");
  assert.equal(options.silenceInitializationMessage, true);
  assert.equal(options.logLevel, "debug");
  assert.equal(options.instrumentations.length, 1);
  assert.equal(options.instrumentations[0].name, "pi");
  assert.equal(options.instrumentations[0].traceScope, "session");
  assert.equal(calls.initialize, 0);

  // The tracer handlers and the lifecycle handlers are both registered.
  for (const event of [
    "session_start",
    "before_agent_start",
    "agent_end",
    "session_shutdown",
    "message_end",
  ]) {
    assert.ok(pi.handlers.has(event), `handler for ${event}`);
  }

  const ctx = createFakeCtx();
  await pi.emit("session_start", { reason: "startup" }, ctx);
  await pi.emit("session_start", { reason: "reload" }, ctx);
  assert.equal(calls.initialize, 1);
  assert.deepEqual(ctx.statuses.at(-1), ["respan", "Respan: tracing"]);

  await pi.emit("agent_end", { messages: [] }, ctx);
  assert.equal(calls.flush, 1);
  await pi.emit(
    "session_compact",
    { compactionEntry: { summary: "s" }, reason: "manual", willRetry: false },
    ctx,
  );
  assert.equal(calls.flush, 2);
  await pi.emit("session_tree", { newLeafId: "a", oldLeafId: "b" }, ctx);
  assert.equal(calls.flush, 3);

  await pi.emit("session_shutdown", { reason: "reload" }, ctx);
  assert.equal(calls.flush, 4);
  assert.equal(calls.shutdown, 0);
  assert.deepEqual(ctx.statuses.at(-1), ["respan", undefined]);

  await pi.emit("session_shutdown", { reason: "quit" }, ctx);
  assert.equal(calls.shutdown, 1);
  assert.ok(logs.some((line) => line.includes("tracing initialized")));
});

test("disabled config registers only session_start and never creates Respan", async () => {
  const { factory, calls } = createFakeRespanFactory();
  const pi = createFakePi();
  createRespanPiExtension({
    config: { enabled: false },
    createRespan: factory,
  })(pi);
  assert.deepEqual([...pi.handlers.keys()], ["session_start"]);
  assert.equal(calls.options.length, 0);

  const ctx = createFakeCtx();
  await pi.emit("session_start", { reason: "startup" }, ctx);
  assert.deepEqual(ctx.statuses, [
    ["respan", "Respan: tracing off (run `respan integrate pi`)"],
  ]);
  await pi.emit("session_start", { reason: "startup" }, undefined); // no ctx → no throw

  // An API key is required even when enabled is true.
  const noKey = createFakePi();
  createRespanPiExtension({ config: { enabled: true }, createRespan: factory })(
    noKey,
  );
  assert.deepEqual([...noKey.handlers.keys()], ["session_start"]);
  assert.equal(calls.options.length, 0);
});

test("handlers never throw when Respan rejects", async () => {
  const { factory, calls } = createFakeRespanFactory({
    rejectInitialize: true,
  });
  const pi = createFakePi();
  createRespanPiExtension({
    config: { enabled: true, apiKey: "sk-test" },
    createRespan: factory,
  })(pi);
  const ctx = createFakeCtx();
  await pi.emit("session_start", { reason: "startup" }, ctx);
  assert.equal(calls.initialize, 1);
  assert.deepEqual(ctx.statuses.at(-1), [
    "respan",
    "Respan: tracing unavailable: no network",
  ]);
  await pi.emit("agent_end", { messages: [] }, ctx);
  assert.equal(
    calls.flush,
    0,
    "flush is skipped when tracing never initialized",
  );
  await pi.emit("session_shutdown", { reason: "quit" }, ctx);
  assert.equal(calls.shutdown, 0);

  const flushFail = createFakeRespanFactory({
    rejectFlush: true,
    rejectShutdown: true,
  });
  const pi2 = createFakePi();
  createRespanPiExtension({
    config: { enabled: true, apiKey: "sk-test" },
    createRespan: flushFail.factory,
  })(pi2);
  const ctx2 = createFakeCtx();
  await pi2.emit("session_start", { reason: "startup" }, ctx2);
  await pi2.emit("agent_end", { messages: [] }, ctx2);
  await pi2.emit("session_shutdown", { reason: "quit" }, ctx2);
  assert.equal(flushFail.calls.flush, 2);
  assert.equal(flushFail.calls.shutdown, 1);
  assert.deepEqual(ctx2.statuses.at(-1), ["respan", undefined]);

  // Malformed events and a missing ctx are tolerated by every handler.
  for (const [event, handlers] of pi2.handlers) {
    for (const handler of handlers) {
      await handler(null, undefined);
      await handler({ type: event, message: 42, messages: "nope" }, {});
    }
  }
});

test("trace_scope from the config reaches the instrumentor", () => {
  const { factory, calls } = createFakeRespanFactory();
  createRespanPiExtension({
    config: { enabled: true, apiKey: "sk-test", traceScope: "session" },
    createRespan: factory,
  })(createFakePi());
  assert.equal(calls.options[0].instrumentations[0].traceScope, "session");
});

test("exports never block pi: run-end flushes run in the background, shutdown is bounded", async () => {
  assert.equal(SHUTDOWN_FLUSH_TIMEOUT_MS, 3000);
  const deferred = createDeferredRespanFactory();
  const pi = createFakePi();
  createRespanPiExtension({
    config: { enabled: true, apiKey: "sk-test" },
    createRespan: deferred.factory,
    shutdownTimeoutMs: 50,
  })(pi);
  const ctx = createFakeCtx();
  await pi.emit("session_start", { reason: "startup" }, ctx);

  // These handlers resolve although every flush is still in flight (an
  // awaited flush would never let pi.emit() resolve here).
  await pi.emit("agent_end", { messages: [] }, ctx);
  await pi.emit(
    "session_compact",
    { compactionEntry: { summary: "s" }, reason: "manual", willRetry: false },
    ctx,
  );
  await pi.emit("session_tree", { newLeafId: "a", oldLeafId: "b" }, ctx);
  assert.equal(deferred.calls.flush, 3);
  assert.equal(
    deferred.pending.length,
    3,
    "flushes were started and are still pending",
  );

  // session_shutdown waits for the final export, but never longer than the bound.
  // A real pi process always has live handles; keep the loop alive here so the
  // extension's `beforeExit` flush hook does not fire while the fake export hangs.
  const keepAlive = setInterval(() => {}, 10);
  const startedAt = Date.now();
  await pi.emit("session_shutdown", { reason: "quit" }, ctx);
  const elapsed = Date.now() - startedAt;
  clearInterval(keepAlive);
  assert.ok(elapsed >= 40 && elapsed < 2000, `bounded wait, took ${elapsed}ms`);
  assert.equal(deferred.calls.flush, 4);
  assert.equal(
    deferred.calls.shutdown,
    0,
    "shutdown() was not reached while the flush hung",
  );
  assert.deepEqual(ctx.statuses.at(-1), ["respan", undefined]);

  // Once the export completes, the shutdown proceeds in the background.
  deferred.settle();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(deferred.calls.shutdown, 1);
  deferred.settle();
});

test("Respan's console chatter is filtered while pi's own console output passes through", async () => {
  const received = [];
  const stdout = [];
  const stderr = [];
  const originalStdoutWrite = process.stdout.write;
  const originalStderrWrite = process.stderr.write;
  await withStubbedTracerProvider(
    // What @respan/tracing prints on EVERY injected span regardless of logLevel.
    (span) => {
      console.debug(
        `[Respan Debug] Processing enriched Respan span: ${span.name}`,
      );
      console.log(
        `[Respan] Sending span "${span.name}" to processor "default"`,
      );
      received.push(span);
    },
    async () => {
      // console.* writes strings. Anything else on these streams is the node
      // test runner's own binary IPC (a previous test's `test:complete`
      // message can land here when files run in a child process), which
      // must be forwarded, not recorded.
      process.stdout.write = (chunk, ...rest) => {
        if (typeof chunk === "string") {
          stdout.push(chunk);
          return true;
        }
        return originalStdoutWrite.call(process.stdout, chunk, ...rest);
      };
      process.stderr.write = (chunk, ...rest) => {
        if (typeof chunk === "string") {
          stderr.push(chunk);
          return true;
        }
        return originalStderrWrite.call(process.stderr, chunk, ...rest);
      };
      try {
        const { factory, calls } = createFakeRespanFactory({ activate: true });
        const pi = createFakePi();
        createRespanPiExtension({
          config: { enabled: true, apiKey: "sk-test", debug: false },
          createRespan: factory,
        })(pi);
        const ctx = createFakeCtx();
        await pi.emit("session_start", { reason: "startup" }, ctx);
        await pi.emit(
          "before_agent_start",
          { prompt: "hi", systemPrompt: "sys" },
          ctx,
        );
        await pi.emit("agent_start", {}, ctx);
        await pi.emit(
          "context",
          { messages: [{ role: "user", content: "hi" }] },
          ctx,
        );
        await pi.emit(
          "message_end",
          {
            message: {
              role: "assistant",
              content: [{ type: "text", text: "hello" }],
              stopReason: "stop",
              usage: { input: 1, output: 1 },
            },
          },
          ctx,
        );
        await pi.emit("agent_end", { messages: [] }, ctx);
        assert.equal(
          received.length,
          2,
          "chat and agent spans reached the processor",
        );
        assert.deepEqual(stdout, [], "no [Respan …] line reached stdout");

        // Everything else is forwarded untouched, even while flushes are in flight.
        console.log("pi output");
        console.debug("[other] extension output");
        console.warn("[Respan] Cannot inject span — SDK not initialized");
        console.warn("other warning");
        assert.deepEqual(stdout, ["pi output\n", "[other] extension output\n"]);
        assert.deepEqual(stderr, ["other warning\n"]);

        await pi.emit("session_shutdown", { reason: "quit" }, ctx);
        assert.equal(calls.shutdown, 1);
        console.log("after quit");
        assert.deepEqual(stdout, [
          "pi output\n",
          "[other] extension output\n",
          "after quit\n",
        ]);
      } finally {
        process.stdout.write = originalStdoutWrite;
        process.stderr.write = originalStderrWrite;
      }
    },
  );
});

test("each factory invocation with an injected Respan gets its own runtime", async () => {
  const first = createFakeRespanFactory();
  const second = createFakeRespanFactory();
  const piA = createFakePi();
  const piB = createFakePi();
  createRespanPiExtension({
    config: { enabled: true, apiKey: "sk-a" },
    createRespan: first.factory,
  })(piA);
  createRespanPiExtension({
    config: { enabled: true, apiKey: "sk-a" },
    createRespan: second.factory,
  })(piB);
  await piA.emit("session_start", { reason: "startup" }, createFakeCtx());
  await piB.emit("session_start", { reason: "startup" }, createFakeCtx());
  assert.equal(first.calls.initialize, 1);
  assert.equal(second.calls.initialize, 1);
});

test("initializes eagerly without session_start (SDK sessions) and flushes on beforeExit", async () => {
  const { factory, calls } = createFakeRespanFactory({ activate: true });
  const extension = createRespanPiExtension({
    config: { enabled: true, apiKey: "sk-test", debug: false },
    createRespan: factory,
    log: () => {},
  });
  const pi = createFakePi();
  extension(pi);
  // No session_start: SDK scripts (`createAgentSession()` + `prompt()`) never emit it.
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(calls.initialize, 1);

  await pi.emit("agent_end", { messages: [] }, createFakeCtx());
  assert.equal(calls.flush, 1);

  // The process-level hook flushes whatever is still batched when the loop drains.
  process.emit("beforeExit", 0);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(calls.flush, 2);
});

test("after each run the TUI shows a link to the trace (Respan cloud only)", async () => {
  const { platformTraceUrl } = await import("../dist/extension.js");
  assert.equal(
    platformTraceUrl(
      "https://api.respan.ai",
      "0123456789abcdef0123456789abcdef",
    ),
    "https://platform.respan.ai/platform/traces?trace_unique_id=0123456789abcdef0123456789abcdef",
  );
  assert.equal(
    platformTraceUrl(
      "https://api.respan.ai/api",
      "0123456789abcdef0123456789abcdef",
    ).startsWith("https://platform.respan.ai/"),
    true,
  );
  assert.equal(
    platformTraceUrl(
      "https://respan.internal.example.com/api",
      "0123456789abcdef0123456789abcdef",
    ),
    undefined,
  );
  assert.equal(platformTraceUrl(undefined, "not-a-trace-id"), undefined);

  const { factory } = createFakeRespanFactory({ activate: true });
  const extension = createRespanPiExtension({
    config: {
      enabled: true,
      apiKey: "sk-test",
      baseURL: "https://api.respan.ai",
      debug: false,
    },
    createRespan: factory,
    log: () => {},
  });
  const pi = createFakePi();
  extension(pi);
  const widgets = [];
  const ctx = createFakeCtx();
  ctx.ui.setWidget = (key, lines, options) =>
    widgets.push([key, lines, options]);
  await pi.emit("session_start", { reason: "startup" }, ctx);
  await pi.emit(
    "before_agent_start",
    { prompt: "hi", systemPrompt: "sys" },
    ctx,
  );
  await pi.emit("agent_start", {}, ctx);
  await pi.emit("agent_end", { messages: [] }, ctx);

  const shown = widgets.find(([key, lines]) => key === "respan-trace" && lines);
  assert.ok(shown, "trace link widget was set");
  assert.match(shown[1][0], /Respan trace \(turn 1\)/);
  assert.match(
    shown[1][1],
    /^https:\/\/platform\.respan\.ai\/platform\/traces\?trace_unique_id=[0-9a-f]{32}$/,
  );
  assert.deepEqual(shown[2], { placement: "belowEditor" });

  await pi.emit("session_shutdown", { reason: "quit" }, ctx);
  assert.deepEqual(widgets.at(-1), ["respan-trace", undefined, undefined]);
});
