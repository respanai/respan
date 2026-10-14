// Baseline wire contract: what a customer configures must reach every span, in
// the shape Respan ingestion reads. These run the real pi SDK in both modes.
//
// Ingestion (`normalize_span_attributes` in the backend) collects ONLY flat
// `respan.metadata.<key>` attributes; a JSON-string `respan.metadata` is
// dropped. Thread and trace group are what group a pi session's runs.
import assert from "node:assert/strict";
import test from "node:test";
import { context, trace } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";
import { PiInstrumentor } from "../dist/index.js";
import { nativeFixture } from "./native_runtime.mjs";

const provider = new BasicTracerProvider();
trace.setGlobalTracerProvider(provider);
const manager = new AsyncLocalStorageContextManager().enable();
context.setGlobalContextManager(manager);
test.after(async () => {
  context.disable();
  manager.disable();
  trace.disable();
  await provider.shutdown();
});

const CUSTOMER_METADATA = { agent_id: "a1", role: "planner", environment: "prod" };

async function session(extension, { prompts = 1, ...options } = {}) {
  const spans = [];
  const instrumentor = new PiInstrumentor({
    emit: (span) => spans.push(span),
    customerIdentifier: "customer-1",
    metadata: CUSTOMER_METADATA,
    ...options,
  });
  instrumentor.activate();
  const fixture = await nativeFixture({
    instrumentor,
    extension,
    responses: [{ text: "first" }, { text: "second" }],
  });
  try {
    for (let i = 0; i < prompts; i++)
      await fixture.session.prompt(`prompt ${i}`);
    return { spans, sessionId: fixture.session.sessionId };
  } finally {
    await fixture.close();
    instrumentor.deactivate();
  }
}

for (const extension of [false, true]) {
  const mode = extension ? "extension" : "attach";

  test(`baseline ${mode}: custom metadata is flat respan.metadata.<key> on every span`, async () => {
    const { spans } = await session(extension);
    assert.ok(spans.length >= 2);
    for (const span of spans) {
      for (const [key, value] of Object.entries(CUSTOMER_METADATA))
        assert.equal(span.attributes[`respan.metadata.${key}`], value, `${span.name} ${key}`);
      assert.equal(span.attributes["respan.metadata"], undefined, span.name);
    }
  });

  test(`baseline ${mode}: thread, session and trace group are the pi session id on every span`, async () => {
    const { spans, sessionId } = await session(extension);
    assert.ok(sessionId);
    for (const span of spans) {
      assert.equal(span.attributes["respan.threads.thread_identifier"], sessionId, span.name);
      assert.equal(span.attributes["respan.sessions.session_identifier"], sessionId, span.name);
      assert.equal(span.attributes["respan.trace.trace_group_identifier"], sessionId, span.name);
    }
  });

  test(`baseline ${mode}: customer identifier on every span`, async () => {
    const { spans } = await session(extension);
    for (const span of spans)
      assert.equal(span.attributes["respan.customer_params.customer_identifier"], "customer-1", span.name);
  });

  test(`baseline ${mode}: two prompts in one session share a thread and a trace`, async () => {
    const { spans, sessionId } = await session(extension, { prompts: 2 });
    const agents = spans.filter((s) => s.attributes["respan.entity.log_type"] === "agent");
    assert.equal(agents.length, 2);
    assert.equal(new Set(spans.map((s) => s.spanContext().traceId)).size, 1);
    assert.deepEqual(
      [...new Set(spans.map((s) => s.attributes["respan.threads.thread_identifier"]))],
      [sessionId],
    );
  });
}

test("baseline attach: attach(session, overrides) sets thread, customer and metadata", async () => {
  const spans = [];
  const instrumentor = new PiInstrumentor({ emit: (span) => spans.push(span) });
  instrumentor.activate();
  const fixture = await nativeFixture({ instrumentor: null });
  const detach = instrumentor.attach(fixture.session, {
    threadIdentifier: "my-thread",
    customerIdentifier: "customer-2",
    metadata: { role: "reviewer" },
  });
  try {
    await fixture.session.prompt("prompt");
  } finally {
    detach();
    await fixture.close();
    instrumentor.deactivate();
  }
  assert.ok(spans.length >= 2);
  for (const span of spans) {
    assert.equal(span.attributes["respan.threads.thread_identifier"], "my-thread", span.name);
    assert.equal(span.attributes["respan.customer_params.customer_identifier"], "customer-2", span.name);
    assert.equal(span.attributes["respan.metadata.role"], "reviewer", span.name);
    assert.equal(span.attributes["respan.trace.trace_group_identifier"], fixture.session.sessionId, span.name);
  }
});
