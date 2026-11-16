// A LiveKit call as a Respan session: one trace per turn, timings on its root.
//
// Spans are made by the real OTel tracer in the shape LiveKit Agents traces a
// call: agent_session at the root, a user_turn and one agent_turn per reply
// under it, llm_node and tts_node under each turn. Children end before their
// turn, as in LiveKit.
import assert from "node:assert/strict";
import test from "node:test";

import { context, trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
} from "@opentelemetry/sdk-trace-base";
import {
  MultiProcessorManager,
  RespanCompositeProcessor,
} from "@respan/tracing/dist/processor/index.js";

import { LiveKitInstrumentor } from "../dist/index.js";

const SESSION = "respan.sessions.session_identifier";
const VOICE = "respan.metadata.voice.";

for (const style of ["semantic", "legacy"]) {
  test(`[${style}] each turn is its own trace in the call's session, with timings`, async () => {
    const harness = createHarness(style);
    const instrumentor = new LiveKitInstrumentor({
      telemetryModule: { tracer: {} },
      syncTracerProvider: false,
    });
    await instrumentor.activate();
    try {
      const call = runCall(harness.tracer);
      const spans = harness.exporter.getFinishedSpans();
      const turns = spans.filter((s) => entity(s) === "agent_turn");
      assert.equal(turns.length, 2);

      const turnTraces = new Set(turns.map((t) => t.spanContext().traceId));
      assert.equal(turnTraces.size, 2);
      assert.equal(turnTraces.has(call), false);
      for (const turn of turns) {
        assert.equal(turn.parentSpanContext?.spanId, undefined);
      }
      // Every span of a turn moved with it, still under the same parent span.
      for (const span of spans.filter((s) =>
        ["llm_node", "tts_node"].includes(entity(s)),
      )) {
        const traceId = span.spanContext().traceId;
        const owner = turns.find((t) => t.spanContext().traceId === traceId);
        assert.ok(owner, `${entity(span)} not in a turn's trace`);
        assert.equal(span.parentSpanContext.spanId, owner.spanContext().spanId);
        assert.equal(span.parentSpanContext.traceId, traceId);
      }
      // The call's own spans stay in the call's trace.
      for (const name of ["agent_session", "user_turn"]) {
        const span = spans.find((s) => entity(s) === name);
        assert.equal(span.spanContext().traceId, call);
      }
      // One session, the call's trace id, on every span; no internal hint left.
      assert.deepEqual(new Set(spans.map((s) => s.attributes[SESSION])), new Set([call]));
      for (const span of spans) {
        for (const key of Object.keys(span.attributes)) {
          assert.equal(key.startsWith("respan.internal."), false, key);
          assert.equal(key.startsWith("lk."), false, key);
        }
      }

      const [first, second] = turns.sort(
        (a, b) => a.attributes[`${VOICE}turn_index`] - b.attributes[`${VOICE}turn_index`],
      );
      assert.deepEqual(voice(first), {
        turn_index: 0,
        wait_ms: 820,
        stt_ms: 210,
        llm_ttft_ms: 390,
        tts_ttfa_ms: 180,
        interrupted: false,
      });
      assert.equal(voice(second).turn_index, 1);
      assert.equal(voice(second).wait_ms, 2650);
      assert.equal(voice(second).interrupted, true);
      for (const span of spans.filter((s) => entity(s) !== "agent_turn")) {
        assert.deepEqual(voice(span), {});
      }
    } finally {
      instrumentor.deactivate();
      await harness.provider.shutdown();
    }
  });
}

test("a session the user set wins, and bad timings are left out", async () => {
  const harness = createHarness("semantic");
  const instrumentor = new LiveKitInstrumentor({
    telemetryModule: { tracer: {} },
    syncTracerProvider: false,
  });
  await instrumentor.activate();
  try {
    const root = harness.tracer.startSpan("agent_session", {
      attributes: { [SESSION]: "mine" },
    });
    const turn = harness.tracer.startSpan(
      "agent_turn",
      { attributes: { [SESSION]: "mine" } },
      trace.setSpan(context.active(), root),
    );
    turn.setAttribute("lk.e2e_latency", -1);
    turn.setAttribute("lk.transcription_delay", Number.NaN);
    turn.end();
    root.end();

    const spans = harness.exporter.getFinishedSpans();
    assert.deepEqual(new Set(spans.map((s) => s.attributes[SESSION])), new Set(["mine"]));
    const exported = spans.find((s) => entity(s) === "agent_turn");
    assert.deepEqual(voice(exported), { turn_index: 0 });
  } finally {
    instrumentor.deactivate();
    await harness.provider.shutdown();
  }
});

function createHarness(spanNameStyle) {
  const exporter = new InMemorySpanExporter();
  const manager = new MultiProcessorManager({ disableBatch: true, spanNameStyle });
  manager.addProcessor({ exporter, name: "default", disableBatch: true });
  const provider = new BasicTracerProvider({
    spanProcessors: [new RespanCompositeProcessor(manager)],
  });
  return { exporter, provider, tracer: provider.getTracer("@livekit/agents") };
}

function runCall(tracer) {
  const session = tracer.startSpan("agent_session");
  const under = trace.setSpan(context.active(), session);
  tracer.startSpan("user_turn", {}, under).end();
  runTurn(tracer, under, { e2e: 0.82, stt: 0.21, ttft: 0.39, ttfa: 0.18, interrupted: false });
  runTurn(tracer, under, { e2e: 2.65, stt: 0.24, ttft: 1.9, ttfa: 0.3, interrupted: true });
  session.end();
  return session.spanContext().traceId;
}

function runTurn(tracer, parent, { e2e, stt, ttft, ttfa, interrupted }) {
  const turn = tracer.startSpan("agent_turn", {}, parent);
  const under = trace.setSpan(parent, turn);
  const node = tracer.startSpan("llm_node", {}, under);
  node.setAttribute("lk.response.ttft", ttft);
  node.end();
  // A follow-up LLM step after a tool call: not what the user waited on.
  const later = tracer.startSpan("llm_node", {}, under);
  later.setAttribute("lk.response.ttft", 9);
  later.end();
  const tts = tracer.startSpan("tts_node", {}, under);
  tts.setAttribute("lk.response.ttfb", ttfa);
  tts.end();
  turn.setAttribute("lk.e2e_latency", e2e);
  turn.setAttribute("lk.transcription_delay", stt);
  turn.setAttribute("lk.interrupted", interrupted);
  turn.end();
}

function entity(span) {
  return span.attributes["traceloop.entity.path"];
}

function voice(span) {
  return Object.fromEntries(
    Object.entries(span.attributes)
      .filter(([key]) => key.startsWith(VOICE))
      .map(([key, value]) => [key.slice(VOICE.length), value]),
  );
}
