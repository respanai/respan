import test from "node:test";
import assert from "node:assert/strict";

import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";
import { RespanCompositeProcessor } from "@respan/tracing/dist/processor/composite.js";
import { EveInstrumentor } from "../dist/index.js";

class RecordingManager {
  started = [];
  ended = [];

  onStart(span) {
    this.started.push(span);
  }

  onEnd(span) {
    this.ended.push(span);
  }

  async shutdown() {}
  async forceFlush() {}
}

function startTurn(tracer, sessionId) {
  return tracer.startSpan("ai.eve.turn", {
    attributes: {
      "ai.telemetry.functionId": "support-agent",
      "eve.environment": "test",
      "eve.session.id": sessionId,
      "eve.version": "0.26.1",
    },
  });
}

function assertCanonicalTurn(span, sessionId) {
  assert.equal(span.attributes["respan.entity.log_type"], "agent");
  assert.equal(span.attributes["traceloop.entity.name"], "support-agent");
  assert.equal(span.attributes["respan.threads.thread_identifier"], sessionId);
  assert.equal(span.attributes["traceloop.workflow.name"], "support-agent");
  assert.equal(span.attributes["eve.session.id"], undefined);
  assert.equal(span.attributes["traceloop.span.kind"], undefined);
}

test("OTel 2.10 transformers support cached tracers, shared ownership, and drain-safe deactivation", async () => {
  const manager = new RecordingManager();
  const composite = new RespanCompositeProcessor(manager);
  const provider = new BasicTracerProvider({ spanProcessors: [composite] });
  const tracerBeforeActivation = provider.getTracer("eve");
  const activeProcessorBefore = provider._activeSpanProcessor;
  const first = new EveInstrumentor();
  const second = new EveInstrumentor();

  try {
    first.activate();
    second.activate();
    assert.equal(provider._activeSpanProcessor, activeProcessorBefore);
    assert.equal(first.isActive(), true);
    assert.equal(second.isActive(), true);

    startTurn(tracerBeforeActivation, "session-both").end();
    assertCanonicalTurn(manager.ended[0], "session-both");

    first.deactivate();
    const tracerAfterActivation = provider.getTracer("eve.after-activation");
    startTurn(tracerAfterActivation, "session-second-owner").end();
    assertCanonicalTurn(manager.ended[1], "session-second-owner");

    const draining = startTurn(tracerBeforeActivation, "session-draining");
    second.deactivate();
    draining.end();
    assertCanonicalTurn(manager.ended[2], "session-draining");

    const inactive = tracerAfterActivation.startSpan("ai.eve.turn", {
      attributes: {
        "eve.session.id": "session-after-deactivation",
        "gen_ai.request.model": "raw-model-keeps-span-routable",
      },
    });
    inactive.end();
    assert.equal(
      manager.ended[3].attributes["respan.entity.log_type"],
      undefined,
    );
    assert.equal(
      manager.ended[3].attributes["eve.session.id"],
      "session-after-deactivation",
    );
  } finally {
    first.deactivate();
    second.deactivate();
    await provider.shutdown();
  }

  const noHost = new EveInstrumentor();
  assert.throws(
    () => noHost.activate(),
    /No compatible Respan span-transformer host is active/,
  );
  assert.equal(noHost.isActive(), false);
});
