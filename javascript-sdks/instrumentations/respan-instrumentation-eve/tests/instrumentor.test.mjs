import test from "node:test";
import assert from "node:assert/strict";

import { trace } from "@opentelemetry/api";
import { EveInstrumentor } from "../dist/index.js";

function makeTurn(sessionId, suffix) {
  const attributes = {
    "ai.telemetry.functionId": "support-agent",
    "eve.environment": "test",
    "eve.session.id": sessionId,
    "eve.version": "0.26.1",
  };
  return {
    name: "ai.eve.turn",
    instrumentationScope: { name: "eve" },
    attributes,
    spanContext() {
      return {
        traceId: suffix.padStart(32, "0"),
        spanId: suffix.padStart(16, "0"),
        traceFlags: 1,
      };
    },
    setAttribute(key, value) {
      attributes[key] = value;
    },
  };
}

function assertCanonicalTurn(span, sessionId) {
  assert.equal(span.attributes["respan.entity.log_type"], "agent");
  assert.equal(span.attributes["traceloop.entity.name"], "support-agent");
  assert.equal(
    span.attributes["respan.threads.thread_identifier"],
    sessionId,
  );
  assert.equal(span.attributes["traceloop.workflow.name"], "support-agent");
  assert.equal(span.attributes["eve.session.id"], undefined);
}

test("wraps the active processor with shared ownership and drain-safe deactivation", () => {
  const delegatedSpans = [];
  const originalProcessor = {
    onStart() {},
    onEnd(span) {
      delegatedSpans.push(span);
    },
    shutdown() {
      return Promise.resolve();
    },
    forceFlush() {
      return Promise.resolve();
    },
  };
  const provider = { activeSpanProcessor: originalProcessor };
  const originalGetTracerProvider = trace.getTracerProvider.bind(trace);

  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value: () => provider,
  });

  const first = new EveInstrumentor();
  const second = new EveInstrumentor();

  try {
    first.activate();
    second.activate();

    const wrapper = provider.activeSpanProcessor;
    assert.notEqual(wrapper, originalProcessor);
    assert.equal(first.isActive(), true);
    assert.equal(second.isActive(), true);

    const both = makeTurn("session-both", "1");
    wrapper.onStart(both, undefined);
    wrapper.onEnd(both);
    assertCanonicalTurn(delegatedSpans[0], "session-both");

    first.deactivate();
    assert.equal(provider.activeSpanProcessor, wrapper);

    const secondOwner = makeTurn("session-second-owner", "2");
    wrapper.onStart(secondOwner, undefined);
    wrapper.onEnd(secondOwner);
    assertCanonicalTurn(delegatedSpans[1], "session-second-owner");

    const draining = makeTurn("session-draining", "3");
    wrapper.onStart(draining, undefined);
    second.deactivate();
    assert.equal(provider.activeSpanProcessor, originalProcessor);
    wrapper.onEnd(draining);
    assertCanonicalTurn(delegatedSpans[2], "session-draining");

    const inactive = makeTurn("session-after-deactivation", "4");
    originalProcessor.onStart(inactive, undefined);
    originalProcessor.onEnd(inactive);
    assert.equal(
      delegatedSpans[3].attributes["respan.entity.log_type"],
      undefined,
    );
  } finally {
    first.deactivate();
    second.deactivate();
    Object.defineProperty(trace, "getTracerProvider", {
      configurable: true,
      writable: true,
      value: originalGetTracerProvider,
    });
  }
});
