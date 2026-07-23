import test from "node:test";
import assert from "node:assert/strict";

import { RespanCompositeProcessor } from "../dist/processor/composite.js";
import { RespanSpanAttributes } from "@respan/respan-sdk";

class RecordingManager {
  constructor() {
    this.started = [];
    this.ended = [];
  }
  onStart(span) {
    this.started.push(span);
  }
  onEnd(span) {
    this.ended.push(span);
  }
  async shutdown() {}
  async forceFlush() {}
}

function span(name, attributes = {}) {
  return { name, attributes };
}

test("composite processor routes structural spans so exporter can drop and reparent them", () => {
  const manager = new RecordingManager();
  const processor = new RespanCompositeProcessor(manager);

  processor.onEnd(
    span("ai.generateText", {
      [RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN]: true,
      "traceloop.entity.path": "agent.triage-service",
      [RespanSpanAttributes.RESPAN_LOG_TYPE]: "text",
    })
  );

  assert.equal(manager.ended.length, 1);
  assert.equal(manager.ended[0].name, "ai.generateText");
});
