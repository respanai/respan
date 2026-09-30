import assert from "node:assert/strict";
import test from "node:test";

import { context } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { RespanSpanAttributes } from "@respan/respan-sdk";

import { propagateAttributes } from "../dist/contexts/propagation.js";
import { RespanCompositeProcessor } from "../dist/processor/composite.js";
import { buildReadableSpan } from "../dist/utils/spanFactory.js";

const METADATA = RespanSpanAttributes.RESPAN_METADATA;

function assertSingleMetadataAttribute(attributes) {
  assert.equal(typeof attributes[METADATA], "string");
  assert.equal(Object.keys(attributes).some((key) => key.startsWith(`${METADATA}.`)), false);
}

test("synthetic spans merge propagated metadata into one canonical JSON attribute", () => {
  const manager = new AsyncLocalStorageContextManager().enable();
  context.setGlobalContextManager(manager);
  try {
    const span = propagateAttributes(
      { metadata: { run_id: "outer", shared: "outer" } },
      () => propagateAttributes(
        { metadata: { run_id: "inner", shared: "propagated", source: "propagated" } },
        () => buildReadableSpan({
          name: "vendor.call",
          attributes: {
            [`${METADATA}.source`]: "sdk",
            [METADATA]: JSON.stringify({
              provider_request_id: "request-1",
              shared: "instrumentation",
            }),
          },
        }),
      ),
    );

    assertSingleMetadataAttribute(span.attributes);
    assert.deepEqual(JSON.parse(span.attributes[METADATA]), {
      provider_request_id: "request-1",
      run_id: "inner",
      shared: "instrumentation",
      source: "sdk",
    });
  } finally {
    context.disable();
  }
});

test("live span processing merges propagated metadata without aliases", async () => {
  const manager = new AsyncLocalStorageContextManager().enable();
  context.setGlobalContextManager(manager);
  const forwarded = [];
  const processor = new RespanCompositeProcessor({
    onStart(span) {
      forwarded.push(span);
    },
    onEnd() {},
    forceFlush: async () => {},
    shutdown: async () => {},
  });
  const span = {
    name: "vendor.call",
    attributes: {
      [`${METADATA}.source`]: "sdk",
      [METADATA]: JSON.stringify({
        provider_request_id: "request-1",
        shared: "instrumentation",
      }),
    },
    setAttribute(key, value) {
      this.attributes[key] = value;
      return this;
    },
  };

  try {
    propagateAttributes(
      { metadata: { run_id: "run-1", shared: "propagated", source: "propagated" } },
      () => processor.onStart(span, context.active()),
    );

    assert.equal(forwarded.length, 1);
    assertSingleMetadataAttribute(span.attributes);
    assert.deepEqual(JSON.parse(span.attributes[METADATA]), {
      provider_request_id: "request-1",
      run_id: "run-1",
      shared: "instrumentation",
      source: "sdk",
    });
  } finally {
    await processor.shutdown();
    context.disable();
  }
});
