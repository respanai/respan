import test from "node:test";
import assert from "node:assert/strict";

import { context } from "@opentelemetry/api";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { RespanSpanAttributes } from "@respan/respan-sdk";

import { buildReadableSpan, propagateAttributes } from "../dist/index.js";
import { RespanCompositeProcessor } from "../dist/processor/composite.js";

const METADATA_ATTR = RespanSpanAttributes.RESPAN_METADATA;
const ALIAS_PREFIX = `${METADATA_ATTR}.`;

class RecordingManager {
  onStart() {}
  onEnd() {}
  async shutdown() {}
  async forceFlush() {}
}

// register() installs the async context manager propagateAttributes reads from.
const provider = new NodeTracerProvider({
  spanProcessors: [new RespanCompositeProcessor(new RecordingManager())],
});
provider.register();
const tracer = provider.getTracer("canonical-metadata-test");

function aliasKeys(attributes) {
  return Object.keys(attributes).filter((key) => key.startsWith(ALIAS_PREFIX));
}

function startedSpanAttributes(attributes, propagated) {
  return propagateAttributes(propagated, () => {
    const span = tracer.startSpan("metadata_span", { attributes }, context.active());
    span.end();
    return span.attributes;
  });
}

const PROPAGATED = {
  metadata: { foo: "propagated", tenant: "acme", shared: "propagated" },
};
const ALIASES = {
  [`${ALIAS_PREFIX}foo`]: "alias",
  [`${ALIAS_PREFIX}shared`]: "alias",
};
const EXPECTED_ALIAS_MERGE = { foo: "alias", tenant: "acme", shared: "alias" };

const SPAN_SOURCES = {
  "live span (composite processor)": startedSpanAttributes,
  "buildReadableSpan": (attributes, propagated) =>
    propagateAttributes(propagated, () =>
      buildReadableSpan({ name: "metadata_span", attributes }).attributes
    ),
};

for (const [source, spanAttributes] of Object.entries(SPAN_SOURCES)) {
  test(`${source}: aliases + propagated metadata export one canonical JSON`, () => {
    const attrs = spanAttributes({ ...ALIASES }, PROPAGATED);

    assert.deepEqual(aliasKeys(attrs), []);
    assert.equal(typeof attrs[METADATA_ATTR], "string");
    assert.deepEqual(JSON.parse(attrs[METADATA_ATTR]), EXPECTED_ALIAS_MERGE);
  });

  test(`${source}: explicit span metadata wins over aliases and propagated`, () => {
    const attrs = spanAttributes(
      {
        ...ALIASES,
        [METADATA_ATTR]: JSON.stringify({ foo: "explicit", own: 1 }),
      },
      PROPAGATED
    );

    assert.deepEqual(aliasKeys(attrs), []);
    assert.deepEqual(JSON.parse(attrs[METADATA_ATTR]), {
      foo: "explicit",
      tenant: "acme",
      shared: "alias",
      own: 1,
    });
  });

  test(`${source}: explicit span metadata wins over propagated`, () => {
    const attrs = spanAttributes(
      { [METADATA_ATTR]: JSON.stringify({ foo: "explicit" }) },
      PROPAGATED
    );

    assert.deepEqual(aliasKeys(attrs), []);
    assert.deepEqual(JSON.parse(attrs[METADATA_ATTR]), {
      foo: "explicit",
      tenant: "acme",
      shared: "propagated",
    });
  });

  test(`${source}: propagated metadata alone becomes canonical JSON`, () => {
    const attrs = spanAttributes({}, PROPAGATED);

    assert.deepEqual(aliasKeys(attrs), []);
    assert.deepEqual(JSON.parse(attrs[METADATA_ATTR]), PROPAGATED.metadata);
  });
}

test("live span without propagated metadata keeps its attributes untouched", () => {
  const span = tracer.startSpan("metadata_span", { attributes: { ...ALIASES } });
  span.end();

  assert.deepEqual(aliasKeys(span.attributes).sort(), Object.keys(ALIASES).sort());
  assert.equal(span.attributes[METADATA_ATTR], undefined);
});
