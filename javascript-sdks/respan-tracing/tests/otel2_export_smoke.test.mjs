// Runtime smoke test for the OpenTelemetry 2.x span shape.
//
// buildReadableSpan hand-constructs ReadableSpan objects via `satisfies`/casts
// that the TS build cannot fully validate against the OTLP wire path. Under OTEL
// 2.x the OTLP transformer reads `record.instrumentationScope.name` and
// `span.parentSpanContext` — a span still carrying the removed 1.x fields
// (`instrumentationLibrary` / `parentSpanId`) throws inside serialization and
// fails the whole export batch. This test drives a built span through the real
// 2.x serializer to guarantee that regression can never ship again.

import test from "node:test";
import assert from "node:assert";

import { JsonTraceSerializer } from "@opentelemetry/otlp-transformer";
import { buildReadableSpan } from "../dist/index.js";

const TRACE_ID = "0af7651916cd43dd8448eb211c80319c";
const PARENT_ID = "b7ad6b7169203331";
const CHILD_ID = "b9c7c989f97918e1";

test("OTEL 2.x: built spans serialize through the OTLP transformer without throwing", () => {
  const parent = buildReadableSpan({
    name: "parent_workflow",
    traceId: TRACE_ID,
    spanId: PARENT_ID,
    attributes: {},
  });
  const child = buildReadableSpan({
    name: "child_task",
    traceId: TRACE_ID,
    spanId: CHILD_ID,
    parentId: PARENT_ID,
    attributes: {},
  });

  // On a 1.x-shaped span this throws (instrumentationScope is undefined) — the
  // crash PR #345 would have shipped. It must serialize cleanly on 2.x.
  let bytes;
  assert.doesNotThrow(() => {
    bytes = JsonTraceSerializer.serializeRequest([parent, child]);
  });
  assert.ok(bytes && bytes.length > 0, "serializer produced output");

  const req = JSON.parse(Buffer.from(bytes).toString("utf8"));

  // buildReadableSpan attaches a fresh resource per span, so spans land in
  // separate resourceSpans groups — flatten across all of them.
  const scopes = (req.resourceSpans ?? []).flatMap((rs) => rs.scopeSpans ?? []);
  assert.ok(scopes.length > 0, "resourceSpans/scopeSpans present");
  const allSpans = scopes.flatMap((ss) =>
    (ss.spans ?? []).map((s) => ({ ...s, scopeName: ss.scope?.name })),
  );

  // instrumentationScope must be present and named on every span (the crash field).
  assert.ok(
    allSpans.every((s) => typeof s.scopeName === "string" && s.scopeName.length > 0),
    "instrumentationScope.name serialized for every span",
  );

  // Parent linkage must survive: the child keeps a parent, the root does not.
  const childOut = allSpans.find((s) => s.name === "child_task");
  const parentOut = allSpans.find((s) => s.name === "parent_workflow");
  assert.ok(
    childOut?.parentSpanId && childOut.parentSpanId.length > 0,
    "child span retains a parent (parentSpanContext propagated to the wire)",
  );
  assert.ok(
    !parentOut?.parentSpanId || parentOut.parentSpanId.length === 0,
    "root span has no parent",
  );
});
