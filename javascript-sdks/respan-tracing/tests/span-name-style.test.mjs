import test from "node:test";
import assert from "node:assert/strict";

import {
  resolveSpanNameStyle,
  semanticSpanNameForSpan,
  SpanNameTransformingExporter,
  transformReadableSpanBatch,
  transformReadableSpanName,
} from "../dist/processor/spanName.js";

function span(name, attributes = {}) {
  return {
    name,
    attributes,
  };
}

test("semantic is the default style; only explicit legacy opts out", () => {
  assert.equal(resolveSpanNameStyle(undefined), "semantic");
  assert.equal(resolveSpanNameStyle("semantic"), "semantic");
  assert.equal(resolveSpanNameStyle("legacy"), "legacy");
  assert.equal(resolveSpanNameStyle("bogus"), "semantic");
  // Case/whitespace normalization — same env value must mean the same thing
  // as in the Python SDK.
  assert.equal(resolveSpanNameStyle("LEGACY"), "legacy");
  assert.equal(resolveSpanNameStyle(" legacy "), "legacy");
  assert.equal(resolveSpanNameStyle("Legacy"), "legacy");
});

test("unrecognized spans keep their original names in semantic style", () => {
  // Pass-through helper spans (no kind/log type) are never renamed.
  assert.equal(semanticSpanNameForSpan(span("http.request", {})), "http.request");
  // Unknown/custom log types are not operations — name preserved.
  assert.equal(
    semanticSpanNameForSpan(
      span("my_special_step", { "respan.entity.log_type": "custom" })
    ),
    "my_special_step"
  );
});

test("completion log types map to llm", () => {
  assert.equal(
    semanticSpanNameForSpan(
      span("openai.completion", {
        "respan.entity.log_type": "completion",
        "gen_ai.request.model": "gpt-4o",
      })
    ),
    "llm.gpt-4o"
  );
});

test("unicode letters survive sanitization", () => {
  assert.equal(
    semanticSpanNameForSpan(
      span("agent run", {
        "traceloop.span.kind": "agent",
        "traceloop.entity.name": "客服 Agent",
      })
    ),
    "agent.客服_Agent"
  );
});

test("semantic span names use operation prefix and entity detail", () => {
  assert.equal(
    semanticSpanNameForSpan(
      span("triage-service.agent", {
        "traceloop.span.kind": "agent",
        "traceloop.entity.name": "triage-service",
      })
    ),
    "agent.triage-service"
  );

  assert.equal(
    semanticSpanNameForSpan(
      span("send_notification.tool", {
        "traceloop.span.kind": "tool",
        "traceloop.entity.name": "send_notification",
      })
    ),
    "tool.send_notification"
  );

  assert.equal(
    semanticSpanNameForSpan(
      span("access-recovery.workflow", {
        "traceloop.span.kind": "workflow",
        "traceloop.entity.name": "access-recovery",
      })
    ),
    "workflow"
  );
});

test("semantic span names use integration hints and strip internal attrs", () => {
  const transformed = transformReadableSpanName(
    span("ai.generateText.doGenerate", {
      "respan.internal.span_name.kind": "generate",
      "respan.internal.span_name.detail": "doGenerate",
      "respan.entity.log_type": "text",
      "gen_ai.request.model": "gpt-4o-mini",
    }),
    "semantic"
  );

  assert.equal(transformed.name, "llm.gpt-4o-mini");
  assert.equal(transformed.attributes["respan.internal.span_name.kind"], undefined);
  assert.equal(transformed.attributes["respan.internal.span_name.detail"], undefined);
  assert.equal(transformed.attributes["respan.entity.log_type"], "text");
});

test("semantic span names use lowercase llm prefix with model suffix", () => {
  assert.equal(
    semanticSpanNameForSpan(
      span("openai.chat", {
        "respan.entity.log_type": "chat",
        "gen_ai.request.model": "gpt-4o",
      })
    ),
    "llm.gpt-4o"
  );

  assert.equal(
    semanticSpanNameForSpan(
      span("llm.doGenerate", {
        "respan.entity.log_type": "text",
        "gen_ai.request.model": "gpt-4.1",
      })
    ),
    "llm.gpt-4.1"
  );

  // Without a resolvable model, emit bare "llm" — never an operation suffix.
  assert.equal(
    semanticSpanNameForSpan(
      span("llm.doGenerate", { "respan.entity.log_type": "text" })
    ),
    "llm"
  );
});

test("semantic span names keep embedding operation name", () => {
  assert.equal(
    semanticSpanNameForSpan(
      span("ai.embed.doEmbed", {
        "respan.entity.log_type": "embedding",
      })
    ),
    "embedding"
  );
});

test("generic structural suffixes are recomputed from attributes", () => {
  // "handoff.task" carries no identity — detail hint wins.
  assert.equal(
    semanticSpanNameForSpan(
      span("handoff.task", {
        "respan.internal.span_name.kind": "handoff",
        "respan.internal.span_name.detail": "triage-service_to_bank-service",
        "respan.entity.log_type": "handoff",
      })
    ),
    "handoff.triage-service_to_bank-service"
  );

  // Without hints or attrs, collapse to the bare operation.
  assert.equal(
    semanticSpanNameForSpan(
      span("handoff.task", { "respan.entity.log_type": "handoff" })
    ),
    "handoff"
  );
});

test("name details are sanitized", () => {
  assert.equal(
    semanticSpanNameForSpan(
      span("agent run", {
        "traceloop.span.kind": "agent",
        "traceloop.entity.name": "Triage Agent (v2)",
      })
    ),
    "agent.Triage_Agent_v2"
  );

  assert.equal(
    semanticSpanNameForSpan(
      span("handoff.task", {
        "respan.internal.span_name.kind": "handoff",
        "respan.internal.span_name.detail": "Triage → Bank",
        "respan.entity.log_type": "handoff",
      })
    ),
    "handoff.Triage_Bank"
  );
});

test("legacy span names only strip internal semantic hint attrs", () => {
  const transformed = transformReadableSpanName(
    span("ai.embed.doEmbed", {
      "respan.internal.span_name.kind": "embed",
      "respan.internal.span_name.detail": "doEmbed",
      "respan.entity.log_type": "embedding",
    }),
    "legacy"
  );

  assert.equal(transformed.name, "ai.embed.doEmbed");
  assert.equal(transformed.attributes["respan.internal.span_name.kind"], undefined);
  assert.equal(transformed.attributes["respan.internal.span_name.detail"], undefined);
  assert.equal(transformed.attributes["respan.entity.log_type"], "embedding");
});

test("semantic export drops drop-marked spans and reparents via export-parent attr", () => {
  const agent = span("agent.triage-service", {
    "traceloop.span.kind": "agent",
    "traceloop.entity.name": "triage-service",
  });

  const wrapper = span("ai.generateText", {
    "respan.entity.log_type": "task",
    "respan.internal.drop_span": true,
  });

  const child = span("ai.generateText.doGenerate", {
    "respan.entity.log_type": "chat",
    "gen_ai.request.model": "gpt-4o",
    "respan.internal.export_parent_span_id": "agent-span",
  });
  child.parentSpanId = "wrapper-span";

  const exported = transformReadableSpanBatch([agent, wrapper, child], "semantic");

  assert.deepEqual(exported.map((item) => item.name), [
    "agent.triage-service",
    "llm.gpt-4o",
  ]);
  assert.equal(exported[1].parentSpanId, "agent-span");
  assert.equal(
    exported[1].attributes["respan.internal.export_parent_span_id"],
    undefined
  );
});

test("drop and reparent work across separate export batches", () => {
  const wrapper = span("ai.generateText", {
    "respan.entity.log_type": "task",
    "respan.internal.drop_span": true,
  });

  const child = span("ai.generateText.doGenerate", {
    "respan.entity.log_type": "chat",
    "gen_ai.request.model": "gpt-4o",
    "respan.internal.export_parent_span_id": "agent-span",
  });
  child.parentSpanId = "wrapper-span";

  // Child flushes in an earlier batch than the wrapper — the decision is
  // attribute-driven per span, so batch boundaries don't matter.
  const firstBatch = transformReadableSpanBatch([child], "semantic");
  const secondBatch = transformReadableSpanBatch([wrapper], "semantic");

  assert.equal(firstBatch.length, 1);
  assert.equal(firstBatch[0].parentSpanId, "agent-span");
  assert.equal(secondBatch.length, 0);
});

test("legacy export keeps drop-marked spans and original parents", () => {
  const wrapper = span("ai.generateText", {
    "respan.entity.log_type": "task",
    "respan.internal.drop_span": true,
  });

  const child = span("ai.generateText.doGenerate", {
    "respan.entity.log_type": "chat",
    "gen_ai.request.model": "gpt-4o",
    "respan.internal.export_parent_span_id": "agent-span",
  });
  child.parentSpanId = "wrapper-span";

  const exported = transformReadableSpanBatch([wrapper, child], "legacy");

  assert.deepEqual(exported.map((item) => item.name), [
    "ai.generateText",
    "ai.generateText.doGenerate",
  ]);
  assert.equal(exported[1].parentSpanId, "wrapper-span");
  // Internal attrs are still stripped in legacy mode.
  assert.equal(exported[0].attributes["respan.internal.drop_span"], undefined);
  assert.equal(
    exported[1].attributes["respan.internal.export_parent_span_id"],
    undefined
  );
});

test("children of a dropped ROOT wrapper are promoted to root (\"\" sentinel)", () => {
  const wrapper = span("ai.generateText", {
    "respan.entity.log_type": "task",
    "respan.internal.drop_span": true,
  });
  wrapper.parentSpanId = undefined;

  const child = span("ai.generateText.doGenerate", {
    "respan.entity.log_type": "chat",
    "gen_ai.request.model": "gpt-4o",
    "respan.internal.export_parent_span_id": "",
  });
  child.parentSpanId = "wrapper-root";

  const exported = transformReadableSpanBatch([wrapper, child], "semantic");

  assert.equal(exported.length, 1);
  assert.equal(exported[0].name, "llm.gpt-4o");
  // The child must NOT reference the dropped wrapper — it becomes the root.
  assert.equal(exported[0].parentSpanId, undefined);
  assert.equal(
    exported[0].attributes["respan.internal.export_parent_span_id"],
    undefined
  );
});

test("SpanNameTransformingExporter delegates export, forceFlush, and shutdown", async () => {
  const calls = { exported: null, flushed: false, shutdown: false };
  const fake = {
    export(spans, cb) {
      calls.exported = spans;
      cb({ code: 0 });
    },
    forceFlush() {
      calls.flushed = true;
      return Promise.resolve();
    },
    shutdown() {
      calls.shutdown = true;
      return Promise.resolve();
    },
  };

  const exporter = new SpanNameTransformingExporter(fake, "semantic");
  let result;
  exporter.export(
    [
      span("openai.chat", {
        "respan.entity.log_type": "chat",
        "gen_ai.request.model": "gpt-4o",
      }),
      span("ai.generateText", {
        "respan.entity.log_type": "task",
        "respan.internal.drop_span": true,
      }),
    ],
    (r) => {
      result = r;
    }
  );

  assert.equal(result.code, 0);
  assert.deepEqual(calls.exported.map((s) => s.name), ["llm.gpt-4o"]);
  await exporter.forceFlush();
  await exporter.shutdown();
  assert.equal(calls.flushed, true);
  assert.equal(calls.shutdown, true);
});
