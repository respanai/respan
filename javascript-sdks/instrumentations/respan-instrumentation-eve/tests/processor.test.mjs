import test from "node:test";
import assert from "node:assert/strict";

import { EveSpanProcessor } from "../dist/_processor.js";

function translate(name, attributes, options = {}) {
  return translateSpan(name, attributes, options).attributes;
}

function translateSpan(name, attributes, options = {}) {
  const translated = { ...attributes };
  const instrumentationScope = {
    name: options.scope ?? "gen_ai",
  };
  const span = {
    name,
    instrumentationScope,
    attributes: translated,
    status: options.status,
    events: options.events,
    spanContext() {
      return {
        spanId: options.spanId ?? "0123456789abcdef",
        traceId:
          options.traceId ?? "11111111111111111111111111111111",
      };
    },
  };
  const writableSpan = {
    name,
    instrumentationScope,
    attributes: translated,
    parentSpanId: options.parentSpanId,
    spanContext: span.spanContext,
    setAttribute(key, value) {
      translated[key] = value;
    },
  };

  const translator = options.translator ?? new EveSpanProcessor();
  translator.onStart(writableSpan, undefined);
  translator.onEnd(span);
  return { span, attributes: translated };
}

function assertRawVendorAttrsStripped(attrs) {
  const rawKeys = Object.keys(attrs).filter(
    (key) => key.startsWith("ai.") || key.startsWith("eve."),
  );
  assert.deepEqual(rawKeys, []);
}

function assertNoOffContractAliases(attrs) {
  for (const key of [
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "span_tools",
    "has_tool_calls",
    "parallel_tool_calls",
    "respan.span.tools",
    "respan.span.tool_calls",
    "respan.span.handoffs",
  ]) {
    assert.equal(attrs[key], undefined, `${key} should be stripped`);
  }
}

test("maps the ai.eve.turn root to an agent and promotes raw Eve context", () => {
  const attrs = translate(
    "ai.eve.turn",
    {
      "ai.telemetry.functionId": "support-agent",
      "eve.version": "0.26.1",
      "eve.environment": "production",
      "eve.session.id": "session-root",
      "eve.turn.id": "turn-root",
    },
    { scope: "eve" },
  );

  assert.equal(attrs["respan.entity.log_type"], "agent");
  assert.equal(attrs["traceloop.entity.name"], "support-agent");
  assert.equal(attrs["traceloop.entity.path"], "");
  assert.equal(attrs["respan.sessions.session_identifier"], "session-root");
  assert.equal(attrs["respan.threads.thread_identifier"], "session-root");
  assert.equal(attrs["respan.trace.trace_group_identifier"], "session-root");
  assert.equal(attrs["respan.environment"], "production");
  assert.equal(attrs["traceloop.workflow.name"], "support-agent");
  assert.equal(attrs["respan.internal.export_parent_span_id"], "");
  assert.equal(attrs["respan.internal.span_name.kind"], "agent");
  assert.equal(attrs["respan.internal.span_name.detail"], "support-agent");
  assert.deepEqual(JSON.parse(attrs["respan.metadata"]), {
    eve: {
      version: "0.26.1",
      environment: "production",
      turn_id: "turn-root",
    },
  });
  assert.deepEqual(JSON.parse(attrs["respan.metadata.eve"]), {
    version: "0.26.1",
    environment: "production",
    turn_id: "turn-root",
  });
  assertRawVendorAttrsStripped(attrs);
  assertNoOffContractAliases(attrs);
});

test("extracts exact AI SDK 7 runtimeContext keys on a gen_ai chat span", () => {
  const attrs = translate("chat openai/gpt-4.1", {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "openai",
    "gen_ai.request.model": "gpt-4.1",
    "gen_ai.input.messages": JSON.stringify([
      { role: "user", parts: [{ type: "text", content: "hello" }] },
    ]),
    "gen_ai.output.messages": JSON.stringify([
      { role: "assistant", parts: [{ type: "text", content: "hi" }] },
    ]),
    "gen_ai.usage.input_tokens": 7,
    "gen_ai.usage.output_tokens": 2,
    "ai.settings.context.eve.version": "0.26.1",
    "ai.settings.context.eve.environment": "development",
    "ai.settings.context.eve.session.id": "session-1",
    "ai.settings.context.eve.turn.id": "turn-1",
    "ai.settings.context.eve.turn.sequence": "3",
    "ai.settings.context.eve.step.index": "1",
    "ai.settings.context.eve.channel.kind": "channel:support",
    "ai.settings.context.eve.retry.reason": "provider-retry",
  });

  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["gen_ai.system"], "openai");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4.1");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.prompt.0.role"], "user");
  assert.equal(attrs["gen_ai.prompt.0.content"], "hello");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], "hi");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 7);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 2);
  assert.equal(attrs["respan.sessions.session_identifier"], "session-1");
  assert.equal(attrs["respan.threads.thread_identifier"], "session-1");
  assert.equal(attrs["respan.trace.trace_group_identifier"], "session-1");
  assert.equal(attrs["respan.environment"], "development");
  assert.deepEqual(JSON.parse(attrs["respan.metadata"]), {
    eve: {
      version: "0.26.1",
      environment: "development",
      turn_id: "turn-1",
      turn_sequence: 3,
      step_index: 1,
      channel_kind: "channel:support",
      retry_reason: "provider-retry",
    },
  });
  assertRawVendorAttrsStripped(attrs);
  assertNoOffContractAliases(attrs);
});

test("maps legacy AI SDK token aliases into canonical usage totals", () => {
  const attrs = translate("ai.generateText.doGenerate", {
    "ai.model.id": "gpt-4o-mini",
    "ai.model.provider": "openai.chat",
    "ai.prompt.messages": JSON.stringify([{ role: "user", content: "hello" }]),
    "ai.response.text": "world",
    "ai.usage.promptTokens": "9",
    "ai.usage.completionTokens": "4",
    "ai.usage.totalTokens": "13",
    "ai.usage.cachedInputTokens": "2",
  });

  assert.equal(attrs["gen_ai.usage.input_tokens"], 9);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 4);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 9);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 4);
  assert.equal(attrs["llm.usage.total_tokens"], 13);
  assert.equal(attrs["llm.usage.cache_read_input_tokens"], 2);
  assertRawVendorAttrsStripped(attrs);
  assertNoOffContractAliases(attrs);
});

test("maps legacy embedding input, vector, and model while stripping raw aliases", () => {
  const attrs = translate("ai.embed.doEmbed", {
    "ai.model.id": "text-embedding-3-small",
    "ai.model.provider": "openai.embedding",
    "ai.value": "embed this",
    "ai.embedding": [0.1, 0.2, 0.3],
    "ai.usage.tokens": 1,
    "gen_ai.usage.input_tokens": 3,
    model: "off-contract-model",
    prompt_tokens: 1,
    "respan.span.tools": JSON.stringify([{ name: "off-contract-tool" }]),
  });

  assert.equal(attrs["respan.entity.log_type"], "embedding");
  assert.equal(attrs["llm.request.type"], "embedding");
  assert.equal(attrs["gen_ai.system"], "openai");
  assert.equal(attrs["gen_ai.request.model"], "text-embedding-3-small");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 3);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 3);
  assert.equal(attrs["traceloop.entity.input"], "embed this");
  assert.equal(attrs["traceloop.entity.output"], JSON.stringify([0.1, 0.2, 0.3]));
  assert.equal(attrs["ai.usage.tokens"], undefined);
  assertRawVendorAttrsStripped(attrs);
  assertNoOffContractAliases(attrs);
});

test("preserves authored step.started runtime context in canonical metadata", () => {
  const cyclic = {};
  cyclic.self = cyclic;

  const attrs = translate("chat openai/gpt-4.1", {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "openai",
    "gen_ai.request.model": "gpt-4.1",
    "ai.settings.context.eve.version": "0.26.1",
    "ai.settings.context.eve.user_override": "must-not-leak",
    "ai.settings.context.tenant": "acme",
    "ai.settings.context.attempt": 2,
    "ai.settings.context.enabled": true,
    "ai.settings.context.nullable": null,
    "ai.settings.context.roles": ["admin", "support"],
    "ai.settings.context.profile.tier": "enterprise",
    "ai.settings.context.snapshot": {
      flags: [true, false],
      nested: { count: 3 },
    },
    "ai.settings.context.unsafe": cyclic,
    "ai.settings.context.missing": undefined,
    "ai.response.text": "AI SDK internal field",
  });

  assert.deepEqual(JSON.parse(attrs["respan.metadata"]), {
    eve: {
      version: "0.26.1",
      runtime_context: {
        tenant: "acme",
        attempt: 2,
        enabled: true,
        nullable: null,
        roles: ["admin", "support"],
        "profile.tier": "enterprise",
        snapshot: {
          flags: [true, false],
          nested: { count: 3 },
        },
      },
    },
  });
  assertRawVendorAttrsStripped(attrs);
  assertNoOffContractAliases(attrs);
});

test("preserves a pre-populated root trace group on Eve child spans", () => {
  const attrs = translate("step 1", {
    "gen_ai.operation.name": "agent_step",
    "ai.settings.context.eve.session.id": "session-child",
    "respan.trace.trace_group_identifier": "session-root",
  });

  assert.equal(attrs["respan.entity.log_type"], "task");
  assert.equal(attrs["respan.sessions.session_identifier"], "session-child");
  assert.equal(attrs["respan.threads.thread_identifier"], "session-child");
  assert.equal(attrs["respan.trace.trace_group_identifier"], "session-root");
  assertRawVendorAttrsStripped(attrs);
  assertNoOffContractAliases(attrs);
});

test("drop-marks structural wrappers and reparents children until wrapper end", () => {
  const translator = new EveSpanProcessor();

  function makeSpan(name, spanId, parentSpanContext) {
    const attributes = {};
    return {
      name,
      instrumentationScope: { name: "ai" },
      attributes,
      parentSpanContext,
      spanContext() {
        return { spanId };
      },
      setAttribute(key, value) {
        attributes[key] = value;
      },
    };
  }

  const wrapper = makeSpan("ai.generateText", "wrapper-span", {
    spanId: "upstream-parent",
  });
  translator.onStart(wrapper, undefined);
  assert.equal(wrapper.attributes["respan.internal.drop_span"], true);
  assert.equal(wrapper.attributes["respan.entity.log_type"], "task");

  const child = makeSpan(
    "ai.generateText.doGenerate",
    "child-span",
    { spanId: "wrapper-span" },
  );
  translator.onStart(child, undefined);
  assert.equal(
    child.attributes["respan.internal.export_parent_span_id"],
    "upstream-parent",
  );

  const outside = makeSpan("ai.toolCall", "outside-span", {
    spanId: "upstream-parent",
  });
  translator.onStart(outside, undefined);
  assert.equal(
    outside.attributes["respan.internal.export_parent_span_id"],
    undefined,
  );

  translator.onEnd(wrapper);
  assert.equal(wrapper.attributes["respan.internal.drop_span"], true);
  assert.equal(wrapper.attributes["respan.entity.log_type"], "task");

  const lateChild = makeSpan(
    "ai.generateText.doGenerate",
    "late-child-span",
    { spanId: "wrapper-span" },
  );
  translator.onStart(lateChild, undefined);
  assert.equal(
    lateChild.attributes["respan.internal.export_parent_span_id"],
    undefined,
  );
});

test("uses an empty export-parent sentinel for children of root wrappers", () => {
  const translator = new EveSpanProcessor();

  function makeSpan(name, spanId, parentSpanContext) {
    const attributes = {};
    return {
      name,
      instrumentationScope: { name: "ai" },
      attributes,
      parentSpanContext,
      spanContext() {
        return { spanId };
      },
      setAttribute(key, value) {
        attributes[key] = value;
      },
    };
  }

  const wrapper = makeSpan("ai.streamText", "root-wrapper", undefined);
  translator.onStart(wrapper, undefined);

  const child = makeSpan(
    "ai.streamText.doStream",
    "root-child",
    { spanId: "root-wrapper" },
  );
  translator.onStart(child, undefined);
  assert.equal(child.attributes["respan.internal.export_parent_span_id"], "");

  translator.onEnd(wrapper);
});

test("admits Eve-scope invoke_agent spans and keeps subagent usage in metadata", () => {
  const attrs = translate(
    "invoke_agent researcher",
    {
      "gen_ai.operation.name": "invoke_agent",
      "gen_ai.agent.name": "researcher",
      "gen_ai.usage.input_tokens": 12,
      "gen_ai.usage.output_tokens": 5,
      "gen_ai.usage.cache_read.input_tokens": 3,
      "gen_ai.usage.cache_creation.input_tokens": 2,
      "ai.settings.context.eve.session.id": "session-1",
    },
    { scope: "eve" },
  );

  assert.equal(attrs["respan.entity.log_type"], "agent");
  assert.equal(attrs["traceloop.entity.name"], "researcher");
  assert.deepEqual(JSON.parse(attrs["respan.metadata"]), {
    eve: {
      subagent: {
        name: "researcher",
        usage: {
          input_tokens: 12,
          output_tokens: 5,
          cache_read_input_tokens: 3,
          cache_creation_input_tokens: 2,
        },
      },
    },
  });
  assert.equal(
    Object.keys(attrs).some((key) => key.startsWith("gen_ai.")),
    false,
  );
  assertRawVendorAttrsStripped(attrs);
  assertNoOffContractAliases(attrs);
});

test("reattaches detached Eve subagent usage to its caller session", () => {
  const translator = new EveSpanProcessor();

  translateSpan(
    "invoke_agent eve-deterministic-researcher",
    {
      "gen_ai.operation.name": "invoke_agent",
      "gen_ai.agent.name": "eve-deterministic-researcher",
      "gen_ai.usage.input_tokens": 19,
      "gen_ai.usage.output_tokens": 6,
      "gen_ai.usage.cache_read.input_tokens": 0,
      "gen_ai.usage.cache_creation.input_tokens": 0,
      "ai.telemetry.functionId": "eve-typescript-run",
      "ai.settings.context.eve.session.id": "session-child",
      "ai.settings.context.__respan_eve.lineage.rootSessionId":
        "session-root",
      "ai.settings.context.__respan_eve.lineage.sessionId":
        "session-parent",
      "ai.settings.context.__respan_eve.lineage.callId": "call-parent",
      "ai.settings.context.__respan_eve.lineage.turn.id": "turn-parent",
      "ai.settings.context.__respan_eve.lineage.turn.sequence": 4,
    },
    { translator },
  );

  const attrs = translate(
    "invoke_agent researcher",
    {
      "gen_ai.operation.name": "invoke_agent",
      "gen_ai.agent.name": "researcher",
      "gen_ai.usage.input_tokens": 19,
      "gen_ai.usage.output_tokens": 6,
      "gen_ai.usage.cache_read.input_tokens": 0,
      "gen_ai.usage.cache_creation.input_tokens": 0,
    },
    { scope: "eve", translator },
  );

  assert.equal(attrs["respan.sessions.session_identifier"], "session-parent");
  assert.equal(attrs["respan.threads.thread_identifier"], "session-parent");
  assert.equal(attrs["respan.trace.trace_group_identifier"], "session-root");
  assert.equal(attrs["traceloop.workflow.name"], "eve-typescript-run");
  assert.deepEqual(JSON.parse(attrs["respan.metadata"]), {
    eve: {
      turn_id: "turn-parent",
      turn_sequence: 4,
      root_session_id: "session-root",
      subagent: {
        name: "researcher",
        usage: {
          input_tokens: 19,
          output_tokens: 6,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      },
    },
  });
  assert.deepEqual(JSON.parse(attrs["respan.metadata.eve"]), {
    turn_id: "turn-parent",
    turn_sequence: 4,
    root_session_id: "session-root",
    subagent: {
      name: "researcher",
      usage: {
        input_tokens: 19,
        output_tokens: 6,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
    },
  });
  assertRawVendorAttrsStripped(attrs);
  assertNoOffContractAliases(attrs);
});

test("inherits the Eve workflow name through nested AI SDK spans", () => {
  const translator = new EveSpanProcessor();
  const rootAttributes = {
    "ai.telemetry.functionId": "eve-typescript-run",
  };
  const root = {
    name: "ai.eve.turn",
    instrumentationScope: { name: "eve" },
    attributes: rootAttributes,
    spanContext() {
      return { spanId: "workflow-root", traceId: "workflow-trace" };
    },
    setAttribute(key, value) {
      rootAttributes[key] = value;
    },
  };
  translator.onStart(root, undefined);

  const childAttributes = {
    "gen_ai.operation.name": "agent_step",
  };
  const child = {
    name: "step 1",
    instrumentationScope: { name: "gen_ai" },
    attributes: childAttributes,
    parentSpanId: "workflow-root",
    spanContext() {
      return { spanId: "workflow-child", traceId: "workflow-trace" };
    },
    setAttribute(key, value) {
      childAttributes[key] = value;
    },
  };
  translator.onStart(child, undefined);

  assert.equal(
    childAttributes["traceloop.workflow.name"],
    "eve-typescript-run",
  );

  // Eve may close its first turn segment before resuming the same workflow
  // trace for a later model/tool step.
  translator.onEnd(child);
  translator.onEnd(root);

  const resumedAttributes = {
    "gen_ai.operation.name": "agent_step",
  };
  const resumed = {
    name: "step 2",
    instrumentationScope: { name: "gen_ai" },
    attributes: resumedAttributes,
    spanContext() {
      return { spanId: "workflow-resumed", traceId: "workflow-trace" };
    },
    setAttribute(key, value) {
      resumedAttributes[key] = value;
    },
  };
  translator.onStart(resumed, undefined);

  assert.equal(
    resumedAttributes["traceloop.workflow.name"],
    "eve-typescript-run",
  );

  translator.onEnd(resumed);
});

test("drops AI SDK invoke_agent wrappers inside an Eve turn", () => {
  const translator = new EveSpanProcessor();

  function makeSpan(name, attributes, spanId, parentSpanId) {
    return {
      name,
      instrumentationScope: { name: "gen_ai" },
      attributes,
      parentSpanId,
      spanContext() {
        return {
          spanId,
          traceId: "22222222222222222222222222222222",
        };
      },
      setAttribute(key, value) {
        attributes[key] = value;
      },
    };
  }

  const wrapper = makeSpan(
    "invoke_agent eve-deterministic-root",
    {
      "gen_ai.operation.name": "invoke_agent",
      "ai.telemetry.functionId": "eve-typescript-run",
      "ai.settings.context.eve.session.id": "session-tool",
    },
    "model-wrapper",
    "eve-turn-root",
  );
  translator.onStart(wrapper, undefined);

  assert.equal(wrapper.attributes["respan.internal.drop_span"], true);

  const task = makeSpan(
    "step 1",
    {
      "gen_ai.operation.name": "agent_step",
      "ai.settings.context.eve.session.id": "session-tool",
    },
    "model-task",
    "model-wrapper",
  );
  translator.onStart(task, undefined);

  assert.equal(
    task.attributes["respan.internal.export_parent_span_id"],
    "eve-turn-root",
  );
  assert.equal(
    task.attributes["traceloop.workflow.name"],
    "eve-typescript-run",
  );
});

test("merges an exact delegated child trace and suppresses late usage duplication", () => {
  const translator = new EveSpanProcessor();
  const parentTraceId = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  const childTraceId = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

  function makeSpan(
    name,
    scope,
    attributes,
    spanId,
    traceId,
    parentSpanId,
  ) {
    return {
      name,
      instrumentationScope: { name: scope },
      attributes,
      parentSpanId,
      spanContext() {
        return { spanId, traceId };
      },
      setAttribute(key, value) {
        attributes[key] = value;
      },
    };
  }

  const parent = makeSpan(
    "ai.eve.turn",
    "eve",
    {
      "ai.telemetry.functionId": "eve-typescript-run",
      "eve.session.id": "session-parent",
      "eve.turn.id": "turn-parent",
    },
    "parent-turn-root",
    parentTraceId,
  );
  translator.onStart(parent, undefined);
  translator.onEnd(parent);

  const lineage = {
    "ai.settings.context.__respan_eve.lineage.rootSessionId":
      "session-parent",
    "ai.settings.context.__respan_eve.lineage.sessionId":
      "session-parent",
    "ai.settings.context.__respan_eve.lineage.turn.id": "turn-parent",
    "ai.settings.context.__respan_eve.lineage.turn.sequence": 0,
  };
  const childRoot = makeSpan(
    "ai.eve.turn",
    "eve",
    {
      "ai.telemetry.functionId": "eve-typescript-run",
      "eve.session.id": "session-child",
      "eve.turn.id": "turn-child",
      "eve.turn.sequence": 0,
    },
    "child-turn-root",
    childTraceId,
  );
  translator.onStart(childRoot, undefined);

  const childWrapper = makeSpan(
    "invoke_agent eve-deterministic-researcher",
    "gen_ai",
    {
      "gen_ai.operation.name": "invoke_agent",
      "gen_ai.agent.name": "eve-deterministic-researcher",
      "gen_ai.usage.input_tokens": 19,
      "gen_ai.usage.output_tokens": 6,
      "ai.telemetry.functionId": "eve-typescript-run",
      "ai.settings.context.eve.session.id": "session-child",
      ...lineage,
    },
    "child-model-wrapper",
    childTraceId,
    "child-turn-root",
  );
  translator.onStart(childWrapper, undefined);

  assert.equal(
    translator.prepareForExport(childWrapper).spanContext().traceId,
    parentTraceId,
  );
  assert.equal(childWrapper.attributes["respan.internal.drop_span"], true);
  assert.equal(
    translator.prepareForExport(childWrapper).attributes[
      "respan.instrumentation.eve.export_trace_id"
    ],
    undefined,
  );

  const childTask = makeSpan(
    "step 1",
    "gen_ai",
    {
      "gen_ai.operation.name": "agent_step",
      "ai.settings.context.eve.session.id": "session-child",
      ...lineage,
    },
    "child-task",
    childTraceId,
    "child-model-wrapper",
  );
  translator.onStart(childTask, undefined);

  assert.equal(
    translator.prepareForExport(childTask).spanContext().traceId,
    parentTraceId,
  );
  assert.equal(
    childTask.attributes["respan.internal.export_parent_span_id"],
    "child-turn-root",
  );
  assert.equal(
    childTask.attributes["traceloop.workflow.name"],
    "eve-typescript-run",
  );

  translator.onEnd(childTask);
  translator.onEnd(childWrapper);
  Object.assign(childRoot.attributes, lineage);
  translator.onEnd(childRoot);

  assert.equal(
    translator.prepareForExport(childRoot).spanContext().traceId,
    parentTraceId,
  );
  assert.equal(
    childRoot.attributes["respan.internal.export_parent_span_id"],
    "parent-turn-root",
  );

  const lateUsage = makeSpan(
    "invoke_agent researcher",
    "eve",
    {
      "gen_ai.operation.name": "invoke_agent",
      "gen_ai.agent.name": "researcher",
      "gen_ai.usage.input_tokens": 19,
      "gen_ai.usage.output_tokens": 6,
    },
    "late-usage",
    "cccccccccccccccccccccccccccccccc",
  );
  translator.onStart(lateUsage, undefined);

  assert.equal(
    translator.prepareForExport(lateUsage).spanContext().traceId,
    parentTraceId,
  );
  assert.equal(
    lateUsage.attributes["respan.internal.export_parent_span_id"],
    "parent-turn-root",
  );
  assert.equal(lateUsage.attributes["respan.internal.drop_span"], true);
});

test("does not guess detached subagent lineage across ambiguous callers", () => {
  const translator = new EveSpanProcessor();

  for (const suffix of ["a", "b"]) {
    translateSpan(
      `invoke_agent child-${suffix}`,
      {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": `child-${suffix}`,
        "gen_ai.usage.input_tokens": 12,
        "gen_ai.usage.output_tokens": 5,
        "ai.settings.context.eve.session.id": `session-child-${suffix}`,
        "ai.settings.context.__respan_eve.lineage.rootSessionId":
          `session-root-${suffix}`,
        "ai.settings.context.__respan_eve.lineage.sessionId":
          `session-parent-${suffix}`,
      },
      { translator },
    );
  }

  const attrs = translate(
    "invoke_agent researcher",
    {
      "gen_ai.operation.name": "invoke_agent",
      "gen_ai.agent.name": "researcher",
      "gen_ai.usage.input_tokens": 12,
      "gen_ai.usage.output_tokens": 5,
    },
    { scope: "eve", translator },
  );

  assert.equal(attrs["respan.sessions.session_identifier"], undefined);
  assert.equal(attrs["respan.threads.thread_identifier"], undefined);
  assert.equal(attrs["respan.trace.trace_group_identifier"], undefined);
  assertRawVendorAttrsStripped(attrs);
  assertNoOffContractAliases(attrs);
});

test("does not mislabel a gen_ai invoke_agent model root as an Eve subagent", () => {
  const attrs = translate("invoke_agent openai/gpt-4.1", {
    "gen_ai.operation.name": "invoke_agent",
    "gen_ai.agent.name": "support-agent",
    "gen_ai.usage.input_tokens": 12,
    "ai.settings.context.eve.session.id": "session-1",
    "ai.settings.context.eve.turn.id": "turn-1",
  });

  assert.equal(attrs["respan.entity.log_type"], "agent");
  assert.equal(attrs["traceloop.entity.name"], "support-agent");
  assert.deepEqual(JSON.parse(attrs["respan.metadata"]), {
    eve: {
      turn_id: "turn-1",
    },
  });
  assert.equal(
    Object.keys(attrs).some((key) => key.startsWith("gen_ai.")),
    false,
  );
  assertRawVendorAttrsStripped(attrs);
  assertNoOffContractAliases(attrs);
});

test("maps AI SDK 7 execute_tool children with canonical tool input and output", () => {
  const attrs = translate("execute_tool search", {
    "gen_ai.operation.name": "execute_tool",
    "gen_ai.tool.name": "search",
    "gen_ai.tool.call.id": "call-1",
    "gen_ai.tool.call.arguments": JSON.stringify({ query: "eve" }),
    "gen_ai.tool.call.result": JSON.stringify({ hits: 2 }),
    "ai.settings.context.eve.session.id": "session-1",
  });

  assert.equal(attrs["respan.entity.log_type"], "tool");
  assert.equal(
    attrs["traceloop.entity.input"],
    JSON.stringify({ name: "search", arguments: { query: "eve" } }),
  );
  assert.equal(
    attrs["traceloop.entity.output"],
    JSON.stringify({ hits: 2 }),
  );
  assert.equal(attrs["respan.internal.span_name.kind"], "tool");
  assert.equal(attrs["respan.internal.span_name.detail"], "search");
  assert.equal(
    Object.keys(attrs).some((key) => key.startsWith("gen_ai.tool.")),
    false,
  );
  assertRawVendorAttrsStripped(attrs);
  assertNoOffContractAliases(attrs);
});

test("preserves an errored span status and events unchanged", () => {
  const status = { code: 2, message: "tool failed" };
  const events = [
    {
      name: "exception",
      attributes: {
        "exception.type": "Error",
        "exception.message": "tool failed",
      },
    },
  ];
  const { span } = translateSpan(
    "execute_tool search",
    {
      "gen_ai.operation.name": "execute_tool",
      "gen_ai.tool.name": "search",
      "gen_ai.tool.call.arguments": JSON.stringify({ query: "eve" }),
    },
    { status, events },
  );

  assert.equal(span.status, status);
  assert.equal(span.events, events);
  assert.deepEqual(span.status, { code: 2, message: "tool failed" });
  assert.deepEqual(span.events, events);
});

test("leaves unrelated OpenTelemetry spans untouched", () => {
  const attrs = translate(
    "http.request",
    { "http.request.method": "GET" },
    { scope: "@opentelemetry/instrumentation-http" },
  );

  assert.deepEqual(attrs, { "http.request.method": "GET" });
});

test("maps delegated session lineage without treating session IDs as span parents", () => {
  const attrs = translate("chat openai/gpt-4.1", {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "openai",
    "gen_ai.request.model": "gpt-4.1",
    "ai.settings.context.eve.session.id": "session-child",
    "ai.settings.context.eve.turn.id": "turn-child",
    "ai.settings.context.__respan_eve.lineage.rootSessionId":
      "session-root",
    "ai.settings.context.__respan_eve.lineage.sessionId":
      "session-parent",
    "ai.settings.context.__respan_eve.lineage.callId": "call-parent",
    "ai.settings.context.__respan_eve.lineage.turn.id": "turn-parent",
    "ai.settings.context.__respan_eve.lineage.turn.sequence": 4,
    "ai.settings.context.tenant": "acme",
  });

  assert.equal(
    attrs["respan.sessions.session_identifier"],
    "session-child",
  );
  assert.equal(
    attrs["respan.threads.thread_identifier"],
    "session-child",
  );
  assert.equal(
    attrs["respan.trace.trace_group_identifier"],
    "session-root",
  );
  assert.equal(attrs["respan.entity.log_parent_id"], undefined);
  assert.equal(attrs["respan.entity.log_root_id"], undefined);
  assert.deepEqual(JSON.parse(attrs["respan.metadata"]), {
    eve: {
      turn_id: "turn-child",
      root_session_id: "session-root",
      parent: {
        session_id: "session-parent",
        call_id: "call-parent",
        turn: {
          id: "turn-parent",
          sequence: 4,
        },
      },
      runtime_context: {
        tenant: "acme",
      },
    },
  });
  assertRawVendorAttrsStripped(attrs);
  assertNoOffContractAliases(attrs);
});

test("maps lineage mirrored directly onto an ai.eve.turn root", () => {
  const attrs = translate(
    "ai.eve.turn",
    {
      "ai.telemetry.functionId": "research-agent",
      "eve.session.id": "session-child",
      "ai.settings.context.__respan_eve.lineage.rootSessionId":
        "session-root",
      "ai.settings.context.__respan_eve.lineage.sessionId":
        "session-parent",
      "ai.settings.context.__respan_eve.lineage.callId": "call-parent",
      "ai.settings.context.__respan_eve.lineage.turn.id": "turn-parent",
      "ai.settings.context.__respan_eve.lineage.turn.sequence": 4,
    },
    { scope: "eve" },
  );

  assert.equal(attrs["respan.entity.log_type"], "agent");
  assert.equal(
    attrs["respan.sessions.session_identifier"],
    "session-child",
  );
  assert.equal(
    attrs["respan.threads.thread_identifier"],
    "session-child",
  );
  assert.equal(
    attrs["respan.trace.trace_group_identifier"],
    "session-root",
  );
  assert.deepEqual(JSON.parse(attrs["respan.metadata"]), {
    eve: {
      root_session_id: "session-root",
      parent: {
        session_id: "session-parent",
        call_id: "call-parent",
        turn: {
          id: "turn-parent",
          sequence: 4,
        },
      },
    },
  });
  assertRawVendorAttrsStripped(attrs);
  assertNoOffContractAliases(attrs);
});
