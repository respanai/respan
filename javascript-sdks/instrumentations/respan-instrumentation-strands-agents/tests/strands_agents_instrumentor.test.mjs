import assert from "node:assert/strict";
import test from "node:test";

import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";
import { RespanCompositeProcessor } from "@respan/tracing/dist/processor/composite.js";

import {
  StrandsAgentsInstrumentor,
  StrandsAgentsSpanProcessor,
  enrichStrandsAgentsSpan,
} from "../dist/index.js";

const RESPAN_LOG_TYPE = "respan.entity.log_type";
const RESPAN_LOG_METHOD = "respan.entity.log_method";
const ENTITY_NAME = "traceloop.entity.name";
const ENTITY_PATH = "traceloop.entity.path";
const ENTITY_INPUT = "traceloop.entity.input";
const ENTITY_OUTPUT = "traceloop.entity.output";
const WORKFLOW_NAME = "traceloop.workflow.name";
const LLM_REQUEST_TYPE = "llm.request.type";
const LLM_REQUEST_FUNCTIONS = "llm.request.functions";
const LLM_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens";

const EVENT_USER_MESSAGE = "gen_ai.user.message";
const EVENT_TOOL_MESSAGE = "gen_ai.tool.message";
const EVENT_CHOICE = "gen_ai.choice";

function makeSpan({
  name,
  attributes = {},
  events = [],
  traceId = "11111111111111111111111111111111",
  spanId = "2222222222222222",
  parentSpanId,
}) {
  const attrs = { ...attributes };
  return {
    name,
    attributes: attrs,
    events,
    parentSpanContext: parentSpanId
      ? { traceId, spanId: parentSpanId, traceFlags: 1 }
      : undefined,
    spanContext() {
      return {
        traceId,
        spanId,
        traceFlags: 1,
      };
    },
  };
}

function event(name, attributes) {
  return { name, attributes };
}

function assertNoOffContractAliases(attrs) {
  for (const key of [
    "respan.span.tools",
    "respan.span.tool_calls",
    "respan.span.handoffs",
    "tools",
    "tool_calls",
    "span_tools",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "has_tool_calls",
    "parallel_tool_calls",
  ]) {
    assert.equal(attrs[key], undefined, `${key} should not be emitted`);
  }
}

test("enriches Strands agent spans with canonical Respan attrs", () => {
  const span = makeSpan({
    name: "invoke_agent WeatherAgent",
    attributes: {
      "gen_ai.system": "strands-agents",
      "gen_ai.operation.name": "invoke_agent",
      "gen_ai.agent.name": "WeatherAgent",
      "gen_ai.agent.id": "weather-agent",
      "gen_ai.agent.tools": JSON.stringify(["get_weather"]),
      "gen_ai.request.model": "gpt-4.1-nano",
      "gen_ai.usage.input_tokens": 30,
      "gen_ai.usage.output_tokens": 6,
    },
    events: [
      event(EVENT_USER_MESSAGE, {
        content: JSON.stringify([{ text: "weather in Seattle" }]),
      }),
      event(EVENT_CHOICE, { message: "It is sunny." }),
    ],
  });

  enrichStrandsAgentsSpan(span);
  const attrs = span.attributes;

  assert.equal(attrs[RESPAN_LOG_TYPE], "agent");
  assert.equal(attrs[RESPAN_LOG_METHOD], "ts_tracing");
  assert.equal(attrs[ENTITY_NAME], "WeatherAgent");
  assert.equal(attrs[ENTITY_PATH], "WeatherAgent");
  assert.equal(attrs[WORKFLOW_NAME], "WeatherAgent");
  assert.deepEqual(JSON.parse(attrs[ENTITY_INPUT]), [
    { role: "user", content: "weather in Seattle" },
  ]);
  assert.deepEqual(JSON.parse(attrs[ENTITY_OUTPUT]), [
    { role: "assistant", content: "It is sunny." },
  ]);
  assert.equal(attrs[LLM_REQUEST_FUNCTIONS], undefined);
  assert.deepEqual(JSON.parse(attrs["respan.metadata"]).strands_agent_tools, [
    "get_weather",
  ]);

  assert.equal(attrs["gen_ai.system"], undefined);
  assert.equal(attrs["gen_ai.request.model"], undefined);
  assert.equal(attrs["gen_ai.usage.input_tokens"], undefined);
  assert.equal(attrs["traceloop.span.kind"], undefined);
  assert.equal(attrs["gen_ai.agent.name"], undefined);
  assertNoOffContractAliases(attrs);
});

test("enriches Strands chat spans with messages, usage, and tool calls", () => {
  const span = makeSpan({
    name: "chat",
    attributes: {
      "gen_ai.system": "strands-agents",
      "gen_ai.operation.name": "chat",
      "gen_ai.request.model": "gpt-4.1-nano",
      "gen_ai.usage.input_tokens": 12,
      "gen_ai.usage.output_tokens": 4,
      "gen_ai.usage.total_tokens": 16,
    },
    events: [
      event(EVENT_USER_MESSAGE, {
        content: JSON.stringify([{ text: "use the tool" }]),
      }),
      event(EVENT_CHOICE, {
        message: JSON.stringify([
          {
            toolUse: {
              toolUseId: "tool_1",
              name: "lookup",
              input: { query: "tracing" },
            },
          },
        ]),
      }),
    ],
  });

  enrichStrandsAgentsSpan(span);
  const attrs = span.attributes;

  assert.equal(attrs[RESPAN_LOG_TYPE], "chat");
  assert.equal(attrs[LLM_REQUEST_TYPE], "chat");
  assert.equal(attrs["gen_ai.system"], "strands-agents");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4.1-nano");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 4);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 4);
  assert.equal(attrs[LLM_USAGE_TOTAL_TOKENS], 16);
  assert.equal(attrs["gen_ai.prompt.0.role"], "user");
  assert.equal(attrs["gen_ai.prompt.0.content"], "use the tool");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      id: "tool_1",
      type: "function",
      function: {
        name: "lookup",
        arguments: JSON.stringify({ query: "tracing" }),
      },
    },
  ]);
  assert.equal(attrs["traceloop.span.kind"], undefined);
  assert.equal(attrs["gen_ai.operation.name"], undefined);
  assertNoOffContractAliases(attrs);
});

test("enriches Strands tool spans with canonical input and output", () => {
  const span = makeSpan({
    name: "execute_tool get_weather",
    attributes: {
      "gen_ai.system": "strands-agents",
      "gen_ai.operation.name": "execute_tool",
      "gen_ai.tool.name": "get_weather",
      "gen_ai.tool.call.id": "tool_1",
      "gen_ai.tool.status": "success",
    },
    events: [
      event(EVENT_TOOL_MESSAGE, { content: '{"city":"Seattle"}' }),
      event(EVENT_CHOICE, {
        message: JSON.stringify([{ text: "Sunny and 72F." }]),
      }),
    ],
  });

  enrichStrandsAgentsSpan(span);
  const attrs = span.attributes;

  assert.equal(attrs[RESPAN_LOG_TYPE], "tool");
  assert.equal(attrs[ENTITY_NAME], "get_weather");
  assert.equal(attrs[ENTITY_PATH], "get_weather");
  assert.deepEqual(JSON.parse(attrs[ENTITY_INPUT]), {
    name: "get_weather",
    id: "tool_1",
    arguments: { city: "Seattle" },
  });
  assert.equal(JSON.parse(attrs[ENTITY_OUTPUT]), "Sunny and 72F.");
  assert.equal(attrs["gen_ai.tool.name"], undefined);
  assert.equal(attrs["gen_ai.tool.call.id"], "tool_1");
  assert.equal(attrs["gen_ai.tool.status"], undefined);
  assert.equal(attrs["gen_ai.system"], undefined);
  assert.equal(attrs[LLM_REQUEST_TYPE], undefined);
  assertNoOffContractAliases(attrs);
});

test("enriches Strands graph and swarm spans as workflows", () => {
  const graphSpan = makeSpan({
    name: "invoke_graph graph-demo",
    attributes: {
      "gen_ai.system": "strands-agents",
      "gen_ai.operation.name": "invoke_graph",
      "gen_ai.agent.id": "graph-demo",
      "gen_ai.agent.input": JSON.stringify("draft a city brief"),
    },
  });

  enrichStrandsAgentsSpan(graphSpan);

  assert.equal(graphSpan.attributes[RESPAN_LOG_TYPE], "workflow");
  assert.equal(graphSpan.attributes[ENTITY_NAME], "graph:graph-demo");
  assert.equal(graphSpan.attributes[ENTITY_PATH], "");
  assert.equal(graphSpan.attributes[WORKFLOW_NAME], "graph:graph-demo");
  assert.equal(graphSpan.attributes["gen_ai.operation.name"], undefined);
  assertNoOffContractAliases(graphSpan.attributes);
});

test("recovers tool-only structured output on its owning agent and clears it", async () => {
  const processor = new StrandsAgentsSpanProcessor();
  const traceId = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  const agentSpanId = "aaaaaaaaaaaaaaaa";
  const cycleSpanId = "bbbbbbbbbbbbbbbb";
  const structured = {
    city: "Tokyo",
    score: 92,
    rationale:
      "Reliable transit, compact neighborhoods, and strong food options.",
  };
  const toolSpan = makeSpan({
    name: "execute_tool strands_structured_output",
    traceId,
    spanId: "cccccccccccccccc",
    parentSpanId: cycleSpanId,
    attributes: {
      "gen_ai.system": "strands-agents",
      "gen_ai.operation.name": "execute_tool",
      "gen_ai.tool.name": "strands_structured_output",
      "gen_ai.tool.call.id": "structured-1",
    },
    events: [
      event(EVENT_TOOL_MESSAGE, { content: JSON.stringify(structured) }),
      event(EVENT_CHOICE, {
        message: JSON.stringify([{ json: structured }]),
      }),
    ],
  });
  const cycleSpan = makeSpan({
    name: "execute_event_loop_cycle",
    traceId,
    spanId: cycleSpanId,
    parentSpanId: agentSpanId,
    attributes: {
      "gen_ai.system": "strands-agents",
      "gen_ai.operation.name": "execute_event_loop_cycle",
    },
  });
  const agentSpan = makeSpan({
    name: "invoke_agent Structured",
    traceId,
    spanId: agentSpanId,
    attributes: {
      "gen_ai.system": "strands-agents",
      "gen_ai.operation.name": "invoke_agent",
      "gen_ai.agent.name": "Structured",
    },
    events: [event(EVENT_CHOICE, { message: "" })],
  });

  processor.onEnd(toolSpan);
  processor.onEnd(cycleSpan);
  processor.onEnd(agentSpan);

  assert.deepEqual(
    JSON.parse(processor.prepareForExport(agentSpan).attributes[ENTITY_OUTPUT]),
    [{ role: "assistant", content: structured }],
  );

  const reusedAgent = makeSpan({
    name: "invoke_agent ReusedTrace",
    traceId,
    spanId: "dddddddddddddddd",
    attributes: {
      "gen_ai.system": "strands-agents",
      "gen_ai.operation.name": "invoke_agent",
      "gen_ai.agent.name": "ReusedTrace",
    },
    events: [event(EVENT_CHOICE, { message: "" })],
  });
  processor.onEnd(reusedAgent);
  assert.deepEqual(
    JSON.parse(
      processor.prepareForExport(reusedAgent).attributes[ENTITY_OUTPUT],
    ),
    [{ role: "assistant", content: "" }],
  );

  await processor.shutdown();
});

test("multiple instrumentors retain the semconv opt-in until the final owner deactivates", async () => {
  const previousOptIn = process.env.OTEL_SEMCONV_STABILITY_OPT_IN;
  process.env.OTEL_SEMCONV_STABILITY_OPT_IN = "custom_semconv";

  const manager = {
    onStart() {},
    onEnd() {},
    async shutdown() {},
    async forceFlush() {},
  };
  const composite = new RespanCompositeProcessor(manager);
  const provider = new BasicTracerProvider({ spanProcessors: [composite] });
  const first = new StrandsAgentsInstrumentor();
  const second = new StrandsAgentsInstrumentor();

  try {
    first.activate();
    second.activate();
    assert.equal(
      process.env.OTEL_SEMCONV_STABILITY_OPT_IN,
      "custom_semconv,gen_ai_tool_definitions",
    );

    first.deactivate();
    assert.equal(second.isActive(), true);
    assert.equal(
      process.env.OTEL_SEMCONV_STABILITY_OPT_IN,
      "custom_semconv,gen_ai_tool_definitions",
      "one owner cannot remove an opt-in still needed by another",
    );

    second.deactivate();
    assert.equal(
      process.env.OTEL_SEMCONV_STABILITY_OPT_IN,
      "custom_semconv",
      "the final owner restores the exact original value",
    );
  } finally {
    first.deactivate();
    second.deactivate();
    await provider.shutdown();
    if (previousOptIn === undefined) {
      delete process.env.OTEL_SEMCONV_STABILITY_OPT_IN;
    } else {
      process.env.OTEL_SEMCONV_STABILITY_OPT_IN = previousOptIn;
    }
  }
});

test("OTel 2.10 cached tracers translate complete events and drain in-flight spans", async () => {
  const delegatedSpans = [];
  const manager = {
    onStart() {},
    onEnd(span) {
      delegatedSpans.push(span);
    },
    async shutdown() {},
    async forceFlush() {},
  };
  const composite = new RespanCompositeProcessor(manager);
  const provider = new BasicTracerProvider({ spanProcessors: [composite] });
  const tracerBeforeActivation = provider.getTracer("strands-agents");
  const activeProcessorBefore = provider._activeSpanProcessor;
  const previousOptIn = process.env.OTEL_SEMCONV_STABILITY_OPT_IN;
  const instrumentor = new StrandsAgentsInstrumentor();
  try {
    instrumentor.activate();
    assert.equal(provider._activeSpanProcessor, activeProcessorBefore);
    assert.ok(
      process.env.OTEL_SEMCONV_STABILITY_OPT_IN.includes(
        "gen_ai_tool_definitions",
      ),
    );

    const span = tracerBeforeActivation.startSpan("chat", {
      attributes: {
        "gen_ai.system": "strands-agents",
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "gpt-4.1-nano",
        "gen_ai.usage.input_tokens": 8,
        "gen_ai.usage.output_tokens": 3,
      },
    });
    span.addEvent(EVENT_USER_MESSAGE, {
      content: JSON.stringify([{ text: "hello from Strands" }]),
    });
    span.addEvent(EVENT_CHOICE, { message: "hello back" });
    span.end();

    assert.equal(delegatedSpans.length, 1);
    assert.equal(delegatedSpans[0].attributes[RESPAN_LOG_TYPE], "chat");
    assert.deepEqual(JSON.parse(delegatedSpans[0].attributes[ENTITY_INPUT]), [
      { role: "user", content: "hello from Strands" },
    ]);
    assert.deepEqual(JSON.parse(delegatedSpans[0].attributes[ENTITY_OUTPUT]), [
      { role: "assistant", content: "hello back" },
    ]);
    assert.equal(
      delegatedSpans[0].attributes["gen_ai.operation.name"],
      undefined,
    );
    assert.equal(delegatedSpans[0].attributes["gen_ai.usage.input_tokens"], 8);
    assertNoOffContractAliases(delegatedSpans[0].attributes);

    const tracerAfterActivation = provider.getTracer("strands-agents.after");
    const draining = tracerAfterActivation.startSpan("chat", {
      attributes: {
        "gen_ai.system": "strands-agents",
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "gpt-4.1-nano",
      },
    });
    draining.addEvent(EVENT_USER_MESSAGE, {
      content: JSON.stringify([{ text: "drain me" }]),
    });
    draining.addEvent(EVENT_CHOICE, { message: "drained" });
    instrumentor.deactivate();
    draining.end();
    assert.equal(delegatedSpans[1].attributes[RESPAN_LOG_TYPE], "chat");
    assert.match(delegatedSpans[1].attributes[ENTITY_OUTPUT], /drained/);

    tracerBeforeActivation
      .startSpan("chat", {
        attributes: {
          "gen_ai.system": "strands-agents",
          "gen_ai.operation.name": "chat",
          "gen_ai.request.model": "raw-after-deactivation",
        },
      })
      .end();
    assert.equal(delegatedSpans[2].attributes[RESPAN_LOG_TYPE], undefined);
    assert.equal(delegatedSpans[2].attributes["gen_ai.operation.name"], "chat");
  } finally {
    instrumentor.deactivate();
    await provider.shutdown();
    if (previousOptIn === undefined) {
      delete process.env.OTEL_SEMCONV_STABILITY_OPT_IN;
    } else {
      process.env.OTEL_SEMCONV_STABILITY_OPT_IN = previousOptIn;
    }
  }

  if (previousOptIn === undefined) {
    assert.equal(process.env.OTEL_SEMCONV_STABILITY_OPT_IN, undefined);
  } else {
    assert.equal(process.env.OTEL_SEMCONV_STABILITY_OPT_IN, previousOptIn);
  }
});

test("failed activation releases semconv ownership and restores the exact environment", () => {
  const previousOptIn = process.env.OTEL_SEMCONV_STABILITY_OPT_IN;
  process.env.OTEL_SEMCONV_STABILITY_OPT_IN = "existing_b,existing_a";
  const noHost = new StrandsAgentsInstrumentor();

  try {
    assert.throws(
      () => noHost.activate(),
      /No compatible Respan span-transformer host is active/,
    );
    assert.equal(noHost.isActive(), false);
    assert.equal(
      process.env.OTEL_SEMCONV_STABILITY_OPT_IN,
      "existing_b,existing_a",
    );

    noHost.deactivate();
    assert.equal(
      process.env.OTEL_SEMCONV_STABILITY_OPT_IN,
      "existing_b,existing_a",
      "deactivate after a failed activation is a no-op",
    );
  } finally {
    if (previousOptIn === undefined) {
      delete process.env.OTEL_SEMCONV_STABILITY_OPT_IN;
    } else {
      process.env.OTEL_SEMCONV_STABILITY_OPT_IN = previousOptIn;
    }
  }
});

test("agent schemas merge inherited metadata without LLM request fields", () => {
  const schema = {
    type: "object",
    properties: Object.fromEntries(
      Array.from({ length: 75 }, (_, i) => [`field${i}`, { type: "string" }]),
    ),
  };
  const definitions = [{ name: "lookup", inputSchema: schema }];
  const span = makeSpan({
    name: "invoke_agent Schema",
    attributes: {
      "gen_ai.system": "strands-agents",
      "gen_ai.operation.name": "invoke_agent",
      "respan.metadata": JSON.stringify({ inherited: "kept" }),
      "gen_ai.agent.tools": JSON.stringify(["lookup"]),
      "gen_ai.tool.definitions": JSON.stringify(definitions),
      "gen_ai.request.model": "gpt-4.1-nano",
      "gen_ai.response.model": "gpt-4.1-nano",
      "gen_ai.usage.reasoning.output_tokens": 3,
      "gen_ai.usage.input_tokens": 5,
      "llm.usage.total_tokens": 5,
    },
  });
  enrichStrandsAgentsSpan(span);
  const metadata = JSON.parse(span.attributes["respan.metadata"]);
  assert.equal(metadata.inherited, "kept");
  assert.deepEqual(metadata.strands_agent_tools, ["lookup"]);
  assert.deepEqual(metadata.strands_tool_definitions, definitions);
  assert.equal(span.attributes["llm.request.functions"], undefined);
  assert.equal(span.attributes["gen_ai.request.model"], undefined);
  assert.equal(span.attributes["gen_ai.response.model"], undefined);
  assert.equal(
    span.attributes["gen_ai.usage.reasoning.output_tokens"],
    undefined,
  );
  assert.equal(span.attributes["gen_ai.usage.input_tokens"], undefined);
  assert.equal(span.attributes["llm.usage.total_tokens"], undefined);
});
