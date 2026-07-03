import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import {
  StrandsAgentsInstrumentor,
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

function makeSpan({ name, attributes = {}, events = [] }) {
  const attrs = { ...attributes };
  return {
    name,
    attributes: attrs,
    events,
    spanContext() {
      return {
        traceId: "11111111111111111111111111111111",
        spanId: "2222222222222222",
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
  assert.deepEqual(JSON.parse(attrs[LLM_REQUEST_FUNCTIONS]), [
    { type: "function", function: { name: "get_weather" } },
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
  assert.equal(attrs["gen_ai.system"], "openai");
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
  assert.equal(attrs[ENTITY_OUTPUT], "Sunny and 72F.");
  assert.equal(attrs["gen_ai.tool.name"], undefined);
  assert.equal(attrs["gen_ai.tool.call.id"], undefined);
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

test("instrumentor wraps and restores the active span processor", () => {
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

  const previousOptIn = process.env.OTEL_SEMCONV_STABILITY_OPT_IN;
  const instrumentor = new StrandsAgentsInstrumentor();
  try {
    instrumentor.activate();
    assert.notEqual(provider.activeSpanProcessor, originalProcessor);
    assert.ok(
      process.env.OTEL_SEMCONV_STABILITY_OPT_IN.includes(
        "gen_ai_tool_definitions",
      ),
    );

    const span = makeSpan({
      name: "chat",
      attributes: {
        "gen_ai.system": "strands-agents",
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "gpt-4.1-nano",
      },
    });
    provider.activeSpanProcessor.onEnd(span);

    assert.equal(delegatedSpans.length, 1);
    assert.equal(delegatedSpans[0].attributes[RESPAN_LOG_TYPE], "chat");
  } finally {
    instrumentor.deactivate();
    Object.defineProperty(trace, "getTracerProvider", {
      configurable: true,
      writable: true,
      value: originalGetTracerProvider,
    });
    if (previousOptIn === undefined) {
      delete process.env.OTEL_SEMCONV_STABILITY_OPT_IN;
    } else {
      process.env.OTEL_SEMCONV_STABILITY_OPT_IN = previousOptIn;
    }
  }

  assert.equal(provider.activeSpanProcessor, originalProcessor);
});
