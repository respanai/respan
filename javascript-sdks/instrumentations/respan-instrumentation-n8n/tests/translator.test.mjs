import assert from "node:assert/strict";
import test from "node:test";

import { N8nTransformingExporter } from "../dist/_exporter.js";
import { N8nSpanProcessor } from "../dist/_processor.js";
import { isN8nSpan } from "../dist/_translator.js";

const TRACE_ID = "11111111111111111111111111111111";
const WORKFLOW_SPAN_ID = "2222222222222222";
const NODE_SPAN_ID = "3333333333333333";
const AGENT_SPAN_ID = "4444444444444444";
const TOOL_SPAN_ID = "5555555555555555";
const LLM_WRAPPER_SPAN_ID = "6666666666666666";
const LLM_SPAN_ID = "7777777777777777";
const AI_TOOL_SPAN_ID = "8888888888888888";
const QUERY_MEMORY_SPAN_ID = "9999999999999999";
const SAVE_MEMORY_SPAN_ID = "aaaaaaaaaaaaaaaa";

class CaptureExporter {
  spans = [];
  shutdownCalls = 0;
  flushCalls = 0;

  export(spans, callback) {
    this.spans = spans;
    callback({ code: 0 });
  }

  async shutdown() {
    this.shutdownCalls += 1;
  }

  async forceFlush() {
    this.flushCalls += 1;
  }
}

function makeSpan({
  name,
  spanId,
  parentSpanId,
  attributes = {},
  scope = "n8n-workflow",
  status = { code: 1 },
  events = [],
}) {
  const span = {
    name,
    attributes: { ...attributes },
    instrumentationScope: { name: scope, version: "2.37.7" },
    parentSpanContext: parentSpanId
      ? { traceId: TRACE_ID, spanId: parentSpanId, traceFlags: 1, isRemote: false }
      : undefined,
    status,
    events,
    links: [],
    resource: { attributes: { "service.name": "n8n" } },
    startTime: [1, 0],
    endTime: [2, 0],
    duration: [1, 0],
    ended: true,
    spanContext() {
      return { traceId: TRACE_ID, spanId, traceFlags: 1, isRemote: false };
    },
    setAttribute(key, value) {
      this.attributes[key] = value;
      return this;
    },
    setStatus(value) {
      this.status = value;
      return this;
    },
    updateName(value) {
      this.name = value;
      return this;
    },
    end() {},
    isRecording() {
      return true;
    },
    addEvent() {
      return this;
    },
    addLink() {
      return this;
    },
    addLinks() {
      return this;
    },
    setAttributes(values) {
      Object.assign(this.attributes, values);
      return this;
    },
    recordException() {},
  };
  return span;
}

function exportWith(processor, spans, options) {
  const capture = new CaptureExporter();
  const exporter = new N8nTransformingExporter(capture, processor, options);
  exporter.export(spans, (result) => assert.equal(result.code, 0));
  return { capture, exporter };
}

function metadata(span) {
  return JSON.parse(span.attributes["respan.metadata"]);
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
    assert.equal(attrs[key], undefined, `${key} must not be exported`);
  }
}

test("maps native workflow and node spans while preserving hierarchy, status, and errors", () => {
  const processor = new N8nSpanProcessor();
  const exceptionEvent = {
    name: "exception",
    attributes: {
      "exception.type": "NodeOperationError",
      "exception.message": "customer lookup failed",
    },
  };
  const workflow = makeSpan({
    name: "workflow.execute",
    spanId: WORKFLOW_SPAN_ID,
    attributes: {
      "n8n.workflow.id": "workflow-1",
      "n8n.workflow.name": "Support flow",
      "n8n.workflow.version_id": "version-2",
      "n8n.workflow.node_count": 3,
      "n8n.execution.id": "execution-7",
      "n8n.execution.mode": "manual",
      "n8n.execution.status": "error",
      "n8n.execution.error_type": "NodeOperationError",
      "http.response.status_code": 422,
      "respan.metadata": JSON.stringify({ run_id: "n8n-deterministic" }),
      tools: "must-be-removed",
    },
    status: { code: 2, message: "failed" },
    events: [exceptionEvent],
  });
  const node = makeSpan({
    name: "node.execute",
    spanId: NODE_SPAN_ID,
    parentSpanId: WORKFLOW_SPAN_ID,
    attributes: {
      "n8n.node.id": "node-1",
      "n8n.node.name": "Fetch customer",
      "n8n.node.type": "n8n-nodes-base.httpRequest",
      "n8n.node.type_version": 4.2,
      "n8n.node.items.input": 1,
      "n8n.node.items.output": 0,
      "n8n.node.termination_reason": "workflow_cancelled",
    },
    status: { code: 2 },
    events: [exceptionEvent],
  });

  processor.onStart(workflow, undefined);
  processor.onStart(node, undefined);
  processor.onEnd(node);
  processor.onEnd(workflow);

  const { capture } = exportWith(processor, [node, workflow]);
  const [exportedNode, exportedWorkflow] = capture.spans;

  assert.equal(exportedWorkflow.name, "workflow");
  assert.equal(exportedWorkflow.attributes["respan.entity.log_type"], "workflow");
  assert.equal(exportedWorkflow.attributes["traceloop.entity.name"], "Support flow");
  assert.equal(exportedWorkflow.attributes["traceloop.workflow.name"], "Support flow");
  assert.equal(exportedWorkflow.attributes["n8n.workflow.id"], undefined);
  assert.deepEqual(metadata(exportedWorkflow), {
    run_id: "n8n-deterministic",
    n8n: {
      "workflow.id": "workflow-1",
      "workflow.name": "Support flow",
      "workflow.version_id": "version-2",
      "workflow.node_count": 3,
      "execution.id": "execution-7",
      "execution.mode": "manual",
      "execution.status": "error",
      "execution.error_type": "NodeOperationError",
    },
  });
  assert.deepEqual(exportedWorkflow.status, { code: 2, message: "failed" });
  assert.equal(exportedWorkflow.attributes.status_code, 422);
  assert.equal(exportedWorkflow.attributes["error.message"], "customer lookup failed");
  assert.equal(exportedWorkflow.attributes["http.response.status_code"], 422);
  assert.strictEqual(exportedWorkflow.events[0], exceptionEvent);

  assert.equal(exportedNode.name, "task");
  assert.equal(exportedNode.attributes["respan.entity.log_type"], "task");
  assert.equal(exportedNode.attributes["traceloop.entity.name"], "Fetch customer");
  assert.equal(exportedNode.attributes["traceloop.entity.path"], "Support flow");
  assert.equal(exportedNode.attributes["traceloop.workflow.name"], "Support flow");
  assert.equal(exportedNode.attributes["n8n.node.id"], undefined);
  assert.equal(exportedNode.parentSpanContext.spanId, WORKFLOW_SPAN_ID);
  assert.deepEqual(metadata(exportedNode).n8n, {
    "node.id": "node-1",
    "node.name": "Fetch customer",
    "node.type": "n8n-nodes-base.httpRequest",
    "node.type_version": 4.2,
    "node.items.input": 1,
    "node.items.output": 0,
    "node.termination_reason": "workflow_cancelled",
  });
  assert.deepEqual(exportedNode.status, { code: 2 });
  assert.equal(exportedNode.attributes.status_code, 500);
  assert.equal(exportedNode.attributes["error.message"], "customer lookup failed");
  assert.strictEqual(exportedNode.events[0], exceptionEvent);
  assert.equal(workflow.attributes["error.message"], undefined);
  assert.equal(node.attributes["error.message"], undefined);
  assertNoOffContractAliases(exportedWorkflow.attributes);
  assertNoOffContractAliases(exportedNode.attributes);
});

test("projects existing and fallback status codes for backend success and error state", () => {
  const processor = new N8nSpanProcessor();
  const success = makeSpan({
    name: "workflow.execute",
    spanId: WORKFLOW_SPAN_ID,
    attributes: {
      "n8n.workflow.id": "workflow-success",
      "n8n.workflow.name": "Successful workflow",
    },
    status: { code: 1 },
    events: [
      {
        name: "exception",
        attributes: { "exception.message": "must not mark a successful span failed" },
      },
    ],
  });
  const unsetWithHttpStatus = makeSpan({
    name: "node.execute",
    spanId: NODE_SPAN_ID,
    parentSpanId: WORKFLOW_SPAN_ID,
    attributes: {
      "n8n.node.id": "node-created",
      "n8n.node.name": "Created record",
      "http.status_code": "201",
    },
    status: { code: 0 },
  });
  const errorWithProviderStatus = makeSpan({
    name: "support-agent.generate",
    spanId: AGENT_SPAN_ID,
    scope: "@n8n/agents",
    attributes: {
      "gen_ai.operation.name": "invoke_agent",
      "gen_ai.agent.name": "support-agent",
      "gen_ai.response.status_code": 429,
    },
    status: { code: 2 },
  });
  const errorWithSuccessfulStatusAlias = makeSpan({
    name: "execute_tool lookup",
    spanId: TOOL_SPAN_ID,
    parentSpanId: AGENT_SPAN_ID,
    scope: "@n8n/agents",
    attributes: {
      "gen_ai.operation.name": "execute_tool",
      "gen_ai.tool.name": "lookup",
      status_code: 200,
    },
    status: { code: 2 },
  });

  for (const span of [success, unsetWithHttpStatus, errorWithProviderStatus, errorWithSuccessfulStatusAlias]) {
    processor.onStart(span, undefined);
    processor.onEnd(span);
  }

  const { capture } = exportWith(processor, [
    success,
    unsetWithHttpStatus,
    errorWithProviderStatus,
    errorWithSuccessfulStatusAlias,
  ]);
  assert.equal(capture.spans[0].attributes.status_code, 200);
  assert.equal(capture.spans[0].attributes["error.message"], undefined);
  assert.equal(capture.spans[1].attributes.status_code, 201);
  assert.equal(capture.spans[1].attributes["http.status_code"], "201");
  assert.equal(capture.spans[2].attributes.status_code, 429);
  assert.equal(capture.spans[2].attributes["gen_ai.response.status_code"], undefined);
  assert.equal(capture.spans[3].attributes.status_code, 500);
  assert.deepEqual(capture.spans.map((span) => span.status.code), [1, 0, 2, 2]);
});

test("maps current n8n Agent and execute_tool spans without duplicate tool aliases", () => {
  const processor = new N8nSpanProcessor();
  const workflow = makeSpan({
    name: "workflow.execute",
    spanId: WORKFLOW_SPAN_ID,
    attributes: {
      "n8n.workflow.id": "workflow-agent",
      "n8n.workflow.name": "Agent workflow",
      "n8n.execution.id": "execution-agent",
    },
  });
  const agent = makeSpan({
    name: "support-agent.stream",
    spanId: AGENT_SPAN_ID,
    parentSpanId: NODE_SPAN_ID,
    scope: "@n8n/agents",
    attributes: {
      "gen_ai.operation.name": "invoke_agent",
      "gen_ai.agent.name": "support-agent",
      "gen_ai.request.model": "openai/gpt-4o-mini",
      "gen_ai.conversation.id": "thread-9",
      "gen_ai.prompt": JSON.stringify({ tool_count: 1, tools: [{ name: "lookup" }] }),
      agent_id: "agent-1",
      project_id: "project-1",
      thread_id: "thread-9",
      source: "workflow",
      execution_id: "execution-agent",
      workflow_id: "workflow-agent",
      node_id: "agent-node",
      "ai.telemetry.metadata.user_id": "user-1",
    },
  });
  const tool = makeSpan({
    name: "execute_tool lookup",
    spanId: TOOL_SPAN_ID,
    parentSpanId: AGENT_SPAN_ID,
    scope: "@n8n/agents",
    attributes: {
      "gen_ai.operation.name": "execute_tool",
      "gen_ai.agent.name": "support-agent",
      "gen_ai.tool.name": "lookup",
      "gen_ai.tool.call.id": "call-1",
      "gen_ai.tool.call.arguments": JSON.stringify({ city: "Paris" }),
      "gen_ai.tool.call.result": JSON.stringify({ temperature: 21 }),
      "ai.toolCall.name": "lookup",
      "ai.toolCall.args": JSON.stringify({ city: "Paris" }),
      "ai.toolCall.result": JSON.stringify({ temperature: 21 }),
      "ai.telemetry.metadata.execution_id": "execution-agent",
      "ai.telemetry.metadata.workflow_id": "workflow-agent",
    },
  });

  processor.onStart(workflow, undefined);
  processor.onStart(agent, undefined);
  processor.onStart(tool, undefined);
  processor.onEnd(tool);
  processor.onEnd(agent);
  processor.onEnd(workflow);

  const { capture } = exportWith(processor, [tool, agent, workflow]);
  const [exportedTool, exportedAgent] = capture.spans;

  assert.equal(exportedAgent.name, "agent.support-agent");
  assert.equal(exportedAgent.attributes["respan.entity.log_type"], "agent");
  assert.equal(exportedAgent.attributes["traceloop.entity.name"], "support-agent");
  assert.equal(
    Object.keys(exportedAgent.attributes).some((key) => key.startsWith("respan.metadata.")),
    false,
  );
  assert.equal(exportedAgent.attributes["traceloop.entity.path"], "Agent workflow");
  assert.equal(exportedAgent.attributes["respan.threads.thread_identifier"], "thread-9");
  assert.deepEqual(
    JSON.parse(exportedAgent.attributes["traceloop.entity.input"]),
    { tool_count: 1, tools: [{ name: "lookup" }] },
  );
  assert.equal(exportedAgent.attributes.agent_id, undefined);
  assert.equal(exportedAgent.attributes["ai.telemetry.metadata.user_id"], undefined);
  assert.equal(
    Object.keys(exportedAgent.attributes).some((key) => key.startsWith("gen_ai.")),
    false,
  );
  assert.deepEqual(metadata(exportedAgent).n8n, {
    agent_id: "agent-1",
    project_id: "project-1",
    thread_id: "thread-9",
    source: "workflow",
    user_id: "user-1",
    execution_id: "execution-agent",
    workflow_id: "workflow-agent",
    node_id: "agent-node",
    model: "openai/gpt-4o-mini",
  });

  assert.equal(exportedTool.name, "tool.lookup");
  assert.equal(exportedTool.attributes["respan.entity.log_type"], "tool");
  assert.equal(exportedTool.attributes["traceloop.entity.name"], "lookup");
  assert.deepEqual(JSON.parse(exportedTool.attributes["traceloop.entity.input"]), {
    name: "lookup",
    arguments: { city: "Paris" },
  });
  assert.deepEqual(JSON.parse(exportedTool.attributes["traceloop.entity.output"]), {
    temperature: 21,
  });
  assert.equal(exportedTool.attributes["gen_ai.tool.name"], undefined);
  assert.equal(exportedTool.attributes["gen_ai.tool.call.arguments"], undefined);
  assert.equal(exportedTool.attributes["gen_ai.tool.call.result"], undefined);
  assert.equal(exportedTool.attributes["ai.toolCall.args"], undefined);
  assert.equal(
    Object.keys(exportedTool.attributes).some((key) => key.startsWith("gen_ai.")),
    false,
  );
  assert.equal(exportedTool.parentSpanContext.spanId, AGENT_SPAN_ID);
  assert.deepEqual(metadata(exportedTool).n8n, {
    execution_id: "execution-agent",
    workflow_id: "workflow-agent",
    tool_call_id: "call-1",
  });
  assertNoOffContractAliases(exportedAgent.attributes);
  assertNoOffContractAliases(exportedTool.attributes);
});

test("maps exact LegacyOpenTelemetry generate spans and suppresses its ai.toolCall duplicate", () => {
  const processor = new N8nSpanProcessor();
  const secret = "Bearer must-not-leave-the-process";
  const metadataSecret = "metadata-secret-must-not-leave-the-process";
  const agent = makeSpan({
    name: "support-agent.generate",
    spanId: AGENT_SPAN_ID,
    scope: "@n8n/agents",
    attributes: {
      "gen_ai.operation.name": "invoke_agent",
      "gen_ai.agent.name": "support-agent",
      agent_id: "agent-current",
      execution_id: "execution-current",
    },
  });
  const wrapper = makeSpan({
    name: "ai.generateText",
    spanId: LLM_WRAPPER_SPAN_ID,
    parentSpanId: AGENT_SPAN_ID,
    scope: "@n8n/agents",
    attributes: {
      "operation.name": "ai.generateText support-agent",
      "resource.name": "support-agent",
      "ai.operationId": "ai.generateText",
      "ai.telemetry.functionId": "support-agent",
      "ai.telemetry.metadata.execution_id": "execution-current",
      "ai.telemetry.metadata.run_id": "agent-run-current",
      "ai.telemetry.metadata.api_key": metadataSecret,
      "ai.model.provider": "openai.chat",
      "ai.model.id": "gpt-4o-mini",
      "ai.prompt": JSON.stringify({
        system: "Answer tersely",
        messages: [{ role: "user", content: "Find customer 7" }],
      }),
      "ai.request.headers.authorization": secret,
    },
  });
  const llm = makeSpan({
    name: "ai.generateText.doGenerate",
    spanId: LLM_SPAN_ID,
    parentSpanId: LLM_WRAPPER_SPAN_ID,
    scope: "@n8n/agents",
    attributes: {
      "operation.name": "ai.generateText.doGenerate support-agent",
      "resource.name": "support-agent",
      "ai.operationId": "ai.generateText.doGenerate",
      "ai.telemetry.functionId": "support-agent",
      "ai.telemetry.metadata.execution_id": "execution-current",
      "ai.telemetry.metadata.run_id": "agent-run-current",
      "ai.telemetry.metadata.api_key": metadataSecret,
      "ai.model.provider": "openai.chat",
      "ai.model.id": "gpt-4o-mini",
      "ai.request.headers.authorization": secret,
      "ai.prompt.messages": JSON.stringify([
        {
          role: "user",
          content: [{ type: "text", text: "Find customer 7" }],
        },
      ]),
      "ai.prompt.tools": [
        JSON.stringify({
          type: "function",
          name: "lookup",
          description: "Look up a customer",
          inputSchema: {
            type: "object",
            properties: { id: { type: "number" } },
          },
        }),
      ],
      "ai.prompt.toolChoice": JSON.stringify({ type: "auto" }),
      "gen_ai.system": "openai.chat",
      "gen_ai.request.model": "gpt-4o-mini",
      "ai.response.finishReason": "tool-calls",
      "ai.response.text": "I will look that up.",
      "ai.response.toolCalls": JSON.stringify([
        { toolCallId: "call-current", toolName: "lookup", input: { id: 7 } },
      ]),
      "ai.response.id": "response-current",
      "ai.response.model": "gpt-4o-mini-2026-08-01",
      "ai.usage.inputTokens": 11,
      "ai.usage.outputTokens": 4,
      "ai.usage.totalTokens": 15,
      "ai.usage.cachedInputTokens": 3,
    },
  });
  const aiTool = makeSpan({
    name: "ai.toolCall",
    spanId: AI_TOOL_SPAN_ID,
    parentSpanId: LLM_SPAN_ID,
    scope: "@n8n/agents",
    attributes: {
      "operation.name": "ai.toolCall support-agent",
      "resource.name": "support-agent",
      "ai.operationId": "ai.toolCall",
      "ai.telemetry.functionId": "support-agent",
      "ai.telemetry.metadata.execution_id": "execution-current",
      "ai.toolCall.name": "lookup",
      "ai.toolCall.id": "call-current",
      "ai.toolCall.args": JSON.stringify({ id: 7 }),
      "ai.toolCall.result": JSON.stringify({ tier: "enterprise" }),
    },
  });
  const ownerTool = makeSpan({
    name: "execute_tool lookup",
    spanId: TOOL_SPAN_ID,
    parentSpanId: AI_TOOL_SPAN_ID,
    scope: "@n8n/agents",
    attributes: {
      "operation.name": "ai.toolCall support-agent",
      "resource.name": "support-agent",
      "ai.operationId": "ai.toolCall",
      "ai.telemetry.functionId": "support-agent",
      "gen_ai.operation.name": "execute_tool",
      "gen_ai.agent.name": "support-agent",
      "gen_ai.tool.name": "lookup",
      "gen_ai.tool.call.id": "call-current",
      "gen_ai.tool.call.arguments": JSON.stringify({ id: 7 }),
      "gen_ai.tool.call.result": JSON.stringify({ tier: "enterprise" }),
      "ai.toolCall.name": "lookup",
      "ai.toolCall.id": "call-current",
      "ai.toolCall.args": JSON.stringify({ id: 7 }),
      "ai.toolCall.result": JSON.stringify({ tier: "enterprise" }),
    },
  });

  for (const span of [agent, wrapper, llm, aiTool, ownerTool]) {
    processor.onStart(span, undefined);
  }
  for (const span of [ownerTool, aiTool, llm, wrapper, agent]) {
    processor.onEnd(span);
  }

  const { capture } = exportWith(processor, [ownerTool, aiTool, llm, wrapper, agent]);
  assert.deepEqual(capture.spans.map((span) => span.name), [
    "tool.lookup",
    "llm.gpt-4o-mini",
    "agent.support-agent",
  ]);
  const [exportedTool, exportedLlm, exportedAgent] = capture.spans;
  assert.equal(exportedLlm.parentSpanContext.spanId, AGENT_SPAN_ID);
  assert.equal(exportedTool.parentSpanContext.spanId, LLM_SPAN_ID);
  assert.equal(exportedAgent.attributes["respan.entity.log_type"], "agent");
  assert.equal(exportedLlm.attributes["respan.entity.log_type"], "text");
  assert.equal(exportedLlm.attributes["llm.request.type"], "chat");
  assert.equal(exportedLlm.attributes["gen_ai.system"], "openai");
  assert.equal(exportedLlm.attributes["gen_ai.request.model"], "gpt-4o-mini");
  assert.equal(exportedLlm.attributes["gen_ai.prompt.0.role"], "user");
  assert.equal(exportedLlm.attributes["gen_ai.prompt.0.content"], "Find customer 7");
  assert.equal(exportedLlm.attributes["gen_ai.completion.0.role"], "assistant");
  assert.equal(
    exportedLlm.attributes["gen_ai.completion.0.content"],
    "I will look that up.",
  );
  assert.deepEqual(
    JSON.parse(exportedLlm.attributes["gen_ai.completion.0.tool_calls"]),
    [
      {
        type: "function",
        id: "call-current",
        function: { name: "lookup", arguments: JSON.stringify({ id: 7 }) },
      },
    ],
  );
  assert.deepEqual(
    JSON.parse(exportedLlm.attributes["llm.request.functions"]),
    [
      {
        type: "function",
        function: {
          name: "lookup",
          description: "Look up a customer",
          parameters: {
            type: "object",
            properties: { id: { type: "number" } },
          },
        },
      },
    ],
  );
  assert.equal(exportedLlm.attributes["gen_ai.usage.input_tokens"], 11);
  assert.equal(exportedLlm.attributes["gen_ai.usage.prompt_tokens"], 11);
  assert.equal(exportedLlm.attributes["gen_ai.usage.output_tokens"], 4);
  assert.equal(exportedLlm.attributes["gen_ai.usage.completion_tokens"], 4);
  assert.equal(exportedLlm.attributes["llm.usage.total_tokens"], 15);
  assert.equal(exportedLlm.attributes["gen_ai.usage.cache_read.input_tokens"], 3);
  assert.equal(exportedLlm.attributes["llm.usage.cache_read_input_tokens"], 3);
  assert.equal(exportedLlm.attributes["traceloop.span.kind"], undefined);
  assert.equal(
    Object.keys(exportedLlm.attributes).some((key) => key.startsWith("ai.")),
    false,
  );
  assert.equal(JSON.stringify(capture.spans).includes(secret), false);
  assert.equal(JSON.stringify(capture.spans).includes(metadataSecret), false);
  assert.deepEqual(metadata(exportedLlm).n8n.ai_sdk, {
    operation: "ai.generateText.doGenerate",
    function_id: "support-agent",
  });
  assert.equal(metadata(exportedLlm).n8n.execution_id, "execution-current");
  assert.deepEqual(metadata(exportedLlm).n8n.telemetry, {
    run_id: "agent-run-current",
    api_key: "[REDACTED]",
  });
  assertNoOffContractAliases(exportedLlm.attributes);
  assertNoOffContractAliases(exportedTool.attributes);
});

test("maps LegacyOpenTelemetry stream detail and keeps structural spans only in legacy mode", () => {
  function makeStreamFixture() {
    const wrapper = makeSpan({
      name: "ai.streamText",
      spanId: LLM_WRAPPER_SPAN_ID,
      parentSpanId: AGENT_SPAN_ID,
      scope: "@n8n/agents",
      attributes: {
        "ai.operationId": "ai.streamText",
        "ai.telemetry.functionId": "stream-agent",
        "ai.model.provider": "anthropic.messages",
        "ai.model.id": "claude-3-5-sonnet",
      },
    });
    const detail = makeSpan({
      name: "ai.streamText.doStream",
      spanId: LLM_SPAN_ID,
      parentSpanId: LLM_WRAPPER_SPAN_ID,
      scope: "@n8n/agents",
      events: [
        {
          name: "ai.stream.firstChunk",
          attributes: { "ai.response.msToFirstChunk": 17 },
        },
        {
          name: "ai.stream.finish",
          attributes: {
            "ai.response.msToFinish": 39,
            "ai.response.avgOutputTokensPerSecond": 51.28,
          },
        },
      ],
      attributes: {
        "ai.operationId": "ai.streamText.doStream",
        "ai.telemetry.functionId": "stream-agent",
        "ai.model.provider": "anthropic.messages",
        "ai.model.id": "claude-3-5-sonnet",
        "ai.prompt.messages": JSON.stringify([{ role: "user", content: "Stream it" }]),
        "gen_ai.system": "anthropic.messages",
        "gen_ai.request.model": "claude-3-5-sonnet",
        "ai.response.finishReason": "stop",
        "ai.response.text": "streamed answer",
        "ai.response.msToFirstChunk": 17,
        "ai.response.msToFinish": 39,
        "ai.usage.inputTokens": 5,
        "ai.usage.outputTokens": 2,
      },
    });
    return { wrapper, detail };
  }

  const semanticProcessor = new N8nSpanProcessor();
  const semantic = makeStreamFixture();
  semanticProcessor.onStart(semantic.wrapper, undefined);
  semanticProcessor.onStart(semantic.detail, undefined);
  semanticProcessor.onEnd(semantic.detail);
  semanticProcessor.onEnd(semantic.wrapper);
  const semanticCapture = exportWith(
    semanticProcessor,
    [semantic.detail, semantic.wrapper],
  ).capture;
  assert.deepEqual(semanticCapture.spans.map((span) => span.name), [
    "llm.claude-3-5-sonnet",
  ]);
  assert.equal(semanticCapture.spans[0].parentSpanContext.spanId, AGENT_SPAN_ID);
  assert.equal(
    JSON.parse(semanticCapture.spans[0].attributes["traceloop.entity.output"]).content,
    "streamed answer",
  );
  assert.equal(
    Object.keys(semanticCapture.spans[0].attributes).some((key) => key.startsWith("ai.")),
    false,
  );
  assert.deepEqual(semanticCapture.spans[0].events, []);
  assert.deepEqual(metadata(semanticCapture.spans[0]).n8n.ai_sdk, {
    operation: "ai.streamText.doStream",
    function_id: "stream-agent",
    time_to_first_output_ms: 17,
    response_time_ms: 39,
  });

  const legacyProcessor = new N8nSpanProcessor();
  const legacy = makeStreamFixture();
  legacyProcessor.onStart(legacy.wrapper, undefined);
  legacyProcessor.onStart(legacy.detail, undefined);
  legacyProcessor.onEnd(legacy.detail);
  legacyProcessor.onEnd(legacy.wrapper);
  const legacyCapture = exportWith(legacyProcessor, [legacy.detail, legacy.wrapper], {
    spanNameStyle: "legacy",
  }).capture;
  assert.deepEqual(legacyCapture.spans.map((span) => span.name), [
    "ai.streamText.doStream",
    "ai.streamText",
  ]);
  for (const span of legacyCapture.spans) {
    assert.equal(span.attributes["respan.internal.drop_span"], undefined);
    assert.equal(span.attributes["respan.internal.export_parent_span_id"], undefined);
    assert.equal(Object.keys(span.attributes).some((key) => key.startsWith("ai.")), false);
    assert.equal(span.events.some((event) => event.name.startsWith("ai.")), false);
  }
});

test("keeps an error-only AI SDK root when no detailed provider span starts", () => {
  const processor = new N8nSpanProcessor();
  const exception = {
    name: "exception",
    attributes: { "exception.type": "ProviderError", "exception.message": "failed early" },
  };
  const wrapper = makeSpan({
    name: "ai.generateText",
    spanId: LLM_WRAPPER_SPAN_ID,
    parentSpanId: AGENT_SPAN_ID,
    scope: "@n8n/agents",
    status: { code: 2, message: "failed early" },
    events: [exception],
    attributes: {
      "ai.operationId": "ai.generateText",
      "ai.telemetry.functionId": "error-agent",
      "ai.model.provider": "openai.chat",
      "ai.model.id": "gpt-4o-mini",
      "ai.prompt": JSON.stringify({ messages: [{ role: "user", content: "fail" }] }),
      "ai.request.headers.authorization": "Bearer error-secret",
    },
  });

  processor.onStart(wrapper, undefined);
  processor.onEnd(wrapper);
  const { capture } = exportWith(processor, [wrapper]);
  assert.equal(capture.spans.length, 1);
  assert.equal(capture.spans[0].name, "llm.gpt-4o-mini");
  assert.equal(capture.spans[0].attributes.status_code, 500);
  assert.strictEqual(capture.spans[0].events[0], exception);
  assert.equal(
    Object.keys(capture.spans[0].attributes).some((key) => key.startsWith("ai.")),
    false,
  );
  assert.equal(JSON.stringify(capture.spans[0].attributes).includes("error-secret"), false);
});

test("keeps an unmatched AI SDK tool call as the canonical tool owner", () => {
  const processor = new N8nSpanProcessor();
  const aiTool = makeSpan({
    name: "ai.toolCall",
    spanId: AI_TOOL_SPAN_ID,
    parentSpanId: LLM_SPAN_ID,
    scope: "@n8n/agents",
    attributes: {
      "ai.operationId": "ai.toolCall",
      "ai.telemetry.functionId": "provider-tool-agent",
      "ai.toolCall.name": "provider_lookup",
      "ai.toolCall.id": "provider-call-1",
      "ai.toolCall.args": JSON.stringify({ id: 9 }),
      "ai.toolCall.result": JSON.stringify({ found: true }),
    },
  });

  processor.onStart(aiTool, undefined);
  processor.onEnd(aiTool);
  const { capture } = exportWith(processor, [aiTool]);
  assert.equal(capture.spans.length, 1);
  assert.equal(capture.spans[0].name, "tool.provider_lookup");
  assert.equal(capture.spans[0].parentSpanContext.spanId, LLM_SPAN_ID);
  assert.deepEqual(JSON.parse(capture.spans[0].attributes["traceloop.entity.input"]), {
    name: "provider_lookup",
    arguments: { id: 9 },
  });
  assert.deepEqual(JSON.parse(capture.spans[0].attributes["traceloop.entity.output"]), {
    found: true,
  });
  assert.equal(
    Object.keys(capture.spans[0].attributes).some((key) => key.startsWith("ai.")),
    false,
  );
});

test("maps n8n 2.37.7 memory spans to canonical tasks and metadata", () => {
  const processor = new N8nSpanProcessor();
  const query = makeSpan({
    name: "query_memory",
    spanId: QUERY_MEMORY_SPAN_ID,
    parentSpanId: AGENT_SPAN_ID,
    scope: "@n8n/agents",
    attributes: {
      "gen_ai.operation.name": "query_memory",
      "gen_ai.agent.name": "support-agent",
      "gen_ai.memory.types": ["session"],
      "gen_ai.memory.owners": ["customer-7"],
      "gen_ai.memory.store.types": ["in_memory"],
      "gen_ai.memory.store.names": ["support-history"],
    },
  });
  const save = makeSpan({
    name: "save_memory",
    spanId: SAVE_MEMORY_SPAN_ID,
    parentSpanId: AGENT_SPAN_ID,
    scope: "@n8n/agents",
    attributes: {
      "gen_ai.operation.name": "save_memory",
      "gen_ai.agent.name": "support-agent",
      "gen_ai.memory.types": ["agent"],
      "gen_ai.memory.owners": ["customer-7"],
      "gen_ai.memory.store.types": ["postgres"],
      "gen_ai.memory.store.names": ["episodic-memory"],
    },
  });

  processor.onStart(query, undefined);
  processor.onStart(save, undefined);
  query.setAttributes({
    "gen_ai.memory.ids": ["message-1"],
    "gen_ai.memory.operations": ["query_memory"],
    "gen_ai.memory.descriptions": ["conversation history"],
  });
  save.setAttributes({
    "gen_ai.memory.ids": ["memory-1"],
    "gen_ai.memory.operations": ["created"],
  });
  processor.onEnd(query);
  processor.onEnd(save);

  const { capture } = exportWith(processor, [query, save]);
  assert.deepEqual(capture.spans.map((span) => span.name), ["task", "task"]);
  const [exportedQuery, exportedSave] = capture.spans;
  assert.equal(exportedQuery.attributes["respan.entity.log_type"], "task");
  assert.equal(exportedQuery.attributes["traceloop.entity.name"], "query_memory");
  assert.equal(exportedSave.attributes["traceloop.entity.name"], "save_memory");
  assert.deepEqual(JSON.parse(exportedQuery.attributes["traceloop.entity.input"]), {
    operation: "query_memory",
    types: ["session"],
    owners: ["customer-7"],
    store_types: ["in_memory"],
    store_names: ["support-history"],
  });
  assert.deepEqual(JSON.parse(exportedQuery.attributes["traceloop.entity.output"]), {
    ids: ["message-1"],
    descriptions: ["conversation history"],
    operations: ["query_memory"],
  });
  assert.deepEqual(metadata(exportedQuery).n8n.memory, {
    types: ["session"],
    owners: ["customer-7"],
    "store.types": ["in_memory"],
    "store.names": ["support-history"],
    ids: ["message-1"],
    operations: ["query_memory"],
    descriptions: ["conversation history"],
    operation: "query_memory",
    agent_name: "support-agent",
  });
  assert.deepEqual(metadata(exportedSave).n8n.memory, {
    types: ["agent"],
    owners: ["customer-7"],
    "store.types": ["postgres"],
    "store.names": ["episodic-memory"],
    ids: ["memory-1"],
    operations: ["created"],
    operation: "save_memory",
    agent_name: "support-agent",
  });
  for (const span of capture.spans) {
    assert.equal(span.attributes["traceloop.span.kind"], undefined);
    assert.equal(
      Object.keys(span.attributes).some((key) => key.startsWith("gen_ai.memory.")),
      false,
    );
    assert.equal(Object.keys(span.attributes).some((key) => key.startsWith("gen_ai.")), false);
    assertNoOffContractAliases(span.attributes);
  }
});

test("does not synthesize agent or tool content when n8n recording is disabled", () => {
  const processor = new N8nSpanProcessor();
  const agent = makeSpan({
    name: "private-agent.generate",
    spanId: AGENT_SPAN_ID,
    scope: "@n8n/agents",
    attributes: {
      "gen_ai.operation.name": "invoke_agent",
      "gen_ai.agent.name": "private-agent",
      agent_id: "agent-private",
      source: "schedule",
    },
  });
  const tool = makeSpan({
    name: "execute_tool private_lookup",
    spanId: TOOL_SPAN_ID,
    parentSpanId: AGENT_SPAN_ID,
    scope: "@n8n/agents",
    attributes: {
      "gen_ai.operation.name": "execute_tool",
      "gen_ai.tool.name": "private_lookup",
    },
  });

  processor.onStart(agent, undefined);
  processor.onStart(tool, undefined);
  processor.onEnd(tool);
  processor.onEnd(agent);

  const { capture } = exportWith(processor, [tool, agent]);
  const [exportedTool, exportedAgent] = capture.spans;
  assert.equal(exportedAgent.attributes["traceloop.entity.input"], undefined);
  assert.equal(exportedAgent.attributes["traceloop.entity.output"], undefined);
  assert.equal(exportedTool.attributes["traceloop.entity.input"], undefined);
  assert.equal(exportedTool.attributes["traceloop.entity.output"], undefined);
});

test("leaves unrelated spans byte-shape compatible and supports legacy names", () => {
  const processor = new N8nSpanProcessor();
  const unrelated = makeSpan({
    name: "workflow.execute",
    spanId: WORKFLOW_SPAN_ID,
    scope: "another-framework",
    attributes: { "vendor.workflow.id": "not-n8n" },
  });
  assert.equal(isN8nSpan(unrelated), false);

  const workflow = makeSpan({
    name: "workflow.execute",
    spanId: NODE_SPAN_ID,
    attributes: { "n8n.workflow.id": "w", "n8n.workflow.name": "Legacy workflow" },
  });
  processor.onStart(workflow, undefined);
  processor.onEnd(workflow);

  const { capture } = exportWith(processor, [unrelated, workflow], {
    spanNameStyle: "legacy",
  });
  assert.strictEqual(capture.spans[0], unrelated);
  assert.equal(capture.spans[1].name, "workflow.execute");
  assert.equal(capture.spans[1].attributes["n8n.workflow.id"], undefined);
});

test("exporter delegates lifecycle methods and fails open on transformation errors", async () => {
  class ThrowingProcessor extends N8nSpanProcessor {
    prepareForExport() {
      throw new Error("synthetic translation failure");
    }
  }

  const original = makeSpan({
    name: "workflow.execute",
    spanId: WORKFLOW_SPAN_ID,
    attributes: { "n8n.workflow.id": "w", "n8n.workflow.name": "Fail open" },
  });
  const sensitive = makeSpan({
    name: "ai.generateText",
    spanId: LLM_WRAPPER_SPAN_ID,
    scope: "@n8n/agents",
    attributes: {
      "ai.operationId": "ai.generateText",
      "ai.request.headers.authorization": "Bearer fail-safe-secret",
      "ai.telemetry.metadata.api_key": "fail-safe-metadata-secret",
      "ai.telemetry.metadata.run_id": "fail-safe-run",
      "ai.settings.context.api_key": "fail-safe-context-secret",
    },
    events: [
      {
        name: "diagnostic",
        attributes: {
          "ai.request.headers.x-api-key": "fail-safe-event-secret",
          retained: "yes",
        },
      },
    ],
  });
  const capture = new CaptureExporter();
  const exporter = new N8nTransformingExporter(capture, new ThrowingProcessor());
  const warnings = [];
  const originalWarn = console.warn;
  console.warn = (...args) => warnings.push(args);
  try {
    exporter.export([original, sensitive], (result) => assert.equal(result.code, 0));
  } finally {
    console.warn = originalWarn;
  }
  assert.strictEqual(capture.spans[0], original);
  assert.notStrictEqual(capture.spans[1], sensitive);
  assert.equal(
    Object.keys(capture.spans[1].attributes).some((key) => key.startsWith("ai.")),
    false,
  );
  assert.deepEqual(capture.spans[1].events[0].attributes, { retained: "yes" });
  assert.equal(
    JSON.stringify({
      attributes: capture.spans[1].attributes,
      events: capture.spans[1].events,
    }).includes("fail-safe"),
    false,
  );
  assert.equal(warnings.length, 1);

  await exporter.forceFlush();
  await exporter.shutdown();
  assert.equal(capture.flushCalls, 1);
  assert.equal(capture.shutdownCalls, 1);
});
