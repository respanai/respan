import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import { CodexSDKInstrumentor } from "../dist/index.js";

const captureState = { spans: [] };
const originalGetTracerProvider = trace.getTracerProvider.bind(trace);

test.before(() => {
  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value() {
      return {
        activeSpanProcessor: {
          onEnd(span) {
            captureState.spans.push(span);
          },
        },
      };
    },
  });
});

test.after(() => {
  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value: originalGetTracerProvider,
  });
});

function createFakeSDK() {
  class FakeThread {
    constructor() {
      this._id = null;
      this._threadOptions = {
        model: "gpt-5.4",
        workingDirectory: "/tmp/codex-demo",
      };
    }

    get id() {
      return this._id;
    }

    async run(input) {
      this._id = "thread_run_123";
      assert.equal(input, "Inspect the repository and summarize the result.");
      return {
        finalResponse: "The repository is ready.",
        usage: {
          input_tokens: 31,
          cached_input_tokens: 4,
          output_tokens: 9,
          reasoning_output_tokens: 3,
        },
        items: [
          {
            id: "reasoning_1",
            type: "reasoning",
            text: "I need to inspect the files first.",
          },
          {
            id: "cmd_1",
            type: "command_execution",
            command: "ls",
            aggregated_output: "package.json\nsrc\n",
            exit_code: 0,
            status: "completed",
          },
          {
            id: "mcp_1",
            type: "mcp_tool_call",
            server: "docs",
            tool: "search",
            arguments: { query: "Codex SDK" },
            result: {
              content: [{ type: "text", text: "Codex SDK docs" }],
              structured_content: { ok: true },
            },
            status: "completed",
          },
          {
            id: "web_1",
            type: "web_search",
            query: "OpenAI Codex SDK",
          },
          {
            id: "file_1",
            type: "file_change",
            changes: [{ path: "README.md", kind: "update" }],
            status: "completed",
          },
          {
            id: "todo_1",
            type: "todo_list",
            items: [{ text: "Inspect repository", completed: true }],
          },
          {
            id: "msg_1",
            type: "agent_message",
            text: "The repository is ready.",
          },
        ],
      };
    }

    async runStreamed() {
      this._id = "thread_stream_123";
      return {
        events: (async function*() {
          yield { type: "thread.started", thread_id: "thread_stream_123" };
          yield {
            type: "item.started",
            item: {
              id: "cmd_stream",
              type: "command_execution",
              command: "pwd",
              aggregated_output: "",
              status: "in_progress",
            },
          };
          yield {
            type: "item.completed",
            item: {
              id: "cmd_stream",
              type: "command_execution",
              command: "pwd",
              aggregated_output: "/tmp/codex-demo\n",
              exit_code: 0,
              status: "completed",
            },
          };
          yield {
            type: "item.completed",
            item: {
              id: "msg_stream",
              type: "agent_message",
              text: "Streaming turn complete.",
            },
          };
          yield {
            type: "turn.completed",
            usage: {
              input_tokens: 12,
              cached_input_tokens: 0,
              output_tokens: 4,
              reasoning_output_tokens: 1,
            },
          };
        })(),
      };
    }
  }

  return { Thread: FakeThread };
}

function spanByLogType(logType) {
  return captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === logType,
  );
}

function spansByLogType(logType) {
  return captureState.spans.filter(
    (span) => span.attributes["respan.entity.log_type"] === logType,
  );
}

function assertNoBannedAliases(span) {
  assert.equal(span.attributes["traceloop.span.kind"], undefined);
  assert.equal(span.attributes["respan.span.tools"], undefined);
  assert.equal(span.attributes["respan.span.tool_calls"], undefined);
  assert.equal(span.attributes.tools, undefined);
  assert.equal(span.attributes.tool_calls, undefined);
  assert.equal(span.attributes.model, undefined);
  assert.equal(span.attributes.prompt_tokens, undefined);
  assert.equal(span.attributes.completion_tokens, undefined);
  assert.equal(span.attributes.total_request_tokens, undefined);
}

test("patches Thread.run and emits canonical Codex turn spans", async () => {
  captureState.spans = [];
  const sdk = createFakeSDK();
  const originalRun = sdk.Thread.prototype.run;
  const instrumentor = new CodexSDKInstrumentor({
    sdkModule: sdk,
    workflowName: "codex-sdk-run-test",
    agentName: "codex-test-agent",
  });
  await instrumentor.activate();

  const thread = new sdk.Thread();
  const result = await thread.run("Inspect the repository and summarize the result.");

  assert.equal(result.finalResponse, "The repository is ready.");
  assert.notEqual(sdk.Thread.prototype.run, originalRun);
  assert.equal(captureState.spans.length, 9);

  const workflowSpan = spanByLogType("workflow");
  const agentSpan = spanByLogType("agent");
  const chatSpan = spanByLogType("chat");
  const toolSpans = spansByLogType("tool");
  const taskSpans = spansByLogType("task");

  assert.ok(workflowSpan);
  assert.ok(agentSpan);
  assert.ok(chatSpan);
  assert.equal(toolSpans.length, 3);
  assert.equal(taskSpans.length, 3);
  assert.equal(workflowSpan.attributes["traceloop.workflow.name"], "codex-sdk-run-test");
  assert.equal(agentSpan.attributes["traceloop.entity.name"], "codex-test-agent");
  assert.equal(agentSpan.attributes["respan.metadata.agent_name"], "codex-test-agent");
  assert.equal(chatSpan.attributes["traceloop.entity.name"], "codex.response");
  assert.equal(chatSpan.attributes["gen_ai.system"], "openai");
  assert.equal(chatSpan.attributes["llm.request.type"], "chat");
  assert.equal(chatSpan.attributes["gen_ai.request.model"], "gpt-5.4");
  assert.equal(chatSpan.attributes["gen_ai.usage.prompt_tokens"], 31);
  assert.equal(chatSpan.attributes["gen_ai.usage.completion_tokens"], 9);
  assert.equal(chatSpan.attributes["llm.usage.total_tokens"], 40);
  assert.equal(chatSpan.attributes["gen_ai.prompt.0.role"], "user");
  assert.equal(
    chatSpan.attributes["gen_ai.prompt.0.content"],
    "Inspect the repository and summarize the result.",
  );
  assert.equal(chatSpan.attributes["gen_ai.completion.0.role"], "assistant");
  assert.equal(chatSpan.attributes["gen_ai.completion.0.content"], "The repository is ready.");
  assert.deepEqual(JSON.parse(chatSpan.attributes["gen_ai.completion.0.tool_calls"]), [
    {
      id: "cmd_1",
      type: "function",
      function: {
        name: "command_execution",
        arguments: "{\"command\":\"ls\"}",
      },
    },
    {
      id: "mcp_1",
      type: "function",
      function: {
        name: "mcp__docs__search",
        arguments: "{\"query\":\"Codex SDK\"}",
      },
    },
    {
      id: "web_1",
      type: "function",
      function: {
        name: "web_search",
        arguments: "{\"query\":\"OpenAI Codex SDK\"}",
      },
    },
  ]);

  const commandSpan = toolSpans.find((span) => span.name === "codex.command");
  assert.ok(commandSpan);
  assert.deepEqual(JSON.parse(commandSpan.attributes["traceloop.entity.input"]), {
    command: "ls",
  });
  assert.deepEqual(JSON.parse(commandSpan.attributes["traceloop.entity.output"]), {
    aggregated_output: "package.json\nsrc\n",
    exit_code: 0,
    status: "completed",
  });

  for (const span of captureState.spans) {
    assertNoBannedAliases(span);
  }

  instrumentor.deactivate();
  assert.equal(sdk.Thread.prototype.run, originalRun);
});

test("wraps Thread.runStreamed and emits spans after stream consumption", async () => {
  captureState.spans = [];
  const sdk = createFakeSDK();
  const instrumentor = new CodexSDKInstrumentor({ sdkModule: sdk });
  await instrumentor.activate();

  const thread = new sdk.Thread();
  const streamed = await thread.runStreamed("Stream a concise result.");
  const eventTypes = [];
  for await (const event of streamed.events) {
    eventTypes.push(event.type);
  }

  assert.deepEqual(eventTypes, [
    "thread.started",
    "item.started",
    "item.completed",
    "item.completed",
    "turn.completed",
  ]);
  assert.equal(captureState.spans.length, 4);

  const chatSpan = spanByLogType("chat");
  const commandSpan = captureState.spans.find((span) => span.name === "codex.command");
  assert.ok(chatSpan);
  assert.ok(commandSpan);
  assert.equal(chatSpan.attributes["gen_ai.completion.0.content"], "Streaming turn complete.");
  assert.equal(chatSpan.attributes["gen_ai.usage.prompt_tokens"], 12);
  assert.equal(chatSpan.attributes["gen_ai.usage.completion_tokens"], 4);
  assert.equal(commandSpan.attributes["respan.threads.thread_identifier"], "thread_stream_123");

  for (const span of captureState.spans) {
    assertNoBannedAliases(span);
  }

  instrumentor.deactivate();
});


test("failed turns add backend status attrs", async () => {
  captureState.spans = [];
  const sdk = createFakeSDK();
  sdk.Thread.prototype.run = async function runFailed() {
    throw new Error("Codex unavailable");
  };
  const instrumentor = new CodexSDKInstrumentor({ sdkModule: sdk });
  await instrumentor.activate();

  const thread = new sdk.Thread();
  await assert.rejects(() => thread.run("fail"), /Codex unavailable/);

  const chatSpan = spanByLogType("chat");
  const agentSpan = spanByLogType("agent");
  assert.ok(chatSpan);
  assert.ok(agentSpan);
  assert.equal(chatSpan.attributes.status_code, 500);
  assert.equal(chatSpan.attributes["error.message"], "Codex unavailable");
  assert.equal(agentSpan.attributes.status_code, 500);

  for (const span of captureState.spans) {
    assertNoBannedAliases(span);
  }

  instrumentor.deactivate();
});
