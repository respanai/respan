import assert from "node:assert/strict";
import test from "node:test";
import { tmpdir } from "node:os";
import { EventEmitter } from "node:events";
import { PassThrough, Writable } from "node:stream";
import * as ClaudeSDK from "@anthropic-ai/claude-agent-sdk";
import { trace } from "@opentelemetry/api";
import { ClaudeAgentSDKInstrumentor } from "../dist/index.js";

const spans = [];
const original = trace.getTracerProvider.bind(trace);
test.before(() => {
  trace.getTracerProvider = () => ({ activeSpanProcessor: { onEnd: span => spans.push(span) } });
});
test.after(() => { trace.getTracerProvider = original; });

// The real SDK owns initialization, hook callbacks, claim and Query controls.
// Only the CLI process is deterministic; no provider credentials are required.
function fixtureProcess({ toolFailure = false } = {}) {
  const process = new EventEmitter();
  process.stdout = new PassThrough();
  process.killed = false;
  process.exitCode = null;
  const requests = [];
  let hooks;
  let hookId = 0;
  const pendingHooks = new Map();
  const send = message => process.stdout.write(`${JSON.stringify(message)}\n`);
  const hook = async (event, input) => {
    for (const matcher of hooks?.[event] ?? []) {
      for (const callbackId of matcher.hookCallbackIds ?? []) {
        const requestId = `hook-${hookId++}`;
        const completed = new Promise(resolve => pendingHooks.set(requestId, resolve));
        send({ type: "control_request", request_id: requestId, request: {
          subtype: "hook_callback", callback_id: callbackId, tool_use_id: "prewarm-tool",
          input: { hook_event_name: event, session_id: "prewarm-session", ...input },
        } });
        await completed;
      }
    }
  };
  process.stdin = new Writable({ write(chunk, _encoding, done) {
    for (const line of chunk.toString().trim().split("\n")) {
      const message = JSON.parse(line);
      requests.push(message);
      if (message.type === "control_request") {
        if (message.request.subtype === "initialize") hooks = message.request.hooks;
        send({ type: "control_response", response: { subtype: "success", request_id: message.request_id,
          response: message.request.subtype === "initialize"
            ? { commands: [], agents: [], models: [], capabilities: {} }
            : { cwd: tmpdir(), session_id: "prewarm-session" },
        } });
      } else if (message.type === "control_response") {
        pendingHooks.get(message.response.request_id)?.();
      } else if (message.type === "user") {
        (async () => {
          await hook("UserPromptSubmit", { prompt: "claim fixture" });
          await hook("PreToolUse", { tool_name: "Read", tool_input: { file_path: "fixture.txt" } });
          await hook(toolFailure ? "PostToolUseFailure" : "PostToolUse", {
            tool_name: "Read", tool_input: { file_path: "fixture.txt" },
            ...(toolFailure ? { error: "fixture rejected" } : { tool_response: "fixture contents" }),
          });
          send({ type: "assistant", session_id: "prewarm-session", message: {
            id: "prewarm-message", model: "claude-sonnet-4-6", role: "assistant",
            content: [{ type: "text", text: "claim fixture completed" }],
            usage: { input_tokens: 4, output_tokens: 3 },
          } });
          send({ type: "result", subtype: "success", session_id: "prewarm-session",
            result: "claim fixture completed", duration_ms: 1, duration_api_ms: 1,
            num_turns: 1, is_error: false, usage: { input_tokens: 4, output_tokens: 3 },
          });
        })().catch(error => process.emit("error", error));
      }
    }
    done();
  } });
  process.kill = () => {
    if (process.killed) return true;
    process.killed = true;
    process.exitCode = 0;
    process.stdout.end();
    process.emit("exit", 0, null);
    return true;
  };
  return { process, requests };
}

for (const toolFailure of [false, true]) {
  test(`native prewarm captures claimed session and ${toolFailure ? "failed" : "successful"} tool`, {
    skip: typeof ClaudeSDK.prewarm !== "function",
  }, async () => {
    spans.length = 0;
    const sdk = { ...ClaudeSDK };
    const instrumentor = new ClaudeAgentSDKInstrumentor({ sdkModule: sdk, agentName: "prewarm-agent" });
    const fixture = fixtureProcess({ toolFailure });
    let userHookCalls = 0;
    await instrumentor.activate();
    const spare = await sdk.prewarm({ initializeTimeoutMs: 1000, options: {
      cwd: tmpdir(), tools: [], permissionMode: "default", settingSources: [],
      spawnClaudeCodeProcess: () => fixture.process,
      hooks: { PreToolUse: [{ hooks: [async () => { userHookCalls++; return {}; }] }] },
    } });
    assert.equal(spans.length, 0, "parked processes must not create agent spans");
    const query = spare.claim({ prompt: "claim fixture", options: { cwd: tmpdir(), model: "claude-sonnet-4-6" } });
    assert.equal(query.then, undefined, "claim remains synchronous");
    assert.equal(query[Symbol.asyncIterator](), query);
    assert.equal((await spare.claimed).sessionId, "prewarm-session");
    await query.setModel("claude-sonnet-4-6");
    assert.throws(() => spare.claim({ prompt: "again", options: { cwd: tmpdir() } }), /once|already/i);
    try {
      for await (const message of query) if (message.type === "result") break;
    } finally {
      spare.close();
      instrumentor.deactivate();
    }
    assert.equal(userHookCalls, 1);
    assert.equal(spans.length, 3);
    const byType = type => spans.find(span => span.attributes["respan.entity.log_type"] === type);
    const agent = byType("agent"), chat = byType("chat"), tool = byType("tool");
    assert.ok(agent && chat && tool);
    assert.equal(chat.parentSpanContext.spanId, agent.spanContext().spanId);
    assert.equal(tool.parentSpanContext.spanId, agent.spanContext().spanId);
    assert.equal(chat.attributes["gen_ai.request.model"], "claude-sonnet-4-6");
    assert.equal(chat.attributes["gen_ai.usage.input_tokens"], 4);
    assert.equal(chat.attributes["gen_ai.usage.output_tokens"], 3);
    assert.match(tool.attributes["traceloop.entity.output"], toolFailure ? /fixture rejected/ : /fixture contents/);
    assert.equal(tool.status.code, toolFailure ? 2 : 1);
    assert.equal(sdk.prewarm, ClaudeSDK.prewarm);
    const claim = fixture.requests.find(message => message.request?.subtype === "claim_session");
    assert.equal(claim.request.hooks, undefined);
  });
}

test("unclaimed spare close and disposal emit no spans; rejected claim can retry", async () => {
  spans.length = 0;
  let attempts = 0, closed = 0;
  class Spare {
    #value = 7;
    get claimed() { return this.#value; }
    claim() {
      if (++attempts === 1) throw new Error("invalid settings; spare remains parked");
      return (async function* () { yield { type: "result", subtype: "success", result: "ok" }; })();
    }
    close() { closed += this.#value; }
    async [Symbol.asyncDispose]() { this.close(); }
  }
  const sdk = { query() {}, async prewarm() { return new Spare(); } };
  const instrumentor = new ClaudeAgentSDKInstrumentor({ sdkModule: sdk });
  await instrumentor.activate();
  const unused = await sdk.prewarm();
  unused.close();
  await unused[Symbol.asyncDispose]();
  assert.equal(spans.length, 0);
  const spare = await sdk.prewarm();
  assert.equal(spare.claimed, 7);
  assert.throws(() => spare.claim({ prompt: "bad", options: {} }), /invalid settings/);
  assert.equal(spans.length, 0);
  const query = spare.claim({ prompt: "retry", options: {} });
  instrumentor.deactivate();
  await query.next();
  await spare[Symbol.asyncDispose]();
  spare.close();
  assert.equal(spans.length, 2);
  assert.equal(closed, 28);
});

test("a rejected claim is recorded when a host closes without consuming its result", async () => {
  spans.length = 0;
  let rejectClaim;
  const claimed = new Promise((_, reject) => { rejectClaim = reject; });
  const sdk = { query() {}, async prewarm() { return {
    claimed,
    claim() { return (async function* () { yield { type: "result", subtype: "error_during_execution", is_error: true }; })(); },
    close() {},
  }; } };
  const instrumentor = new ClaudeAgentSDKInstrumentor({ sdkModule: sdk });
  await instrumentor.activate();
  const spare = await sdk.prewarm();
  spare.claim({ prompt: "refused claim", options: {} });
  assert.equal(spare.claimed, claimed);
  rejectClaim(new Error("not_claimed: cwd_not_found"));
  await assert.rejects(spare.claimed, /cwd_not_found/);
  spare.close();
  instrumentor.deactivate();
  assert.equal(spans.length, 2);
  assert.ok(spans.every(span => span.status.code === 2));
});
