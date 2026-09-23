import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";

import { context, ROOT_CONTEXT, trace } from "@opentelemetry/api";
import { PROPAGATED_ATTRIBUTES_KEY } from "@respan/tracing";

import * as publicApi from "../dist/index.js";
import { mergeDifyPropagatedMetadata } from "../dist/_otel_emitter.js";
import { buildDifySpanAttributes } from "../dist/_translator.js";

const { DifyInstrumentor } = publicApi;

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

function createFakeSdk({ streamError = false } = {}) {
  class HttpClient {
    constructor() {
      this.rawCalls = 0;
    }

    async request(options) {
      const response = await this.requestRaw(options);
      return {
        data: response.data,
        status: response.status,
        headers: response.headers,
        requestId: response.requestId,
      };
    }

    async requestStream(options) {
      await this.requestRaw({ ...options, responseType: "stream" });
      const data = new EventEmitter();
      const stream = {
        data,
        status: 200,
        headers: { "content-type": "text/event-stream" },
        requestId: "request-stream",
        toReadable() {
          return data;
        },
        async toText() {
          let text = "";
          for await (const event of stream) {
            text += event.data.answer ?? "";
          }
          return text;
        },
        async *[Symbol.asyncIterator]() {
          yield {
            event: "message",
            data: {
              event: "message",
              task_id: "task-stream",
              message_id: "message-stream",
              conversation_id: "conversation-stream",
              answer: "Hel",
            },
            raw: '{"event":"message","answer":"Hel"}',
          };
          if (streamError) throw new Error("stream failed");
          yield {
            event: "message_end",
            data: {
              event: "message_end",
              task_id: "task-stream",
              message_id: "message-stream",
              conversation_id: "conversation-stream",
              answer: "lo",
              metadata: {
                usage: {
                  model: "dify/test-model",
                  prompt_tokens: 2,
                  completion_tokens: 1,
                  total_tokens: 3,
                },
              },
            },
            raw: '{"event":"message_end","answer":"lo"}',
          };
          data.emit("end");
        },
      };
      return stream;
    }

    async requestBinaryStream(options) {
      await this.requestRaw({ ...options, responseType: "stream" });
      const data = new EventEmitter();
      return {
        data,
        status: 200,
        headers: { "content-type": "audio/mpeg" },
        toReadable() {
          return data;
        },
      };
    }

    async requestRaw(options) {
      this.rawCalls += 1;
      if (options.path === "/fail") {
        const error = new Error("Dify unavailable");
        error.statusCode = 503;
        throw error;
      }
      if (options.responseType === "stream") {
        return {
          data: new EventEmitter(),
          status: 200,
          headers: { "content-type": "text/event-stream" },
          requestId: "raw-stream",
          url: `https://api.dify.ai/v1${options.path}`,
        };
      }
      if (options.path.includes("/credentials/validate")) {
        return {
          data: {
            result: "success",
            provider: {
              credentials: { access_token: "js-response-access-token" },
            },
          },
          status: 200,
          headers: { "content-type": "application/json" },
          requestId: "request-credentials",
          url: `https://api.dify.ai/v1${options.path}`,
        };
      }
      return {
        data: {
          event: "message",
          task_id: "task-1",
          message_id: "message-1",
          conversation_id: "conversation-1",
          answer: "Hello from Dify.",
          metadata: {
            usage: {
              model: "dify/test-model",
              prompt_tokens: 4,
              completion_tokens: 3,
              total_tokens: 7,
            },
          },
        },
        status: 200,
        headers: { "content-type": "application/json" },
        requestId: "request-1",
        url: `https://api.dify.ai/v1${options.path}`,
      };
    }
  }
  return { HttpClient };
}

function createSynchronousContextManager() {
  let active = ROOT_CONTEXT;
  return {
    active: () => active,
    bind: (_context, target) => target,
    disable() {
      active = ROOT_CONTEXT;
      return this;
    },
    enable() {
      return this;
    },
    with(scopedContext, fn, thisArg, ...args) {
      const previous = active;
      active = scopedContext;
      try {
        return fn.call(thisArg, ...args);
      } finally {
        active = previous;
      }
    },
  };
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
  assert.equal(
    Object.keys(span.attributes).some((key) => key.startsWith("respan.metadata.")),
    false,
  );
}

test("public surface exports only the instrumentor and compatibility alias", () => {
  assert.deepEqual(
    Object.keys(publicApi).sort(),
    ["DifyAIInstrumentor", "DifyInstrumentor"],
  );
});

test("maps blocking chat responses to the canonical contract", async () => {
  captureState.spans = [];
  const sdk = createFakeSdk();
  const original = sdk.HttpClient.prototype.request;
  const instrumentor = new DifyInstrumentor({ sdkModule: sdk });
  await instrumentor.activate();
  try {
    const client = new sdk.HttpClient();
    const response = await client.request({
      method: "POST",
      path: "/chat-messages",
      data: {
        inputs: {},
        query: "Hello?",
        user: "user-123",
        conversation_id: "conversation-1",
        response_mode: "blocking",
      },
    });

    assert.equal(response.data.answer, "Hello from Dify.");
    assert.equal(client.rawCalls, 1);
    assert.equal(captureState.spans.length, 1);
    const span = captureState.spans[0];
    assert.equal(span.attributes["respan.entity.log_type"], "chat");
    assert.equal(span.attributes["respan.entity.log_method"], "ts_tracing");
    assert.equal(span.attributes["gen_ai.system"], "dify");
    assert.equal(span.attributes["gen_ai.request.model"], "dify/test-model");
    assert.equal(span.attributes["llm.request.type"], "chat");
    assert.equal(span.attributes["gen_ai.prompt.0.content"], "Hello?");
    assert.equal(span.attributes["gen_ai.completion.0.content"], "Hello from Dify.");
    assert.equal(span.attributes["gen_ai.usage.input_tokens"], 4);
    assert.equal(span.attributes["gen_ai.usage.output_tokens"], 3);
    assert.equal(span.attributes["llm.usage.total_tokens"], 7);
    assert.equal(span.attributes["respan.customer_params.customer_identifier"], "user-123");
    assert.equal(span.attributes["respan.threads.thread_identifier"], "conversation-1");
    const metadata = JSON.parse(span.attributes["respan.metadata"]);
    assert.equal(metadata["dify.endpoint"], "/chat-messages");
    assert.equal(metadata["dify.task_id"], "task-1");
    assertNoBannedAliases(span);
  } finally {
    instrumentor.deactivate();
  }
  assert.equal(sdk.HttpClient.prototype.request, original);
});

test("merges propagated run metadata into the single canonical attribute", () => {
  const attributes = buildDifySpanAttributes({
    request: { method: "GET", path: "/parameters" },
    response: { data: { opening_statement: "Hello" }, status: 200, headers: {} },
  });
  const merged = mergeDifyPropagatedMetadata(attributes, {
    metadata: {
      run_id: "run-123",
      example: "dify-loopback",
      "dify.endpoint": "must-not-replace-call-metadata",
    },
  });

  const metadata = JSON.parse(merged["respan.metadata"]);
  assert.equal(metadata["dify.endpoint"], "/parameters");
  assert.equal(metadata["dify.method"], "GET");
  assert.equal(metadata.run_id, "run-123");
  assert.equal(metadata.example, "dify-loopback");
  assert.equal(
    Object.keys(merged).some((key) => key.startsWith("respan.metadata.")),
    false,
  );
});

test("emits a streaming chat span only after AsyncIterable consumption", async () => {
  captureState.spans = [];
  const sdk = createFakeSdk();
  const instrumentor = new DifyInstrumentor({ sdkModule: sdk });
  await instrumentor.activate();
  try {
    const client = new sdk.HttpClient();
    const stream = await client.requestStream({
      method: "POST",
      path: "/chat-messages",
      data: { query: "Stream", user: "user-1", response_mode: "streaming" },
    });
    assert.equal(captureState.spans.length, 0);
    const text = await stream.toText();
    assert.equal(text, "Hello");
    assert.equal(captureState.spans.length, 1);
    const span = captureState.spans[0];
    assert.equal(span.attributes["gen_ai.completion.0.content"], "Hello");
    assert.equal(span.attributes["gen_ai.usage.input_tokens"], 2);
    assert.equal(span.attributes["gen_ai.usage.output_tokens"], 1);
    const metadata = JSON.parse(span.attributes["respan.metadata"]);
    assert.equal(metadata["dify.event"], "message_end");
    assert.equal(metadata["dify.task_id"], "task-stream");
    assert.equal(metadata["dify.message_id"], "message-stream");
    assert.equal(metadata["dify.conversation_id"], "conversation-stream");
    assertNoBannedAliases(span);
    assert.equal(client.rawCalls, 1);
  } finally {
    instrumentor.deactivate();
  }
});

test("deferred stream retains propagated metadata after its scope exits", async () => {
  captureState.spans = [];
  context.disable();
  context.setGlobalContextManager(createSynchronousContextManager());
  const sdk = createFakeSdk();
  const instrumentor = new DifyInstrumentor({ sdkModule: sdk });
  await instrumentor.activate();
  try {
    const propagatedContext = context.active().setValue(
      PROPAGATED_ATTRIBUTES_KEY,
      {
        customer_identifier: "customer-delayed-stream",
        metadata: { run_id: "run-delayed-stream", example: "outside-scope" },
      },
    );
    let stream;
    await context.with(propagatedContext, async () => {
      stream = await new sdk.HttpClient().requestStream({
        method: "POST",
        path: "/chat-messages",
        data: { query: "Deferred", response_mode: "streaming" },
      });
    });

    assert.equal(
      context.active().getValue(PROPAGATED_ATTRIBUTES_KEY),
      undefined,
    );
    await stream.toText();
    assert.equal(captureState.spans.length, 1);
    const span = captureState.spans[0];
    const metadata = JSON.parse(span.attributes["respan.metadata"]);
    assert.equal(metadata.run_id, "run-delayed-stream");
    assert.equal(metadata.example, "outside-scope");
    assert.equal(metadata["dify.endpoint"], "/chat-messages");
    assert.equal(
      span.attributes["respan.customer_params.customer_identifier"],
      "customer-delayed-stream",
    );
    assertNoBannedAliases(span);
  } finally {
    instrumentor.deactivate();
    context.disable();
  }
});

test("early stream return and stream errors each emit exactly once", async () => {
  captureState.spans = [];
  const sdk = createFakeSdk();
  const instrumentor = new DifyInstrumentor({ sdkModule: sdk });
  await instrumentor.activate();
  try {
    const client = new sdk.HttpClient();
    const stream = await client.requestStream({
      method: "POST",
      path: "/chat-messages",
      data: { query: "Partial", user: "user-1", response_mode: "streaming" },
    });
    for await (const _event of stream) break;
    assert.equal(captureState.spans.length, 1);
    assert.equal(captureState.spans[0].attributes["gen_ai.completion.0.content"], "Hel");
  } finally {
    instrumentor.deactivate();
  }

  captureState.spans = [];
  const failingSdk = createFakeSdk({ streamError: true });
  const failingInstrumentor = new DifyInstrumentor({ sdkModule: failingSdk });
  await failingInstrumentor.activate();
  try {
    const stream = await new failingSdk.HttpClient().requestStream({
      method: "POST",
      path: "/chat-messages",
      data: { query: "Fail", user: "user-1", response_mode: "streaming" },
    });
    await assert.rejects(async () => {
      for await (const _event of stream) {
        // consume
      }
    }, /stream failed/);
    assert.equal(captureState.spans.length, 1);
    assert.equal(captureState.spans[0].status.code, 2);
    assert.match(captureState.spans[0].status.message, /stream failed/);
  } finally {
    failingInstrumentor.deactivate();
  }
});

test("binary and direct raw stream lifecycles emit without duplicate request spans", async () => {
  captureState.spans = [];
  const sdk = createFakeSdk();
  const instrumentor = new DifyInstrumentor({ sdkModule: sdk });
  await instrumentor.activate();
  try {
    const client = new sdk.HttpClient();
    const binary = await client.requestBinaryStream({
      method: "POST",
      path: "/text-to-audio",
      data: { text: "hello", user: "user-1", streaming: true },
    });
    assert.equal(captureState.spans.length, 0);
    binary.data.emit("end");
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(captureState.spans.length, 1);

    const raw = await client.requestRaw({
      method: "POST",
      path: "/workflows/run",
      data: { inputs: {}, user: "user-1", response_mode: "streaming" },
      responseType: "stream",
    });
    assert.equal(captureState.spans.length, 1);
    raw.data.emit("close");
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(captureState.spans.length, 2);
    assert.equal(captureState.spans[1].attributes["respan.entity.log_type"], "workflow");
  } finally {
    instrumentor.deactivate();
  }
});

test("request failures preserve status and privacy mode omits content", async () => {
  captureState.spans = [];
  const sdk = createFakeSdk();
  const instrumentor = new DifyInstrumentor({ sdkModule: sdk, includeContent: false });
  await instrumentor.activate();
  try {
    await assert.rejects(
      () => new sdk.HttpClient().request({
        method: "POST",
        path: "/fail",
        data: { query: "private prompt", user: "user-private" },
      }),
      /Dify unavailable/,
    );
    assert.equal(captureState.spans.length, 1);
    const span = captureState.spans[0];
    assert.equal(span.status.code, 2);
    assert.equal(span.attributes.status_code, 503);
    assert.equal(span.attributes["error.message"], "Dify unavailable");
    assert.equal(span.attributes["traceloop.entity.input"], undefined);
    assert.equal(span.attributes["traceloop.entity.output"], undefined);
    assert.equal(span.attributes["gen_ai.prompt.0.content"], undefined);
    assert.equal(span.attributes["gen_ai.completion.0.content"], undefined);
    assert.equal(span.attributes["gen_ai.request.model"], undefined);
    assert.equal(span.attributes["gen_ai.usage.input_tokens"], undefined);
    assert.equal(span.attributes["gen_ai.usage.output_tokens"], undefined);
    assert.equal(JSON.stringify(span.attributes).includes("private prompt"), false);
  } finally {
    instrumentor.deactivate();
  }
});

test("generic Workspace credential validation recursively redacts secrets", async () => {
  captureState.spans = [];
  const sdk = createFakeSdk();
  const instrumentor = new DifyInstrumentor({ sdkModule: sdk });
  const credentials = {
    openai_api_key: "js-provider-key",
    apiKey: "js-camel-key",
    Authorization: "Bearer js-authorization",
    nested: {
      refresh_token: "js-refresh-token",
      password: "js-password",
      password_hash: "js-password-hash",
      aws_secret_access_key: "js-aws-secret-access-key",
      safe_region: "us-east-1",
    },
  };
  await instrumentor.activate();
  try {
    await new sdk.HttpClient().request({
      method: "POST",
      path: "/workspaces/current/model-providers/openai/credentials/validate",
      data: credentials,
    });

    assert.equal(captureState.spans.length, 1);
    const serializedAttributes = JSON.stringify(captureState.spans[0].attributes);
    for (const secret of [
      "js-provider-key",
      "js-camel-key",
      "js-authorization",
      "js-refresh-token",
      "js-password",
      "js-password-hash",
      "js-aws-secret-access-key",
      "js-response-access-token",
    ]) {
      assert.equal(serializedAttributes.includes(secret), false);
    }
    assert.equal(serializedAttributes.includes("[REDACTED]"), true);
    assert.equal(serializedAttributes.includes("us-east-1"), true);
    assert.equal(credentials.openai_api_key, "js-provider-key");
    assert.equal(credentials.nested.refresh_token, "js-refresh-token");
  } finally {
    instrumentor.deactivate();
  }
});

test("concurrent instrumentors preserve the first content policy", async () => {
  captureState.spans = [];
  const sdk = createFakeSdk();
  const first = new DifyInstrumentor({ sdkModule: sdk, includeContent: false });
  const conflicting = new DifyInstrumentor({ sdkModule: sdk, includeContent: true });
  await first.activate();
  await conflicting.activate();
  try {
    await new sdk.HttpClient().request({
      method: "POST",
      path: "/chat-messages",
      data: { query: "private prompt", user: "user-private" },
    });
    assert.equal(captureState.spans.at(-1).attributes["gen_ai.prompt.0.content"], undefined);

    conflicting.deactivate();
    await new sdk.HttpClient().request({
      method: "POST",
      path: "/chat-messages",
      data: { query: "still private", user: "user-private" },
    });
    assert.equal(captureState.spans.at(-1).attributes["gen_ai.prompt.0.content"], undefined);
  } finally {
    conflicting.deactivate();
    first.deactivate();
  }

  const replacement = new DifyInstrumentor({ sdkModule: sdk, includeContent: true });
  await replacement.activate();
  try {
    await new sdk.HttpClient().request({
      method: "POST",
      path: "/chat-messages",
      data: { query: "visible prompt", user: "user-visible" },
    });
    assert.equal(captureState.spans.at(-1).attributes["gen_ai.prompt.0.content"], "visible prompt");
  } finally {
    replacement.deactivate();
  }
});

test("deactivation preserves a later foreign patch across shared references", async () => {
  const sdk = createFakeSdk();
  const original = sdk.HttpClient.prototype.request;
  const first = new DifyInstrumentor({ sdkModule: sdk });
  const second = new DifyInstrumentor({ sdkModule: sdk });
  await first.activate();
  await second.activate();
  assert.notEqual(sdk.HttpClient.prototype.request, original);

  const foreignPatch = async function foreignPatch() {
    return { data: { answer: "foreign" }, status: 200, headers: {} };
  };
  sdk.HttpClient.prototype.request = foreignPatch;
  first.deactivate();
  assert.equal(sdk.HttpClient.prototype.request, foreignPatch);
  second.deactivate();

  assert.equal(sdk.HttpClient.prototype.request, foreignPatch);
  assert.equal(first.isActive(), false);
  assert.equal(second.isActive(), false);
});

test("translator maps workflow, completion, and RAG pipeline semantics", () => {
  const completion = buildDifySpanAttributes({
    request: {
      method: "POST",
      path: "/completion-messages",
      data: { inputs: { query: "Translate hello" }, user: "u" },
    },
    response: { data: { answer: "Hola" }, status: 200, headers: {} },
  });
  assert.equal(completion["respan.entity.log_type"], "text");
  assert.equal(completion["gen_ai.prompt.0.content"], "Translate hello");
  assert.equal(completion["gen_ai.completion.0.content"], "Hola");

  const workflow = buildDifySpanAttributes({
    request: {
      method: "POST",
      path: "/datasets/dataset-1/pipeline/run",
      data: { inputs: { source: "docs" }, response_mode: "blocking" },
    },
    response: {
      data: { data: { status: "succeeded", outputs: { documents: 2 } } },
      status: 200,
      headers: {},
    },
  });
  assert.equal(workflow["respan.entity.log_type"], "workflow");
  assert.equal(workflow["llm.request.type"], undefined);
  assertNoBannedAliases({ attributes: workflow });
});
