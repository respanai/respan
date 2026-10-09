import { AsyncLocalStorage } from "node:async_hooks";
import { createRequire } from "node:module";
import { types } from "node:util";
import { pathToFileURL } from "node:url";
import {
  context,
  createContextKey,
  SpanStatusCode,
  trace,
  type Span,
  type Context,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import {
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";

type RecordValue = Record<string, any>;
type Fn = (...args: any[]) => any;
type Gate = {
  allowed: boolean;
  content: boolean;
  traceId?: string;
  parentId?: string;
  parent?: object;
  ctx?: Context;
};
type Operation = {
  gate: Gate;
  name: string;
  request?: RecordValue;
  response?: RecordValue;
  chunks: RecordValue[];
  started: Date;
  ended: boolean;
  defer?: boolean;
  cancelled?: boolean;
  httpStatus?: number;
  requestModel?: string;
  nativeMetadata?: RecordValue;
  span?: Span;
};
type Patch = () => void;
const require = createRequire(import.meta.url);
const { version } = require("../package.json") as { version: string };
const scope = { name: "@respan/instrumentation-openrouter", version };
// Compatibility with instrumentations using this private language-model suppression key.
const localLanguageModelSuppression = createContextKey(
  "suppress_language_model_instrumentation",
);
const activeOperation = new AsyncLocalStorage<{
  gate: Gate;
  operation?: Operation;
}>();
const requests = new WeakMap<object, Operation>();
const modelResults = new WeakMap<object, Gate>();
const ancestorVetoes = new WeakSet<object>();
const owners = new Set<OpenRouterInstrumentor>();
const pendingOwners = new Set<OpenRouterInstrumentor>();
const patches: Patch[] = [];
const settlementPatches: Patch[] = [];
const pendingOperations = new Set<Operation>();
let settlementInstalled = false;
let installing: Promise<void> | undefined;
let patchEpoch = 0;

export interface OpenRouterInstrumentationOptions {
  traceContent?: boolean;
}

export class OpenRouterInstrumentor {
  readonly name = "openrouter";
  readonly traceContent: boolean;
  private enabled = false;
  private epoch = 0;
  private activation?: Promise<void>;
  constructor(options: OpenRouterInstrumentationOptions = {}) {
    this.traceContent = options.traceContent !== false;
  }
  isActive(): boolean {
    return this.enabled;
  }
  enable(): void {
    void this.activate().catch(() => undefined);
  }
  activate(): Promise<void> {
    if (this.enabled) return Promise.resolve();
    if (this.activation) return this.activation;
    const epoch = ++this.epoch;
    pendingOwners.add(this);
    const pending = (async () => {
      if (!patches.length && !installing)
        installing = install().finally(() => {
          installing = undefined;
        });
      if (installing) await installing;
      if (this.epoch !== epoch) return;
      if (patches.length) {
        owners.add(this);
        this.enabled = true;
      }
    })();
    this.activation = pending;
    void pending
      .finally(() => {
        pendingOwners.delete(this);
        if (this.activation === pending) this.activation = undefined;
        if (!owners.size && !pendingOwners.size && !installing) restore();
      })
      .catch(() => undefined);
    return pending;
  }
  disable(): void {
    ++this.epoch;
    this.enabled = false;
    owners.delete(this);
    if (!owners.size && !pendingOwners.size && !installing) restore();
  }
  async deactivate(): Promise<void> {
    this.disable();
    if (this.activation) await this.activation.catch(() => undefined);
  }
}
export function instrumentOpenRouter(
  options?: OpenRouterInstrumentationOptions,
): OpenRouterInstrumentor {
  const instrumentor = new OpenRouterInstrumentor(options);
  instrumentor.enable();
  return instrumentor;
}
export default OpenRouterInstrumentor;

async function sdkModule(path: string): Promise<RecordValue> {
  const host = createRequire(`${process.cwd()}/package.json`);
  let resolved: string;
  try {
    resolved = host.resolve(`@openrouter/sdk/${path}`);
  } catch {
    resolved = require.resolve(`@openrouter/sdk/${path}`);
  }
  return import(pathToFileURL(resolved).href);
}
function patch(
  proto: RecordValue | undefined,
  key: string,
  make: (original: Fn) => Fn,
  settlement = false,
): void {
  if (!proto || typeof proto[key] !== "function") return;
  const descriptor = Object.getOwnPropertyDescriptor(proto, key);
  if (!descriptor || typeof descriptor.value !== "function") return;
  const original = descriptor.value;
  const token = patchEpoch;
  const implementation = make(original);
  const wrapped = function (this: any) {
    return settlement || token === patchEpoch
      ? implementation.apply(this, arguments as any)
      : original.apply(this, arguments as any);
  };
  Object.defineProperty(proto, key, { ...descriptor, value: wrapped });
  (settlement ? settlementPatches : patches).push(() => {
    if (proto[key] === wrapped) Object.defineProperty(proto, key, descriptor);
  });
}
function restore(): void {
  ++patchEpoch;
  for (const undo of patches.splice(0).reverse()) undo();
  restoreSettlementIfIdle();
}
function restoreSettlementIfIdle(): void {
  if (owners.size || pendingOperations.size) return;
  for (const undo of settlementPatches.splice(0).reverse()) undo();
  settlementInstalled = false;
}
async function install(): Promise<void> {
  ++patchEpoch;
  try {
    const { ClientSDK } = await sdkModule("lib/sdks.js");
    patch(
      ClientSDK?.prototype,
      "_createRequest",
      (original) =>
        function (this: any, nativeContext: any, conf: any) {
          const result = original.apply(this, arguments as any);
          if (!owners.size) return result;
          const kind = operationName(nativeContext?.operationID);
          if (!kind || !result?.ok || !result.value) return result;
          telemetry(() => {
            const inherited = activeOperation.getStore();
            const gate = inherited?.gate ?? captureGate();
            refreshGate(gate, true);
            if (!gate.allowed) return;
            const op: Operation =
              inherited?.operation ?? startOperation(gate, kind);
            if (!op.span?.isRecording()) return;
            // The SDK has already validated and encoded this JSON. Telemetry never visits customer getters.
            if (
              gate.content &&
              op.span?.isRecording() &&
              typeof conf?.body === "string"
            )
              op.request = JSON.parse(conf.body);
            else op.request = {};
            requests.set(result.value, op);
          });
          return result;
        },
    );
    patch(
      ClientSDK?.prototype,
      "_do",
      (original) =>
        function (this: any, request: any) {
          const op = requests.get(request);
          const result = original.apply(this, arguments as any);
          if (op)
            observe(
              result,
              (native) => {
                if (!native?.ok) {
                  finish(op, native?.error);
                  return;
                }
                observeResponse(native.value, op);
              },
              (error) => finish(op, error),
            );
          return result;
        },
    );
    for (const [file, cls, method, name] of [
      ["sdk/chat.js", "Chat", "send", "chat.send"],
      ["sdk/embeddings.js", "Embeddings", "generate", "embeddings.generate"],
      ["sdk/responses.js", "Responses", "send", "responses.send"],
      ["sdk/betaresponses.js", "BetaResponses", "send", "responses.send"],
    ]) {
      let mod: RecordValue;
      try {
        mod = await sdkModule(file);
      } catch {
        continue;
      }
      patch(
        mod[cls]?.prototype,
        method,
        (original) =>
          function (this: any) {
            if (!owners.size) return original.apply(this, arguments as any);
            const gate = captureGate();
            if (!gate.allowed) return original.apply(this, arguments as any);
            const body = dataProperty(
              arguments[0],
              name.startsWith("chat")
                ? "chatRequest"
                : name.startsWith("embeddings")
                  ? "requestBody"
                  : "responsesRequest",
            );
            const nativeModel = dataProperty(body ?? arguments[0], "model");
            const op = startOperation(gate, name);
            op.requestModel =
              typeof nativeModel === "string" ? nativeModel : undefined;
            let result: any;
            try {
              result = activeOperation.run({ gate, operation: op }, () =>
                original.apply(this, arguments as any),
              );
            } catch (error) {
              finish(op, error);
              throw error;
            }
            observe(
              result,
              (value) => {
                if (!(value instanceof ReadableStream)) finish(op);
              },
              (error) => finish(op, error),
            );
            return result;
          },
      );
    }
    const asyncModule = await sdkModule("types/async.js");
    const inspect = asyncModule.APIPromise?.prototype.$inspect;
    const observed = new WeakSet<object>();
    const observeAPI = (api: object, promise: any) => {
      if (observed.has(api)) return;
      observed.add(api);
      observe(promise, (tuple) => {
        const [result, meta] = tuple;
        const op = meta?.request && requests.get(meta.request);
        if (op) {
          collectMetadata(op, result.value);
          if (!result.ok || !(result.value instanceof ReadableStream))
            finish(op, result.ok ? undefined : result.error);
        }
      });
    };
    if (typeof inspect === "function" && !settlementInstalled) {
      settlementInstalled = true;
      patch(
        asyncModule.APIPromise.prototype,
        "then",
        (original) =>
          function (this: any) {
            telemetry(() => observeAPI(this, inspect.call(this)));
            return original.apply(this, arguments as any);
          },
        true,
      );
      patch(
        asyncModule.APIPromise.prototype,
        "$inspect",
        (original) =>
          function (this: any) {
            const result = original.apply(this, arguments as any);
            telemetry(() => observeAPI(this, result));
            return result;
          },
        true,
      );
    }
    const sdk = await sdkModule("sdk/sdk.js");
    patch(
      sdk.OpenRouter?.prototype,
      "callModel",
      (original) =>
        function (this: any) {
          const gate = captureGate();
          const result = original.apply(this, arguments as any);
          if (result && typeof result === "object")
            modelResults.set(result, gate);
          return result;
        },
    );
    let model: RecordValue;
    try {
      model = await sdkModule("lib/model-result.js");
    } catch {
      return;
    }
    // Retain creation-time privacy/parentage across the native lazy execution and continuation paths.
    for (const method of [
      "initStream",
      "makeFollowupRequest",
      "continueWithUnsentResults",
      "processApprovalDecisions",
    ]) {
      patch(
        model.ModelResult?.prototype,
        method,
        (original) =>
          function (this: any) {
            const gate = modelResults.get(this);
            return gate
              ? activeOperation.run({ gate }, () =>
                  original.apply(this, arguments as any),
                )
              : original.apply(this, arguments as any);
          },
      );
    }
  } catch (error) {
    restore();
    throw error;
  }
}
function operationName(operation: unknown): string | undefined {
  if (operation === "sendChatCompletionRequest") return "chat.send";
  if (operation === "createEmbeddings") return "embeddings.generate";
  if (operation === "createResponses") return "responses.send";
  return undefined;
}
function captureGate(): Gate {
  try {
    const ctx = context.active();
    const parent = trace.getSpan(ctx);
    if (parent && types.isProxy(parent))
      return { allowed: false, content: false };
    const spanContext = parent?.spanContext();
    const key = ctx.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT);
    if (parent && key === false) ancestorVetoes.add(parent);
    const gate: Gate = {
      allowed:
        owners.size > 0 &&
        !isTracingSuppressed(ctx) &&
        !ctx.getValue(localLanguageModelSuppression) &&
        (!spanContext || (spanContext.traceFlags & 1) !== 0),
      content:
        key !== false &&
        !(parent && ancestorVetoes.has(parent)) &&
        [...owners].every((owner) => owner.traceContent) &&
        !denied(process.env.RESPAN_TRACE_CONTENT) &&
        !denied(process.env.TRACELOOP_TRACE_CONTENT),
      traceId: spanContext?.traceId,
      parentId: spanContext?.spanId,
      parent,
      ctx,
    };
    refreshGate(gate, false);
    return gate;
  } catch {
    return { allowed: false, content: false };
  }
}
function startOperation(gate: Gate, name: string): Operation {
  const span = gate.allowed
    ? trace
        .getTracer(scope.name, scope.version)
        .startSpan(`openrouter.${name.split(".")[0]}`, {}, gate.ctx)
    : undefined;
  const operation = {
    gate,
    name,
    span,
    started: new Date(),
    chunks: [],
    ended: false,
  };
  if (span?.isRecording()) pendingOperations.add(operation);
  return operation;
}
function refreshGate(gate: Gate, evaluateContext: boolean): void {
  if (
    denied(process.env.RESPAN_TRACE_CONTENT) ||
    denied(process.env.TRACELOOP_TRACE_CONTENT)
  )
    gate.content = false;
  if (evaluateContext) {
    const ctx = context.active();
    if (ctx.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT) === false)
      gate.content = false;
    const currentParent = trace.getSpan(ctx);
    if (
      isTracingSuppressed(ctx) ||
      ctx.getValue(localLanguageModelSuppression) === true ||
      (currentParent && (currentParent.spanContext().traceFlags & 1) === 0)
    ) {
      gate.allowed = false;
      gate.content = false;
    }
  }
  if (gate.parent) {
    const attributes = dataProperty(gate.parent, "attributes");
    if (denied(dataProperty(attributes, "allow_trace_content")))
      ancestorVetoes.add(gate.parent);
    if (ancestorVetoes.has(gate.parent)) gate.content = false;
    if (!attributes && !(gate.parent as any).spanContext().isRemote)
      gate.content = false;
  }
}
function collectMetadata(op: Operation, native: unknown): void {
  const metadata: RecordValue = {};
  for (const key of ["model", "id"]) {
    const value = dataProperty(native, key);
    if (typeof value === "string") metadata[key] = value;
  }
  const usage = dataProperty(native, "usage");
  if (usage) {
    const safe: RecordValue = {};
    for (const [out, candidates] of Object.entries({
      input_tokens: ["inputTokens", "promptTokens"],
      output_tokens: ["outputTokens", "completionTokens"],
      total_tokens: ["totalTokens"],
    })) {
      for (const key of candidates) {
        const value = dataProperty(usage, key);
        if (typeof value === "number") {
          safe[out] = value;
          break;
        }
      }
    }
    metadata.usage = safe;
  }
  op.nativeMetadata = metadata;
}
function denied(value: unknown): boolean {
  return value === false || value === "false" || value === "0";
}
function telemetry(fn: () => void): void {
  try {
    fn();
  } catch {
    /* Telemetry must not change native outcomes. */
  }
}
function observe(
  promise: any,
  success: (value: any) => void,
  failure?: (error: any) => void,
): void {
  telemetry(() => {
    Promise.prototype.then.call(
      promise,
      (value: any) => {
        telemetry(() => success(value));
      },
      (error: any) => {
        if (failure) telemetry(() => failure(error));
      },
    );
  });
}
function observeResponse(response: Response, op: Operation): void {
  telemetry(() => {
    op.httpStatus = response.status;
    const text = response.text;
    Object.defineProperty(response, "text", {
      configurable: true,
      writable: true,
      value: function (this: Response) {
        const promise = text.call(this);
        observe(
          promise,
          (body) => {
            telemetry(() => {
              if (op.gate.content) op.response = JSON.parse(body);
            });
          },
          (error) => finish(op, error),
        );
        return promise;
      },
    });
    if (
      !response.body ||
      !response.headers.get("content-type")?.includes("text/event-stream")
    )
      return;
    const body = response.body;
    const getReader = body.getReader;
    const decoder = new TextDecoder();
    let buffer = "";
    Object.defineProperty(body, "getReader", {
      configurable: true,
      writable: true,
      value: function (this: ReadableStream) {
        const reader: any = getReader.apply(this, arguments as any);
        const read = reader.read;
        reader.read = function () {
          const promise = read.apply(this, arguments as any);
          observe(
            promise,
            (item) => {
              if (item.done) {
                if (buffer) consumeEvent(buffer, op);
                finish(op);
              } else if (op.gate.content && item.value instanceof Uint8Array) {
                buffer += decoder.decode(item.value, { stream: true });
                const events = buffer.split(/\r?\n\r?\n/);
                buffer = events.pop() ?? "";
                for (const event of events) consumeEvent(event, op);
              }
            },
            (error) => finish(op, error),
          );
          return promise;
        };
        const cancel = reader.cancel;
        reader.cancel = function (reason?: unknown) {
          const promise = cancel.call(this, reason);
          observe(promise, () => {
            op.cancelled = reason !== "done";
            finish(op, reason instanceof Error ? reason : undefined);
          });
          return promise;
        };
        return reader;
      },
    });
  });
}
function consumeEvent(event: string, op: Operation): void {
  if (!op.gate.content) return;
  const data = event
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n");
  if (!data || data === "[DONE]") return;
  telemetry(() => {
    const value = JSON.parse(data);
    op.chunks.push(value);
    if (value.response) op.response = value.response;
    else if (value.choices || value.usage || value.model || value.id)
      op.response = { ...op.response, ...value };
    if (value.type === "response.failed" || value.error)
      finish(op, value.response?.error ?? value.error);
  });
}
function finish(op: Operation, error?: unknown): void {
  if (op.ended || !op.span?.isRecording()) return;
  op.ended = true;
  refreshGate(op.gate, false);
  telemetry(() => {
    const attrs: RecordValue = {
      [SpanAttributes.TRACELOOP_ENTITY_NAME]: `openrouter.${op.name.split(".")[0]}`,
      [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
      [SpanAttributes.LLM_REQUEST_TYPE]: op.name.startsWith("embeddings")
        ? "embedding"
        : "chat",
      [RespanSpanAttributes.RESPAN_LOG_TYPE]: op.name.startsWith("embeddings")
        ? RespanLogType.EMBEDDING
        : RespanLogType.CHAT,
      [RespanSpanAttributes.RESPAN_LOG_METHOD]: "ts_tracing",
      [ATTR_GEN_AI_SYSTEM]: "openrouter",
      "respan.metadata.openrouter_operation": op.name,
    };
    set(attrs, "http.response.status_code", op.httpStatus);
    if (op.cancelled) attrs["respan.metadata.openrouter_cancelled"] = true;
    const request = op.request ?? {};
    const response = op.response ?? {};
    attachScalarMetadata(
      attrs,
      { model: op.requestModel ?? request.model },
      op.nativeMetadata ?? response,
    );
    if (op.gate.content) attachContent(attrs, op, request, response);
    const nativeError = error ?? response.error;
    const errorName =
      dataProperty(nativeError, "name") ?? dataProperty(nativeError, "code");
    const errorMessage = dataProperty(nativeError, "message");
    if (nativeError !== undefined && nativeError !== null) {
      attrs["error.type"] = typeof errorName === "string" ? errorName : "Error";
      if (op.gate.content && typeof errorMessage === "string")
        attrs["error.message"] = errorMessage;
    }
    const span = op.span!;
    canonicalMetadata(attrs);
    span.setAttributes(attrs);
    span.setStatus(
      nativeError !== undefined && nativeError !== null
        ? {
            code: SpanStatusCode.ERROR,
            ...(op.gate.content && typeof errorMessage === "string"
              ? { message: errorMessage }
              : {}),
          }
        : { code: SpanStatusCode.UNSET },
    );
    privacySpan(
      span as Span & { attributes: RecordValue; status: any },
      op.gate,
    );
    try {
      span.end();
    } finally {
      pendingOperations.delete(op);
      restoreSettlementIfIdle();
    }
  });
}
function canonicalMetadata(attrs: RecordValue): void {
  const metadata: RecordValue = {};
  for (const key of Object.keys(attrs))
    if (key.startsWith("respan.metadata.")) {
      metadata[key.slice("respan.metadata.".length)] = attrs[key];
      delete attrs[key];
    }
  if (Object.keys(metadata).length)
    attrs[RespanSpanAttributes.RESPAN_METADATA] = JSON.stringify(metadata);
}
function attachScalarMetadata(
  attrs: RecordValue,
  request: RecordValue,
  response: RecordValue,
): void {
  set(attrs, ATTR_GEN_AI_REQUEST_MODEL, request.model);
  set(attrs, ATTR_GEN_AI_RESPONSE_MODEL, response.model);
  set(attrs, ATTR_GEN_AI_RESPONSE_ID, response.id);
  const usage = response.usage;
  if (usage) {
    const input = usage.input_tokens ?? usage.prompt_tokens;
    const output = usage.output_tokens ?? usage.completion_tokens;
    set(attrs, ATTR_GEN_AI_USAGE_INPUT_TOKENS, input);
    set(attrs, ATTR_GEN_AI_USAGE_PROMPT_TOKENS, input);
    set(attrs, ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, output);
    set(attrs, ATTR_GEN_AI_USAGE_COMPLETION_TOKENS, output);
    set(attrs, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens);
  }
}
function attachContent(
  attrs: RecordValue,
  op: Operation,
  request: RecordValue,
  response: RecordValue,
): void {
  set(attrs, ATTR_GEN_AI_REQUEST_MODEL, request.model);
  set(attrs, ATTR_GEN_AI_RESPONSE_MODEL, response.model);
  set(attrs, ATTR_GEN_AI_RESPONSE_ID, response.id);
  set(attrs, "respan.metadata.openrouter_provider", response.provider);
  set(attrs, "respan.metadata.openrouter_status", response.status);
  set(attrs, "respan.metadata.stream", request.stream);
  const usage = response.usage;
  if (usage) {
    const input = usage.prompt_tokens ?? usage.input_tokens;
    const output = usage.completion_tokens ?? usage.output_tokens;
    set(attrs, ATTR_GEN_AI_USAGE_INPUT_TOKENS, input);
    set(attrs, ATTR_GEN_AI_USAGE_PROMPT_TOKENS, input);
    set(attrs, ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, output);
    set(attrs, ATTR_GEN_AI_USAGE_COMPLETION_TOKENS, output);
    set(attrs, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens);
    set(
      attrs,
      SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
      usage.prompt_tokens_details?.cached_tokens ??
        usage.input_tokens_details?.cached_tokens,
    );
    set(
      attrs,
      SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
      usage.completion_tokens_details?.reasoning_tokens ??
        usage.output_tokens_details?.reasoning_tokens,
    );
  }
  if (op.name.startsWith("embeddings")) {
    set(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, json(request.input));
    set(
      attrs,
      SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
      json(response.data?.map((item: any) => item.embedding)),
    );
    return;
  }
  const input = request.messages ?? request.input;
  set(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, json(input));
  const messages = Array.isArray(input)
    ? input
    : typeof input === "string"
      ? [{ role: "user", content: input }]
      : [];
  for (const [index, message] of messages.entries()) {
    const base = `${ATTR_GEN_AI_PROMPT}.${index}`;
    set(attrs, `${base}.role`, message.role);
    set(
      attrs,
      `${base}.content`,
      content(
        message.content ??
          (message.type === "function_call_output"
            ? message.output
            : undefined),
      ),
    );
    set(
      attrs,
      `${base}.tool_calls`,
      json(
        message.tool_calls ??
          (message.type === "function_call" ? [message] : undefined),
      ),
    );
  }
  set(attrs, SpanAttributes.LLM_REQUEST_FUNCTIONS, json(request.tools));
  if (op.name === "responses.send") {
    set(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json(response.output));
    const emittedCalls = response.output?.filter(
      (item: any) => item.type === "function_call",
    );
    const outputMessages =
      response.output?.filter((item: any) => item.type === "message") ?? [];
    outputMessages.forEach((message: any, index: number) => {
      const base = `${ATTR_GEN_AI_COMPLETION}.${index}`;
      set(attrs, `${base}.role`, message.role);
      set(attrs, `${base}.content`, content(message.content));
    });
    if (emittedCalls?.length)
      set(attrs, `${ATTR_GEN_AI_COMPLETION}.0.tool_calls`, json(emittedCalls));
  } else {
    const choices = aggregateChat(response, op.chunks);
    set(
      attrs,
      SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
      json(choices?.map((choice: any) => choice.message)),
    );
    choices?.forEach((choice: any, index: number) => {
      const base = `${ATTR_GEN_AI_COMPLETION}.${choice.index ?? index}`;
      set(attrs, `${base}.role`, choice.message?.role);
      set(attrs, `${base}.content`, content(choice.message?.content));
      set(attrs, `${base}.tool_calls`, json(choice.message?.tool_calls));
      set(
        attrs,
        `respan.metadata.openrouter_finish_reason.${choice.index ?? index}`,
        choice.finish_reason,
      );
    });
  }
}
function aggregateChat(
  response: RecordValue,
  chunks: RecordValue[],
): any[] | undefined {
  if (!chunks.length) return response.choices;
  if (!chunks.some((chunk) => Array.isArray(chunk.choices))) return undefined;
  const choices = new Map<number, any>();
  for (const chunk of chunks)
    for (const choice of chunk.choices ?? []) {
      const index = choice.index;
      let current = choices.get(index);
      if (!current) {
        current = { index, message: {} };
        choices.set(index, current);
      }
      if (choice.finish_reason !== undefined)
        current.finish_reason = choice.finish_reason;
      const delta = choice.delta ?? choice.message;
      if (!delta) continue;
      if (delta.role !== undefined) current.message.role = delta.role;
      if (delta.content !== undefined)
        current.message.content =
          (current.message.content ?? "") + delta.content;
      if (delta.tool_calls) {
        current.tools ??= new Map();
        for (const call of delta.tool_calls) {
          const prior = current.tools.get(call.index) ?? {};
          const fn = { ...prior.function, ...call.function };
          for (const key of ["name", "arguments"])
            if (call.function?.[key] !== undefined)
              fn[key] = (prior.function?.[key] ?? "") + call.function[key];
          current.tools.set(call.index, { ...prior, ...call, function: fn });
        }
        current.message.tool_calls = [...current.tools.values()];
      }
    }
  return [...choices.values()];
}
function dataProperty(value: any, key: string): any {
  if (
    !value ||
    (typeof value !== "object" && typeof value !== "function") ||
    types.isProxy(value)
  )
    return undefined;
  for (let object = value; object; object = Object.getPrototypeOf(object)) {
    const descriptor = Object.getOwnPropertyDescriptor(object, key);
    if (descriptor) return "value" in descriptor ? descriptor.value : undefined;
  }
}
function set(attrs: RecordValue, key: string, value: any): void {
  if (value !== undefined) attrs[key] = value === null ? "null" : value;
}
function json(value: any): string | undefined {
  return value === undefined ? undefined : JSON.stringify(value);
}
function content(value: any): string | undefined {
  return typeof value === "string" ? value : json(value);
}
function privacySpan<T extends { attributes: any; status: any }>(
  span: T,
  gate: Gate,
): T {
  const spanStatus = span.status.code;
  const allowed = new Set([
    SpanAttributes.TRACELOOP_ENTITY_NAME,
    SpanAttributes.TRACELOOP_ENTITY_PATH,
    SpanAttributes.LLM_REQUEST_TYPE,
    RespanSpanAttributes.RESPAN_LOG_TYPE,
    RespanSpanAttributes.RESPAN_LOG_METHOD,
    ATTR_GEN_AI_SYSTEM,
    "respan.metadata.openrouter_operation",
    "error.type",
    "http.response.status_code",
    "respan.metadata.openrouter_cancelled",
  ]);
  const safe = new Set([
    ...allowed,
    ATTR_GEN_AI_REQUEST_MODEL,
    ATTR_GEN_AI_RESPONSE_MODEL,
    ATTR_GEN_AI_RESPONSE_ID,
    ATTR_GEN_AI_USAGE_INPUT_TOKENS,
    ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
    ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
    ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
    SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
    SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
  ]);
  let state = span.attributes;
  const refresh = () => {
    refreshGate(gate, false);
    if (denied(dataProperty(state, "allow_trace_content")))
      gate.content = false;
    if (gate.content) return state;
    const values: RecordValue = {};
    for (const key of safe) {
      const value = dataProperty(state, key);
      if (["string", "number", "boolean"].includes(typeof value))
        values[key] = value;
    }
    return values;
  };
  const attrs = new Proxy(
    {},
    {
      get(_target, key) {
        return typeof key === "string"
          ? dataProperty(refresh(), key)
          : undefined;
      },
      ownKeys() {
        return Object.keys(refresh());
      },
      getOwnPropertyDescriptor(_target, key) {
        const values = refresh();
        return typeof key === "string" && Object.hasOwn(values, key)
          ? {
              value: values[key],
              enumerable: true,
              configurable: true,
              writable: true,
            }
          : undefined;
      },
      set(_target, key, value) {
        if (key === "allow_trace_content" && denied(value))
          gate.content = false;
        if (gate.content && typeof key === "string") state[key] = value;
        return true;
      },
      defineProperty(_target, key, descriptor) {
        if (key === "allow_trace_content" && denied(descriptor.value))
          gate.content = false;
        if (gate.content && "value" in descriptor && typeof key === "string")
          state[key] = descriptor.value;
        return true;
      },
      deleteProperty(_target, key) {
        if (gate.content && typeof key === "string") delete state[key];
        return true;
      },
      setPrototypeOf() {
        return false;
      },
    },
  );
  let events = (span as any).events;
  let status = span.status;
  Object.defineProperties(span, {
    attributes: {
      configurable: false,
      get: () => {
        refresh();
        return attrs;
      },
      set: (value) => {
        if (denied(dataProperty(value, "allow_trace_content")))
          gate.content = false;
        if (gate.content && value && !types.isProxy(value)) state = value;
      },
    },
    events: {
      configurable: false,
      get: () => {
        refresh();
        return gate.content ? events : [];
      },
      set: (value) => {
        refresh();
        if (gate.content) events = value;
      },
    },
    status: {
      configurable: false,
      get: () => {
        refresh();
        return gate.content ? status : { code: spanStatus };
      },
      set: (value) => {
        refresh();
        if (gate.content) status = value;
      },
    },
  });
  return span;
}
