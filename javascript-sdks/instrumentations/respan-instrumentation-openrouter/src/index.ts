import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { context, trace } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { buildReadableSpan, injectSpan } from "@respan/tracing";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

const INSTRUMENTATION_NAME = "@respan/instrumentation-openrouter";
const INSTRUMENTATION_VERSION = "0.1.0";
const OPENROUTER_SYSTEM = "openrouter";
const TRACE_METHOD = "ts_tracing";
// OpenTelemetry/Traceloop JS packages used here do not export a cache-read token constant yet.
const CACHE_READ_INPUT_TOKENS = "llm.usage.cache_read_input_tokens";
const COMPLETION_ZERO = `${ATTR_GEN_AI_COMPLETION}.0`;

type AnyRecord = Record<string, any>;
type AnyFunction = (...args: any[]) => any;
type Patch = () => void;
type ParentContext = { traceId?: string; parentId?: string };

const instrumentationScope = {
  name: INSTRUMENTATION_NAME,
  version: INSTRUMENTATION_VERSION,
};
// Resolve Traceloop-owned keys from the installed package first, with canonical-name fallbacks for older runtime copies.
const TL = SpanAttributes as unknown as Record<string, string>;
const ATTR_TRACELOOP_ENTITY_NAME = TL.TRACELOOP_ENTITY_NAME ?? "traceloop.entity.name";
const ATTR_TRACELOOP_ENTITY_PATH = TL.TRACELOOP_ENTITY_PATH ?? "traceloop.entity.path";
const ATTR_TRACELOOP_ENTITY_INPUT = TL.TRACELOOP_ENTITY_INPUT ?? "traceloop.entity.input";
const ATTR_TRACELOOP_ENTITY_OUTPUT = TL.TRACELOOP_ENTITY_OUTPUT ?? "traceloop.entity.output";
const ATTR_LLM_REQUEST_TYPE = TL.LLM_REQUEST_TYPE ?? "llm.request.type";
const ATTR_LLM_REQUEST_FUNCTIONS = TL.LLM_REQUEST_FUNCTIONS ?? "llm.request.functions";
const ATTR_LLM_USAGE_TOTAL_TOKENS = TL.LLM_USAGE_TOTAL_TOKENS ?? "llm.usage.total_tokens";

export class OpenRouterInstrumentor {
  public readonly name = "openrouter";

  private static readonly _sharedState = {
    activeInstances: 0,
    patches: [] as Patch[],
    activation: undefined as Promise<boolean> | undefined,
  };

  private enabled = false;
  private activation?: Promise<void>;

  async activate(): Promise<void> {
    if (this.enabled) return;

    if (!this.activation) {
      this.activation = this.acquireActivation();
    }
    const activation = this.activation;
    try {
      await activation;
    } finally {
      if (this.activation === activation) this.activation = undefined;
    }
  }

  private async acquireActivation(): Promise<void> {
    const shared = OpenRouterInstrumentor._sharedState;
    let patchesInstalled: boolean;
    if (shared.activation) {
      patchesInstalled = await shared.activation;
    } else if (shared.patches.length > 0) {
      patchesInstalled = true;
    } else {
      const activation = this.installPatches();
      shared.activation = activation;
      try {
        patchesInstalled = await activation;
      } finally {
        if (shared.activation === activation) {
          shared.activation = undefined;
        }
      }
    }
    if (!patchesInstalled || this.enabled) return;

    shared.activeInstances += 1;
    this.enabled = true;
  }

  async deactivate(): Promise<void> {
    this.disable();
  }

  enable(): void {
    void this.activate().catch(() => undefined);
  }

  isActive(): boolean {
    return this.enabled;
  }

  disable(): void {
    if (!this.enabled) return;

    const shared = OpenRouterInstrumentor._sharedState;
    shared.activeInstances = Math.max(0, shared.activeInstances - 1);
    this.enabled = false;
    if (shared.activeInstances === 0) this.restorePatches();
  }

  private async installPatches(): Promise<boolean> {
    try {
      await this.patchChat();
      await this.patchEmbeddings();
    } catch (error) {
      this.restorePatches();
      throw error;
    }
    return OpenRouterInstrumentor._sharedState.patches.length > 0;
  }

  private restorePatches(): void {
    const shared = OpenRouterInstrumentor._sharedState;
    for (const undo of [...shared.patches].reverse()) undo();
    shared.patches = [];
  }

  private async patchChat(): Promise<void> {
    const mod = await importOpenRouterSdkModule("@openrouter/sdk/sdk/chat.js");
    const proto = (mod as AnyRecord).Chat?.prototype;
    if (!proto || proto.__respanPatchedSend || typeof proto.send !== "function") return;

    const original = proto.send as AnyFunction;
    const patchedSend: AnyFunction = function patchedSend(
      this: unknown,
      request: AnyRecord,
      options?: AnyRecord,
    ) {
      if (OpenRouterInstrumentor._sharedState.activeInstances === 0) {
        return original.apply(this, arguments as any);
      }
      const parent = activeParentContext();
      const startedAt = new Date();
      const chatRequest = resolveChatRequest(request);
      const emit = (response: unknown, error?: unknown, chunks?: unknown[]) => {
        injectSpan(buildChatSpan(chatRequest, response, startedAt, parent, error, chunks));
      };

      try {
        const result = original.apply(this, arguments as any);
        if (isPromiseLike(result)) {
          return result.then((value: any) => {
            if (isStreamingRequest(chatRequest) && isAsyncIterable(value)) return wrapStream(value, emit);
            emit(value);
            return value;
          }, (error: unknown) => {
            emit(undefined, error);
            throw error;
          });
        }
        if (isStreamingRequest(chatRequest) && isAsyncIterable(result)) return wrapStream(result, emit);
        emit(result);
        return result;
      } catch (error) {
        emit(undefined, error);
        throw error;
      }
    };
    proto.send = patchedSend;
    proto.__respanPatchedSend = true;
    OpenRouterInstrumentor._sharedState.patches.push(() => {
      if (proto.send === patchedSend) proto.send = original;
      delete proto.__respanPatchedSend;
    });
  }

  private async patchEmbeddings(): Promise<void> {
    const mod = await importOpenRouterSdkModule("@openrouter/sdk/sdk/embeddings.js");
    const proto = (mod as AnyRecord).Embeddings?.prototype;
    if (
      !proto ||
      proto.__respanPatchedGenerate ||
      typeof proto.generate !== "function"
    ) return;

    const original = proto.generate as AnyFunction;
    const patchedGenerate: AnyFunction = function patchedGenerate(
      this: unknown,
      request: AnyRecord,
      options?: AnyRecord,
    ) {
      if (OpenRouterInstrumentor._sharedState.activeInstances === 0) {
        return original.apply(this, arguments as any);
      }
      const parent = activeParentContext();
      const startedAt = new Date();
      const requestBody = resolveEmbeddingRequestBody(request);
      const emit = (response: unknown, error?: unknown) => {
        injectSpan(buildEmbeddingSpan(requestBody, response, startedAt, parent, error));
      };

      try {
        const result = original.apply(this, arguments as any);
        if (isPromiseLike(result)) {
          return result.then((value: any) => {
            emit(value);
            return value;
          }, (error: unknown) => {
            emit(undefined, error);
            throw error;
          });
        }
        emit(result);
        return result;
      } catch (error) {
        emit(undefined, error);
        throw error;
      }
    };
    proto.generate = patchedGenerate;
    proto.__respanPatchedGenerate = true;
    OpenRouterInstrumentor._sharedState.patches.push(() => {
      if (proto.generate === patchedGenerate) proto.generate = original;
      delete proto.__respanPatchedGenerate;
    });
  }
}

export function instrumentOpenRouter(): OpenRouterInstrumentor {
  const instrumentor = new OpenRouterInstrumentor();
  instrumentor.enable();
  return instrumentor;
}

export default OpenRouterInstrumentor;

async function importOpenRouterSdkModule(specifier: string): Promise<AnyRecord> {
  try {
    const hostRequire = createRequire(`${process.cwd()}/package.json`);
    return await import(pathToFileURL(hostRequire.resolve(specifier)).href);
  } catch {
    return await import(specifier);
  }
}

function buildChatSpan(
  request: AnyRecord,
  response: unknown,
  startedAt: Date,
  parent: ParentContext,
  error?: unknown,
  streamChunks?: unknown[],
): ReadableSpan {
  const responseRecord = response as AnyRecord | undefined;
  const completion = extractCompletion(responseRecord, streamChunks);
  const usage = mergeUsage(responseRecord?.usage, streamChunks);
  const model = responseRecord?.model ?? request?.model;
  const attributes: AnyRecord = {
    [ATTR_TRACELOOP_ENTITY_NAME]: "openrouter.chat",
    [ATTR_TRACELOOP_ENTITY_PATH]: "openrouter.chat.send",
    [ATTR_TRACELOOP_ENTITY_INPUT]: safeStringify(request?.messages ?? []),
    [ATTR_TRACELOOP_ENTITY_OUTPUT]: safeStringify(completion ?? responseRecord ?? null),
    [ATTR_LLM_REQUEST_TYPE]: "chat",
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: TRACE_METHOD,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.CHAT,
    [ATTR_GEN_AI_SYSTEM]: OPENROUTER_SYSTEM,
    [ATTR_GEN_AI_REQUEST_MODEL]: model,
    "respan.metadata.openrouter_operation": "chat.send",
    "respan.metadata.stream": String(Boolean(request?.stream)),
  };

  attachPromptAttributes(attributes, request?.messages);
  attachCompletionAttributes(attributes, completion);
  attachRequestToolAttributes(attributes, request?.tools);
  attachUsageAttributes(attributes, usage);
  attachResponseMetadata(attributes, responseRecord);
  if (error) attachErrorAttributes(attributes, error);

  return buildReadableSpan({
    name: "openrouter.chat",
    startTimeIso: startedAt.toISOString(),
    endTimeIso: new Date().toISOString(),
    attributes: prune(attributes),
    traceId: parent.traceId,
    parentId: parent.parentId,
  });
}

function buildEmbeddingSpan(
  request: AnyRecord,
  response: unknown,
  startedAt: Date,
  parent: ParentContext,
  error?: unknown,
): ReadableSpan {
  const responseRecord = response as AnyRecord | undefined;
  const usage = responseRecord?.usage;
  const summary = {
    id: responseRecord?.id,
    model: responseRecord?.model,
    object: responseRecord?.object,
    data_count: Array.isArray(responseRecord?.data) ? responseRecord.data.length : undefined,
  };
  const attributes: AnyRecord = {
    [ATTR_TRACELOOP_ENTITY_NAME]: "openrouter.embeddings",
    [ATTR_TRACELOOP_ENTITY_PATH]: "openrouter.embeddings.generate",
    [ATTR_TRACELOOP_ENTITY_INPUT]: safeStringify(request?.input ?? null),
    [ATTR_TRACELOOP_ENTITY_OUTPUT]: safeStringify(prune(summary)),
    [ATTR_LLM_REQUEST_TYPE]: "embedding",
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: TRACE_METHOD,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.EMBEDDING,
    [ATTR_GEN_AI_SYSTEM]: OPENROUTER_SYSTEM,
    [ATTR_GEN_AI_REQUEST_MODEL]: responseRecord?.model ?? request?.model,
    "respan.metadata.openrouter_operation": "embeddings.generate",
  };
  attachUsageAttributes(attributes, usage);
  if (error) attachErrorAttributes(attributes, error);

  return buildReadableSpan({
    name: "openrouter.embeddings",
    startTimeIso: startedAt.toISOString(),
    endTimeIso: new Date().toISOString(),
    attributes: prune(attributes),
    traceId: parent.traceId,
    parentId: parent.parentId,
  });
}

function resolveChatRequest(request: AnyRecord = {}): AnyRecord {
  return request.chatRequest ?? request;
}

function resolveEmbeddingRequestBody(request: AnyRecord = {}): AnyRecord {
  return request.requestBody ?? request;
}

function activeParentContext(): ParentContext {
  const spanContext = trace.getSpan(context.active())?.spanContext();
  return spanContext ? { traceId: spanContext.traceId, parentId: spanContext.spanId } : {};
}

function isStreamingRequest(request: AnyRecord): boolean {
  return Boolean(request?.stream);
}

function isPromiseLike(value: unknown): value is Promise<unknown> {
  return Boolean(value && typeof (value as AnyRecord).then === "function");
}

function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return Boolean(value && typeof (value as { [Symbol.asyncIterator]?: unknown })[Symbol.asyncIterator] === "function");
}

function wrapStream(stream: AsyncIterable<unknown>, emit: (response: unknown, error?: unknown, chunks?: unknown[]) => void): ReadableStream<unknown> {
  const chunks: unknown[] = [];
  let emitted = false;
  const finish = (error?: unknown) => {
    if (emitted) return;
    emitted = true;
    emit(undefined, error, chunks);
  };

  return new ReadableStream<unknown>({
    async start(controller) {
      try {
        for await (const chunk of stream) {
          chunks.push(chunk);
          controller.enqueue(chunk);
        }
        finish();
        controller.close();
      } catch (error) {
        finish(error);
        controller.error(error);
      }
    },
    cancel() {
      finish(new Error("OpenRouter stream cancelled before completion"));
      const cancel = (stream as AnyRecord).cancel;
      if (typeof cancel === "function") return cancel.call(stream);
      return undefined;
    },
  });
}

function attachPromptAttributes(attributes: AnyRecord, messages: unknown): void {
  if (!Array.isArray(messages)) return;
  messages.forEach((message, index) => {
    const msg = message as AnyRecord;
    const base = `${ATTR_GEN_AI_PROMPT}.${index}`;
    attributes[`${base}.role`] = msg.role;
    attributes[`${base}.content`] = contentToString(msg.content);
    if (Array.isArray(msg.toolCalls)) attributes[`${base}.tool_calls`] = safeStringify(normalizeToolCalls(msg.toolCalls));
    if (Array.isArray(msg.tool_calls)) attributes[`${base}.tool_calls`] = safeStringify(normalizeToolCalls(msg.tool_calls));
  });
}

function attachCompletionAttributes(attributes: AnyRecord, completion: AnyRecord | undefined): void {
  if (!completion) return;
  attributes[`${COMPLETION_ZERO}.role`] = completion.role;
  attributes[`${COMPLETION_ZERO}.content`] = contentToString(completion.content);
  const toolCalls = completion.toolCalls ?? completion.tool_calls;
  if (Array.isArray(toolCalls)) attributes[`${COMPLETION_ZERO}.tool_calls`] = safeStringify(normalizeToolCalls(toolCalls));
}

function attachRequestToolAttributes(attributes: AnyRecord, tools: unknown): void {
  if (!Array.isArray(tools) || tools.length === 0) return;
  attributes[ATTR_LLM_REQUEST_FUNCTIONS] = safeStringify(tools.map((tool) => {
    const record = tool as AnyRecord;
    const fn = record.function ?? {};
    return prune({ type: record.type, name: fn.name, description: fn.description, parameters: fn.parameters });
  }));
}

function attachUsageAttributes(attributes: AnyRecord, usage: AnyRecord | undefined): void {
  if (!usage) return;
  const promptTokens = usage.promptTokens ?? usage.prompt_tokens ?? usage.input_tokens;
  const completionTokens = usage.completionTokens ?? usage.completion_tokens ?? usage.output_tokens;
  const totalTokens = usage.totalTokens ?? usage.total_tokens;
  const cacheReadTokens = usage.promptTokensDetails?.cachedTokens ?? usage.prompt_tokens_details?.cached_tokens;
  if (promptTokens !== undefined) {
    attributes[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = promptTokens;
    attributes[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = promptTokens;
  }
  if (completionTokens !== undefined) {
    attributes[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = completionTokens;
    attributes[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = completionTokens;
  }
  if (totalTokens !== undefined) attributes[ATTR_LLM_USAGE_TOTAL_TOKENS] = totalTokens;
  if (cacheReadTokens !== undefined) attributes[CACHE_READ_INPUT_TOKENS] = cacheReadTokens;
}

function attachResponseMetadata(attributes: AnyRecord, response: AnyRecord | undefined): void {
  if (!response) return;
  const choice = Array.isArray(response.choices) ? response.choices[0] : undefined;
  attributes["respan.metadata.openrouter_response_id"] = response.id;
  attributes["respan.metadata.openrouter_finish_reason"] = choice?.finishReason ?? choice?.finish_reason;
  attributes["respan.metadata.openrouter_service_tier"] = response.serviceTier ?? response.service_tier;
  attributes["respan.metadata.openrouter_provider"] = response.provider;
  attributes["respan.metadata.openrouter_cost"] = response.cost;
}

function attachErrorAttributes(attributes: AnyRecord, error: unknown): void {
  const record = error as AnyRecord;
  attributes["error.type"] = record?.name ?? "Error";
  attributes["error.message"] = record?.message ?? String(error);
}

function extractCompletion(response: AnyRecord | undefined, chunks?: unknown[]): AnyRecord | undefined {
  if (chunks?.length) {
    const content: string[] = [];
    const toolCalls: unknown[] = [];
    let role: string | undefined;
    for (const chunk of chunks) {
      const delta = (chunk as AnyRecord)?.choices?.[0]?.delta;
      if (!delta) continue;
      if (delta.role) role = delta.role;
      if (delta.content) content.push(delta.content);
      if (Array.isArray(delta.toolCalls)) toolCalls.push(...delta.toolCalls);
      if (Array.isArray(delta.tool_calls)) toolCalls.push(...delta.tool_calls);
    }
    return prune({ role: role ?? "assistant", content: content.join(""), toolCalls });
  }
  return response?.choices?.[0]?.message ?? response?.choices?.[0]?.delta;
}

function mergeUsage(usage: AnyRecord | undefined, chunks?: unknown[]): AnyRecord | undefined {
  if (usage) return usage;
  if (!chunks?.length) return undefined;
  return [...chunks].reverse().map((chunk) => (chunk as AnyRecord).usage).find(Boolean);
}

function normalizeToolCalls(toolCalls: unknown[]): unknown[] {
  return toolCalls.map((toolCall) => {
    const record = toolCall as AnyRecord;
    const fn = record.function ?? {};
    return prune({ id: record.id, type: record.type, name: fn.name, arguments: fn.arguments });
  });
}

function contentToString(content: unknown): string | undefined {
  if (content === undefined || content === null) return undefined;
  return typeof content === "string" ? content : safeStringify(content);
}

function safeStringify(value: unknown): string {
  return JSON.stringify(value, (_key, current) => {
    if (typeof current === "function" || typeof current === "symbol") return undefined;
    return current;
  });
}

function prune<T extends AnyRecord>(record: T): T {
  return Object.fromEntries(Object.entries(record).filter(([, value]) => value !== undefined && value !== null && value !== "")) as T;
}
