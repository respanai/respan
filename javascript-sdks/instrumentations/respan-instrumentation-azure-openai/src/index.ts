import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

/**
 * Respan instrumentation plugin for Azure OpenAI TypeScript clients.
 *
 * Supports the current `openai` package's `AzureOpenAI` client and the older
 * `@azure/openai` `OpenAIClient` methods.
 */

import { context, trace, TraceFlags } from "@opentelemetry/api";
import { hrTime } from "@opentelemetry/core";
import {
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_REQUEST_FREQUENCY_PENALTY,
  ATTR_GEN_AI_REQUEST_MAX_TOKENS,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_REQUEST_PRESENCE_PENALTY,
  ATTR_GEN_AI_REQUEST_TEMPERATURE,
  ATTR_GEN_AI_REQUEST_TOP_P,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import {
  buildReadableSpan,
  injectSpan,
} from "@respan/tracing";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";

const PACKAGE_VERSION = "1.0.0";
const INSTRUMENTATION_NAME = "@respan/instrumentation-azure-openai";
const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";
const PATCH_MARKER = Symbol.for("@respan/instrumentation-azure-openai.original");
const MAX_ATTRIBUTE_CHARS = 16000;
const MAX_SERIALIZATION_DEPTH = 6;

const GEN_AI_COMPLETION_PREFIX = `${ATTR_GEN_AI_COMPLETION}.0`;
const GEN_AI_COMPLETION_ROLE = `${GEN_AI_COMPLETION_PREFIX}.role`;
const GEN_AI_COMPLETION_CONTENT = `${GEN_AI_COMPLETION_PREFIX}.content`;
const GEN_AI_COMPLETION_TOOL_CALLS = `${GEN_AI_COMPLETION_PREFIX}.tool_calls`;

type AnyRecord = Record<string, any>;
type AzureOperationKind = "chat" | "completion" | "embedding";
type WrappedFunction = ((...args: any[]) => any) & { [PATCH_MARKER]?: (...args: any[]) => any };

type PatchRecord = {
  target: AnyRecord;
  methodName: string;
  original: (...args: any[]) => any;
};

type OperationConfig = {
  kind: AzureOperationKind;
  spanName: string;
  logType: RespanLogType;
};

export interface AzureOpenAIInstrumentorOptions {
  /**
   * The imported `openai` module. Passing the module avoids module-load ordering
   * issues and is recommended when the application imports `AzureOpenAI` before
   * initializing Respan.
   */
  openAIModule?: AnyRecord;
  /**
   * Optional older `@azure/openai` module. This is only needed for applications
   * still using `OpenAIClient`.
   */
  azureOpenAIModule?: AnyRecord;
  traceContent?: boolean;
  exceptionLogger?: (error: Error) => void;
}

const OPERATION_CONFIG: Record<AzureOperationKind, OperationConfig> = {
  chat: {
    kind: "chat",
    spanName: "azure_openai.chat",
    logType: RespanLogType.CHAT,
  },
  completion: {
    kind: "completion",
    spanName: "azure_openai.completion",
    logType: RespanLogType.TEXT,
  },
  embedding: {
    kind: "embedding",
    spanName: "azure_openai.embedding",
    logType: RespanLogType.EMBEDDING,
  },
};

export class AzureOpenAIInstrumentor {
  public readonly name = "azure-openai";

  private static readonly _sharedState = {
    activeInstances: 0,
    patches: [] as PatchRecord[],
  };

  private readonly _options: AzureOpenAIInstrumentorOptions;
  private _isInstrumented = false;

  constructor(options: AzureOpenAIInstrumentorOptions = {}) {
    this._options = options;
  }

  async activate(): Promise<void> {
    if (this._isInstrumented) {
      return;
    }

    const sharedState = AzureOpenAIInstrumentor._sharedState;
    if (sharedState.activeInstances === 0) {
      await this._installPatches(sharedState.patches);
    }

    if (sharedState.patches.length === 0) {
      console.warn(
        "[respan] AzureOpenAIInstrumentor failed to activate: install the `openai` package or pass an openAIModule.",
      );
      return;
    }

    sharedState.activeInstances += 1;
    this._isInstrumented = true;
  }

  deactivate(): void {
    if (!this._isInstrumented) {
      return;
    }

    const sharedState = AzureOpenAIInstrumentor._sharedState;
    sharedState.activeInstances = Math.max(0, sharedState.activeInstances - 1);
    this._isInstrumented = false;

    if (sharedState.activeInstances > 0) {
      return;
    }

    for (const patch of sharedState.patches) {
      try {
        patch.target[patch.methodName] = patch.original;
      } catch {
        // Ignore restore failures; instrumentation must not break shutdown.
      }
    }
    sharedState.patches = [];
  }

  private async _installPatches(patches: PatchRecord[]): Promise<void> {
    const openAI = await this._loadOpenAIModule();
    if (openAI?.AzureOpenAI) {
      this._patchModernOpenAI(openAI, patches);
    }

    const legacyAzure = await this._loadLegacyAzureOpenAIModule();
    if (legacyAzure?.OpenAIClient) {
      this._patchLegacyAzureOpenAI(legacyAzure, patches);
    }
  }

  private async _loadOpenAIModule(): Promise<AnyRecord | null> {
    if (this._options.openAIModule) {
      return this._options.openAIModule;
    }
    return await importSdkModule("openai", "index.mjs");
  }

  private async _loadLegacyAzureOpenAIModule(): Promise<AnyRecord | null> {
    if (this._options.azureOpenAIModule) {
      return this._options.azureOpenAIModule;
    }
    return await importSdkModule("@azure/openai");
  }

  private _patchModernOpenAI(openAI: AnyRecord, patches: PatchRecord[]): void {
    const AzureOpenAI = openAI.AzureOpenAI;
    const resources = [
      {
        target: AzureOpenAI.Chat?.Completions?.prototype,
        methodName: "create",
        operation: OPERATION_CONFIG.chat,
        resolveCall: (args: any[], receiver: AnyRecord) => ({
          client: receiver?._client,
          params: asRecord(args[0]),
          callArgs: [withoutExtraAttributes(args[0]), ...args.slice(1)],
        }),
      },
      {
        target: AzureOpenAI.Completions?.prototype,
        methodName: "create",
        operation: OPERATION_CONFIG.completion,
        resolveCall: (args: any[], receiver: AnyRecord) => ({
          client: receiver?._client,
          params: asRecord(args[0]),
          callArgs: [withoutExtraAttributes(args[0]), ...args.slice(1)],
        }),
      },
      {
        target: AzureOpenAI.Embeddings?.prototype,
        methodName: "create",
        operation: OPERATION_CONFIG.embedding,
        resolveCall: (args: any[], receiver: AnyRecord) => ({
          client: receiver?._client,
          params: asRecord(args[0]),
          callArgs: [withoutExtraAttributes(args[0]), ...args.slice(1)],
        }),
      },
    ];

    for (const resource of resources) {
      this._patchMethod(
        patches,
        resource.target,
        resource.methodName,
        (original) => {
          const instrumentor = this;
          return function patchedModernAzureOpenAIMethod(this: AnyRecord, ...args: any[]) {
            const call = resource.resolveCall(args, this);
            if (!isModernAzureOpenAIClient(call.client, AzureOpenAI)) {
              return original.apply(this, args);
            }
            return instrumentor._traceOperation({
              operation: resource.operation,
              original,
              receiver: this,
              args: call.callArgs,
              params: call.params,
              client: call.client,
            });
          };
        },
      );
    }
  }

  private _patchLegacyAzureOpenAI(azureOpenAI: AnyRecord, patches: PatchRecord[]): void {
    const OpenAIClient = azureOpenAI.OpenAIClient;
    const legacyResources = [
      {
        target: OpenAIClient.prototype,
        methodName: "getChatCompletions",
        operation: OPERATION_CONFIG.chat,
        resolveCall: (args: any[], receiver: AnyRecord) => ({
          client: receiver,
          params: {
            model: args[0],
            messages: Array.isArray(args[1]) ? args[1] : [],
            ...(asRecord(args[2])),
          },
          callArgs: [args[0], args[1], withoutExtraAttributes(args[2]), ...args.slice(3)],
        }),
      },
      {
        target: OpenAIClient.prototype,
        methodName: "getCompletions",
        operation: OPERATION_CONFIG.completion,
        resolveCall: (args: any[], receiver: AnyRecord) => ({
          client: receiver,
          params: {
            model: args[0],
            prompt: args[1],
            ...(asRecord(args[2])),
          },
          callArgs: [args[0], args[1], withoutExtraAttributes(args[2]), ...args.slice(3)],
        }),
      },
      {
        target: OpenAIClient.prototype,
        methodName: "getEmbeddings",
        operation: OPERATION_CONFIG.embedding,
        resolveCall: (args: any[], receiver: AnyRecord) => ({
          client: receiver,
          params: {
            model: args[0],
            input: args[1],
            ...(asRecord(args[2])),
          },
          callArgs: [args[0], args[1], withoutExtraAttributes(args[2]), ...args.slice(3)],
        }),
      },
      {
        target: OpenAIClient.prototype,
        methodName: "streamChatCompletions",
        operation: OPERATION_CONFIG.chat,
        resolveCall: (args: any[], receiver: AnyRecord) => ({
          client: receiver,
          params: {
            model: args[0],
            messages: Array.isArray(args[1]) ? args[1] : [],
            stream: true,
            ...(asRecord(args[2])),
          },
          callArgs: [args[0], args[1], withoutExtraAttributes(args[2]), ...args.slice(3)],
        }),
      },
      {
        target: OpenAIClient.prototype,
        methodName: "streamCompletions",
        operation: OPERATION_CONFIG.completion,
        resolveCall: (args: any[], receiver: AnyRecord) => ({
          client: receiver,
          params: {
            model: args[0],
            prompt: args[1],
            stream: true,
            ...(asRecord(args[2])),
          },
          callArgs: [args[0], args[1], withoutExtraAttributes(args[2]), ...args.slice(3)],
        }),
      },
    ];

    for (const resource of legacyResources) {
      this._patchMethod(
        patches,
        resource.target,
        resource.methodName,
        (original) => {
          const instrumentor = this;
          return function patchedLegacyAzureOpenAIMethod(this: AnyRecord, ...args: any[]) {
            const call = resource.resolveCall(args, this);
            return instrumentor._traceOperation({
              operation: resource.operation,
              original,
              receiver: this,
              args: call.callArgs,
              params: call.params,
              client: call.client,
            });
          };
        },
      );
    }
  }

  private _patchMethod(
    patches: PatchRecord[],
    target: AnyRecord | undefined,
    methodName: string,
    wrap: (original: (...args: any[]) => any) => (...args: any[]) => any,
  ): void {
    if (!target || typeof target[methodName] !== "function") {
      return;
    }

    const existing = target[methodName] as WrappedFunction;
    if (existing[PATCH_MARKER]) {
      return;
    }

    const original = existing as (...args: any[]) => any;
    const wrapped = wrap(original) as WrappedFunction;
    Object.defineProperty(wrapped, PATCH_MARKER, {
      value: original,
      enumerable: false,
    });
    target[methodName] = wrapped;
    patches.push({ target, methodName, original });
  }

  private _traceOperation(opts: {
    operation: OperationConfig;
    original: (...args: any[]) => any;
    receiver: AnyRecord;
    args: any[];
    params: AnyRecord;
    client: AnyRecord;
  }): any {
    const startTime = hrTime();
    let result: any;

    try {
      result = opts.original.apply(opts.receiver, opts.args);
    } catch (error) {
      this._emitErrorSpan(opts.operation, opts.params, opts.client, startTime, error);
      throw error;
    }

    if (isStreamingRequest(opts.params)) {
      return Promise.resolve(result).then(
        (stream) => this._wrapStream(opts.operation, opts.params, opts.client, startTime, stream),
        (error) => {
          this._emitErrorSpan(opts.operation, opts.params, opts.client, startTime, error);
          throw error;
        },
      );
    }

    return Promise.resolve(result).then(
      (value) => {
        this._emitSuccessSpan(opts.operation, opts.params, opts.client, startTime, value);
        return value;
      },
      (error) => {
        this._emitErrorSpan(opts.operation, opts.params, opts.client, startTime, error);
        throw error;
      },
    );
  }

  private _wrapStream(
    operation: OperationConfig,
    params: AnyRecord,
    client: AnyRecord,
    startTime: [number, number],
    stream: AsyncIterable<any>,
  ): AsyncIterable<any> {
    const instrumentor = this;

    return {
      async *[Symbol.asyncIterator]() {
        const state = createStreamState(operation.kind, params, client);
        try {
          for await (const chunk of stream) {
            updateStreamState(state, chunk);
            yield chunk;
          }
          instrumentor._emitSuccessSpan(operation, params, client, startTime, finalizeStreamState(state));
        } catch (error) {
          instrumentor._emitErrorSpan(operation, params, client, startTime, error);
          throw error;
        }
      },
    };
  }

  private _emitSuccessSpan(
    operation: OperationConfig,
    params: AnyRecord,
    client: AnyRecord,
    startTime: [number, number],
    result: AnyRecord,
  ): void {
    try {
      const attrs = buildBaseAttributes(operation, params, client, this._shouldTraceContent());
      enrichSuccessAttributes(attrs, operation.kind, params, result, this._shouldTraceContent());
      this._emitSpan(operation.spanName, attrs, startTime);
    } catch (error) {
      this._logException(error);
    }
  }

  private _emitErrorSpan(
    operation: OperationConfig,
    params: AnyRecord,
    client: AnyRecord,
    startTime: [number, number],
    error: unknown,
  ): void {
    try {
      const attrs = buildBaseAttributes(operation, params, client, this._shouldTraceContent());
      const errorMessage = error instanceof Error ? error.message : String(error);
      attrs["error.message"] = errorMessage;
      this._emitSpan(operation.spanName, attrs, startTime, errorMessage);
    } catch (innerError) {
      this._logException(innerError);
    }
  }

  private _emitSpan(
    name: string,
    attributes: Record<string, any>,
    startTime: [number, number],
    errorMessage?: string,
  ): void {
    const activeSpanContext = trace.getSpan(context.active())?.spanContext();
    const span = buildReadableSpan({
      name,
      traceId: activeSpanContext?.traceId,
      parentId: activeSpanContext?.spanId,
      startTimeHr: startTime,
      endTimeHr: hrTime(),
      attributes,
      errorMessage,
    }) as ReturnType<typeof buildReadableSpan> & {
      instrumentationLibrary?: { name: string; version?: string };
      spanContext: () => ReturnType<ReturnType<typeof buildReadableSpan>["spanContext"]>;
    };

    const originalSpanContext = span.spanContext.bind(span);
    span.spanContext = () => ({
      ...originalSpanContext(),
      traceFlags: activeSpanContext?.traceFlags ?? TraceFlags.SAMPLED,
    });
    span.instrumentationLibrary = {
      name: INSTRUMENTATION_NAME,
      version: PACKAGE_VERSION,
    };
    injectSpan(span);
  }

  private _shouldTraceContent(): boolean {
    const contextValue = context.active().getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT);
    if (contextValue !== undefined) {
      return Boolean(contextValue);
    }
    return this._options.traceContent ?? true;
  }

  private _logException(error: unknown): void {
    const normalized = error instanceof Error ? error : new Error(String(error));
    if (this._options.exceptionLogger) {
      this._options.exceptionLogger(normalized);
    }
  }
}

function buildBaseAttributes(
  operation: OperationConfig,
  params: AnyRecord,
  client: AnyRecord,
  traceContent: boolean,
): Record<string, any> {
  const attrs: Record<string, any> = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: operation.spanName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: operation.spanName,
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: operation.logType,
    [ATTR_GEN_AI_SYSTEM]: "azure",
    [SpanAttributes.LLM_REQUEST_TYPE]: getBackendRequestType(operation.kind),
  };

  const model = resolveModel(params, client);
  if (model) {
    attrs[ATTR_GEN_AI_REQUEST_MODEL] = model;
  }

  setRequestOptions(attrs, params);
  setExtraAttributes(attrs, params.extraAttributes);

  if (!traceContent) {
    return attrs;
  }

  if (operation.kind === "chat") {
    setChatInputAttributes(attrs, params.messages);
    setToolDefinitionAttributes(attrs, params.tools ?? params.functions);
  } else if (operation.kind === "completion") {
    setCompletionInputAttributes(attrs, params.prompt);
  } else if (operation.kind === "embedding") {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson({
      input: params.input,
      model,
    });
  }

  return attrs;
}

function getBackendRequestType(kind: AzureOperationKind): string {
  if (kind === "completion") {
    return "chat";
  }
  return kind;
}

function enrichSuccessAttributes(
  attrs: Record<string, any>,
  kind: AzureOperationKind,
  params: AnyRecord,
  result: AnyRecord,
  traceContent: boolean,
): void {
  const responseModel = result?.model ?? resolveModel(params, {});
  if (responseModel) {
    attrs[ATTR_GEN_AI_RESPONSE_MODEL] = responseModel;
  }

  setUsageAttributes(attrs, result?.usage);

  if (!traceContent) {
    return;
  }

  if (kind === "chat") {
    const message = result?.choices?.[0]?.message ?? {};
    attrs[GEN_AI_COMPLETION_ROLE] = message.role ?? "assistant";
    attrs[GEN_AI_COMPLETION_CONTENT] = stringifyMessageContent(message.content ?? "");

    const toolCalls = normalizeToolCalls(message.tool_calls ?? message.toolCalls);
    if (toolCalls.length > 0) {
      attrs[GEN_AI_COMPLETION_TOOL_CALLS] = safeJson(toolCalls);
    }
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson([
      {
        role: attrs[GEN_AI_COMPLETION_ROLE],
        content: attrs[GEN_AI_COMPLETION_CONTENT],
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      },
    ]);
  } else if (kind === "completion") {
    const text = result?.choices?.[0]?.text ?? "";
    attrs[GEN_AI_COMPLETION_ROLE] = "assistant";
    attrs[GEN_AI_COMPLETION_CONTENT] = stringifyMessageContent(text);
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson({
      text: attrs[GEN_AI_COMPLETION_CONTENT],
    });
  } else if (kind === "embedding") {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson({
      model: responseModel,
      embedding_count: Array.isArray(result?.data) ? result.data.length : undefined,
    });
  }
}

function setRequestOptions(attrs: Record<string, any>, params: AnyRecord): void {
  const numericOptions: Array<[string, string]> = [
    ["max_tokens", ATTR_GEN_AI_REQUEST_MAX_TOKENS],
    ["temperature", ATTR_GEN_AI_REQUEST_TEMPERATURE],
    ["top_p", ATTR_GEN_AI_REQUEST_TOP_P],
    ["frequency_penalty", ATTR_GEN_AI_REQUEST_FREQUENCY_PENALTY],
    ["presence_penalty", ATTR_GEN_AI_REQUEST_PRESENCE_PENALTY],
  ];

  for (const [sourceKey, attrKey] of numericOptions) {
    if (params[sourceKey] !== undefined && attrKey) {
      attrs[attrKey] = params[sourceKey];
    }
  }
}

function setExtraAttributes(attrs: Record<string, any>, extraAttributes: unknown): void {
  if (!extraAttributes || typeof extraAttributes !== "object" || Array.isArray(extraAttributes)) {
    return;
  }
  for (const [key, value] of Object.entries(extraAttributes as Record<string, unknown>)) {
    const normalized = toAttributeValue(value);
    if (normalized !== undefined) {
      attrs[key] = normalized;
    }
  }
}

function setChatInputAttributes(attrs: Record<string, any>, messages: unknown): void {
  const normalizedMessages = Array.isArray(messages) ? messages : [];
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(normalizedMessages);

  normalizedMessages.forEach((message, index) => {
    const normalized = asRecord(message);
    const prefix = `${ATTR_GEN_AI_PROMPT}.${index}`;
    attrs[`${prefix}.role`] = String(normalized.role ?? "user");
    attrs[`${prefix}.content`] = stringifyMessageContent(normalized.content ?? "");

    const toolCalls = normalizeToolCalls(normalized.tool_calls ?? normalized.toolCalls);
    if (toolCalls.length > 0) {
      attrs[`${prefix}.tool_calls`] = safeJson(toolCalls);
    }
  });
}

function setCompletionInputAttributes(attrs: Record<string, any>, prompt: unknown): void {
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson({ prompt });
  attrs[`${ATTR_GEN_AI_PROMPT}.0.role`] = "user";
  attrs[`${ATTR_GEN_AI_PROMPT}.0.content`] = stringifyMessageContent(prompt);
}

function setToolDefinitionAttributes(attrs: Record<string, any>, tools: unknown): void {
  const normalized = normalizeToolDefinitions(tools);
  if (normalized.length === 0) {
    return;
  }
  attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(normalized);
}

function setUsageAttributes(attrs: Record<string, any>, usage: unknown): void {
  if (!usage || typeof usage !== "object") {
    return;
  }
  const usageRecord = usage as Record<string, any>;
  const promptTokens =
    usageRecord.prompt_tokens ?? usageRecord.promptTokens ?? usageRecord.input_tokens ?? usageRecord.inputTokens;
  const completionTokens =
    usageRecord.completion_tokens ??
    usageRecord.completionTokens ??
    usageRecord.output_tokens ??
    usageRecord.outputTokens;
  const totalTokens = usageRecord.total_tokens ?? usageRecord.totalTokens;
  if (typeof promptTokens === "number") {
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = promptTokens;
    attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = promptTokens;
  }
  if (typeof completionTokens === "number") {
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = completionTokens;
    attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = completionTokens;
  }
  if (typeof totalTokens === "number") {
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = totalTokens;
  } else if (typeof promptTokens === "number" && typeof completionTokens === "number") {
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = promptTokens + completionTokens;
  }
}

function normalizeToolDefinitions(tools: unknown): any[] {
  if (!Array.isArray(tools)) {
    return [];
  }

  return tools.map((tool) => {
    const record = asRecord(tool);
    if (record.type === "function" && record.function) {
      return {
        type: "function",
        function: toSerializableValue(record.function),
      };
    }
    if (record.name) {
      return {
        type: "function",
        function: {
          name: record.name,
          description: record.description,
          parameters: record.parameters,
        },
      };
    }
    return toSerializableValue(record);
  });
}

function normalizeToolCalls(toolCalls: unknown): any[] {
  if (!Array.isArray(toolCalls)) {
    return [];
  }

  return toolCalls.map((toolCall) => {
    const record = asRecord(toolCall);
    const fn = asRecord(record.function);
    return {
      id: String(record.id ?? ""),
      type: String(record.type ?? "function"),
      function: {
        name: String(fn.name ?? record.name ?? ""),
        arguments: stringifyMessageContent(fn.arguments ?? record.arguments ?? ""),
      },
    };
  });
}

function resolveModel(params: AnyRecord, client: AnyRecord): string | undefined {
  const model = params.model || client?.deploymentName || params.deployment || params.deploymentName;
  if (typeof model === "string" && model.length > 0) {
    return model;
  }
  return undefined;
}

function isModernAzureOpenAIClient(client: AnyRecord, AzureOpenAI: any): boolean {
  if (!client) {
    return false;
  }
  if (typeof AzureOpenAI === "function" && client instanceof AzureOpenAI) {
    return true;
  }
  if (client.constructor?.name === "AzureOpenAI") {
    return true;
  }
  const baseURL = typeof client.baseURL === "string" ? client.baseURL.toLowerCase() : "";
  return typeof client.apiVersion === "string" && (baseURL.includes("azure") || client.deploymentName);
}

function isStreamingRequest(params: AnyRecord): boolean {
  return params.stream === true;
}

function withoutExtraAttributes(value: unknown): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return value;
  }
  const { extraAttributes: _extraAttributes, ...rest } = value as Record<string, unknown>;
  return rest;
}

function asRecord(value: unknown): AnyRecord {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value as AnyRecord;
}

function stringifyMessageContent(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (value === undefined || value === null) {
    return "";
  }
  return safeJson(value);
}

function toAttributeValue(value: unknown): string | number | boolean | Array<string | number | boolean> | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  if (Array.isArray(value)) {
    const primitiveValues = value.filter((item) =>
      typeof item === "string" || typeof item === "number" || typeof item === "boolean",
    ) as Array<string | number | boolean>;
    if (primitiveValues.length === value.length) {
      return primitiveValues;
    }
  }
  return safeJson(value);
}

function toSerializableValue(value: unknown, depth = 0): unknown {
  if (depth > MAX_SERIALIZATION_DEPTH) {
    return String(value);
  }
  if (value === undefined || value === null) {
    return value;
  }
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  if (typeof value === "bigint") {
    return value.toString();
  }
  if (typeof value === "function" || typeof value === "symbol") {
    return undefined;
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (value instanceof Error) {
    return {
      error: value.name,
      message: value.message,
    };
  }
  if (Array.isArray(value)) {
    return value
      .map((item) => toSerializableValue(item, depth + 1))
      .filter((item) => item !== undefined);
  }
  if (typeof value === "object") {
    const maybeToJSON = (value as { toJSON?: () => unknown }).toJSON;
    if (typeof maybeToJSON === "function") {
      try {
        return toSerializableValue(maybeToJSON.call(value), depth + 1);
      } catch {
        // Continue with structural serialization below.
      }
    }

    const normalized: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      if (key.startsWith("_")) {
        continue;
      }
      const serialized = toSerializableValue(item, depth + 1);
      if (serialized !== undefined) {
        normalized[key] = serialized;
      }
    }
    return normalized;
  }
  return String(value);
}

function safeJson(value: unknown): string {
  let serialized: string | undefined;
  try {
    serialized = JSON.stringify(toSerializableValue(value), (_key, innerValue) =>
      typeof innerValue === "bigint" ? innerValue.toString() : innerValue,
    );
  } catch {
    serialized = String(value);
  }

  if (typeof serialized !== "string") {
    serialized = String(serialized);
  }

  if (serialized.length <= MAX_ATTRIBUTE_CHARS) {
    return serialized;
  }
  return `${serialized.slice(0, MAX_ATTRIBUTE_CHARS)}...`;
}

type StreamState = {
  kind: AzureOperationKind;
  model?: string;
  content: string;
  text: string;
  toolCalls: Array<{
    id: string;
    type: string;
    function: {
      name: string;
      arguments: string;
    };
  }>;
  usage?: AnyRecord;
};

function createStreamState(kind: AzureOperationKind, params: AnyRecord, client: AnyRecord): StreamState {
  return {
    kind,
    model: resolveModel(params, client),
    content: "",
    text: "",
    toolCalls: [],
  };
}

function updateStreamState(state: StreamState, chunk: AnyRecord): void {
  state.model = chunk?.model ?? state.model;
  if (chunk?.usage) {
    state.usage = chunk.usage;
  }

  const choice = chunk?.choices?.[0];
  if (!choice) {
    return;
  }

  if (state.kind === "chat") {
    state.content += choice.delta?.content ?? "";
    for (const toolCallDelta of choice.delta?.tool_calls ?? []) {
      const index = Number(toolCallDelta.index ?? state.toolCalls.length);
      state.toolCalls[index] ??= {
        id: "",
        type: "function",
        function: {
          name: "",
          arguments: "",
        },
      };
      const current = state.toolCalls[index];
      current.id += toolCallDelta.id ?? "";
      current.type = toolCallDelta.type ?? current.type;
      current.function.name += toolCallDelta.function?.name ?? "";
      current.function.arguments += toolCallDelta.function?.arguments ?? "";
    }
  } else if (state.kind === "completion") {
    state.text += choice.text ?? "";
  }
}

function finalizeStreamState(state: StreamState): AnyRecord {
  if (state.kind === "chat") {
    return {
      model: state.model,
      usage: state.usage,
      choices: [
        {
          message: {
            role: "assistant",
            content: state.content,
            ...(state.toolCalls.length > 0 ? { tool_calls: state.toolCalls } : {}),
          },
        },
      ],
    };
  }

  if (state.kind === "completion") {
    return {
      model: state.model,
      usage: state.usage,
      choices: [{ text: state.text }],
    };
  }

  return {
    model: state.model,
    usage: state.usage,
    data: [],
  };
}

async function importSdkModule(packageName: string, esmEntryFile?: string): Promise<AnyRecord | null> {
  try {
    const hostRequire = createRequire(`${process.cwd()}/package.json`);
    const resolved = hostRequire.resolve(packageName);
    const entry = esmEntryFile ? join(dirname(resolved), esmEntryFile) : resolved;
    return await import(pathToFileURL(existsSync(entry) ? entry : resolved).href);
  } catch {
    try {
      return await import(packageName);
    } catch {
      return null;
    }
  }
}
