import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

/**
 * Respan instrumentation plugin for the Google Vertex AI TypeScript SDK.
 *
 * Patches generation and chat methods from `@google-cloud/vertexai` to emit
 * canonical Respan chat spans for sync-like promises, streaming responses, and
 * chat sessions.
 */

import {
  CHAT_SESSION_CLASS_NAME,
  GENERATE_CONTENT_METHOD_NAME,
  GENERATE_CONTENT_STREAM_METHOD_NAME,
  GENERATIVE_MODEL_CLASS_NAME,
  SEND_MESSAGE_METHOD_NAME,
  SEND_MESSAGE_STREAM_METHOD_NAME,
  START_CHAT_METHOD_NAME,
  VERTEXAI_CHAT_SPAN_NAME,
  VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
  VERTEXAI_INSTRUMENTATION_NAME,
} from "./_constants.js";
import { emitGenerateContentSpan } from "./_otel_emitter.js";
import { requestPayloadFromCall, type VertexAIRequestPayload } from "./_translator.js";

type AnyFunction = (...args: any[]) => any;
type PatchablePrototype = Record<string, unknown>;

export interface VertexAIModule {
  GenerativeModel?: { prototype?: PatchablePrototype };
  ChatSession?: { prototype?: PatchablePrototype };
}

export interface VertexAIInstrumentorOptions {
  sdkModule?: VertexAIModule;
}

interface PatchedMethod {
  methodName: string;
  original: AnyFunction;
  prototype: PatchablePrototype;
}

interface TraceCallOptions {
  args: unknown[];
  instance: unknown;
  isChatMethod?: boolean;
  isStreamMethod?: boolean;
  original: AnyFunction;
  spanName: string;
}

const PATCHED_BY_RESPAN = Symbol.for("respan.instrumentation.vertexai.patched");

function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  return Boolean(value && typeof (value as PromiseLike<unknown>).then === "function");
}

function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return Boolean(value && typeof (value as AsyncIterable<unknown>)[Symbol.asyncIterator] === "function");
}

function isStreamResult(value: unknown): boolean {
  return Boolean(value && typeof value === "object" && "stream" in value);
}

async function collectAsyncChunks(iterable: AsyncIterable<unknown>): Promise<unknown[]> {
  const chunks: unknown[] = [];
  for await (const chunk of iterable) {
    chunks.push(chunk);
  }
  return chunks;
}

function emitSuccess(
  requestPayload: VertexAIRequestPayload,
  startTimeIso: string,
  spanName: string,
  responseOrChunks: unknown,
): void {
  emitGenerateContentSpan({
    requestPayload,
    startTimeIso,
    responseOrChunks,
    spanName,
  });
}

function emitError(
  requestPayload: VertexAIRequestPayload,
  startTimeIso: string,
  spanName: string,
  error: unknown,
): void {
  emitGenerateContentSpan({
    requestPayload,
    startTimeIso,
    spanName,
    errorMessage: error instanceof Error ? error.message : String(error),
    statusCode: 500,
  });
}

function instrumentResponsePromise<T>(
  responsePromise: PromiseLike<T>,
  requestPayload: VertexAIRequestPayload,
  startTimeIso: string,
  spanName: string,
): Promise<T> {
  return Promise.resolve(responsePromise).then(
    (response) => {
      emitSuccess(requestPayload, startTimeIso, spanName, response);
      return response;
    },
    (error) => {
      emitError(requestPayload, startTimeIso, spanName, error);
      throw error;
    },
  );
}

function instrumentResolvedResult<T>(
  result: T,
  requestPayload: VertexAIRequestPayload,
  startTimeIso: string,
  spanName: string,
): T {
  if (isAsyncIterable(result)) {
    const wrapped = (async function* wrappedVertexAIStream() {
      const chunks: unknown[] = [];
      try {
        for await (const chunk of result) {
          chunks.push(chunk);
          yield chunk;
        }
        emitSuccess(requestPayload, startTimeIso, spanName, chunks);
      } catch (error) {
        emitError(requestPayload, startTimeIso, spanName, error);
        throw error;
      }
    })();
    return wrapped as T;
  }

  if (result && typeof result === "object" && "response" in result) {
    const responseValue = (result as Record<string, any>).response;
    if (isPromiseLike(responseValue)) {
      const instrumentedResponse = instrumentResponsePromise(
        responseValue,
        requestPayload,
        startTimeIso,
        spanName,
      );
      try {
        (result as Record<string, any>).response = instrumentedResponse;
      } catch {
        instrumentedResponse.catch(() => undefined);
      }
      return result;
    }

    emitSuccess(requestPayload, startTimeIso, spanName, responseValue);
    return result;
  }

  if (isStreamResult(result)) {
    const stream = (result as Record<string, unknown>).stream;
    if (isAsyncIterable(stream)) {
      collectAsyncChunks(stream).then(
        (chunks) => emitSuccess(requestPayload, startTimeIso, spanName, chunks),
        (error) => emitError(requestPayload, startTimeIso, spanName, error),
      );
      return result;
    }
  }

  emitSuccess(requestPayload, startTimeIso, spanName, result);
  return result;
}

function traceVertexAICall<T>(opts: TraceCallOptions): T {
  const startTimeIso = new Date().toISOString();
  const requestPayload = requestPayloadFromCall(opts.instance, opts.args, {
    isChatMethod: opts.isChatMethod,
    isStreamMethod: opts.isStreamMethod,
  });

  try {
    const result = opts.original.apply(opts.instance, opts.args);
    if (isPromiseLike(result)) {
      return Promise.resolve(result).then(
        (resolved) => instrumentResolvedResult(
          resolved,
          requestPayload,
          startTimeIso,
          opts.spanName,
        ),
        (error) => {
          emitError(requestPayload, startTimeIso, opts.spanName, error);
          throw error;
        },
      ) as T;
    }

    return instrumentResolvedResult(result, requestPayload, startTimeIso, opts.spanName);
  } catch (error) {
    emitError(requestPayload, startTimeIso, opts.spanName, error);
    throw error;
  }
}

function wrapMethod(
  original: AnyFunction,
  spanName: string,
  opts: { isChatMethod?: boolean; isStreamMethod?: boolean } = {},
): AnyFunction {
  const wrapped = function wrappedVertexAIMethod(this: unknown, ...args: unknown[]) {
    return traceVertexAICall({
      args,
      instance: this,
      isChatMethod: opts.isChatMethod,
      isStreamMethod: opts.isStreamMethod,
      original,
      spanName,
    });
  };
  Object.defineProperty(wrapped, PATCHED_BY_RESPAN, {
    enumerable: false,
    value: true,
  });
  return wrapped;
}

export class VertexAIInstrumentor {
  public readonly name = VERTEXAI_INSTRUMENTATION_NAME;

  private static readonly _sharedState = {
    activeInstances: 0,
    patchedMethods: [] as PatchedMethod[],
    patchedChatInstances: new WeakSet<object>(),
  };

  private readonly _sdkModule?: VertexAIModule;
  private _isInstrumented = false;

  constructor(options: VertexAIInstrumentorOptions = {}) {
    this._sdkModule = options.sdkModule;
  }

  async activate(): Promise<void> {
    if (this._isInstrumented) return;

    const module = await this._resolveSdkModule();
    if (!module) {
      console.warn(
        "[Respan] Failed to activate Vertex AI instrumentation - @google-cloud/vertexai not found",
      );
      return;
    }

    const sharedState = VertexAIInstrumentor._sharedState;
    try {
      this._patchGenerationPrototype(module.GenerativeModel?.prototype);
      this._patchChatPrototype(module.ChatSession?.prototype);

      if (sharedState.patchedMethods.length === 0) {
        console.warn(
          "[Respan] Failed to activate Vertex AI instrumentation - no compatible prototypes found",
        );
        return;
      }

      sharedState.activeInstances += 1;
      this._isInstrumented = true;
    } catch (error) {
      console.warn("[Respan] Failed to activate Vertex AI instrumentation:", error);
      this.deactivate();
    }
  }

  deactivate(): void {
    if (!this._isInstrumented) return;

    const sharedState = VertexAIInstrumentor._sharedState;
    sharedState.activeInstances = Math.max(0, sharedState.activeInstances - 1);
    this._isInstrumented = false;

    if (sharedState.activeInstances > 0) return;

    for (const patched of sharedState.patchedMethods) {
      patched.prototype[patched.methodName] = patched.original;
    }
    sharedState.patchedMethods = [];
    sharedState.patchedChatInstances = new WeakSet<object>();
  }

  isActive(): boolean {
    return this._isInstrumented;
  }

  private async _resolveSdkModule(): Promise<VertexAIModule | undefined> {
    if (this._sdkModule) return this._sdkModule;
    const moduleName = "@google-cloud/vertexai";
    try {
      const hostRequire = createRequire(`${process.cwd()}/package.json`);
      return (await import(pathToFileURL(hostRequire.resolve(moduleName)).href)) as VertexAIModule;
    } catch {
      try {
        return (await import(moduleName)) as VertexAIModule;
      } catch {
        return undefined;
      }
    }
  }

  private _patchGenerationPrototype(prototype: PatchablePrototype | undefined): void {
    if (!prototype) return;

    this._patchMethod(
      prototype,
      GENERATE_CONTENT_METHOD_NAME,
      VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
    );
    this._patchMethod(
      prototype,
      GENERATE_CONTENT_STREAM_METHOD_NAME,
      VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
      { isStreamMethod: true },
    );
    this._patchStartChat(prototype);
  }

  private _patchChatPrototype(prototype: PatchablePrototype | undefined): void {
    if (!prototype) return;

    this._patchMethod(
      prototype,
      SEND_MESSAGE_METHOD_NAME,
      VERTEXAI_CHAT_SPAN_NAME,
      { isChatMethod: true },
    );
    this._patchMethod(
      prototype,
      SEND_MESSAGE_STREAM_METHOD_NAME,
      VERTEXAI_CHAT_SPAN_NAME,
      { isChatMethod: true, isStreamMethod: true },
    );
  }

  private _patchStartChat(prototype: PatchablePrototype): void {
    const original = prototype[START_CHAT_METHOD_NAME];
    if (typeof original !== "function") return;
    if ((original as unknown as Record<symbol, unknown>)[PATCHED_BY_RESPAN]) return;

    const instrumentor = this;
    const wrapped = function wrappedStartChat(this: unknown, ...args: unknown[]) {
      const chat = (original as AnyFunction).apply(this, args);
      instrumentor._patchChatInstance(chat);
      return chat;
    };
    Object.defineProperty(wrapped, PATCHED_BY_RESPAN, {
      enumerable: false,
      value: true,
    });

    prototype[START_CHAT_METHOD_NAME] = wrapped;
    VertexAIInstrumentor._sharedState.patchedMethods.push({
      methodName: START_CHAT_METHOD_NAME,
      original: original as AnyFunction,
      prototype,
    });
  }

  private _patchChatInstance(chat: unknown): void {
    if (!chat || typeof chat !== "object") return;
    const sharedState = VertexAIInstrumentor._sharedState;
    if (sharedState.patchedChatInstances.has(chat)) return;

    const prototype = Object.getPrototypeOf(chat) as PatchablePrototype | undefined;
    if (!prototype) return;

    this._patchChatPrototype(prototype);
    sharedState.patchedChatInstances.add(chat);
  }

  private _patchMethod(
    prototype: PatchablePrototype,
    methodName: string,
    spanName: string,
    opts: { isChatMethod?: boolean; isStreamMethod?: boolean } = {},
  ): void {
    const original = prototype[methodName];
    if (typeof original !== "function") return;
    if ((original as unknown as Record<symbol, unknown>)[PATCHED_BY_RESPAN]) return;

    prototype[methodName] = wrapMethod(original as AnyFunction, spanName, opts);
    VertexAIInstrumentor._sharedState.patchedMethods.push({
      methodName,
      original: original as AnyFunction,
      prototype,
    });
  }
}

export { buildGenerateContentAttrs } from "./_otel_emitter.js";
export {
  extractToolCalls,
  extractTools,
  extractUsage,
  formatInput,
  formatOutput,
  normalizeInputMessages,
  requestPayloadFromCall,
} from "./_translator.js";
