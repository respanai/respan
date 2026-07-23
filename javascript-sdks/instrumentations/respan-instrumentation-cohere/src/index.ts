/**
 * Respan instrumentation plugin for the Cohere TypeScript SDK.
 *
 * The current Cohere SDK exposes both v1 and v2 clients. This package patches
 * the SDK methods directly so chat, streaming chat, generation, embeddings,
 * and rerank all emit spans in the Respan span contract.
 */

import {
  COHERE_PATCHED,
} from "./_constants.js";
import {
  captureStreamEvent,
  createStreamState,
  emitSpanRecord,
  startSpanRecord,
  streamResultFromState,
  type CohereApiVersion,
  type CohereOperation,
  type OperationConfig,
  type SpanRecord,
} from "./_mapping.js";

export {
  applySuccessAttributes,
  buildStartAttributes,
  createStreamState,
} from "./_mapping.js";
export {
  isCohereSpan,
  normalizeCohereAttrs,
  normalizeCohereSpan,
} from "./_translator.js";

export interface CohereInstrumentorOptions {
  sdkModule?: any;
  traceContent?: boolean;
}

interface PatchedMethod {
  target: any;
  method: string;
  original: (...args: any[]) => any;
}

const V1_METHODS: Array<[CohereOperation, boolean]> = [
  ["chat", false],
  ["chatStream", true],
  ["generate", false],
  ["generateStream", true],
  ["embed", false],
  ["rerank", false],
];

const V2_METHODS: Array<[CohereOperation, boolean]> = [
  ["chat", false],
  ["chatStream", true],
  ["embed", false],
  ["rerank", false],
];

function isThenable(value: any): boolean {
  return value !== null && value !== undefined && typeof value.then === "function";
}

function findPrototypeWithMethod(instance: any, method: string): any | null {
  let target = instance;
  while (target) {
    const descriptor = Object.getOwnPropertyDescriptor(target, method);
    if (descriptor && typeof descriptor.value === "function") return target;
    target = Object.getPrototypeOf(target);
  }
  return null;
}

function patchMethod(
  target: any,
  method: string,
  config: OperationConfig,
  traceContent: boolean,
): PatchedMethod | null {
  if (!target || typeof target[method] !== "function") return null;
  if ((target[method] as any)[COHERE_PATCHED]) return null;

  const original = target[method];
  const wrapped = function wrappedCohereMethod(this: any, ...args: any[]) {
    const request = args[0] ?? {};
    const record = startSpanRecord(config, request, traceContent);

    let result: any;
    try {
      result = original.apply(this, args);
    } catch (error) {
      emitSpanRecord(config, record, undefined, error);
      throw error;
    }

    if (config.streaming) {
      return instrumentStreamingResult(result, config, record);
    }

    instrumentPromiseResult(result, config, record);
    return result;
  };

  Object.defineProperty(wrapped, COHERE_PATCHED, {
    configurable: true,
    enumerable: false,
    value: true,
  });
  target[method] = wrapped;
  return { target, method, original };
}

function instrumentPromiseResult(
  result: any,
  config: OperationConfig,
  record: SpanRecord,
): void {
  if (!isThenable(result)) {
    emitSpanRecord(config, record, result);
    return;
  }

  let emitted = false;
  Promise.resolve(result).then(
    (value) => {
      if (emitted) return;
      emitted = true;
      emitSpanRecord(config, record, value);
    },
    (error) => {
      if (emitted) return;
      emitted = true;
      emitSpanRecord(config, record, undefined, error);
    },
  );
}

function instrumentStreamingResult(
  result: any,
  config: OperationConfig,
  record: SpanRecord,
): any {
  if (!isThenable(result)) {
    return wrapStream(result, config, record);
  }

  let emitted = false;
  return Promise.resolve(result).then(
    (stream: any) => wrapStream(stream, config, record, () => {
      emitted = true;
    }),
    (error: unknown) => {
      if (!emitted) {
        emitted = true;
        emitSpanRecord(config, record, undefined, error);
      }
      throw error;
    },
  );
}

function wrapStream(
  stream: any,
  config: OperationConfig,
  record: SpanRecord,
  markEmitted?: () => void,
): any {
  if (!stream || typeof stream[Symbol.asyncIterator] !== "function") {
    emitSpanRecord(config, record, stream);
    markEmitted?.();
    return stream;
  }

  const originalAsyncIterator = stream[Symbol.asyncIterator].bind(stream);
  const state = createStreamState();
  let emitted = false;

  const emitFinal = (error?: unknown) => {
    if (emitted) return;
    emitted = true;
    markEmitted?.();
    if (error !== undefined) {
      emitSpanRecord(config, record, undefined, error);
      return;
    }
    emitSpanRecord(config, record, streamResultFromState(config, state));
  };

  stream[Symbol.asyncIterator] = function patchedCohereAsyncIterator() {
    const iterator = originalAsyncIterator();
    return {
      async next(...args: any[]) {
        try {
          const item = await iterator.next(...args);
          if (item.done) {
            emitFinal();
          } else {
            captureStreamEvent(state, item.value);
          }
          return item;
        } catch (error) {
          emitFinal(error);
          throw error;
        }
      },
      async return(value?: any) {
        try {
          const result = typeof iterator.return === "function"
            ? await iterator.return(value)
            : { done: true, value };
          emitFinal();
          return result;
        } catch (error) {
          emitFinal(error);
          throw error;
        }
      },
      async throw(error?: any) {
        emitFinal(error);
        if (typeof iterator.throw === "function") {
          return iterator.throw(error);
        }
        throw error;
      },
      [Symbol.asyncIterator]() {
        return this;
      },
    };
  };

  return stream;
}

function createConfig(
  operation: CohereOperation,
  apiVersion: CohereApiVersion,
  streaming: boolean,
): OperationConfig {
  return { operation, apiVersion, streaming };
}

export class CohereInstrumentor {
  public readonly name = "cohere";
  private static readonly _sharedState = {
    activeInstances: 0,
    patchedMethods: [] as PatchedMethod[],
  };

  private readonly _sdkModule?: any;
  private readonly _traceContent: boolean;
  private _isInstrumented = false;

  constructor(options: CohereInstrumentorOptions = {}) {
    this._sdkModule = options.sdkModule;
    this._traceContent = options.traceContent ?? true;
  }

  async activate(): Promise<void> {
    if (this._isInstrumented) return;

    const sdkModule = this._sdkModule ?? await import("cohere-ai").catch(() => null);
    if (!sdkModule?.CohereClient) {
      console.warn(
        "[Respan] Failed to activate Cohere instrumentation - cohere-ai not found",
      );
      return;
    }

    const shared = CohereInstrumentor._sharedState;
    if (shared.activeInstances === 0) {
      try {
        this._patchSdkModule(sdkModule);
      } catch (error) {
        console.warn("[Respan] Failed to activate Cohere instrumentation:", error);
        this._restorePatchedMethods();
        return;
      }
    }

    shared.activeInstances += 1;
    this._isInstrumented = true;
  }

  deactivate(): void {
    if (!this._isInstrumented) return;

    const shared = CohereInstrumentor._sharedState;
    shared.activeInstances = Math.max(0, shared.activeInstances - 1);
    this._isInstrumented = false;

    if (shared.activeInstances === 0) {
      this._restorePatchedMethods();
    }
  }

  private _patchSdkModule(sdkModule: any): void {
    const client = new sdkModule.CohereClient({ token: "respan-placeholder" });
    for (const [operation, streaming] of V1_METHODS) {
      this._patchInstanceMethod(client, operation, createConfig(operation, "v1", streaming));
    }

    const v2Client = client.v2;
    if (v2Client) {
      for (const [operation, streaming] of V2_METHODS) {
        this._patchInstanceMethod(v2Client, operation, createConfig(operation, "v2", streaming));
      }
    }
  }

  private _patchInstanceMethod(
    instance: any,
    operation: CohereOperation,
    config: OperationConfig,
  ): void {
    const target = findPrototypeWithMethod(instance, operation);
    const patched = patchMethod(target, operation, config, this._traceContent);
    if (patched) {
      CohereInstrumentor._sharedState.patchedMethods.push(patched);
    }
  }

  private _restorePatchedMethods(): void {
    const shared = CohereInstrumentor._sharedState;
    for (const patched of shared.patchedMethods.reverse()) {
      if (patched.target?.[patched.method]?.[COHERE_PATCHED]) {
        patched.target[patched.method] = patched.original;
      }
    }
    shared.patchedMethods = [];
  }
}
