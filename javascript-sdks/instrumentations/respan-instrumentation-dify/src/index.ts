import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { emitDifyCall, createDifyCallState, type DifyCallState } from "./_otel_emitter.js";
import type { DifyRequestOptionsLike } from "./_translator.js";

type AnyMethod = (...args: any[]) => any;
type PatchablePrototype = Record<string | symbol, any>;

export interface DifySdkModule {
  HttpClient?: { prototype?: PatchablePrototype };
}

export interface DifyInstrumentorOptions {
  includeContent?: boolean;
  /** Use the exact Dify module instance resolved by the host application. */
  sdkModule?: DifySdkModule;
}

interface PatchedMethod {
  methodName: string;
  original: AnyMethod;
  prototype: PatchablePrototype;
  replacement: AnyMethod;
}

const INTERNAL_REQUEST = Symbol.for("respan.dify.internal_request");
const STREAM_INSTRUMENTED = Symbol.for("respan.dify.stream.instrumented");

const isRecord = (value: unknown): value is Record<string | symbol, any> =>
  Boolean(value && typeof value === "object" && !Array.isArray(value));

const errorFrom = (value: unknown): Error =>
  value instanceof Error ? value : new Error(String(value));

const markedRequest = (request: unknown): Record<string | symbol, any> => ({
  ...(isRecord(request) ? request : {}),
  [INTERNAL_REQUEST]: true,
});

function observeReadable(
  readable: unknown,
  onSuccess: () => void,
  onError: (error: unknown) => void,
): void {
  if (!isRecord(readable) || typeof readable.once !== "function") return;
  let scheduled = false;
  const scheduleSuccess = () => {
    if (scheduled) return;
    scheduled = true;
    setImmediate(onSuccess);
  };
  readable.once("end", scheduleSuccess);
  readable.once("close", scheduleSuccess);
  readable.once("error", onError);
}

function instrumentDifyStream(
  stream: any,
  state: DifyCallState,
): any {
  if (!isRecord(stream) || stream[STREAM_INSTRUMENTED]) return stream;
  const originalAsyncIterator = stream[Symbol.asyncIterator]?.bind(stream);
  if (typeof originalAsyncIterator !== "function") {
    emitDifyCall(state, { response: stream });
    return stream;
  }

  Object.defineProperty(stream, STREAM_INSTRUMENTED, {
    configurable: true,
    enumerable: false,
    value: true,
  });
  const events: unknown[] = [];
  const emitSuccess = () => emitDifyCall(state, { response: stream, streamEvents: events });
  const emitError = (error: unknown) =>
    emitDifyCall(state, { response: stream, streamEvents: events, error: errorFrom(error) });

  stream[Symbol.asyncIterator] = function () {
    const iterator = originalAsyncIterator();
    return {
      async next(...args: any[]) {
        try {
          const result = await iterator.next(...args);
          if (result.done) emitSuccess();
          else events.push(result.value);
          return result;
        } catch (error) {
          emitError(error);
          throw error;
        }
      },
      async return(value?: any) {
        try {
          const result = typeof iterator.return === "function"
            ? await iterator.return(value)
            : { done: true, value };
          emitSuccess();
          return result;
        } catch (error) {
          emitError(error);
          throw error;
        }
      },
      async throw(error?: any) {
        emitError(error);
        if (typeof iterator.throw === "function") return iterator.throw(error);
        throw error;
      },
      [Symbol.asyncIterator]() {
        return this;
      },
    };
  };

  // The documented AsyncIterable and `toText()` paths are captured above.
  // These listeners also finish a span when callers consume `data` or
  // `toReadable()` directly, without putting the stream into flowing mode.
  observeReadable(stream.data, emitSuccess, emitError);
  return stream;
}

function instrumentBinaryStream(stream: any, state: DifyCallState): any {
  if (!isRecord(stream)) {
    emitDifyCall(state, { response: stream });
    return stream;
  }
  observeReadable(
    stream.data,
    () => emitDifyCall(state, { response: stream }),
    (error) => emitDifyCall(state, { response: stream, error: errorFrom(error) }),
  );
  return stream;
}

function instrumentRawResult(response: any, state: DifyCallState): any {
  if (state.request.responseType !== "stream") {
    emitDifyCall(state, { response });
    return response;
  }
  if (!isRecord(response)) {
    emitDifyCall(state, { response });
    return response;
  }
  observeReadable(
    response.data,
    () => emitDifyCall(state, { response }),
    (error) => emitDifyCall(state, { response, error: errorFrom(error) }),
  );
  return response;
}

function wrapRequestMethod(
  original: AnyMethod,
  includeContent: () => boolean,
): AnyMethod {
  return function wrappedRequest(this: unknown, request: DifyRequestOptionsLike) {
    const state = createDifyCallState(request ?? {}, includeContent());
    try {
      const result = original.call(this, markedRequest(request));
      return Promise.resolve(result).then(
        (response) => {
          emitDifyCall(state, { response });
          return response;
        },
        (error) => {
          emitDifyCall(state, { error: errorFrom(error) });
          throw error;
        },
      );
    } catch (error) {
      emitDifyCall(state, { error: errorFrom(error) });
      throw error;
    }
  };
}

function wrapStreamMethod(
  original: AnyMethod,
  includeContent: () => boolean,
): AnyMethod {
  return function wrappedStream(this: unknown, request: DifyRequestOptionsLike) {
    const state = createDifyCallState(request ?? {}, includeContent());
    try {
      return Promise.resolve(original.call(this, markedRequest(request))).then(
        (stream) => instrumentDifyStream(stream, state),
        (error) => {
          emitDifyCall(state, { error: errorFrom(error) });
          throw error;
        },
      );
    } catch (error) {
      emitDifyCall(state, { error: errorFrom(error) });
      throw error;
    }
  };
}

function wrapBinaryStreamMethod(
  original: AnyMethod,
  includeContent: () => boolean,
): AnyMethod {
  return function wrappedBinaryStream(this: unknown, request: DifyRequestOptionsLike) {
    const state = createDifyCallState(request ?? {}, includeContent());
    try {
      return Promise.resolve(original.call(this, markedRequest(request))).then(
        (stream) => instrumentBinaryStream(stream, state),
        (error) => {
          emitDifyCall(state, { error: errorFrom(error) });
          throw error;
        },
      );
    } catch (error) {
      emitDifyCall(state, { error: errorFrom(error) });
      throw error;
    }
  };
}

function wrapRawMethod(
  original: AnyMethod,
  includeContent: () => boolean,
): AnyMethod {
  return function wrappedRaw(this: unknown, request: DifyRequestOptionsLike) {
    if (isRecord(request) && request[INTERNAL_REQUEST]) {
      return original.call(this, request);
    }
    const state = createDifyCallState(request ?? {}, includeContent());
    try {
      return Promise.resolve(original.call(this, request)).then(
        (response) => instrumentRawResult(response, state),
        (error) => {
          emitDifyCall(state, { error: errorFrom(error) });
          throw error;
        },
      );
    } catch (error) {
      emitDifyCall(state, { error: errorFrom(error) });
      throw error;
    }
  };
}

/** Instrument the official `dify-client` Node SDK. */
export class DifyInstrumentor {
  public readonly name = "dify";

  private static readonly sharedState = {
    activeInstances: 0,
    includeContent: true,
    patchedMethods: [] as PatchedMethod[],
  };

  private readonly includeContent: boolean;
  private readonly sdkModule?: DifySdkModule;
  private instrumented = false;

  constructor(options: DifyInstrumentorOptions = {}) {
    this.includeContent = options.includeContent ?? true;
    this.sdkModule = options.sdkModule;
  }

  async activate(): Promise<void> {
    if (this.instrumented) return;
    const module = await this.resolveSdkModule();
    const prototype = module?.HttpClient?.prototype;
    if (!prototype) {
      console.warn("[Respan] Failed to activate Dify instrumentation - dify-client >=3.1.0 not found");
      return;
    }

    const shared = DifyInstrumentor.sharedState;
    if (shared.activeInstances === 0) {
      shared.includeContent = this.includeContent;
    } else if (shared.includeContent !== this.includeContent) {
      console.warn(
        `[Respan] Dify instrumentation is already active with includeContent=${shared.includeContent}; ` +
          "keeping the first active configuration",
      );
    }
    if (shared.patchedMethods.length === 0) {
      const wrappers: Array<[string, (original: AnyMethod, includeContent: () => boolean) => AnyMethod]> = [
        ["request", wrapRequestMethod],
        ["requestStream", wrapStreamMethod],
        ["requestBinaryStream", wrapBinaryStreamMethod],
        ["requestRaw", wrapRawMethod],
      ];
      for (const [methodName, wrapper] of wrappers) {
        const original = prototype[methodName];
        if (typeof original !== "function") continue;
        const replacement = wrapper(
          original,
          () => DifyInstrumentor.sharedState.includeContent,
        );
        shared.patchedMethods.push({ methodName, original, prototype, replacement });
        prototype[methodName] = replacement;
      }
    }

    if (shared.patchedMethods.length === 0) {
      console.warn("[Respan] Failed to activate Dify instrumentation - compatible HttpClient methods not found");
      return;
    }
    shared.activeInstances += 1;
    this.instrumented = true;
  }

  deactivate(): void {
    if (!this.instrumented) return;
    const shared = DifyInstrumentor.sharedState;
    shared.activeInstances = Math.max(0, shared.activeInstances - 1);
    this.instrumented = false;
    if (shared.activeInstances > 0) return;
    for (const patched of shared.patchedMethods) {
      if (patched.prototype[patched.methodName] === patched.replacement) {
        patched.prototype[patched.methodName] = patched.original;
      } else {
        console.warn(
          `[Respan] Dify method HttpClient.${patched.methodName} changed after activation; ` +
            "leaving the later patch installed",
        );
      }
    }
    shared.patchedMethods = [];
  }

  isActive(): boolean {
    return this.instrumented;
  }

  private async resolveSdkModule(): Promise<DifySdkModule | undefined> {
    if (this.sdkModule) return this.sdkModule;
    try {
      const hostRequire = createRequire(`${process.cwd()}/package.json`);
      return (await import(pathToFileURL(hostRequire.resolve("dify-client")).href)) as DifySdkModule;
    } catch {
      try {
        return (await import("dify-client")) as DifySdkModule;
      } catch {
        return undefined;
      }
    }
  }
}

export const DifyAIInstrumentor = DifyInstrumentor;
