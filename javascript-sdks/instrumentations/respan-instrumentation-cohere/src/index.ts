/** Native Cohere adapter: preserve SDK objects and observe their lifecycle. */
import { context } from "@opentelemetry/api";
import { ATTR_HTTP_RESPONSE_STATUS_CODE } from "@opentelemetry/semantic-conventions";
import {
  capture,
  data,
  internalCall,
  type CaptureOptions,
} from "./_privacy.js";
import {
  startSpanRecord,
  emitSpanRecord,
  createStreamState,
  captureStreamEvent,
  streamResultFromState,
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
export interface CohereInstrumentorOptions extends CaptureOptions {
  sdkModule?: any;
}
type Patch = {
  target: any;
  method: string;
  original: Function;
  wrapped: Function;
  owners: Set<CohereInstrumentor>;
};
const patches = new WeakMap<object, Map<string, Patch>>();
const methods = [
  "chat",
  "chatStream",
  "generate",
  "generateStream",
  "embed",
  "rerank",
] as const;
function findTarget(instance: any, method: string): any {
  for (let target = instance; target; target = Object.getPrototypeOf(target)) {
    if (
      typeof Object.getOwnPropertyDescriptor(target, method)?.value ===
      "function"
    )
      return target;
  }
}
function observe(
  result: any,
  config: OperationConfig,
  record: SpanRecord,
): void {
  const resolved = (value: any) => {
    try {
      if (config.streaming) wrapStream(value, config, record);
      else emitSpanRecord(config, record, value);
    } catch {
      emitSpanRecord(config, record, undefined);
    }
  };
  const rejected = (error: unknown) => {
    emitSpanRecord(config, record, undefined, error);
  };
  try {
    const rawPromise = data(result, "innerPromise");
    if (rawPromise && typeof rawPromise.then === "function")
      rawPromise.then(
        (value: any) => {
          const status = data(data(value, "rawResponse"), "status");
          if (
            record.admitted &&
            typeof status === "number" &&
            Number.isInteger(status) &&
            status >= 100 &&
            status <= 599
          )
            record.span.setAttribute(ATTR_HTTP_RESPONSE_STATUS_CODE, status);
        },
        () => undefined,
      );
    if (result && typeof result.then === "function") {
      // An observer branch leaves HttpResponsePromise and withRawResponse() intact.
      result.then(resolved, rejected);
    } else resolved(result);
  } catch {
    /* Telemetry must not affect native results. */
  }
}
function wrapStream(
  stream: any,
  config: OperationConfig,
  record: SpanRecord,
): void {
  if (!stream || typeof stream[Symbol.asyncIterator] !== "function") {
    emitSpanRecord(config, record, stream);
    return;
  }
  const original = stream[Symbol.asyncIterator];
  const state = createStreamState();
  const finish = (error?: unknown) =>
    emitSpanRecord(
      config,
      record,
      error === undefined ? streamResultFromState(config, state) : undefined,
      error,
    );
  const wrapped = function (this: any, ...args: any[]) {
    const iterator = original.apply(this, args);
    for (const method of ["next", "return", "throw"] as const) {
      const originalMethod = iterator[method];
      if (typeof originalMethod !== "function") continue;
      Object.defineProperty(iterator, method, {
        configurable: true,
        value: function (this: any, ...callArgs: any[]) {
          let pending: any;
          try {
            pending = originalMethod.apply(this, callArgs);
          } catch (error) {
            finish(error);
            throw error;
          }
          Promise.resolve(pending).then(
            (item: any) => {
              try {
                if (method === "throw") finish(callArgs[0]);
                else if (method === "return" || item.done) finish();
                else captureStreamEvent(state, item.value, record.policy);
              } catch {
                /* Keep native iterator results unchanged. */
              }
            },
            (error) => finish(error),
          );
          return pending;
        },
      });
    }
    return iterator;
  };
  try {
    Object.defineProperty(stream, Symbol.asyncIterator, {
      configurable: true,
      writable: true,
      value: wrapped,
    });
  } catch {
    finish();
  }
  const controller = stream.controller;
  if (controller?.signal)
    controller.signal.addEventListener("abort", () => finish(), { once: true });
}
function acquire(
  target: any,
  method: string,
  config: OperationConfig,
  owner: CohereInstrumentor,
): Patch | undefined {
  if (!target) return;
  let map = patches.get(target);
  if (!map) patches.set(target, (map = new Map()));
  let patch = map.get(method);
  if (patch && target[method] === patch.wrapped) {
    patch.owners.add(owner);
    return patch;
  }
  const original = target[method];
  patch = {
    target,
    method,
    original,
    wrapped: undefined as any,
    owners: new Set([owner]),
  };
  const owned = patch;
  owned.wrapped = function (this: any, ...args: any[]) {
    const ctx = context.active();
    if (ctx.getValue(internalCall)) return original.apply(this, args);
    const owners = [...owned.owners].filter((item) => item.isActive());
    if (!owners.length) return original.apply(this, args);
    const policy = capture(
      {
        traceContent: owners.every(
          (item) => item.options.traceContent !== false,
        ),
        recordInputs: owners.every(
          (item) => item.options.recordInputs !== false,
        ),
        recordOutputs: owners.every(
          (item) => item.options.recordOutputs !== false,
        ),
      },
      ctx,
    );
    if (!policy.emit) return original.apply(this, args);
    const record = startSpanRecord(config, args[0], policy, ctx);
    let result: any;
    try {
      result = context.with(record.context.setValue(internalCall, true), () =>
        original.apply(this, args),
      );
    } catch (error) {
      emitSpanRecord(config, record, undefined, error);
      throw error;
    }
    observe(result, config, record);
    return result;
  };
  Object.defineProperty(target, method, {
    ...Object.getOwnPropertyDescriptor(target, method),
    value: owned.wrapped,
  });
  map.set(method, owned);
  return owned;
}
export class CohereInstrumentor {
  readonly name = "cohere";
  readonly options: Readonly<CohereInstrumentorOptions>;
  private active = false;
  private generation = 0;
  private activation?: Promise<void>;
  private owned: Patch[] = [];
  constructor(options: CohereInstrumentorOptions = {}) {
    this.options = Object.freeze({ ...options });
  }
  activate(): Promise<void> {
    if (this.active) return Promise.resolve();
    if (this.activation) return this.activation;
    const generation = this.generation;
    const pending = this.install(generation);
    this.activation = pending;
    void pending.finally(() => {
      if (this.activation === pending) this.activation = undefined;
    });
    return pending;
  }
  private async install(generation: number): Promise<void> {
    const sdk =
      this.options.sdkModule ??
      (await import("cohere-ai").catch(() => undefined));
    if (generation !== this.generation || !sdk?.CohereClient) return;
    try {
      const client = new sdk.CohereClient({ token: "respan-placeholder" });
      const targets: Array<[any, "v1" | "v2"]> = [[client, "v1"]];
      if (client.v2) targets.push([client.v2, "v2"]);
      if (sdk.CohereClientV2) {
        const v2 = new sdk.CohereClientV2({ token: "respan-placeholder" });
        if (v2.clientV2) targets.push([v2.clientV2, "v2"]);
      }
      for (const [target, apiVersion] of targets)
        for (const method of [
          ...methods,
          ...(apiVersion === "v2" ? ["parse" as const] : []),
        ]) {
          const operation = method;
          const patch = acquire(
            findTarget(target, method),
            method,
            { operation, apiVersion, streaming: method.endsWith("Stream") },
            this,
          );
          if (patch && !this.owned.includes(patch)) this.owned.push(patch);
        }
      this.active = this.owned.length > 0;
    } catch {
      this.deactivate();
    }
  }
  deactivate(): void {
    this.generation++;
    this.activation = undefined;
    this.active = false;
    for (const patch of this.owned) {
      patch.owners.delete(this);
      if (patch.owners.size) continue;
      if (patch.target[patch.method] === patch.wrapped)
        Object.defineProperty(patch.target, patch.method, {
          ...Object.getOwnPropertyDescriptor(patch.target, patch.method),
          value: patch.original,
        });
      if (patches.get(patch.target)?.get(patch.method) === patch)
        patches.get(patch.target)!.delete(patch.method);
    }
    this.owned = [];
  }
  isActive(): boolean {
    return this.active;
  }
}
