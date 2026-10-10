/** Native AWS Bedrock instrumentation; transport and SDK return values stay native. */
import { context } from "@opentelemetry/api";
import {
  AWS_BEDROCK_INSTRUMENTATION_NAME,
  SUPPORTED_OPERATIONS,
  STREAMING_OPERATIONS,
} from "./_constants.js";
import { startBedrockSession, type BedrockSession } from "./_otel_emitter.js";
import { data, type CaptureOptions } from "./_privacy.js";

type AnyFunction = (...args: any[]) => any;
type PatchablePrototype = { send?: AnyFunction };
type PatchableClientConstructor = { prototype?: PatchablePrototype };
export interface AWSBedrockRuntimeModule {
  BedrockRuntimeClient?: PatchableClientConstructor;
}
export interface AWSBedrockInstrumentorOptions extends CaptureOptions {
  sdkModule?: AWSBedrockRuntimeModule;
  clientClass?: PatchableClientConstructor;
}
interface Patch {
  prototype: PatchablePrototype;
  original: AnyFunction;
  wrapped: AnyFunction;
  descriptor?: PropertyDescriptor;
  owners: Set<AWSBedrockInstrumentor>;
}
const registryKey = Symbol.for("respan.instrumentation.awsBedrock.registry.v2");
const globalRegistry = globalThis as typeof globalThis & {
  [registryKey]?: WeakMap<object, Patch>;
};
const registry = (globalRegistry[registryKey] ??= new WeakMap<object, Patch>());
function operation(command: any): string | undefined {
  const proto =
    command && typeof command === "object"
      ? Object.getPrototypeOf(command)
      : undefined;
  const ctor = data(proto, "constructor");
  const name = data(ctor, "name");
  if (typeof name !== "string" || !name.endsWith("Command")) return undefined;
  const op = name.slice(0, -7);
  return SUPPORTED_OPERATIONS.has(op) ? op : undefined;
}
function wrapStream(stream: any, session: BedrockSession): boolean {
  const original = stream?.[Symbol.asyncIterator];
  if (typeof original !== "function") return false;
  const observed = new WeakSet<object>();
  const wrapped = function (this: any) {
    const iterator = Reflect.apply(original, this, []) as any;
    if (observed.has(iterator)) return iterator;
    observed.add(iterator);
    for (const key of ["next", "return", "throw"] as const) {
      const method = iterator[key];
      if (typeof method !== "function") continue;
      Object.defineProperty(iterator, key, {
        configurable: true,
        writable: true,
        value: function (this: unknown, ...args: unknown[]) {
          let result: any;
          try {
            result = context.with(session.ctx, () =>
              Reflect.apply(method, this, args),
            );
          } catch (error) {
            session.finish(undefined, error);
            throw error;
          }
          const observe = (value: any) => {
            if (key === "next" && !value.done) session.event(value.value);
            if (value.done || key === "return" || key === "throw")
              session.finish();
          };
          if (result && typeof result.then === "function")
            void result
              .then(observe, (error: unknown) =>
                session.finish(undefined, error),
              )
              .catch(() => {});
          else observe(result);
          return result;
        },
      });
    }
    return iterator;
  };
  try {
    Object.defineProperty(stream, Symbol.asyncIterator, {
      value: wrapped,
      writable: true,
      configurable: true,
    });
    return true;
  } catch {
    return false;
  }
}
function complete(
  session: BedrockSession,
  op: string,
  response: any,
  error?: unknown,
): void {
  if (error !== undefined) {
    session.finish(undefined, error);
    return;
  }
  session.response(response);
  if (STREAMING_OPERATIONS.has(op)) {
    const stream = data(response, "stream") ?? data(response, "body");
    if (stream && wrapStream(stream, session)) return;
  }
  session.finish(response);
}
function instrument(patch: Patch, instance: unknown, args: unknown[]): unknown {
  const op = operation(args[0]);
  if (!op || !patch.owners.size)
    return Reflect.apply(patch.original, instance, args);
  const options: CaptureOptions = {};
  for (const owner of patch.owners) {
    if (owner.options.traceContent === false) options.traceContent = false;
    if (owner.options.recordInputs === false) options.recordInputs = false;
    if (owner.options.recordOutputs === false) options.recordOutputs = false;
  }
  let session: BedrockSession | undefined;
  try {
    session = startBedrockSession(op, data(args[0], "input"), options);
  } catch {
    /* Fail open for telemetry only. */
  }
  if (!session) return Reflect.apply(patch.original, instance, args);
  const callArgs = [...args];
  const cbIndex =
    typeof callArgs.at(-1) === "function" ? callArgs.length - 1 : -1;
  if (cbIndex >= 0) {
    const callback = callArgs[cbIndex] as AnyFunction;
    callArgs[cbIndex] = function (this: unknown, ...values: unknown[]) {
      complete(session!, op, values[1], values[0] ?? undefined);
      return Reflect.apply(callback, this, values);
    };
  }
  try {
    const result = context.with(session.ctx, () =>
      Reflect.apply(patch.original, instance, callArgs),
    );
    if (cbIndex < 0) {
      if (result && typeof result.then === "function")
        void result
          .then(
            (response: any) => complete(session!, op, response),
            (error: unknown) => session!.finish(undefined, error),
          )
          .catch(() => {});
      else complete(session, op, result);
    }
    return result;
  } catch (error) {
    session.finish(undefined, error);
    throw error;
  }
}
export class AWSBedrockInstrumentor {
  public readonly name = AWS_BEDROCK_INSTRUMENTATION_NAME;
  readonly options: AWSBedrockInstrumentorOptions;
  private patch?: Patch;
  private activation?: Promise<void>;
  private epoch = 0;
  constructor(options: AWSBedrockInstrumentorOptions = {}) {
    this.options = { ...options };
  }
  isActive(): boolean {
    return !!this.patch;
  }
  activate(): Promise<void> {
    if (this.patch) return Promise.resolve();
    if (this.activation) return this.activation;
    const epoch = this.epoch;
    const pending = (async () => {
      const ctor =
        this.options.clientClass ??
        this.options.sdkModule?.BedrockRuntimeClient ??
        (await import("@aws-sdk/client-bedrock-runtime")).BedrockRuntimeClient;
      if (epoch !== this.epoch) return;
      const prototype = ctor?.prototype;
      if (!prototype || typeof prototype.send !== "function")
        throw new Error(
          "AWSBedrockInstrumentor requires BedrockRuntimeClient.prototype.send.",
        );
      let patch = registry.get(prototype);
      if (!patch || prototype.send !== patch.wrapped) {
        const original = prototype.send;
        patch = {
          prototype,
          original,
          wrapped: undefined as unknown as AnyFunction,
          descriptor: Object.getOwnPropertyDescriptor(prototype, "send"),
          owners: new Set(),
        };
        const state = patch;
        patch.wrapped = function (this: unknown, ...args: unknown[]) {
          return instrument(state, this, args);
        };
        Object.defineProperty(prototype, "send", {
          value: patch.wrapped,
          writable: true,
          configurable: true,
        });
        registry.set(prototype, patch);
      }
      patch.owners.add(this);
      this.patch = patch;
    })();
    this.activation = pending;
    void pending
      .finally(() => {
        if (this.activation === pending) this.activation = undefined;
      })
      .catch(() => {});
    return pending;
  }
  deactivate(): void {
    this.epoch++;
    this.activation = undefined;
    const patch = this.patch;
    this.patch = undefined;
    if (!patch) return;
    patch.owners.delete(this);
    if (!patch.owners.size) {
      if (patch.prototype.send === patch.wrapped) {
        if (patch.descriptor)
          Object.defineProperty(patch.prototype, "send", patch.descriptor);
        else delete patch.prototype.send;
      }
      if (registry.get(patch.prototype) === patch)
        registry.delete(patch.prototype);
    }
  }
}
export { buildBedrockAttrs, emitBedrockSpan } from "./_otel_emitter.js";
export {
  parseBedrockRequest,
  parseBedrockResponse,
  parseBedrockStreamResponse,
} from "./_translator.js";
