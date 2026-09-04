/**
 * Respan instrumentation for `@helicone/helpers`.
 *
 * `HeliconeManualLogger.sendLog()` is the common successful-call chokepoint for
 * direct sends, builders, request helpers, and stream helpers. This package
 * patches it once and adds narrow outer wrappers solely to capture operations
 * that reject before `sendLog()` is reached.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { context, trace } from "@opentelemetry/api";
import {
  getEntityPath,
  getPropagatedAttributes,
  injectSpan,
} from "@respan/tracing";
import {
  buildHeliconeSpan,
  type HeliconeCapture,
  type HeliconeParentContext,
} from "./_translator.js";

type AnyFunction = (this: any, ...args: any[]) => any;
type AnyRecord = Record<string, any>;

interface HeliconeHelpersModule {
  HeliconeManualLogger?: {
    prototype?: any;
  };
}

interface InvocationState {
  parent?: InvocationState;
  captureReached: boolean;
  callbackDepth: number;
  parentContext: HeliconeParentContext;
  startedAt: number;
  operation: string;
  request: unknown;
  additionalHeaders?: unknown;
  propagatedAttributes?: unknown;
  provider?: unknown;
}

interface InstalledPatch {
  target: Record<string, AnyFunction>;
  method: string;
  original: AnyFunction;
  wrapped: AnyFunction;
}

export interface HeliconeInstrumentorOptions {
  /** Application-resolved module override, primarily useful for tests/loaders. */
  sdkModule?: HeliconeHelpersModule;
  /** Capture request/response bodies. Defaults to true. */
  traceContent?: boolean;
}

const INSTRUMENTATION_NAME = "@respan/instrumentation-helicone";
const packageRequire = createRequire(import.meta.url);
const { version: INSTRUMENTATION_VERSION } = packageRequire("../package.json") as {
  version: string;
};
const instrumentationScope = {
  name: INSTRUMENTATION_NAME,
  version: INSTRUMENTATION_VERSION,
};

const invocationStorage = new AsyncLocalStorage<InvocationState>();
const OUTER_ERROR_METHODS = [
  "logRequest",
  "logStream",
  "logSingleStream",
  "logSingleRequest",
] as const;

const sharedState = {
  activeInstances: 0,
  contentCaptureDisabledInstances: 0,
  patches: [] as InstalledPatch[],
  activation: undefined as Promise<boolean> | undefined,
};

export class HeliconeInstrumentor {
  public readonly name = "helicone";

  private readonly sdkModule?: HeliconeHelpersModule;
  private readonly traceContent: boolean;
  private active = false;
  private activation?: Promise<void>;

  constructor(options: HeliconeInstrumentorOptions = {}) {
    this.sdkModule = options.sdkModule;
    this.traceContent = options.traceContent ?? true;
  }

  async activate(): Promise<void> {
    if (this.active) return;
    if (!this.activation) this.activation = this.acquireActivation();
    const activation = this.activation;
    try {
      await activation;
    } finally {
      if (this.activation === activation) this.activation = undefined;
    }
  }

  deactivate(): void {
    if (!this.active) return;
    const shared = sharedState;
    shared.activeInstances = Math.max(0, shared.activeInstances - 1);
    if (!this.traceContent) {
      shared.contentCaptureDisabledInstances = Math.max(
        0,
        shared.contentCaptureDisabledInstances - 1,
      );
    }
    this.active = false;
    if (shared.activeInstances === 0) restorePatches();
  }

  isActive(): boolean {
    return this.active;
  }

  enable(): void {
    void this.activate().catch(() => undefined);
  }

  disable(): void {
    this.deactivate();
  }

  private async acquireActivation(): Promise<void> {
    const shared = sharedState;
    let installed: boolean;
    if (shared.activation) {
      installed = await shared.activation;
    } else if (shared.patches.length > 0) {
      installed = true;
    } else {
      const activation = installPatches(this.sdkModule);
      shared.activation = activation;
      try {
        installed = await activation;
      } finally {
        if (shared.activation === activation) shared.activation = undefined;
      }
    }

    if (!installed || this.active) return;
    shared.activeInstances += 1;
    if (!this.traceContent) shared.contentCaptureDisabledInstances += 1;
    this.active = true;
  }
}

export async function instrumentHelicone(
  options: HeliconeInstrumentorOptions = {},
): Promise<HeliconeInstrumentor> {
  const instrumentor = new HeliconeInstrumentor(options);
  await instrumentor.activate();
  return instrumentor;
}

export { buildHeliconeSpan } from "./_translator.js";
export default HeliconeInstrumentor;

async function installPatches(
  suppliedModule?: HeliconeHelpersModule,
): Promise<boolean> {
  const helpers = suppliedModule ?? await importHeliconeHelpers();
  const prototype = helpers.HeliconeManualLogger?.prototype;
  if (!prototype || typeof prototype.sendLog !== "function") {
    throw new Error(
      "@respan/instrumentation-helicone requires @helicone/helpers " +
      ">=1.8.3 <2 with HeliconeManualLogger.sendLog().",
    );
  }

  try {
    patchSendLog(prototype);
    patchLogBuilder(prototype);
    for (const method of OUTER_ERROR_METHODS) patchOuterErrorPath(prototype, method);
  } catch (error) {
    restorePatches();
    throw error;
  }
  return sharedState.patches.length > 0;
}

function patchSendLog(prototype: Record<string, AnyFunction>): void {
  const original = prototype.sendLog;
  if (typeof original !== "function") return;

  const wrapped: AnyFunction = async function respanHeliconeSendLog(
    this: unknown,
    request: unknown,
    response: unknown,
    options: unknown,
  ) {
    if (sharedState.activeInstances === 0) {
      return await original.apply(this, arguments as any);
    }

    const state = invocationStorage.getStore();
    const callbackOwnedSend = (state?.callbackDepth ?? 0) > 0;
    if (!callbackOwnedSend) markCaptureReached(state);
    const parent = callbackOwnedSend
      ? activeParentContext()
      : state?.parentContext ?? activeParentContext();
    const propagatedAttributes =
      callbackOwnedSend
        ? snapshotPropagatedAttributes()
        : state?.propagatedAttributes ?? snapshotPropagatedAttributes();
    let thrown: unknown;
    try {
      return await original.apply(this, arguments as any);
    } catch (error) {
      thrown = error;
      throw error;
    } finally {
      emitCapture({
        request,
        response,
        options,
        loggerHeaders: asRecord(this).headers,
        error: thrown,
        parent,
        fallbackOperation: callbackOwnedSend ? undefined : state?.operation,
        propagatedAttributes,
        traceContent: shouldCaptureContent(),
        instrumentationScope,
      });
    }
  };

  installMethodPatch(prototype, "sendLog", original, wrapped);
}

function patchLogBuilder(prototype: Record<string, AnyFunction>): void {
  const original = prototype.logBuilder;
  if (typeof original !== "function") return;

  const wrapped: AnyFunction = function respanHeliconeLogBuilder(
    this: unknown,
    request: unknown,
    additionalHeaders?: unknown,
  ) {
    const builder = original.apply(this, arguments as any);
    if (sharedState.activeInstances === 0 || !builder) return builder;

    const builderRecord = asRecord(builder);
    const originalSendLog = builderRecord.sendLog;
    if (typeof originalSendLog !== "function") return builder;
    const parentState = invocationStorage.getStore();
    const snapshot = {
      parent: parentState,
      parentContext: activeParentContext(),
      propagatedAttributes: snapshotPropagatedAttributes(),
      request,
      additionalHeaders,
      loggerHeaders: asRecord(this).headers,
    };

    builderRecord.sendLog = async function respanHeliconeBuilderSendLog(
      this: unknown,
      ...args: unknown[]
    ) {
      if (sharedState.activeInstances === 0) {
        return await originalSendLog.apply(this, args);
      }
      const state: InvocationState = {
        parent: snapshot.parent,
        captureReached: false,
        callbackDepth: 0,
        parentContext: snapshot.parentContext,
        propagatedAttributes: snapshot.propagatedAttributes,
        startedAt: Date.now(),
        operation: "logBuilder",
        request: snapshot.request,
        additionalHeaders: snapshot.additionalHeaders,
      };
      return await invocationStorage.run(state, async () => {
        try {
          return await originalSendLog.apply(this, args);
        } catch (error) {
          if (!state.captureReached) {
            markCaptureReached(state);
            emitCapture({
              request: state.request,
              options: {
                startTime: state.startedAt,
                endTime: Date.now(),
                status: 500,
                additionalHeaders: state.additionalHeaders,
              },
              loggerHeaders: snapshot.loggerHeaders,
              error,
              parent: state.parentContext,
              fallbackOperation: state.operation,
              propagatedAttributes: state.propagatedAttributes,
              traceContent: shouldCaptureContent(),
              instrumentationScope,
            });
          }
          throw error;
        }
      });
    };
    return builder;
  };

  installMethodPatch(prototype, "logBuilder", original, wrapped);
}

function patchOuterErrorPath(
  prototype: Record<string, AnyFunction>,
  method: typeof OUTER_ERROR_METHODS[number],
): void {
  const original = prototype[method];
  if (typeof original !== "function") return;

  const wrapped: AnyFunction = async function respanHeliconeOuterCall(
    this: unknown,
    ...args: unknown[]
  ) {
    if (sharedState.activeInstances === 0) {
      return await original.apply(this, args);
    }

    const parentState = invocationStorage.getStore();
    const state: InvocationState = {
      parent: parentState,
      captureReached: false,
      callbackDepth: 0,
      parentContext: activeParentContext(),
      propagatedAttributes: snapshotPropagatedAttributes(),
      startedAt: Date.now(),
      operation: method,
      request: args[0],
      ...outerCorrelation(method, args),
    };

    return await invocationStorage.run(state, async () => {
      const operationArgs = wrapUserCallback(method, args, state);
      try {
        return await original.apply(this, operationArgs);
      } catch (error) {
        if (!state.captureReached) {
          markCaptureReached(state);
          emitCapture({
            request: state.request,
            response: undefined,
            options: {
              startTime: state.startedAt,
              endTime: Date.now(),
              status: 500,
              additionalHeaders: state.additionalHeaders,
              provider: state.provider,
            },
            loggerHeaders: asRecord(this).headers,
            error,
            parent: state.parentContext,
            fallbackOperation: state.operation,
            propagatedAttributes: state.propagatedAttributes,
            traceContent: shouldCaptureContent(),
            instrumentationScope,
          });
        }
        throw error;
      }
    });
  };

  installMethodPatch(prototype, method, original, wrapped);
}

function wrapUserCallback(
  method: typeof OUTER_ERROR_METHODS[number],
  args: unknown[],
  state: InvocationState,
): unknown[] {
  if (method !== "logRequest" && method !== "logStream") return args;
  const callback = args[1];
  if (typeof callback !== "function") return args;
  const wrapped = async function respanHeliconeUserCallback(
    this: unknown,
    ...callbackArgs: unknown[]
  ) {
    state.callbackDepth += 1;
    try {
      return await callback.apply(this, callbackArgs);
    } finally {
      state.callbackDepth = Math.max(0, state.callbackDepth - 1);
    }
  };
  const operationArgs = [...args];
  operationArgs[1] = wrapped;
  return operationArgs;
}

function outerCorrelation(
  method: typeof OUTER_ERROR_METHODS[number],
  args: unknown[],
): Pick<InvocationState, "additionalHeaders" | "provider"> {
  if (method === "logRequest") {
    return { additionalHeaders: args[2], provider: args[3] };
  }
  if (method === "logStream" || method === "logSingleStream") {
    return { additionalHeaders: args[2] };
  }
  const options = asRecord(args[2]);
  return { additionalHeaders: options.additionalHeaders };
}

function installMethodPatch(
  target: Record<string, AnyFunction>,
  method: string,
  original: AnyFunction,
  wrapped: AnyFunction,
): void {
  target[method] = wrapped;
  sharedState.patches.push({ target, method, original, wrapped });
}

function restorePatches(): void {
  const shared = sharedState;
  for (const patch of [...shared.patches].reverse()) {
    if (patch.target[patch.method] === patch.wrapped) {
      patch.target[patch.method] = patch.original;
    }
  }
  shared.patches = [];
  shared.contentCaptureDisabledInstances = 0;
}

function markCaptureReached(state: InvocationState | undefined): void {
  if (state) state.captureReached = true;
}

function shouldCaptureContent(): boolean {
  const shared = sharedState;
  return shared.activeInstances > 0 && shared.contentCaptureDisabledInstances === 0;
}

function emitCapture(capture: HeliconeCapture): void {
  try {
    injectSpan(buildHeliconeSpan(capture));
  } catch {
    // Instrumentation must never alter Helicone application behavior.
  }
}

function activeParentContext(): HeliconeParentContext {
  const activeContext = context.active();
  const spanContext = trace.getSpan(activeContext)?.spanContext();
  let entityPath: string | undefined;
  try {
    entityPath = getEntityPath(activeContext);
  } catch {
    entityPath = undefined;
  }
  return {
    traceId: spanContext?.traceId,
    parentId: spanContext?.spanId,
    entityPath,
  };
}

function snapshotPropagatedAttributes(): unknown {
  const propagated = asRecord(getPropagatedAttributes());
  if (Object.keys(propagated).length === 0) return {};
  return {
    ...propagated,
    ...(isPlainRecord(propagated.metadata)
      ? { metadata: { ...propagated.metadata } }
      : {}),
  };
}

async function importHeliconeHelpers(): Promise<HeliconeHelpersModule> {
  try {
    const hostRequire = createRequire(`${process.cwd()}/package.json`);
    const resolved = hostRequire.resolve("@helicone/helpers");
    return await import(pathToFileURL(resolved).href) as HeliconeHelpersModule;
  } catch {
    return await import("@helicone/helpers") as HeliconeHelpersModule;
  }
}

function asRecord(value: unknown): AnyRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as AnyRecord
    : {};
}

function isPlainRecord(value: unknown): value is AnyRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
