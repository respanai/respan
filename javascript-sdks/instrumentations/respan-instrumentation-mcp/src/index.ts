/** Native MCP v1 and v2 protocol instrumentation. */
import {
  context,
  createContextKey,
  trace,
  SpanKind,
  SpanStatusCode,
  type Span,
} from "@opentelemetry/api";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  MCP_INSTRUMENTATION_NAME,
  MCP_INSTRUMENTATION_PACKAGE,
} from "./_constants.js";
import {
  environmentDenied,
  guardSpan,
  isGuarded,
  own,
  suppressed,
  type CapturePolicy,
} from "./_privacy.js";
import {
  clientMethodInput,
  errorOutput,
  findLastFunctionIndex,
  resolveClientEntityName,
  resolveClientSpanName,
  safeJson,
  toSerializableValue,
} from "./_utils.js";

type AnyFunction = (...args: any[]) => any;
type Prototype = Record<string, any>;
export interface MCPClientModule {
  Client?: { prototype?: object };
}
export interface MCPServerModule {
  McpServer?: { prototype?: object };
  Server?: { prototype?: object };
}
export interface MCPInstrumentorOptions {
  clientModule?: MCPClientModule;
  serverModule?: MCPServerModule;
  /** Additional independently loaded SDK copies, including the split v2 packages. */
  clientModules?: MCPClientModule[];
  serverModules?: MCPServerModule[];
  captureClientOperations?: boolean;
  captureServerTools?: boolean;
  traceContent?: boolean;
  recordInputs?: boolean;
  recordOutputs?: boolean;
}
interface Owner {
  policy: CapturePolicy;
}
interface Slot {
  prototype: Prototype;
  key: string;
  original: AnyFunction;
  wrapper: AnyFunction;
  descriptor?: PropertyDescriptor;
  owners: Set<Owner>;
  transports: Set<object>;
}
const incomingRequestId = createContextKey("respan.mcp.native-request-id");
const slots = new WeakMap<object, Map<string, Slot>>();
const callbackSlots = new WeakMap<AnyFunction, Slot>();
const transportOriginals = new WeakMap<
  object,
  { original: AnyFunction; wrapper: AnyFunction; slots: Set<Slot> }
>();
function policyFor(slot: Slot): CapturePolicy | undefined {
  if (!slot.owners.size) return undefined;
  const policy: CapturePolicy = {
    traceContent: true,
    recordInputs: true,
    recordOutputs: true,
  };
  for (const owner of slot.owners) {
    policy.traceContent &&= owner.policy.traceContent;
    policy.recordInputs &&= owner.policy.recordInputs;
    policy.recordOutputs &&= owner.policy.recordOutputs;
  }
  return policy;
}
function safely(fn: () => void): void {
  try {
    fn();
  } catch {
    /* Instrumentation cannot change SDK behavior. */
  }
}
function traceOperation(
  policy: CapturePolicy,
  side: "client" | "server",
  method: string,
  params: unknown,
  run: () => any,
  requestId?: unknown,
  signal?: unknown,
): any {
  if (suppressed()) return run();
  const tool = method === "tools/call";
  const nativeName = own(params, "name");
  const entity =
    tool && typeof nativeName === "string" && nativeName
      ? nativeName
      : `mcp.${method}`;
  const parent = context.active();
  const span = trace.getTracer(MCP_INSTRUMENTATION_PACKAGE, "1.0.0").startSpan(
    tool ? `mcp.tool.${entity}` : `mcp.${method}`,
    {
      kind: side === "client" ? SpanKind.CLIENT : SpanKind.SERVER,
      attributes: {
        [RespanSpanAttributes.RESPAN_LOG_METHOD]: "ts_tracing",
        [RespanSpanAttributes.RESPAN_LOG_TYPE]: tool ? "tool" : "task",
        [SpanAttributes.TRACELOOP_ENTITY_NAME]: entity,
        [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
        "mcp.method": method,
        "mcp.operation": `${side}.${method}`,
        ...(typeof requestId === "number" || typeof requestId === "string"
          ? { "mcp.request.id": requestId }
          : {}),
      },
    },
    parent,
  );
  const state = guardSpan(span, policy);
  const active = trace.setSpan(parent, span);
  safely(() => {
    if (state.input())
      span.setAttribute(
        SpanAttributes.TRACELOOP_ENTITY_INPUT,
        safeJson(
          tool
            ? { name: nativeName, arguments: own(params, "arguments") }
            : { method, params },
        ),
      );
  });
  let ended = false;
  let removeAbortListener: (() => void) | undefined;
  const finish = (result: unknown, error?: unknown) => {
    if (ended) return;
    ended = true;
    removeAbortListener?.();
    safely(() => {
      if (error !== undefined) {
        if (state.allowed()) {
          span.recordException(
            error instanceof Error ? error : errorOutput(error).message,
          );
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: errorOutput(error).message,
          });
        } else span.setStatus({ code: SpanStatusCode.ERROR });
      } else {
        if (state.output())
          span.setAttribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            safeJson(result),
          );
        span.setStatus({
          code:
            own(result, "isError") === true
              ? SpanStatusCode.ERROR
              : SpanStatusCode.OK,
        });
      }
    });
    span.end();
  };
  if (signal instanceof AbortSignal) {
    const onAbort = () =>
      finish(undefined, signal.reason ?? new Error("MCP request cancelled"));
    signal.addEventListener("abort", onAbort, { once: true });
    removeAbortListener = () => signal.removeEventListener("abort", onAbort);
    if (signal.aborted) onAbort();
  }
  try {
    const result = context.with(active, run);
    if (result instanceof Promise) {
      // Observe settlement without replacing the native promise or its helpers.
      void result.then(
        (value) => {
          finish(value);
        },
        (error) => {
          finish(undefined, error);
        },
      );
    } else finish(result);
    return result;
  } catch (error) {
    finish(undefined, error);
    throw error;
  }
}
function wrapCallback(
  handler: AnyFunction,
  slot: Slot,
  side: "client" | "server",
): AnyFunction {
  if (callbackSlots.has(handler)) return handler;
  const wrapped = function (this: unknown, ...args: unknown[]) {
    const policy = policyFor(slot);
    const request = args[0];
    const method = own(request, "method");
    if (!policy || typeof method !== "string") return handler.apply(this, args);
    return traceOperation(
      policy,
      side,
      method,
      own(request, "params"),
      () => handler.apply(this, args),
      own(request, "id") ??
        own(args[1], "requestId") ??
        own(own(args[1], "mcpReq"), "id") ??
        context.active().getValue(incomingRequestId),
      own(args[1], "signal") ?? own(own(args[1], "mcpReq"), "signal"),
    );
  };
  callbackSlots.set(wrapped, slot);
  return wrapped;
}
function wrapExistingHandlers(
  instance: unknown,
  slot: Slot,
  side: "client" | "server",
): void {
  for (const key of ["_requestHandlers", "_notificationHandlers"]) {
    const handlers = own(instance, key);
    if (!(handlers instanceof Map)) continue;
    for (const [method, handler] of Map.prototype.entries.call(handlers)) {
      if (typeof handler === "function" && !callbackSlots.has(handler))
        Map.prototype.set.call(
          handlers,
          method,
          wrapCallback(handler, slot, side),
        );
    }
  }
}
function observeTransport(transport: unknown, slot: Slot): void {
  if (!transport || typeof transport !== "object") return;
  const existing = transportOriginals.get(transport);
  if (existing) {
    existing.slots.add(slot);
    slot.transports.add(transport);
    return;
  }
  const send = (transport as Prototype).send;
  if (typeof send !== "function") return;
  const wrapper = function (this: unknown, ...args: unknown[]) {
    const message = args[0];
    const id = own(message, "id");
    const method = own(message, "method");
    if (
      typeof method === "string" &&
      (typeof id === "string" || typeof id === "number") &&
      isGuarded(trace.getSpan(context.active()))
    )
      safely(() =>
        trace.getSpan(context.active())?.setAttribute("mcp.request.id", id),
      );
    return send.apply(this, args);
  };
  try {
    (transport as Prototype).send = wrapper;
    transportOriginals.set(transport, {
      original: send,
      wrapper,
      slots: new Set([slot]),
    });
    slot.transports.add(transport);
  } catch {
    /* Immutable transport. */
  }
}
function acquire(
  prototype: object,
  key: string,
  owner: Owner,
  side: "client" | "server",
  mode:
    | "request"
    | "notification"
    | "handler"
    | "connect"
    | "legacyTool"
    | "dispatch",
): Slot | undefined {
  let map = slots.get(prototype);
  if (!map) {
    map = new Map();
    slots.set(prototype, map);
  }
  const target = prototype as Prototype;
  let slot = map.get(key);
  if (
    !slot ||
    (!slot.owners.size &&
      target[key] !== slot.original &&
      target[key] !== slot.wrapper)
  ) {
    const original = target[key];
    if (typeof original !== "function") return undefined;
    slot = {
      prototype: target,
      key,
      original,
      wrapper: original,
      descriptor: Object.getOwnPropertyDescriptor(target, key),
      owners: new Set(),
      transports: new Set(),
    };
    const current = slot;
    current.wrapper = function (this: unknown, ...args: unknown[]) {
      const policy = policyFor(current);
      if (!policy) return original.apply(this, args);
      if (mode === "dispatch")
        return context.with(
          context.active().setValue(incomingRequestId, own(args[0], "id")),
          () => original.apply(this, args),
        );
      if (mode === "handler") {
        const index = findLastFunctionIndex(args);
        if (index >= 0) {
          const known = new Set<AnyFunction>();
          for (const key of ["_requestHandlers", "_notificationHandlers"]) {
            const handlers = own(this, key);
            if (handlers instanceof Map)
              for (const value of Map.prototype.values.call(handlers))
                known.add(value);
          }
          const patched = [...args];
          patched[index] = wrapCallback(
            args[index] as AnyFunction,
            current,
            side,
          );
          const result = original.apply(this, patched);
          // Native SDK validation wraps callbacks again; remember that outer native wrapper.
          for (const key of ["_requestHandlers", "_notificationHandlers"]) {
            const handlers = own(this, key);
            if (handlers instanceof Map)
              for (const value of Map.prototype.values.call(handlers))
                if (typeof value === "function" && !known.has(value))
                  callbackSlots.set(value, current);
          }
          return result;
        }
        return original.apply(this, args);
      }
      if (mode === "connect") {
        wrapExistingHandlers(this, current, side);
        observeTransport(args[0], current);
        return traceOperation(policy, side, "connect", undefined, () =>
          original.apply(this, args),
        );
      }
      if (mode === "legacyTool") {
        const index = findLastFunctionIndex(args);
        if (index < 0) return original.apply(this, args);
        const callback = args[index] as AnyFunction;
        const patched = [...args];
        patched[index] = function (this: unknown, ...handlerArgs: unknown[]) {
          const latest = policyFor(current);
          if (!latest) return callback.apply(this, handlerArgs);
          const name = typeof args[0] === "string" ? args[0] : "tool";
          return traceOperation(
            latest,
            "server",
            "tools/call",
            { name, arguments: handlerArgs[0] },
            () => callback.apply(this, handlerArgs),
          );
        };
        return original.apply(this, patched);
      }
      observeTransport(own(this, "_transport"), current);
      const method = own(args[0], "method");
      if (typeof method !== "string") return original.apply(this, args);
      return traceOperation(policy, side, method, own(args[0], "params"), () =>
        original.apply(this, args),
      );
    };
    map.set(key, current);
  }
  if (!slot.owners.size) {
    try {
      Object.defineProperty(target, key, {
        configurable: true,
        writable: true,
        value: slot.wrapper,
      });
    } catch {
      return undefined;
    }
  }
  slot.owners.add(owner);
  return slot;
}
function release(slot: Slot, owner: Owner): void {
  slot.owners.delete(owner);
  if (slot.owners.size) return;
  for (const transport of slot.transports) {
    const patch = transportOriginals.get(transport);
    patch?.slots.delete(slot);
    if (patch && !patch.slots.size) {
      if ((transport as Prototype).send === patch.wrapper)
        (transport as Prototype).send = patch.original;
      transportOriginals.delete(transport);
    }
  }
  slot.transports.clear();
  if (slot.prototype[slot.key] !== slot.wrapper) return;
  if (slot.descriptor)
    Object.defineProperty(slot.prototype, slot.key, slot.descriptor);
  else delete slot.prototype[slot.key];
}
async function optionalImport(name: string): Promise<any | undefined> {
  try {
    return await import(name);
  } catch {
    return undefined;
  }
}
export class MCPInstrumentor implements Owner {
  public readonly name = MCP_INSTRUMENTATION_NAME;
  public readonly policy: CapturePolicy;
  private readonly options: MCPInstrumentorOptions;
  private readonly owned = new Set<Slot>();
  private active = false;
  private generation = 0;
  private activation?: Promise<void>;
  constructor(options: MCPInstrumentorOptions = {}) {
    this.options = options;
    this.policy = Object.freeze({
      traceContent: options.traceContent !== false && !environmentDenied(),
      recordInputs: options.recordInputs !== false,
      recordOutputs: options.recordOutputs !== false,
    });
  }
  activate(): Promise<void> {
    if (this.activation) return this.activation;
    if (this.active) return Promise.resolve();
    const generation = this.generation;
    const pending = this.activateGeneration(generation);
    this.activation = pending;
    void pending.finally(() => {
      if (this.activation === pending) this.activation = undefined;
    });
    return pending;
  }
  private async activateGeneration(generation: number): Promise<void> {
    await Promise.resolve();
    const clients =
      this.options.clientModules ??
      (this.options.clientModule
        ? [this.options.clientModule]
        : (
            await Promise.all([
              optionalImport("@modelcontextprotocol/sdk/client/index.js"),
              optionalImport("@modelcontextprotocol/client"),
            ])
          ).filter(Boolean));
    const servers =
      this.options.serverModules ??
      (this.options.serverModule
        ? [this.options.serverModule]
        : [
            Object.assign(
              {},
              await optionalImport("@modelcontextprotocol/sdk/server/index.js"),
              await optionalImport("@modelcontextprotocol/sdk/server/mcp.js"),
            ),
            await optionalImport("@modelcontextprotocol/server"),
          ].filter(Boolean));
    if (generation !== this.generation) return;
    const attach = (
      prototype: object,
      side: "client" | "server",
      key: string,
      mode: Parameters<typeof acquire>[4],
    ) => {
      const slot = acquire(prototype, key, this, side, mode);
      if (slot) this.owned.add(slot);
    };
    if (this.options.captureClientOperations !== false)
      for (const module of clients)
        if (module?.Client?.prototype) {
          const p = module.Client.prototype;
          attach(p, "client", "_onrequest", "dispatch");
          attach(p, "client", "connect", "connect");
          attach(p, "client", "request", "request");
          attach(p, "client", "notification", "notification");
          attach(p, "client", "setRequestHandler", "handler");
          attach(p, "client", "setNotificationHandler", "handler");
        }
    if (this.options.captureServerTools !== false)
      for (const module of servers) {
        if (module?.Server?.prototype) {
          const p = module.Server.prototype;
          attach(p, "server", "_onrequest", "dispatch");
          attach(p, "server", "connect", "connect");
          attach(p, "server", "request", "request");
          attach(p, "server", "notification", "notification");
          attach(p, "server", "setRequestHandler", "handler");
          attach(p, "server", "setNotificationHandler", "handler");
        } else if (module?.McpServer?.prototype) {
          attach(
            module.McpServer.prototype,
            "server",
            "registerTool",
            "legacyTool",
          );
          attach(module.McpServer.prototype, "server", "tool", "legacyTool");
        }
      }
    this.active = this.owned.size > 0;
  }
  deactivate(): void {
    this.generation++;
    for (const slot of this.owned) release(slot, this);
    this.owned.clear();
    this.active = false;
    this.activation = undefined;
  }
  isActive(): boolean {
    return this.active;
  }
}
// Preserve existing public utility exports.
export {
  clientMethodInput,
  resolveClientEntityName,
  resolveClientSpanName,
  safeJson,
  toSerializableValue,
};
