import { context, createContextKey } from "@opentelemetry/api";
import { data, refresh, type CaptureOptions } from "./_privacy.js";
import { start, type Operation, type Session } from "./_span_emitter.js";
import { observeIterable, observePromise } from "./_streaming.js";
type Fn = (...args: any[]) => any;
export interface Owner {
  options: CaptureOptions;
}
export interface Patch {
  target: any;
  key: PropertyKey;
  original: Fn;
  wrapped: Fn;
  descriptor?: PropertyDescriptor;
  owners: Set<Owner>;
}
const registryKey = Symbol.for(
  "respan.instrumentation.anthropic.native.registry.v2",
);
const globals = globalThis as any;
const registry: WeakMap<object, Map<PropertyKey, Patch>> = (globals[
  registryKey
] ??= new WeakMap());
const drainContext = createContextKey("respan.anthropic.admittedRunner");
const activeAgents: Set<Session> = (globals[
  Symbol.for("respan.anthropic.activeAgents")
] ??= new Set());
const pendingReleases: Set<Patch> = (globals[
  Symbol.for("respan.anthropic.pendingReleases")
] ??= new Set());
function restore(patch: Patch): void {
  if (patch.owners.size) return;
  if (patch.target[patch.key] === patch.wrapped) {
    if (patch.descriptor)
      Object.defineProperty(patch.target, patch.key, patch.descriptor);
    else delete patch.target[patch.key];
  }
  const map = registry.get(patch.target);
  if (map?.get(patch.key) === patch) map.delete(patch.key);
  pendingReleases.delete(patch);
}
function lease(session: Session): void {
  activeAgents.add(session);
  session.ctx = session.ctx.setValue(drainContext, session);
  const finish = session.finish;
  let closed = false;
  session.finish = (...args) => {
    finish(...args);
    if (closed) return;
    closed = true;
    activeAgents.delete(session);
    if (!activeAgents.size)
      for (const patch of [...pendingReleases]) restore(patch);
  };
}
function options(owners: Set<Owner>): CaptureOptions {
  const out: CaptureOptions = {};
  for (const owner of owners) {
    if (owner.options.traceContent === false) out.traceContent = false;
    if (owner.options.recordInputs === false) out.recordInputs = false;
    if (owner.options.recordOutputs === false) out.recordOutputs = false;
  }
  return out;
}
const toolWrappers = new WeakMap<Fn, { original: Fn; session: Session }>();
function runTool(tool: any, original: Fn, session: Session): Fn {
  return function (this: any, ...args: any[]) {
    const toolUse = data(args[1], "toolUse") ?? data(args[1], "toolUseBlock");
    const child = start(
      "tool",
      { name: data(tool, "name"), arguments: args[0] },
      optionsFrom(session),
      session.ctx,
      data(toolUse, "id"),
    );
    let result: any;
    try {
      result = context.with(child?.ctx ?? session.ctx, () =>
        Reflect.apply(original, this, args),
      );
    } catch (error) {
      child?.finish(undefined, error);
      throw error;
    }
    if (result instanceof Promise)
      void result
        .then(
          (value) => child?.finish(value),
          (error) => child?.finish(undefined, error),
        )
        .catch(() => {});
    else child?.finish(result);
    return result;
  };
}
function optionsFrom(session: Session): CaptureOptions {
  return {
    recordInputs: session.policy.inputs,
    recordOutputs: session.policy.outputs,
  };
}
function prepareTools(params: any, session: Session): any {
  if (!session.policy.inputs || !params || typeof params !== "object")
    return params;
  const definitions = data(params, "tools");
  if (!Array.isArray(definitions)) return params;
  // Preserve descriptors and functions; callbacks must be present before the native helper clones its tools.
  const out = Object.create(
    Object.getPrototypeOf(params),
    Object.getOwnPropertyDescriptors(params),
  );
  const converted = definitions.map((tool) => {
    const existing = data(tool, "run");
    if (typeof existing !== "function") return tool;
    const known = toolWrappers.get(existing);
    if (known?.session === session) return tool;
    const original = known?.original ?? existing;
    const clone = Object.create(
      Object.getPrototypeOf(tool),
      Object.getOwnPropertyDescriptors(tool),
    );
    const wrapped = runTool(tool, original, session);
    toolWrappers.set(wrapped, { original, session });
    Object.defineProperty(clone, "run", {
      value: wrapped,
      writable: true,
      configurable: true,
      enumerable: true,
    });
    return clone;
  });
  Object.defineProperty(out, "tools", {
    value: converted,
    writable: true,
    configurable: true,
    enumerable: true,
  });
  return out;
}
function runner(
  original: Fn,
  instance: any,
  args: any[],
  config: CaptureOptions,
): any {
  const session = start("agent", args[0], config);
  if (!session) return Reflect.apply(original, instance, args);
  lease(session);
  let value: any;
  try {
    value = context.with(session.ctx, () =>
      Reflect.apply(original, instance, [
        prepareTools(args[0], session),
        ...args.slice(1),
      ]),
    );
  } catch (error) {
    session.finish(undefined, error);
    throw error;
  }
  // done() is a passive native completion promise and does not start iteration.
  const completed = value.done?.();
  if (completed instanceof Promise)
    void completed
      .then(
        (v) => session.finish(v),
        (e) => session.finish(undefined, e),
      )
      .catch(() => {});
  observeIterable(value, session, "agent");
  for (const key of [
    "runUntilDone",
    "generateToolResponse",
    "done",
    "setMessagesParams",
    "addTools",
  ]) {
    const method = value[key];
    if (typeof method !== "function") continue;
    Object.defineProperty(value, key, {
      configurable: true,
      writable: true,
      value: function (this: any, ...callArgs: any[]) {
        if (key === "addTools")
          callArgs = callArgs.map(
            (tool) => prepareTools({ tools: [tool] }, session).tools[0],
          );
        if (key === "setMessagesParams") {
          const first = callArgs[0];
          callArgs[0] =
            typeof first === "function"
              ? function (this: any, params: any) {
                  return prepareTools(
                    Reflect.apply(first, this, [params]),
                    session,
                  );
                }
              : prepareTools(first, session);
        }
        let result: any;
        try {
          result = context.with(session.ctx, () =>
            Reflect.apply(method, this, callArgs),
          );
        } catch (error) {
          session.finish(undefined, error);
          throw error;
        }
        if (
          (key === "runUntilDone" || key === "done") &&
          result instanceof Promise
        )
          void result
            .then(
              (v) => session.finish(v),
              (e) => session.finish(undefined, e),
            )
            .catch(() => {});
        return result;
      },
    });
  }
  return value;
}
export function acquire(
  target: any,
  key: PropertyKey,
  operation: Operation,
  owner: Owner,
): Patch | undefined {
  const original = target?.[key];
  if (typeof original !== "function") return;
  let map = registry.get(target);
  if (!map) {
    map = new Map();
    registry.set(target, map);
  }
  let patch = map.get(key);
  if (!patch || target[key] !== patch.wrapped) {
    patch = {
      target,
      key,
      original,
      wrapped: undefined as unknown as Fn,
      descriptor: Object.getOwnPropertyDescriptor(target, key),
      owners: new Set(),
    };
    const state = patch;
    patch.wrapped = function (this: any, ...args: any[]) {
      const admitted = context.active().getValue(drainContext) as
        Session | undefined;
      if (!state.owners.size && !(admitted && activeAgents.has(admitted)))
        return Reflect.apply(state.original, this, args);
      const config =
        admitted && activeAgents.has(admitted)
          ? optionsFrom(admitted)
          : options(state.owners);
      if (operation === "agent")
        return runner(state.original, this, args, config);
      const session = start(operation, args[0], config);
      if (!session) return Reflect.apply(state.original, this, args);
      try {
        const result = context.with(session.ctx, () =>
          Reflect.apply(state.original, this, args),
        );
        return observePromise(
          result,
          session,
          data(args[0], "stream") === true,
          operation === "batches.results",
        );
      } catch (error) {
        session.finish(undefined, error);
        throw error;
      }
    };
    Object.defineProperty(target, key, {
      value: patch.wrapped,
      writable: true,
      configurable: true,
    });
    map.set(key, patch);
  }
  patch.owners.add(owner);
  return patch;
}
export function release(patch: Patch, owner: Owner): void {
  patch.owners.delete(owner);
  if (patch.owners.size) return;
  if (activeAgents.size) {
    pendingReleases.add(patch);
    return;
  }
  restore(patch);
}
