/** Observes released LlamaIndex methods and native tool execution events. */
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import {
  observe,
  ToolObserver,
  type Operation,
  type Options,
} from "./_emitter.js";
import { data } from "./_privacy.js";

export interface LlamaIndexInstrumentorOptions extends Options {
  /** Explicit module(s), including separately installed provider/core copies. */
  llamaIndexModule?: Record<string, any> | Record<string, any>[];
}
interface Patch {
  target: any;
  method: string;
  descriptor: PropertyDescriptor;
  wrapped: Function;
  owners: Set<LlamaIndexInstrumentor>;
  event: boolean;
}
const patches = new Map<any, Map<string, Patch>>();

export class LlamaIndexInstrumentor {
  readonly name = "llama-index";
  private active = false;
  private activation?: Promise<void>;
  private readonly targets = new Set<Patch>();
  private readonly toolObserver: ToolObserver;
  constructor(private readonly options: LlamaIndexInstrumentorOptions = {}) {
    this.options = { ...options };
    this.toolObserver = new ToolObserver(this.options);
    this.toolObserver.onDrain = () => {
      if (!this.active) this.deactivate();
    };
  }
  isActive(): boolean {
    return this.active;
  }
  activate(): Promise<void> {
    if (this.activation)
      return this.active
        ? this.activation
        : this.activation.then(() => this.activate());
    if (this.active) return Promise.resolve();
    this.active = true;
    this.activation = this.install()
      .catch((error) => {
        this.deactivate();
        throw error;
      })
      .finally(() => {
        this.activation = undefined;
      });
    return this.activation;
  }
  private async install(): Promise<void> {
    const explicit = this.options.llamaIndexModule;
    const modules = explicit
      ? Array.isArray(explicit)
        ? explicit
        : [explicit]
      : await nativeModules();
    if (!this.active) return;
    for (const module of modules) {
      for (const [name, value] of Object.entries(module)) {
        if (typeof value !== "function" || !value.prototype) continue;
        const target = value.prototype;
        if (name === "CallbackManager")
          this.patch(
            target,
            "dispatchEvent",
            "task",
            name,
            true,
            module.getEventCaller,
          );
        for (const method of Object.getOwnPropertyNames(target)) {
          const operation = classify(name, method);
          if (operation)
            this.patch(target, method, operation, `${name}.${method}`);
        }
      }
    }
  }
  private patch(
    target: any,
    method: string,
    operation: Operation,
    name: string,
    event = false,
    getEventCaller?: () => unknown,
  ): void {
    const descriptor = Object.getOwnPropertyDescriptor(target, method);
    if (
      !descriptor ||
      typeof descriptor.value !== "function" ||
      descriptor.writable === false
    )
      return;
    let methods = patches.get(target);
    if (!methods) patches.set(target, (methods = new Map()));
    let patch = methods.get(method);
    if (!patch) {
      const original = descriptor.value;
      const owners = new Set<LlamaIndexInstrumentor>();
      const wrapped = function (this: any, ...args: any[]) {
        const startsEvent = event && /-(start|call)$/.test(args[0]);
        const owner = startsEvent
          ? [...owners].find((owner) => owner.active)
          : owners.values().next().value;
        if (!owner) return original.apply(this, args);
        const effective = {
          ...owner.options,
          traceContent: [...owners].every(
            (o) => o.options.traceContent !== false,
          ),
          recordInputs: [...owners].every(
            (o) => o.options.recordInputs !== false,
          ),
          recordOutputs: [...owners].every(
            (o) => o.options.recordOutputs !== false,
          ),
        };
        if (event) {
          try {
            for (const listener of startsEvent ? [owner] : owners)
              listener.toolObserver.event(
                args[0],
                args[1],
                this,
                getEventCaller?.(),
                effective,
              );
          } catch {}
          return original.apply(this, args);
        }
        return observe(original, this, args, operation, name, effective);
      };
      patch = { target, method, descriptor, wrapped, owners, event };
      Object.defineProperty(target, method, { ...descriptor, value: wrapped });
      methods.set(method, patch);
    }
    patch.owners.add(this);
    this.targets.add(patch);
  }
  deactivate(): void {
    this.active = false;
    for (const patch of this.targets) {
      if (patch.event && this.toolObserver.hasPending()) continue;
      this.targets.delete(patch);
      patch.owners.delete(this);
      if (patch.owners.size) continue;
      if (data(patch.target, patch.method) === patch.wrapped)
        Object.defineProperty(patch.target, patch.method, patch.descriptor);
      patches.get(patch.target)?.delete(patch.method);
      if (!patches.get(patch.target)?.size) patches.delete(patch.target);
    }
  }
}
function classify(name: string, method: string): Operation | undefined {
  if (method === "getOpenAIEmbedding") return "embedding";
  if (method === "complete" && name === "BaseLLM") return "text";
  if (method === "chat") {
    if (/Agent|Runner/.test(name)) return "agent";
    if (/ChatEngine/.test(name)) return "workflow";
    if (/OpenAI|LLM/.test(name)) return "chat";
  }
  if (method === "query" && /QueryEngine/.test(name)) return "workflow";
  if (
    ["retrieve", "synthesize", "getNodesFromDocuments", "splitText"].includes(
      method,
    )
  )
    return "task";
  return undefined;
}
async function nativeModules(): Promise<Record<string, any>[]> {
  const modules: Record<string, any>[] = [];
  const require = createRequire(import.meta.url);
  const specs = [
    "llamaindex",
    "@llamaindex/openai",
    "@llamaindex/core/global",
    "@llamaindex/core/llms",
    "@llamaindex/core/agent",
    "@llamaindex/core/query-engine",
    "@llamaindex/core/retriever",
    "@llamaindex/core/response-synthesizers",
    "@llamaindex/core/node-parser",
  ];
  const roots = [require];
  for (const spec of ["llamaindex", "@llamaindex/openai"]) {
    try {
      roots.push(createRequire(require.resolve(spec)));
    } catch {}
  }
  const seen = new Set<string>();
  for (const resolver of roots)
    for (const spec of specs) {
      try {
        // Resolve ESM with import condition: require entrypoints may contain a separate SDK copy.
        const location = resolver
          .resolve(spec)
          .replace(/\/index\.cjs$/, "/index.js")
          .replace(/\/dist\/cjs\/index\.js$/, "/dist/index.js");
        if (seen.has(location)) continue;
        seen.add(location);
        modules.push(await import(pathToFileURL(location).href));
      } catch {
        /* Optional provider and core modules need not be installed. */
      }
    }
  if (!modules.length)
    throw new Error(
      "Install llamaindex >=0.12.0 before activating LlamaIndexInstrumentor.",
    );
  return modules;
}
