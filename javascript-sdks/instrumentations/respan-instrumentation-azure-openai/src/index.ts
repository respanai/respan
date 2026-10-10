import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import { context } from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import { observeRequest, type Operation } from "./_request.js";

export interface AzureOpenAIInstrumentorOptions {
  /** The imported openai module or AzureOpenAI constructor; resolves both module formats by default. */
  openAIModule?: Record<string, any>;
  /** Optional legacy @azure/openai module. Version 2 is a types companion. */
  azureOpenAIModule?: Record<string, any>;
  traceContent?: boolean;
  recordInputs?: boolean;
  recordOutputs?: boolean;
  exceptionLogger?: (error: Error) => void;
}
interface Patch {
  target: any;
  method: string;
  original: any;
  wrapped: any;
  owners: Set<AzureOpenAIInstrumentor>;
}

/** Observes real Azure SDK calls without replacing native promise or stream objects. */
export class AzureOpenAIInstrumentor {
  readonly name = "azure-openai";
  private static patches = new Map<any, Map<string, Patch>>();
  private readonly patches = new Set<Patch>();
  private active = false;
  private activation?: Promise<void>;
  private readonly options: AzureOpenAIInstrumentorOptions;
  constructor(options: AzureOpenAIInstrumentorOptions = {}) {
    this.options = { ...options };
  }

  isActive(): boolean {
    return this.active;
  }
  activate(): Promise<void> {
    if (this.activation) return this.activation.then(() => this.activate());
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
    const modern = this.options.openAIModule
      ? [this.options.openAIModule]
      : await loadModules("openai", "index.mjs");
    const legacy = this.options.azureOpenAIModule
      ? [this.options.azureOpenAIModule]
      : await loadModules("@azure/openai");
    if (!this.active) return;
    for (const module of modern) {
      const Azure = module.AzureOpenAI ?? module.default?.AzureOpenAI ?? module;
      if (typeof Azure !== "function") continue;
      for (const [target, operation] of [
        [Azure.Chat?.Completions?.prototype, "chat"],
        [Azure.Completions?.prototype, "completion"],
        [Azure.Responses?.prototype, "responses"],
        [Azure.Embeddings?.prototype, "embedding"],
      ] as const)
        this.patch(target, "create", operation, false, Azure);
    }
    for (const module of legacy) {
      const Client = module.OpenAIClient ?? module.default?.OpenAIClient;
      for (const [method, operation] of [
        ["getChatCompletions", "chat"],
        ["getCompletions", "completion"],
        ["getEmbeddings", "embedding"],
        ["listChatCompletions", "chat"],
        ["listCompletions", "completion"],
        ["streamChatCompletions", "chat"],
        ["streamCompletions", "completion"],
      ] as const)
        this.patch(Client?.prototype, method, operation, true);
    }
  }
  private patch(
    target: any,
    method: string,
    operation: Operation,
    legacy: boolean,
    Azure?: any,
  ): void {
    if (typeof target?.[method] !== "function") return;
    let methods = AzureOpenAIInstrumentor.patches.get(target);
    if (!methods)
      AzureOpenAIInstrumentor.patches.set(target, (methods = new Map()));
    let patch = methods.get(method);
    // A foreign replacement stays in charge until a future activation.
    if (patch && target[method] !== patch.wrapped) return;
    if (!patch) {
      const original = target[method];
      const owners = new Set<AzureOpenAIInstrumentor>();
      const wrapped = function (this: any, ...args: any[]) {
        const owner = owners.values().next().value;
        const client = legacy ? this : this?._client;
        if (
          !owner ||
          isTracingSuppressed(context.active()) ||
          (!legacy && !(client instanceof Azure))
        )
          return original.apply(this, args);
        return observeRequest(
          original,
          this,
          args,
          operation,
          {
            ...owner.options,
            traceContent: [...owners].every(
              (item) => item.options.traceContent !== false,
            ),
            recordInputs: [...owners].every(
              (item) => item.options.recordInputs !== false,
            ),
            recordOutputs: [...owners].every(
              (item) => item.options.recordOutputs !== false,
            ),
          },
          legacy,
        );
      };
      patch = { target, method, original, wrapped, owners };
      target[method] = wrapped;
      methods.set(method, patch);
    }
    patch.owners.add(this);
    this.patches.add(patch);
  }
  deactivate(): void {
    this.active = false;
    for (const patch of this.patches) {
      patch.owners.delete(this);
      if (patch.owners.size) continue;
      if (patch.target[patch.method] === patch.wrapped)
        patch.target[patch.method] = patch.original;
      const methods = AzureOpenAIInstrumentor.patches.get(patch.target);
      methods?.delete(patch.method);
      if (!methods?.size) AzureOpenAIInstrumentor.patches.delete(patch.target);
    }
    this.patches.clear();
  }
}
async function loadModules(name: string, esm?: string): Promise<any[]> {
  let require = createRequire(`${process.cwd()}/package.json`);
  let resolved: string;
  try {
    resolved = require.resolve(name);
  } catch {
    require = createRequire(import.meta.url);
    try {
      resolved = require.resolve(name);
    } catch {
      return [];
    }
  }
  const modules: any[] = [];
  try {
    modules.push(require(resolved));
  } catch {
    modules.push(await import(pathToFileURL(resolved).href));
  }
  const entry = esm ? join(dirname(resolved), esm) : undefined;
  if (entry && existsSync(entry))
    modules.push(await import(pathToFileURL(entry).href));
  return modules;
}
