/** Native Anthropic SDK adapter using the active OpenTelemetry tracer. */
import { loadAnthropicConstructors } from "./_helpers.js";
import { acquire, release, type Patch } from "./_instrumentation.js";
import type { CaptureOptions } from "./_privacy.js";
export interface AnthropicInstrumentorOptions extends CaptureOptions {
  sdkModule?: { default?: any; Anthropic?: any };
  clientClass?: new (...args: any[]) => any;
}
export class AnthropicInstrumentor {
  readonly name = "anthropic";
  readonly options: AnthropicInstrumentorOptions;
  private patches: Patch[] = [];
  private activation?: Promise<void>;
  private epoch = 0;
  constructor(options: AnthropicInstrumentorOptions = {}) {
    this.options = { ...options };
  }
  isActive(): boolean {
    return this.patches.length > 0;
  }
  activate(): Promise<void> {
    if (this.isActive()) return Promise.resolve();
    if (this.activation) return this.activation;
    const epoch = this.epoch;
    const pending = (async () => {
      const provided =
        this.options.clientClass ??
        this.options.sdkModule?.default ??
        this.options.sdkModule?.Anthropic;
      const constructors = provided
        ? [provided]
        : await loadAnthropicConstructors();
      if (epoch !== this.epoch) return;
      try {
        for (const Constructor of constructors) {
          const client = new Constructor({
            apiKey: "fixture-instrumentation-discovery",
          });
          for (const resource of [client.messages, client.beta?.messages]) {
            if (!resource) continue;
            const prototype = Object.getPrototypeOf(resource);
            for (const [key, op] of [
              ["create", "messages"],
              ["countTokens", "countTokens"],
              ["toolRunner", "agent"],
            ] as const) {
              const patch = acquire(prototype, key, op, this);
              if (patch && !this.patches.includes(patch))
                this.patches.push(patch);
            }
            if (resource.batches) {
              const p = Object.getPrototypeOf(resource.batches);
              for (const [key, op] of [
                ["create", "batches.create"],
                ["results", "batches.results"],
              ] as const) {
                const patch = acquire(p, key, op, this);
                if (patch && !this.patches.includes(patch))
                  this.patches.push(patch);
              }
            }
          }
          if (client.completions) {
            const patch = acquire(
              Object.getPrototypeOf(client.completions),
              "create",
              "completion",
              this,
            );
            if (patch && !this.patches.includes(patch))
              this.patches.push(patch);
          }
        }
      } catch (error) {
        this.deactivate();
        throw error;
      }
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
    for (const patch of this.patches) release(patch, this);
    this.patches = [];
  }
}
