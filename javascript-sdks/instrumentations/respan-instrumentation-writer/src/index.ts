/** Native observation of the official Writer SDK's lazy Stainless boundaries. */
import {
  registerSpanTransformer,
  type SpanTransformerRegistration,
} from "@respan/tracing";
import { patchWriterMethod, type PatchedMethodTarget } from "./_streaming.js";
import { observeAncestor, type CaptureOptions } from "./_privacy.js";
export interface WriterInstrumentorOptions extends CaptureOptions {
  sdkModule?: any;
}
const patches = new Set<PatchedMethodTarget>();
const transformer = { onStart: observeAncestor, onEnd: observeAncestor };
export class WriterInstrumentor {
  public readonly name = "writer";
  private active = false;
  private generation = 0;
  private pending?: Promise<void>;
  private owned: PatchedMethodTarget[] = [];
  private registration?: SpanTransformerRegistration;
  private readonly options: WriterInstrumentorOptions;
  constructor(options: WriterInstrumentorOptions = {}) {
    this.options = { ...options };
  }
  activate(): Promise<void> {
    if (this.active) return Promise.resolve();
    if (this.pending) return this.pending;
    const generation = ++this.generation;
    const operation = this.activateGeneration(generation);
    this.pending = operation;
    void operation
      .finally(() => {
        if (this.pending === operation) this.pending = undefined;
      })
      .catch(() => {});
    return operation;
  }
  private async activateGeneration(generation: number): Promise<void> {
    const sdk = this.options.sdkModule ?? (await import("writer-sdk"));
    if (generation !== this.generation) return;
    const Writer = sdk.default ?? sdk.Writer;
    if (typeof Writer !== "function")
      throw new Error("Writer constructor not found");
    const client = new Writer({ apiKey: "respan-placeholder" });
    try {
      for (const [target, method, type] of [
        [Object.getPrototypeOf(client.chat), "chat", "chat"],
        [Object.getPrototypeOf(client.chat), "parse", "chat"],
        [Object.getPrototypeOf(client.completions), "create", "completion"],
      ] as const) {
        if (typeof target?.[method] !== "function") continue;
        let patch = [...patches].find(
          (p) =>
            p.target === target &&
            p.methodName === method &&
            target[method] === p.wrappedMethod,
        );
        if (!patch) {
          patch =
            patchWriterMethod(target, method, type, new Set()) ?? undefined;
          if (patch) patches.add(patch);
        }
        if (patch) {
          patch.owners.add(this.options);
          this.owned.push(patch);
        }
      }
      // Raw OTel providers can use the instrumentor too; the registry is optional.
      try {
        this.registration = registerSpanTransformer(
          "@respan/instrumentation-writer",
          transformer,
        );
      } catch {
        /* no Respan host */
      }
      this.active = this.owned.length > 0;
    } catch (error) {
      this.release();
      throw error;
    }
  }
  deactivate(): void {
    this.generation += 1;
    this.pending = undefined;
    if (!this.active) return;
    this.active = false;
    this.release();
  }
  isActive(): boolean {
    return this.active;
  }
  private release(): void {
    this.registration?.unregister();
    this.registration = undefined;
    for (const patch of this.owned) {
      patch.owners.delete(this.options);
      if (patch.owners.size === 0) {
        if (patch.target[patch.methodName] === patch.wrappedMethod)
          patch.target[patch.methodName] = patch.originalMethod;
        patches.delete(patch);
      }
    }
    this.owned = [];
  }
}
export {
  buildErrorAttrs,
  buildSuccessAttrs,
  emitOperationError,
  emitOperationSuccess,
} from "./_span_emitter.js";
export {
  buildChatCompletionFromStreamState,
  buildCompletionFromStreamState,
  createChatStreamState,
  createTextStreamState,
  updateChatStreamState,
  updateTextStreamState,
} from "./_helpers.js";
