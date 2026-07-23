/**
 * Respan instrumentation plugin for the official Writer TypeScript SDK.
 *
 * Patches `writer-sdk` resource prototypes to emit canonical Respan LLM spans
 * for chat, structured parse, streaming chat, and text completion calls.
 */

import { patchWriterMethod, type PatchedMethodTarget } from "./_streaming.js";

export interface WriterInstrumentorOptions {
  /**
   * Optional `writer-sdk` module instance. Pass this when an application resolves a
   * different mutable SDK copy than the instrumentor would import by default.
   */
  sdkModule?: any;
}

export class WriterInstrumentor {
  public readonly name = "writer";

  private static readonly _sharedState = {
    activeInstances: 0,
    patchedTargets: [] as PatchedMethodTarget[],
  };

  private _isInstrumented = false;
  private readonly _sdkModule?: any;

  constructor(options: WriterInstrumentorOptions = {}) {
    this._sdkModule = options.sdkModule;
  }

  async activate(): Promise<void> {
    if (this._isInstrumented) return;

    const sharedState = WriterInstrumentor._sharedState;

    try {
      const writerModule = this._sdkModule ?? await import("writer-sdk");
      const Writer = writerModule.default ?? writerModule.Writer;
      if (typeof Writer !== "function") {
        console.warn(
          "[Respan] Failed to activate Writer instrumentation — compatible Writer constructor not found",
        );
        return;
      }

      const tempClient = new Writer({ apiKey: "respan-placeholder" });
      const chatPrototype = Object.getPrototypeOf(tempClient.chat);
      const completionsPrototype = Object.getPrototypeOf(tempClient.completions);

      const targets: Array<[any, string, "chat" | "completion"]> = [
        [chatPrototype, "chat", "chat"],
        [completionsPrototype, "create", "completion"],
      ];

      for (const [target, methodName, type] of targets) {
        if (
          !target ||
          typeof target[methodName] !== "function" ||
          sharedState.patchedTargets.some(
            (patched) => patched.target === target && patched.methodName === methodName,
          )
        ) {
          continue;
        }

        const patchedTarget = patchWriterMethod(target, methodName, type);
        if (patchedTarget) {
          sharedState.patchedTargets.push(patchedTarget);
        }
      }

      if (sharedState.patchedTargets.length === 0) {
        console.warn(
          "[Respan] Failed to activate Writer instrumentation — no compatible Writer resource methods found",
        );
        return;
      }

      sharedState.activeInstances += 1;
      this._isInstrumented = true;
    } catch (err) {
      console.warn("[Respan] Failed to activate Writer instrumentation:", err);
    }
  }

  deactivate(): void {
    if (!this._isInstrumented) return;

    const sharedState = WriterInstrumentor._sharedState;
    sharedState.activeInstances = Math.max(0, sharedState.activeInstances - 1);
    this._isInstrumented = false;

    if (sharedState.activeInstances > 0 || sharedState.patchedTargets.length === 0) return;

    try {
      for (const patchedTarget of sharedState.patchedTargets) {
        patchedTarget.target[patchedTarget.methodName] = patchedTarget.originalMethod;
      }
    } catch {
      /* ignore */
    }

    sharedState.patchedTargets = [];
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
