/**
 * Respan instrumentation plugin for the Together AI TypeScript SDK.
 *
 * Patches the Stainless-generated `together-ai` resource prototypes and emits
 * Respan-compatible spans for chat, completions, embeddings, image generation,
 * rerank, speech, transcription, and translation calls.
 */

import { RespanLogType } from "@respan/respan-sdk";
import { loadTogetherConstructors } from "./_helpers.js";
import { patchResourceMethod } from "./_patching.js";
import type { PatchedResourceTarget, TogetherOperationSpec } from "./_types.js";

const TOGETHER_OPERATION_SPECS = {
  chatCompletions: {
    kind: "chat",
    method: "create",
    spanName: "together.chat.completions",
    logType: RespanLogType.CHAT,
    requestType: RespanLogType.CHAT,
  },
  completions: {
    kind: "text",
    method: "create",
    spanName: "together.completions",
    logType: RespanLogType.TEXT,
    requestType: RespanLogType.CHAT,
  },
  embeddings: {
    kind: "embedding",
    method: "create",
    spanName: "together.embeddings",
    logType: RespanLogType.EMBEDDING,
    requestType: RespanLogType.EMBEDDING,
  },
  images: {
    kind: "image",
    method: "generate",
    spanName: "together.images.generate",
    logType: RespanLogType.GENERATION,
    requestType: "image",
  },
  rerank: {
    kind: "rerank",
    method: "create",
    spanName: "together.rerank",
    logType: RespanLogType.CUSTOM,
    requestType: "rerank",
  },
  speech: {
    kind: "speech",
    method: "create",
    spanName: "together.audio.speech",
    logType: RespanLogType.SPEECH,
    requestType: RespanLogType.SPEECH,
  },
  transcriptions: {
    kind: "transcription",
    method: "create",
    spanName: "together.audio.transcriptions",
    logType: RespanLogType.TRANSCRIPTION,
    requestType: RespanLogType.TRANSCRIPTION,
  },
  translations: {
    kind: "translation",
    method: "create",
    spanName: "together.audio.translations",
    logType: RespanLogType.CUSTOM,
    requestType: "translation",
  },
} satisfies Record<string, TogetherOperationSpec>;

function prototypeOf(value: any): Record<string, any> | null {
  if (!value || typeof value !== "object") return null;
  return Object.getPrototypeOf(value);
}

function collectPatchTargets(client: any): Array<{
  prototype: Record<string, any> | null;
  spec: TogetherOperationSpec;
}> {
  return [
    {
      prototype: prototypeOf(client?.chat?.completions),
      spec: TOGETHER_OPERATION_SPECS.chatCompletions,
    },
    {
      prototype: prototypeOf(client?.completions),
      spec: TOGETHER_OPERATION_SPECS.completions,
    },
    {
      prototype: prototypeOf(client?.embeddings),
      spec: TOGETHER_OPERATION_SPECS.embeddings,
    },
    {
      prototype: prototypeOf(client?.images),
      spec: TOGETHER_OPERATION_SPECS.images,
    },
    {
      prototype: prototypeOf(client?.rerank),
      spec: TOGETHER_OPERATION_SPECS.rerank,
    },
    {
      prototype: prototypeOf(client?.audio?.speech),
      spec: TOGETHER_OPERATION_SPECS.speech,
    },
    {
      prototype: prototypeOf(client?.audio?.transcriptions),
      spec: TOGETHER_OPERATION_SPECS.transcriptions,
    },
    {
      prototype: prototypeOf(client?.audio?.translations),
      spec: TOGETHER_OPERATION_SPECS.translations,
    },
  ];
}

export class TogetherAIInstrumentor {
  public readonly name = "together-ai";
  private static readonly _sharedState = {
    activeInstances: 0,
    patchedTargets: [] as PatchedResourceTarget[],
  };

  private _isInstrumented = false;

  async activate(): Promise<void> {
    if (this._isInstrumented) return;

    const togetherConstructors = await loadTogetherConstructors();
    if (togetherConstructors.length === 0) {
      console.warn(
        "[Respan] Failed to activate Together AI instrumentation — together-ai not found",
      );
      return;
    }

    const sharedState = TogetherAIInstrumentor._sharedState;

    try {
      for (const Together of togetherConstructors) {
        const tempClient = new Together({ apiKey: "sk-placeholder" });
        for (const { prototype, spec } of collectPatchTargets(tempClient)) {
          if (
            !prototype ||
            sharedState.patchedTargets.some(
              (target) => target.prototype === prototype && target.method === spec.method,
            )
          ) {
            continue;
          }

          const patchedTarget = patchResourceMethod(prototype, spec);
          if (patchedTarget) sharedState.patchedTargets.push(patchedTarget);
        }
      }

      if (sharedState.patchedTargets.length === 0) {
        console.warn(
          "[Respan] Failed to activate Together AI instrumentation — no compatible Together AI resource prototypes found",
        );
        return;
      }

      sharedState.activeInstances += 1;
      this._isInstrumented = true;
    } catch (err) {
      console.warn("[Respan] Failed to activate Together AI instrumentation:", err);
    }
  }

  deactivate(): void {
    if (!this._isInstrumented) return;

    const sharedState = TogetherAIInstrumentor._sharedState;
    sharedState.activeInstances = Math.max(0, sharedState.activeInstances - 1);
    this._isInstrumented = false;

    if (sharedState.activeInstances > 0 || sharedState.patchedTargets.length === 0) return;

    try {
      for (const patchedTarget of sharedState.patchedTargets) {
        patchedTarget.prototype[patchedTarget.method] = patchedTarget.original;
      }
    } catch {
      // Ignore teardown failures.
    }

    sharedState.patchedTargets = [];
  }
}

export { TogetherAIInstrumentor as TogetherAIInstrumentation };
