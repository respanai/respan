/**
 * Respan instrumentation plugin for the OpenAI Codex TypeScript SDK.
 *
 * Patches `Thread.run()` and `Thread.runStreamed()` from `@openai/codex-sdk`
 * and emits Codex turn events into the active Respan OTEL pipeline.
 */

import {
  createCodexTurnState,
  emitCodexTurnSpans,
  finalizeCodexTurnState,
  markCodexTurnError,
  trackCodexEvent,
  trackCodexRunResult,
  type CodexEmitterOptions,
  type CodexTurnState,
} from "./_otel_emitter.js";

type AnyFunction = (...args: unknown[]) => unknown;
type PatchablePrototype = Record<string, unknown>;
type CodexThreadConstructor = { prototype?: object };

export interface CodexSDKModule {
  Thread?: CodexThreadConstructor;
}

export interface CodexSDKInstrumentorOptions extends CodexEmitterOptions {
  sdkModule?: CodexSDKModule;
}

export class CodexSDKInstrumentor {
  public readonly name = "codex-sdk";

  private static _originalRun: AnyFunction | null = null;
  private static _originalRunStreamed: AnyFunction | null = null;
  private static _patchCount = 0;
  private static _activeOptions: CodexEmitterOptions = {};

  private readonly _sdkModule?: CodexSDKModule;
  private readonly _options: CodexEmitterOptions;
  private _isInstrumented = false;
  private _threadPrototype?: PatchablePrototype;

  constructor(options: CodexSDKInstrumentorOptions = {}) {
    const { sdkModule, ...emitterOptions } = options;
    this._sdkModule = sdkModule;
    this._options = emitterOptions;
  }

  async activate(): Promise<void> {
    if (this._isInstrumented) {
      return;
    }

    const sdkModule = await this._resolveSDKModule();
    const threadPrototype = sdkModule.Thread?.prototype;
    if (!threadPrototype) {
      throw new Error(
        "CodexSDKInstrumentor requires sdkModule.Thread.prototype.",
      );
    }

    this._patchThreadPrototype(threadPrototype);
    this._isInstrumented = true;
  }

  deactivate(): void {
    if (!this._isInstrumented || !this._threadPrototype) {
      return;
    }

    CodexSDKInstrumentor._patchCount = Math.max(
      0,
      CodexSDKInstrumentor._patchCount - 1,
    );

    if (CodexSDKInstrumentor._patchCount === 0) {
      if (CodexSDKInstrumentor._originalRun) {
        this._threadPrototype.run = CodexSDKInstrumentor._originalRun;
      }
      if (CodexSDKInstrumentor._originalRunStreamed) {
        this._threadPrototype.runStreamed =
          CodexSDKInstrumentor._originalRunStreamed;
      }
      CodexSDKInstrumentor._originalRun = null;
      CodexSDKInstrumentor._originalRunStreamed = null;
      CodexSDKInstrumentor._activeOptions = {};
    }

    this._isInstrumented = false;
  }

  isActive(): boolean {
    return this._isInstrumented;
  }

  private async _resolveSDKModule(): Promise<CodexSDKModule> {
    if (this._sdkModule) {
      return this._sdkModule;
    }

    return (await import("@openai/codex-sdk")) as unknown as CodexSDKModule;
  }

  private _patchThreadPrototype(prototype: object): void {
    const patchablePrototype = prototype as PatchablePrototype;
    this._threadPrototype = patchablePrototype;
    if (CodexSDKInstrumentor._patchCount === 0) {
      const originalRun = patchablePrototype.run;
      const originalRunStreamed = patchablePrototype.runStreamed;
      if (typeof originalRun !== "function") {
        throw new Error(
          "CodexSDKInstrumentor requires Thread.prototype.run to be a function.",
        );
      }
      if (typeof originalRunStreamed !== "function") {
        throw new Error(
          "CodexSDKInstrumentor requires Thread.prototype.runStreamed to be a function.",
        );
      }

      CodexSDKInstrumentor._originalRun = originalRun as AnyFunction;
      CodexSDKInstrumentor._originalRunStreamed =
        originalRunStreamed as AnyFunction;
      CodexSDKInstrumentor._activeOptions = this._options;

      patchablePrototype.run = async function instrumentedCodexRun(
        this: unknown,
        ...args: unknown[]
      ): Promise<unknown> {
        const state = createCodexTurnState({
          input: args[0],
          thread: this,
          options: CodexSDKInstrumentor._activeOptions,
        });

        try {
          const result = await CodexSDKInstrumentor._originalRun?.apply(
            this,
            args,
          );
          trackCodexRunResult(state, result, this);
          return result;
        } catch (error) {
          markCodexTurnError(state, error);
          throw error;
        } finally {
          finalizeCodexTurnState(state, this);
          emitCodexTurnSpans(state);
        }
      };

      patchablePrototype.runStreamed =
        async function instrumentedCodexRunStreamed(
          this: unknown,
          ...args: unknown[]
        ): Promise<unknown> {
          const state = createCodexTurnState({
            input: args[0],
            thread: this,
            options: CodexSDKInstrumentor._activeOptions,
          });

          try {
            const result =
              await CodexSDKInstrumentor._originalRunStreamed?.apply(
                this,
                args,
              );
            if (!isRecord(result) || !isAsyncIterable(result.events)) {
              finalizeCodexTurnState(state, this);
              emitCodexTurnSpans(state);
              return result;
            }
            return {
              ...result,
              events: wrapCodexEvents(result.events, state, this),
            };
          } catch (error) {
            markCodexTurnError(state, error);
            finalizeCodexTurnState(state, this);
            emitCodexTurnSpans(state);
            throw error;
          }
        };
    }

    CodexSDKInstrumentor._patchCount += 1;
  }
}

async function* wrapCodexEvents(
  events: AsyncIterable<unknown>,
  state: CodexTurnState,
  thread: unknown,
): AsyncGenerator<unknown> {
  try {
    for await (const event of events) {
      trackCodexEvent(state, event);
      yield event;
    }
  } catch (error) {
    markCodexTurnError(state, error);
    throw error;
  } finally {
    finalizeCodexTurnState(state, thread);
    emitCodexTurnSpans(state);
  }
}

function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return Boolean(
    value &&
      typeof value === "object" &&
      Symbol.asyncIterator in value &&
      typeof (value as AsyncIterable<unknown>)[Symbol.asyncIterator] ===
        "function",
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}
