/**
 * Respan instrumentation plugin for the Claude Agent SDK.
 *
 * This plugin patches `query()` on a mutable Claude Agent SDK module, merges in
 * tool lifecycle hooks, and emits OTEL ReadableSpan objects into the active
 * Respan tracing pipeline.
 *
 * ```typescript
 * import * as _ClaudeAgentSDK from "@anthropic-ai/claude-agent-sdk";
 * import { Respan } from "@respan/respan";
 * import { ClaudeAgentSDKInstrumentor } from "@respan/instrumentation-claude-agent-sdk";
 *
 * const ClaudeAgentSDK = { ..._ClaudeAgentSDK };
 *
 * const respan = new Respan({
 *   instrumentations: [
 *     new ClaudeAgentSDKInstrumentor({ sdkModule: ClaudeAgentSDK }),
 *   ],
 * });
 * await respan.initialize();
 * ```
 */

import {
  createQueryState,
  emitAgentSpan,
  emitCompletedTool,
  registerPendingTool,
  registerPromptSubmit,
  trackClaudeMessage,
  type QueryState,
} from "./_otel_emitter.js";

export interface ClaudeAgentSDKInstrumentorOptions {
  agentName?: string;
  sdkModule: Record<string, unknown>;
}

type HookCallback = (
  input: Record<string, unknown>,
  toolUseId?: string,
) => Promise<Record<string, unknown>>;

type HookGroup = {
  matcher?: string;
  hooks?: HookCallback[];
};

export class ClaudeAgentSDKInstrumentor {
  public readonly name = "claude-agent-sdk";

  private readonly _agentName?: string;
  private readonly _sdkModule: Record<string, unknown>;
  private _isInstrumented = false;
  private _originalQuery: ((...args: unknown[]) => unknown) | null = null;

  constructor({ sdkModule, agentName }: ClaudeAgentSDKInstrumentorOptions) {
    this._sdkModule = sdkModule;
    this._agentName = agentName;
  }

  async activate(): Promise<void> {
    if (this._isInstrumented) {
      return;
    }

    const query = this._sdkModule.query;
    if (typeof query !== "function") {
      throw new Error(
        "ClaudeAgentSDKInstrumentor requires sdkModule.query to be a function.",
      );
    }

    this._originalQuery = query as (...args: unknown[]) => unknown;
    const instrumentor = this;

    this._sdkModule.query = async function instrumentedQuery(
      args: unknown,
    ): Promise<unknown> {
      const normalizedArgs =
        args && typeof args === "object" && !Array.isArray(args)
          ? ({ ...(args as Record<string, unknown>) } as Record<string, unknown>)
          : {};

      const options =
        normalizedArgs.options &&
        typeof normalizedArgs.options === "object" &&
        !Array.isArray(normalizedArgs.options)
          ? ({ ...(normalizedArgs.options as Record<string, unknown>) } as Record<string, unknown>)
          : {};

      const state = createQueryState({
        prompt: normalizedArgs.prompt,
        options,
        agentName: instrumentor._agentName,
      });

      normalizedArgs.options = instrumentor._buildHooks(options, state);

      let originalResult: unknown;
      try {
        originalResult = await instrumentor._originalQuery?.call(
          instrumentor._sdkModule,
          normalizedArgs,
        );
      } catch (error) {
        state.statusCode = 500;
        state.errorMessage = error instanceof Error ? error.message : String(error);
        emitAgentSpan(state);
        throw error;
      }

      if (
        !originalResult ||
        typeof originalResult !== "object" ||
        !(Symbol.asyncIterator in (originalResult as Record<string, unknown>))
      ) {
        emitAgentSpan(state);
        return originalResult;
      }

      return instrumentor._wrapAsyncIterable(
        originalResult as AsyncIterable<unknown>,
        state,
      );
    };

    this._isInstrumented = true;
  }

  deactivate(): void {
    if (!this._isInstrumented || !this._originalQuery) {
      return;
    }

    this._sdkModule.query = this._originalQuery;
    this._originalQuery = null;
    this._isInstrumented = false;
  }

  private _buildHooks(
    options: Record<string, unknown>,
    state: QueryState,
  ): Record<string, unknown> {
    const hooks =
      options.hooks && typeof options.hooks === "object" && !Array.isArray(options.hooks)
        ? ({ ...(options.hooks as Record<string, unknown>) } as Record<string, unknown>)
        : {};

    const appendHook = (eventName: string, callback: HookCallback): void => {
      const existingHooks = Array.isArray(hooks[eventName])
        ? ([...(hooks[eventName] as HookGroup[])] as HookGroup[])
        : [];
      existingHooks.push({ hooks: [callback] });
      hooks[eventName] = existingHooks;
    };

    appendHook("UserPromptSubmit", async (input) => {
      try {
        registerPromptSubmit(state, input);
      } catch (error) {
        console.warn("[respan] ClaudeAgentSDKInstrumentor UserPromptSubmit hook failed:", error);
      }
      return {};
    });

    appendHook("PreToolUse", async (input, toolUseId) => {
      try {
        registerPendingTool(state, input, toolUseId);
      } catch (error) {
        console.warn("[respan] ClaudeAgentSDKInstrumentor PreToolUse hook failed:", error);
      }
      return {};
    });

    appendHook("PostToolUse", async (input, toolUseId) => {
      try {
        emitCompletedTool(state, input, toolUseId);
      } catch (error) {
        console.warn("[respan] ClaudeAgentSDKInstrumentor PostToolUse hook failed:", error);
      }
      return {};
    });

    return {
      ...options,
      hooks,
    };
  }

  private _wrapAsyncIterable(
    originalResult: AsyncIterable<unknown>,
    state: QueryState,
  ): AsyncIterable<unknown> {
    return {
      [Symbol.asyncIterator]: async function*() {
        try {
          for await (const message of originalResult) {
            try {
              trackClaudeMessage(state, message);
            } catch (error) {
              console.warn("[respan] ClaudeAgentSDKInstrumentor message tracking failed:", error);
            }
            yield message;
          }
        } catch (error) {
          state.statusCode = 500;
          state.errorMessage = error instanceof Error ? error.message : String(error);
          throw error;
        } finally {
          emitAgentSpan(state);
        }
      },
    };
  }
}
