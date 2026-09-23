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

const HOOK_EVENT_USER_PROMPT_SUBMIT = "UserPromptSubmit";
const HOOK_EVENT_PRE_TOOL_USE = "PreToolUse";
const HOOK_EVENT_POST_TOOL_USE = "PostToolUse";
const HOOK_EVENT_POST_TOOL_USE_FAILURE = "PostToolUseFailure";
const HOOK_EVENT_POST_TOOL_BATCH = "PostToolBatch";
const INSTRUMENTOR_LOG_PREFIX = "[respan] ClaudeAgentSDKInstrumentor";

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

    const originalQuery = this._originalQuery;
    this._sdkModule.query = function instrumentedQuery(
      args: unknown,
    ): unknown {
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
        originalResult = originalQuery.call(
          instrumentor._sdkModule,
          normalizedArgs,
        );
      } catch (error) {
        state.statusCode = 500;
        state.errorMessage = error instanceof Error ? error.message : String(error);
        emitAgentSpan(state);
        throw error;
      }

      const wrapResult = (result: unknown): unknown => {
        if (!result || typeof result !== "object" || !(Symbol.asyncIterator in result)) {
          emitAgentSpan(state);
          return result;
        }
        return instrumentor._wrapAsyncIterable(result as AsyncIterable<unknown>, state);
      };
      // Real SDK queries return Query immediately. Preserve Promise-returning
      // adapters as well without making the native API asynchronous.
      if (originalResult && typeof (originalResult as PromiseLike<unknown>).then === "function") {
        return Promise.resolve(originalResult).then(wrapResult, (error) => {
          state.statusCode = 500;
          state.errorMessage = error instanceof Error ? error.message : String(error);
          emitAgentSpan(state);
          throw error;
        });
      }
      return wrapResult(originalResult);
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

    appendHook(HOOK_EVENT_USER_PROMPT_SUBMIT, async (input) => {
      try {
        registerPromptSubmit(state, input);
      } catch (error) {
        console.warn(
          `${INSTRUMENTOR_LOG_PREFIX} ${HOOK_EVENT_USER_PROMPT_SUBMIT} hook failed:`,
          error,
        );
      }
      return {};
    });

    appendHook(HOOK_EVENT_PRE_TOOL_USE, async (input, toolUseId) => {
      try {
        registerPendingTool(state, input, toolUseId);
      } catch (error) {
        console.warn(
          `${INSTRUMENTOR_LOG_PREFIX} ${HOOK_EVENT_PRE_TOOL_USE} hook failed:`,
          error,
        );
      }
      return {};
    });

    appendHook(HOOK_EVENT_POST_TOOL_USE, async (input, toolUseId) => {
      try {
        emitCompletedTool(state, input, toolUseId);
      } catch (error) {
        console.warn(
          `${INSTRUMENTOR_LOG_PREFIX} ${HOOK_EVENT_POST_TOOL_USE} hook failed:`,
          error,
        );
      }
      return {};
    });

    appendHook(HOOK_EVENT_POST_TOOL_USE_FAILURE, async (input, toolUseId) => {
      try {
        emitCompletedTool(state, input, toolUseId);
      } catch (error) {
        console.warn(
          `${INSTRUMENTOR_LOG_PREFIX} ${HOOK_EVENT_POST_TOOL_USE_FAILURE} hook failed:`,
          error,
        );
      }
      return {};
    });

    appendHook(HOOK_EVENT_POST_TOOL_BATCH, async (input) => {
      try {
        const toolCalls = Array.isArray(input.tool_calls) ? input.tool_calls : [];
        for (const toolCall of toolCalls) {
          if (!toolCall || typeof toolCall !== "object" || Array.isArray(toolCall)) {
            continue;
          }
          const toolCallRecord = toolCall as Record<string, unknown>;
          const toolUseId =
            typeof toolCallRecord.tool_use_id === "string"
              ? toolCallRecord.tool_use_id
              : undefined;
          if (toolUseId && !state.pendingTools.has(toolUseId)) {
            continue;
          }
          emitCompletedTool(
            state,
            {
              session_id: input.session_id,
              ...toolCallRecord,
            },
            toolUseId,
          );
        }
      } catch (error) {
        console.warn(
          `${INSTRUMENTOR_LOG_PREFIX} ${HOOK_EVENT_POST_TOOL_BATCH} hook failed:`,
          error,
        );
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
    const iterator = originalResult[Symbol.asyncIterator]();
    let finished = false;
    const finish = (error?: unknown): void => {
      if (finished) return;
      finished = true;
      if (error !== undefined) {
        state.statusCode = 500;
        state.errorMessage = error instanceof Error ? error.message : String(error);
      }
      emitAgentSpan(state);
    };
    const invoke = async (method: "next" | "return" | "throw", args: unknown[]) => {
      try {
        const operation = iterator[method];
        if (!operation) {
          if (method === "throw") throw args[0];
          finish();
          return { done: true, value: args[0] };
        }
        const result = await (operation as (...values: unknown[]) => Promise<IteratorResult<unknown>>).apply(iterator, args);
        if (!result.done) {
          try {
            trackClaudeMessage(state, result.value);
          } catch (error) {
            console.warn("[respan] ClaudeAgentSDKInstrumentor message tracking failed:", error);
          }
        }
        if (result.done || method === "return") finish();
        return result;
      } catch (error) {
        finish(error);
        throw error;
      }
    };
    const proxy = new Proxy(originalResult, {
      get(target, property) {
        if (property === Symbol.asyncIterator) return () => proxy;
        if (property === "next" || property === "return" || property === "throw") {
          return (...args: unknown[]) => invoke(property, args);
        }
        const value = Reflect.get(target, property, target);
        if (property === "close" && typeof value === "function") {
          return (...args: unknown[]) => {
            try { return value.apply(target, args); }
            finally { finish(); }
          };
        }
        // Query methods use private fields; their receiver must remain Query.
        return typeof value === "function" ? value.bind(target) : value;
      },
    });
    return proxy;
  }
}
