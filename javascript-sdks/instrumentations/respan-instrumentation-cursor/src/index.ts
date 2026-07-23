/**
 * Respan instrumentation plugin for the Cursor TypeScript SDK.
 *
 * Cursor SDK's ESM namespace is immutable, so pass a mutable module copy:
 *
 * ```ts
 * import * as CursorSDK from "@cursor/sdk";
 * const cursorSdkModule = { ...CursorSDK };
 *
 * new CursorSDKInstrumentor({ sdkModule: cursorSdkModule });
 * ```
 */

import {
  createCursorRunState,
  emitCursorToolExecution,
  emitFinalCursorRunSpans,
  markCursorRunError,
  recordCursorRunResult,
  registerCursorToolDefinitions,
  trackCursorCallback,
  trackCursorMessage,
  type CursorRunState,
} from "./_otel_emitter.js";

type AnyRecord = Record<string, any>;
type AnyFunction = (...args: any[]) => any;
type Patch = () => void;

export interface CursorSDKInstrumentorOptions {
  agentName?: string;
  sdkModule: Record<string, unknown>;
}

const INSTRUMENTOR_LOG_PREFIX = "[respan] CursorSDKInstrumentor";

export class CursorSDKInstrumentor {
  public readonly name = "cursor-sdk";

  private readonly _agentName?: string;
  private readonly _sdkModule: Record<string, unknown>;
  private readonly _agentOptions = new WeakMap<object, AnyRecord | undefined>();
  private readonly _patchedAgents = new WeakSet<object>();
  private _patchedRuns = new WeakMap<object, CursorRunState>();
  private _isInstrumented = false;
  private _patches: Patch[] = [];
  private _suppressNestedPromptInstrumentation = 0;

  constructor({ sdkModule, agentName }: CursorSDKInstrumentorOptions) {
    this._sdkModule = sdkModule;
    this._agentName = agentName;
  }

  activate(): void {
    if (this._isInstrumented) return;
    const Agent = (this._sdkModule as AnyRecord).Agent;
    if (!Agent || (typeof Agent !== "function" && typeof Agent !== "object")) {
      throw new Error("CursorSDKInstrumentor requires sdkModule.Agent from @cursor/sdk.");
    }
    this._patchAgentStatics(Agent as AnyRecord);
    this._isInstrumented = true;
  }

  deactivate(): void {
    for (const undo of [...this._patches].reverse()) {
      try { undo(); } catch { /* best effort */ }
    }
    this._patches = [];
    this._patchedRuns = new WeakMap<object, CursorRunState>();
    this._suppressNestedPromptInstrumentation = 0;
    this._isInstrumented = false;
  }

  private _patchAgentStatics(Agent: AnyRecord): void {
    this._patchMethod(Agent, "create", (original) => {
      const instrumentor = this;
      return async function patchedCreate(this: unknown, options?: AnyRecord) {
        if (instrumentor._suppressNestedPromptInstrumentation > 0) {
          return await original.call(this, options);
        }
        const agent = await original.call(this, instrumentor._wrapAgentOptions(options));
        return instrumentor._wrapAgent(agent, options);
      };
    });

    this._patchMethod(Agent, "resume", (original) => {
      const instrumentor = this;
      return async function patchedResume(this: unknown, agentId: string, options?: AnyRecord) {
        if (instrumentor._suppressNestedPromptInstrumentation > 0) {
          return await original.call(this, agentId, options);
        }
        const agent = await original.call(this, agentId, instrumentor._wrapAgentOptions(options));
        return instrumentor._wrapAgent(agent, { ...(options ?? {}), agentId });
      };
    });

    this._patchMethod(Agent, "prompt", (original) => {
      const instrumentor = this;
      return async function patchedPrompt(this: unknown, message: unknown, options?: AnyRecord) {
        const state = createCursorRunState({
          agentName: instrumentor._agentName,
          agentOptions: options,
          message,
          operation: "Agent.prompt",
          options,
        });
        const wrappedOptions = instrumentor._wrapAgentOptions(options, state);
        instrumentor._suppressNestedPromptInstrumentation += 1;
        try {
          const result = await original.call(this, message, wrappedOptions);
          recordCursorRunResult(state, result);
          return result;
        } catch (error) {
          markCursorRunError(state, error);
          throw error;
        } finally {
          instrumentor._suppressNestedPromptInstrumentation -= 1;
          emitFinalCursorRunSpans(state);
        }
      };
    });

    this._patchMethod(Agent, "getRun", (original) => {
      const instrumentor = this;
      return async function patchedGetRun(this: unknown, runId: string, options?: AnyRecord) {
        const run = await original.call(this, runId, options);
        const state = createCursorRunState({
          agentName: instrumentor._agentName,
          operation: "Agent.getRun",
        });
        state.runId = runId;
        return instrumentor._wrapRun(run, state);
      };
    });
  }

  private _wrapAgent(agent: unknown, agentOptions?: AnyRecord): unknown {
    if (!agent || typeof agent !== "object") return agent;
    const agentObject = agent as AnyRecord;
    if (this._patchedAgents.has(agentObject)) return agent;
    this._patchedAgents.add(agentObject);
    this._agentOptions.set(agentObject, agentOptions);

    this._patchMethod(agentObject, "send", (original) => {
      const instrumentor = this;
      return async function patchedSend(this: AnyRecord, message: unknown, options?: AnyRecord) {
        if (instrumentor._suppressNestedPromptInstrumentation > 0) {
          return await original.call(this, message, options);
        }
        const state = createCursorRunState({
          agent: this,
          agentName: instrumentor._agentName,
          agentOptions: instrumentor._agentOptions.get(this),
          message,
          operation: "SDKAgent.send",
          options,
        });
        const wrappedOptions = instrumentor._wrapSendOptions(options, state);
        try {
          const run = await original.call(this, message, wrappedOptions);
          return instrumentor._wrapRun(run, state);
        } catch (error) {
          markCursorRunError(state, error);
          emitFinalCursorRunSpans(state);
          throw error;
        }
      };
    });

    return agent;
  }

  private _wrapRun(run: unknown, state: CursorRunState): unknown {
    if (!run || typeof run !== "object") return run;
    const runObject = run as AnyRecord;
    if (this._patchedRuns.has(runObject)) return run;
    this._patchedRuns.set(runObject, state);

    state.runId = stringValue(runObject.id) ?? state.runId;
    state.requestId = stringValue(runObject.requestId) ?? state.requestId;
    state.agentId = stringValue(runObject.agentId) ?? state.agentId;
    state.model = resolveModel(runObject.model) ?? state.model;

    this._patchMethod(runObject, "stream", (original) => {
      return function patchedStream(this: AnyRecord) {
        const stream = original.call(this);
        return (async function*() {
          try {
            for await (const message of stream as AsyncIterable<unknown>) {
              try {
                trackCursorMessage(state, message);
              } catch (error) {
                console.warn(`${INSTRUMENTOR_LOG_PREFIX} message tracking failed:`, error);
              }
              yield message;
            }
          } catch (error) {
            markCursorRunError(state, error);
            throw error;
          } finally {
            emitFinalCursorRunSpans(state);
          }
        })();
      };
    });

    this._patchMethod(runObject, "wait", (original) => {
      return async function patchedWait(this: AnyRecord) {
        try {
          const result = await original.call(this);
          recordCursorRunResult(state, result);
          return result;
        } catch (error) {
          markCursorRunError(state, error);
          throw error;
        } finally {
          emitFinalCursorRunSpans(state);
        }
      };
    });

    return run;
  }

  private _wrapAgentOptions(options?: AnyRecord, state?: CursorRunState): AnyRecord | undefined {
    if (!options) return options;
    return this._wrapOptions(options, state);
  }

  private _wrapSendOptions(options: AnyRecord | undefined, state: CursorRunState): AnyRecord | undefined {
    if (!options) return options;
    return this._wrapOptions(options, state);
  }

  private _wrapOptions(options: AnyRecord, state?: CursorRunState): AnyRecord {
    const wrapped: AnyRecord = { ...options };
    if (typeof options.onStep === "function" && state) {
      const originalOnStep = options.onStep;
      wrapped.onStep = async (...args: unknown[]) => {
        trackCursorCallback(state, "onStep", args[0]);
        return await originalOnStep(...args);
      };
    }
    if (typeof options.onDelta === "function" && state) {
      const originalOnDelta = options.onDelta;
      wrapped.onDelta = async (...args: unknown[]) => {
        trackCursorCallback(state, "onDelta", args[0]);
        return await originalOnDelta(...args);
      };
    }

    const local = asRecord(options.local);
    const customTools = asRecord(local?.customTools);
    if (customTools && state) {
      const wrappedTools: AnyRecord = {};
      const toolDefinitions: AnyRecord[] = [];
      for (const [toolName, tool] of Object.entries(customTools)) {
        const toolRecord = asRecord(tool);
        if (!toolRecord || typeof toolRecord.execute !== "function") {
          wrappedTools[toolName] = tool;
          continue;
        }
        toolDefinitions.push({
          type: "function",
          function: {
            name: toolName,
            ...(typeof toolRecord.description === "string" ? { description: toolRecord.description } : {}),
            ...(toolRecord.inputSchema !== undefined ? { parameters: toolRecord.inputSchema } : {}),
          },
        });
        const originalExecute = toolRecord.execute;
        wrappedTools[toolName] = {
          ...toolRecord,
          execute: async (...args: unknown[]) => {
            const toolArgs = args[0];
            const context = asRecord(args[1]);
            try {
              const result = await originalExecute.apply(toolRecord, args);
              emitCursorToolExecution({
                args: toolArgs,
                callId: stringValue(context?.toolCallId),
                result,
                state,
                toolName,
              });
              return result;
            } catch (error) {
              emitCursorToolExecution({
                args: toolArgs,
                callId: stringValue(context?.toolCallId),
                error,
                state,
                toolName,
              });
              throw error;
            }
          },
        };
      }
      registerCursorToolDefinitions(state, toolDefinitions);
      wrapped.local = { ...local, customTools: wrappedTools };
    }

    return wrapped;
  }

  private _patchMethod(target: AnyRecord, methodName: string, createWrapper: (original: AnyFunction) => AnyFunction): void {
    const original = target[methodName];
    if (typeof original !== "function") return;
    const wrapped = createWrapper(original);
    target[methodName] = wrapped;
    this._patches.push(() => { target[methodName] = original; });
  }
}

function asRecord(value: unknown): AnyRecord | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  return value as AnyRecord;
}

function stringValue(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  const text = String(value);
  return text ? text : undefined;
}

function resolveModel(model: unknown): string | undefined {
  if (typeof model === "string" && model) return model;
  const record = asRecord(model);
  return stringValue(record?.id) ?? stringValue(record?.model);
}
