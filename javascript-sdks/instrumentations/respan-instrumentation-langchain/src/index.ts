/**
 * Respan instrumentation plugin for LangChain JS, LangGraph JS, and
 * Langflow-style component flows.
 *
 * The JavaScript LangChain ecosystem uses callback handlers for chain, tool,
 * retriever, LLM, and graph lifecycle events. This package provides a Respan
 * callback handler that emits native OpenTelemetry spans into the active
 * `@respan/tracing` OpenTelemetry pipeline.
 */

import {
  addRespanCallback,
  getCallbackHandler,
  RespanCallbackHandler,
  type LangChainCallbackConfig,
  type RespanCallbackHandlerOptions,
} from "./_callback.js";

export {
  addRespanCallback,
  getCallbackHandler,
  RespanCallbackHandler,
  type LangChainCallbackConfig,
  type RespanCallbackHandlerOptions,
} from "./_callback.js";

export interface LangChainInstrumentorOptions {
  callbackHandler?: RespanCallbackHandler;
  callbackHandlerOptions?: RespanCallbackHandlerOptions;
}

export class LangChainInstrumentor {
  public readonly name = "langchain";
  public readonly callbackHandler: RespanCallbackHandler;

  private _active = false;

  constructor(options: LangChainInstrumentorOptions = {}) {
    this.callbackHandler =
      options.callbackHandler ??
      getCallbackHandler(options.callbackHandlerOptions);
    this.callbackHandler._setActive(false, this);
  }

  activate(): void {
    this._active = true;
    this.callbackHandler._setActive(true, this);
  }

  deactivate(): void {
    this._active = false;
    this.callbackHandler._setActive(false, this);
  }

  isActive(): boolean {
    return this._active;
  }

  addCallback<TConfig extends object = LangChainCallbackConfig>(
    config: TConfig = {} as TConfig,
  ): TConfig {
    return addRespanCallback(config, this.callbackHandler);
  }
}

export { LangChainInstrumentor as LangchainInstrumentor };
