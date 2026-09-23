import { AsyncLocalStorage } from "node:async_hooks";
import { OpenAIChatCompletionsModel, OpenAIResponsesModel } from "@openai/agents";

type ModelTraceContext = { streaming: boolean; captureContent: boolean };
const modelContext = new AsyncLocalStorage<ModelTraceContext>();
const patches: Array<{ prototype: any; method: string; original: any; wrapped: any }> = [];

export function isStreaming(): boolean {
  return modelContext.getStore()?.streaming === true;
}

export function shouldCaptureContent(): boolean {
  return modelContext.getStore()?.captureContent !== false;
}

export function installStreamPatches(): void {
  if (patches.length) return;
  for (const model of [OpenAIChatCompletionsModel, OpenAIResponsesModel]) {
    const prototype = model.prototype as any;
    for (const method of ["getResponse", "getStreamedResponse"]) {
      const original = prototype[method];
      if (typeof original !== "function") continue;
      const wrapped = function (this: any, ...args: any[]) {
        const state = {
          streaming: method === "getStreamedResponse",
          captureContent: shouldCaptureContent() && args[0]?.tracing !== "enabled_without_data",
        };
        if (!state.streaming) return modelContext.run(state, () => original.apply(this, args));
        const source = modelContext.run(state, () => original.apply(this, args));
        return {
          [Symbol.asyncIterator]() { return this; },
          next(value?: any) { return modelContext.run(state, () => source.next(value)); },
          return(value?: any) { return modelContext.run(state, () => source.return(value)); },
          throw(error?: any) { return modelContext.run(state, () => source.throw(error)); },
        };
      };
      prototype[method] = wrapped;
      patches.push({ prototype, method, original, wrapped });
    }
  }
}

export function removeStreamPatches(): void {
  for (const { prototype, method, original, wrapped } of patches.splice(0)) {
    if (prototype[method] === wrapped) prototype[method] = original;
  }
}
