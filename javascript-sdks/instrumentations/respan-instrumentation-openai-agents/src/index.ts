/**
 * Respan instrumentation plugin for the OpenAI Agents SDK.
 *
 * Registers a TracingProcessor that converts OpenAI Agents SDK
 * traces/spans to OTEL ReadableSpan objects and injects them into
 * the unified OTEL pipeline.
 *
 * ```typescript
 * import { Respan } from "@respan/respan";
 * import { OpenAIAgentsInstrumentor } from "@respan/instrumentation-openai-agents";
 *
 * const respan = new Respan({
 *   instrumentations: [new OpenAIAgentsInstrumentor()],
 * });
 * await respan.initialize();
 * ```
 */

import {
  setTraceProcessors,
  type TracingProcessor,
  type Trace,
  type Span,
} from "@openai/agents";
import {
  clearSdkTrace,
  clearSdkTraceContexts,
  emitSdkItem,
  registerSdkTrace,
} from "./_otel_emitter.js";

import { installStreamPatches, removeStreamPatches } from "./_streaming.js";

class _RespanTracingProcessor implements TracingProcessor {
  enabled = true;
  async onTraceStart(traceObj: Trace): Promise<void> {
    if (this.enabled) registerSdkTrace(traceObj);
  }

  async onTraceEnd(traceObj: Trace): Promise<void> {
    if (!this.enabled) return;
    try {
      emitSdkItem(traceObj);
    } finally {
      clearSdkTrace(traceObj.traceId);
    }
  }

  async onSpanStart(_span: Span<any>): Promise<void> {
    // no-op
  }

  async onSpanEnd(span: Span<any>): Promise<void> {
    if (this.enabled) emitSdkItem(span);
  }

  async shutdown(): Promise<void> {
    // no-op
  }

  async forceFlush(): Promise<void> {
    // no-op
  }
}

let sharedProcessor: _RespanTracingProcessor | null = null;
let activationCount = 0;

export class OpenAIAgentsInstrumentor {
  public readonly name = "openai-agents";
  private _processor: _RespanTracingProcessor | null = null;

  activate(): void {
    if (this._processor) return;
    if (!sharedProcessor) {
      installStreamPatches();
      sharedProcessor = new _RespanTracingProcessor();
      setTraceProcessors([sharedProcessor]);
    }
    this._processor = sharedProcessor;
    activationCount += 1;
  }

  deactivate(): void {
    if (!this._processor) return;
    this._processor = null;
    activationCount -= 1;
    if (activationCount) return;
    if (sharedProcessor) sharedProcessor.enabled = false;
    sharedProcessor = null;
    removeStreamPatches();
    clearSdkTraceContexts();
  }
}
