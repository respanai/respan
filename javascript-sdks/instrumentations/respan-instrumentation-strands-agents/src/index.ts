/**
 * Respan instrumentation plugin for the Strands Agents TypeScript SDK.
 *
 * Strands already emits OpenTelemetry spans for agents, model calls, tools,
 * graph/swarm orchestration, and node execution. This plugin installs a small
 * processor wrapper ahead of the active Respan processor so those spans are
 * translated into the canonical Respan span contract before export.
 */

import { trace } from "@opentelemetry/api";
import type { Context } from "@opentelemetry/api";
import type {
  ReadableSpan,
  Span,
  SpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import {
  enrichStrandsAgentsSpan,
  StrandsAgentsSpanProcessor,
} from "./_processor.js";
import { STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN } from "./_constants.js";

export interface StrandsAgentsInstrumentorOptions {
  includeToolDefinitions?: boolean;
}

type ProcessorProperty = "activeSpanProcessor" | "_activeSpanProcessor";

interface ProcessorPatchState {
  originalProcessor: SpanProcessor;
  processorProperty: ProcessorProperty;
  provider: Record<string, unknown>;
  wrapper: StrandsProcessorWrapper;
}

const WRAPPED_PROCESSOR = Symbol.for(
  "respan.instrumentation.strandsAgents.wrappedProcessor",
);

class StrandsProcessorWrapper implements SpanProcessor {
  readonly [WRAPPED_PROCESSOR] = true;

  constructor(
    private readonly _delegate: SpanProcessor,
    private readonly _translator: SpanProcessor,
  ) {}

  onStart(span: Span, parentContext: Context): void {
    this._translator.onStart(span, parentContext);
    this._delegate.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    this._translator.onEnd(span);
    this._delegate.onEnd(span);
  }

  shutdown(): Promise<void> {
    return this._delegate.shutdown();
  }

  forceFlush(): Promise<void> {
    return this._delegate.forceFlush();
  }
}

export class StrandsAgentsInstrumentor {
  public readonly name = "strands-agents";

  private static _patchCount = 0;
  private static _patchState: ProcessorPatchState | null = null;

  private readonly _includeToolDefinitions: boolean;
  private _previousSemconvOptIn: string | undefined;
  private _processor: StrandsAgentsSpanProcessor | null = null;
  private _isInstrumented = false;

  constructor(options: StrandsAgentsInstrumentorOptions = {}) {
    this._includeToolDefinitions = options.includeToolDefinitions ?? true;
  }

  activate(): void {
    if (this._isInstrumented) {
      return;
    }

    this._enableSemconvOptIns();
    if (!this._processor) {
      this._processor = new StrandsAgentsSpanProcessor();
    }
    this._installProcessor(this._processor);
    this._isInstrumented = true;
  }

  deactivate(): void {
    if (!this._isInstrumented) {
      return;
    }

    this._restoreProcessor();
    this._restoreSemconvOptIns();
    this._isInstrumented = false;
  }

  isActive(): boolean {
    return this._isInstrumented;
  }

  private _enableSemconvOptIns(): void {
    if (!this._includeToolDefinitions) {
      return;
    }
    this._previousSemconvOptIn = process.env.OTEL_SEMCONV_STABILITY_OPT_IN;
    const values = new Set(
      (this._previousSemconvOptIn ?? "")
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean),
    );
    values.add(STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN);
    process.env.OTEL_SEMCONV_STABILITY_OPT_IN = [...values].sort().join(",");
  }

  private _restoreSemconvOptIns(): void {
    if (!this._includeToolDefinitions) {
      return;
    }
    if (this._previousSemconvOptIn === undefined) {
      delete process.env.OTEL_SEMCONV_STABILITY_OPT_IN;
    } else {
      process.env.OTEL_SEMCONV_STABILITY_OPT_IN = this._previousSemconvOptIn;
    }
    this._previousSemconvOptIn = undefined;
  }

  private _installProcessor(processor: StrandsAgentsSpanProcessor): void {
    const provider = resolveWritableTracerProvider();
    const { activeProcessor, processorProperty } =
      resolveActiveSpanProcessor(provider);

    if ((activeProcessor as unknown as Record<symbol, unknown>)[WRAPPED_PROCESSOR]) {
      StrandsAgentsInstrumentor._patchCount += 1;
      return;
    }

    const wrapper = new StrandsProcessorWrapper(activeProcessor, processor);
    setActiveSpanProcessor(provider, processorProperty, wrapper);

    StrandsAgentsInstrumentor._patchState = {
      originalProcessor: activeProcessor,
      processorProperty,
      provider,
      wrapper,
    };
    StrandsAgentsInstrumentor._patchCount = 1;
  }

  private _restoreProcessor(): void {
    if (StrandsAgentsInstrumentor._patchCount === 0) {
      return;
    }

    StrandsAgentsInstrumentor._patchCount -= 1;
    if (StrandsAgentsInstrumentor._patchCount > 0) {
      return;
    }

    const patchState = StrandsAgentsInstrumentor._patchState;
    if (!patchState) {
      return;
    }

    if (patchState.provider[patchState.processorProperty] === patchState.wrapper) {
      setActiveSpanProcessor(
        patchState.provider,
        patchState.processorProperty,
        patchState.originalProcessor,
      );
    }
    StrandsAgentsInstrumentor._patchState = null;
  }
}

function resolveWritableTracerProvider(): Record<string, unknown> {
  const provider = trace.getTracerProvider() as unknown as Record<string, unknown>;
  const delegated = provider?._delegate as Record<string, unknown> | undefined;
  return delegated ?? provider;
}

function resolveActiveSpanProcessor(provider: Record<string, unknown>): {
  activeProcessor: SpanProcessor;
  processorProperty: ProcessorProperty;
} {
  for (const property of [
    "activeSpanProcessor",
    "_activeSpanProcessor",
  ] as const) {
    const candidate = provider[property];
    if (isSpanProcessor(candidate)) {
      return { activeProcessor: candidate, processorProperty: property };
    }
  }
  throw new Error(
    "StrandsAgentsInstrumentor requires an active OpenTelemetry SpanProcessor. Initialize Respan before activating this instrumentor.",
  );
}

function setActiveSpanProcessor(
  provider: Record<string, unknown>,
  property: ProcessorProperty,
  processor: SpanProcessor,
): void {
  try {
    provider[property] = processor;
    if (provider[property] === processor) {
      return;
    }
  } catch {
    // Fall through to defineProperty.
  }

  Object.defineProperty(provider, property, {
    configurable: true,
    value: processor,
    writable: true,
  });
}

function isSpanProcessor(value: unknown): value is SpanProcessor {
  return Boolean(
    value &&
      typeof value === "object" &&
      typeof (value as SpanProcessor).onStart === "function" &&
      typeof (value as SpanProcessor).onEnd === "function",
  );
}

export { enrichStrandsAgentsSpan, StrandsAgentsSpanProcessor };
