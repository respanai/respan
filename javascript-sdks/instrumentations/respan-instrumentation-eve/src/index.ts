/**
 * Respan instrumentation plugin for the Eve agent framework.
 *
 * Eve enables its vendored AI SDK OpenTelemetry integration whenever
 * `agent/instrumentation.ts` is present. This plugin installs a package-local
 * processor wrapper ahead of the active Respan processor so Eve turn, model,
 * and tool spans are translated before Respan filters and exports them.
 */

import { trace } from "@opentelemetry/api";
import type { Context } from "@opentelemetry/api";
import type {
  ReadableSpan,
  Span,
  SpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { EveSpanProcessor } from "./_processor.js";

export { withEveLineage } from "./lineage.js";

type ProcessorProperty = "activeSpanProcessor" | "_activeSpanProcessor";

interface ProcessorPatchState {
  originalProcessor: SpanProcessor;
  processorProperty: ProcessorProperty;
  provider: Record<string, unknown>;
  wrapper: EveProcessorWrapper;
}

const WRAPPED_PROCESSOR = Symbol.for(
  "respan.instrumentation.eve.wrappedProcessor",
);

class EveProcessorWrapper implements SpanProcessor {
  readonly [WRAPPED_PROCESSOR] = true;

  constructor(
    private readonly _delegate: SpanProcessor,
    private readonly _translator: EveSpanProcessor,
  ) {}

  onStart(span: Span, parentContext: Context): void {
    this._translator.onStart(span, parentContext);
    this._delegate.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    this._translator.onEnd(span);
    this._delegate.onEnd(this._translator.prepareForExport(span));
  }

  shutdown(): Promise<void> {
    return this._delegate.shutdown();
  }

  forceFlush(): Promise<void> {
    return this._delegate.forceFlush();
  }
}

export class EveInstrumentor {
  public readonly name = "eve";

  private static _patchCount = 0;
  private static _patchState: ProcessorPatchState | null = null;
  private static readonly _processor = new EveSpanProcessor({
    initiallyActive: false,
  });

  private _active = false;

  activate(): void {
    if (this._active) {
      return;
    }

    EveInstrumentor._processor.acquire();
    try {
      this._installProcessor();
      this._active = true;
    } catch (error) {
      EveInstrumentor._processor.release();
      throw error;
    }
  }

  deactivate(): void {
    if (!this._active) {
      return;
    }

    this._restoreProcessor();
    EveInstrumentor._processor.release();
    this._active = false;
  }

  isActive(): boolean {
    return this._active;
  }

  private _installProcessor(): void {
    const provider = resolveWritableTracerProvider();
    const { activeProcessor, processorProperty } =
      resolveActiveSpanProcessor(provider);

    if (
      (activeProcessor as unknown as Record<symbol, unknown>)[WRAPPED_PROCESSOR]
    ) {
      EveInstrumentor._patchCount += 1;
      return;
    }

    const wrapper = new EveProcessorWrapper(
      activeProcessor,
      EveInstrumentor._processor,
    );
    setActiveSpanProcessor(provider, processorProperty, wrapper);
    EveInstrumentor._patchState = {
      originalProcessor: activeProcessor,
      processorProperty,
      provider,
      wrapper,
    };
    EveInstrumentor._patchCount = 1;
  }

  private _restoreProcessor(): void {
    if (EveInstrumentor._patchCount === 0) {
      return;
    }

    EveInstrumentor._patchCount -= 1;
    if (EveInstrumentor._patchCount > 0) {
      return;
    }

    const patchState = EveInstrumentor._patchState;
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
    EveInstrumentor._patchState = null;
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
    "EveInstrumentor requires an active OpenTelemetry SpanProcessor. Initialize Respan before activating this instrumentor.",
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

export { EveSpanProcessor };
