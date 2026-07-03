import { trace, type Span, type TracerProvider } from "@opentelemetry/api";
import { LIVEKIT_INSTRUMENTATION_NAME } from "./_constants.js";
import {
  getSpanAttributes,
  type MutableAttributes,
  translateLiveKitSpan,
} from "./_translator.js";

const PATCHED_SPAN = Symbol.for("respan.instrumentation.livekit.patchedSpan");
const TRANSLATED_SPAN = Symbol.for("respan.instrumentation.livekit.translatedSpan");

type AnyFunction = (...args: any[]) => any;
type PatchableTracer = Record<string, unknown>;

export interface LiveKitTelemetryModule {
  tracer?: {
    startSpan?: AnyFunction;
    startActiveSpan?: AnyFunction;
    startActiveSpanSync?: AnyFunction;
    setProvider?: (provider: TracerProvider) => void;
  };
  setTracerProvider?: (...args: any[]) => void;
}

export interface LiveKitAgentsModule {
  telemetry?: LiveKitTelemetryModule;
}

export interface LiveKitInstrumentorOptions {
  livekitModule?: LiveKitAgentsModule;
  telemetryModule?: LiveKitTelemetryModule;
  syncTracerProvider?: boolean;
}

export class LiveKitInstrumentor {
  public readonly name = LIVEKIT_INSTRUMENTATION_NAME;

  private static _originals: Map<string, AnyFunction> = new Map();
  private static _patchCount = 0;
  private static _patchedTracer?: PatchableTracer;

  private readonly _livekitModule?: LiveKitAgentsModule;
  private readonly _telemetryModule?: LiveKitTelemetryModule;
  private readonly _syncTracerProvider: boolean;
  private _isInstrumented = false;

  constructor(options: LiveKitInstrumentorOptions = {}) {
    this._livekitModule = options.livekitModule;
    this._telemetryModule = options.telemetryModule;
    this._syncTracerProvider = options.syncTracerProvider ?? true;
  }

  async activate(): Promise<void> {
    if (this._isInstrumented) {
      return;
    }

    const telemetryModule = await this._resolveTelemetryModule();
    if (!telemetryModule?.tracer) {
      return;
    }

    if (this._syncTracerProvider) {
      syncLiveKitTracerProvider(telemetryModule);
    }

    this._patchTracer(telemetryModule.tracer as PatchableTracer);
    this._isInstrumented = true;
  }

  deactivate(): void {
    if (!this._isInstrumented) {
      return;
    }

    LiveKitInstrumentor._patchCount = Math.max(0, LiveKitInstrumentor._patchCount - 1);
    if (LiveKitInstrumentor._patchCount === 0 && LiveKitInstrumentor._patchedTracer) {
      for (const [methodName, original] of LiveKitInstrumentor._originals) {
        LiveKitInstrumentor._patchedTracer[methodName] = original;
      }
      LiveKitInstrumentor._originals.clear();
      LiveKitInstrumentor._patchedTracer = undefined;
    }

    this._isInstrumented = false;
  }

  isActive(): boolean {
    return this._isInstrumented;
  }

  private async _resolveTelemetryModule(): Promise<LiveKitTelemetryModule | undefined> {
    if (this._telemetryModule) {
      return this._telemetryModule;
    }
    if (this._livekitModule?.telemetry) {
      return this._livekitModule.telemetry;
    }

    try {
      const livekit = await import("@livekit/agents") as unknown as LiveKitAgentsModule;
      return livekit.telemetry;
    } catch {
      return undefined;
    }
  }

  private _patchTracer(tracer: PatchableTracer): void {
    if (LiveKitInstrumentor._patchCount === 0) {
      LiveKitInstrumentor._patchedTracer = tracer;

      const startSpan = tracer.startSpan;
      if (typeof startSpan === "function") {
        LiveKitInstrumentor._originals.set("startSpan", startSpan as AnyFunction);
        tracer.startSpan = function instrumentedLiveKitStartSpan(
          this: unknown,
          options: { name?: string; attributes?: MutableAttributes } = {},
        ): Span {
          const span = (startSpan as AnyFunction).call(this, options);
          return patchSpan(span, options?.name, options?.attributes);
        };
      }

      const startActiveSpan = tracer.startActiveSpan;
      if (typeof startActiveSpan === "function") {
        LiveKitInstrumentor._originals.set("startActiveSpan", startActiveSpan as AnyFunction);
        tracer.startActiveSpan = function instrumentedLiveKitStartActiveSpan<T>(
          this: unknown,
          fn: (span: Span) => Promise<T>,
          options: { name?: string; attributes?: MutableAttributes } = {},
        ): Promise<T> {
          return (startActiveSpan as AnyFunction).call(
            this,
            async (span: Span) => await fn(patchSpan(span, options?.name, options?.attributes)),
            options,
          );
        };
      }

      const startActiveSpanSync = tracer.startActiveSpanSync;
      if (typeof startActiveSpanSync === "function") {
        LiveKitInstrumentor._originals.set("startActiveSpanSync", startActiveSpanSync as AnyFunction);
        tracer.startActiveSpanSync = function instrumentedLiveKitStartActiveSpanSync<T>(
          this: unknown,
          fn: (span: Span) => T,
          options: { name?: string; attributes?: MutableAttributes } = {},
        ): T {
          return (startActiveSpanSync as AnyFunction).call(
            this,
            (span: Span) => fn(patchSpan(span, options?.name, options?.attributes)),
            options,
          );
        };
      }
    }

    if (LiveKitInstrumentor._originals.size > 0) {
      LiveKitInstrumentor._patchCount += 1;
    }
  }
}

function syncLiveKitTracerProvider(telemetryModule: LiveKitTelemetryModule): void {
  const provider = resolveActiveTracerProvider();
  if (!provider) {
    return;
  }

  if (typeof telemetryModule.setTracerProvider === "function") {
    try {
      telemetryModule.setTracerProvider(provider);
      return;
    } catch {
      // Fall back to the tracer-level setter below.
    }
  }

  if (typeof telemetryModule.tracer?.setProvider === "function") {
    try {
      telemetryModule.tracer.setProvider(provider as TracerProvider);
    } catch {
      // Non-fatal: the method patch still works if LiveKit already uses the global provider.
    }
  }
}

function resolveActiveTracerProvider(): unknown {
  const provider = trace.getTracerProvider() as unknown as { _delegate?: unknown };
  return provider?._delegate ?? provider;
}

function patchSpan(
  span: Span,
  spanName?: string,
  initialAttributes?: MutableAttributes,
): Span {
  const patchableSpan = span as Span & Record<string | symbol, any>;
  if (patchableSpan[PATCHED_SPAN]) {
    return span;
  }

  const attrs = getSpanAttributes(span);
  if (initialAttributes) {
    for (const [key, value] of Object.entries(initialAttributes)) {
      attrs[key] = value;
    }
  }

  const originalSetAttribute = span.setAttribute.bind(span);
  patchableSpan.setAttribute = (key: string, value: unknown): Span => {
    attrs[key] = value;
    return originalSetAttribute(key, value as any);
  };

  const originalSetAttributes = span.setAttributes.bind(span);
  patchableSpan.setAttributes = (attributes: MutableAttributes): Span => {
    for (const [key, value] of Object.entries(attributes)) {
      attrs[key] = value;
    }
    return originalSetAttributes(attributes as any);
  };

  const originalEnd = span.end.bind(span);
  patchableSpan.end = (...args: unknown[]): void => {
    translateOnce(span, attrs, spanName);
    (originalEnd as AnyFunction)(...args);
  };

  Object.defineProperty(patchableSpan, PATCHED_SPAN, {
    enumerable: false,
    value: true,
  });

  return span;
}

function translateOnce(span: Span, attrs: MutableAttributes, spanName?: string): void {
  const patchableSpan = span as Span & Record<string | symbol, any>;
  if (patchableSpan[TRANSLATED_SPAN]) {
    return;
  }
  Object.defineProperty(patchableSpan, TRANSLATED_SPAN, {
    enumerable: false,
    value: true,
  });

  translateLiveKitSpan(span, { attributes: attrs, spanName });
}

export { translateLiveKitSpan };
