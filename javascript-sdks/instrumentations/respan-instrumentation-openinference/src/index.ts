import { trace } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  isOpenInferenceSpan,
  prepareOpenInferenceSpanForExport,
  translateOpenInferenceSpan,
} from "./_translator.js";

export {
  OpenInferenceTranslator,
  isOpenInferenceSpan,
  prepareOpenInferenceSpanForExport,
  translateOpenInferenceSpan,
} from "./_translator.js";

/**
 * Generic Respan instrumentation wrapper for any OpenInference instrumentor.
 *
 * OpenInference instrumentors (from `@arizeai/openinference-instrumentation-*`)
 * may expose either:
 * - `.instrument({ tracerProvider })` — standard OTEL instrumentor interface
 * - SpanProcessor interface (added via `tracerProvider.addSpanProcessor()`)
 *
 * This wrapper detects the interface and handles both patterns.
 *
 * Translation is additive on the live span so Respan callbacks and routing
 * still see the original OpenInference attributes. Export-only cleanup is
 * applied later at the processor-manager boundary using a cloned span.
 *
 * ```typescript
 * import { Respan } from "@respan/respan";
 * import { OpenInferenceInstrumentor } from "@respan/instrumentation-openinference";
 * import { GoogleADKInstrumentor } from "@arizeai/openinference-instrumentation-google-adk";
 *
 * const respan = new Respan({
 *   instrumentations: [new OpenInferenceInstrumentor(GoogleADKInstrumentor)],
 * });
 * await respan.initialize();
 * ```
 */
export class OpenInferenceInstrumentor {
  public readonly name: string;
  private _instrumentorClass: any;
  private _instrumentor: any = null;
  private _isInstrumented = false;
  private _isSpanProcessor = false;
  private _ownsTranslatorHook = false;

  private static _translatorHookRefCount = 0;
  private static _patchedProcessor: any = null;
  private static _originalProcessorOnEnd: ((span: ReadableSpan) => void) | null = null;
  private static _wrappedProcessorOnEnd: ((span: ReadableSpan) => void) | null = null;
  private static _patchedProcessorManager: any = null;
  private static _originalManagerOnEnd: ((span: ReadableSpan) => void) | null = null;
  private static _wrappedManagerOnEnd: ((span: ReadableSpan) => void) | null = null;

  private _sdkModule: any;

  /**
   * @param instrumentorClass - The OI instrumentor class (e.g. GoogleADKInstrumentor)
   * @param sdkModule - Optional SDK module for ESM manual instrumentation.
   *   Required for instrumentors that use `manuallyInstrument(module)` instead of
   *   auto-patching (e.g. Claude Agent SDK in ESM environments).
   */
  constructor(instrumentorClass: any, sdkModule?: any) {
    this._instrumentorClass = instrumentorClass;
    this._sdkModule = sdkModule;
    this.name = `openinference-${instrumentorClass.name || "unknown"}`;
  }

  private static _getActiveSpanProcessor(): any {
    const tracerProvider = trace.getTracerProvider() as any;
    return (
      tracerProvider?.activeSpanProcessor ??
      tracerProvider?._activeSpanProcessor ??
      tracerProvider?._delegate?.activeSpanProcessor ??
      tracerProvider?._delegate?._activeSpanProcessor ??
      tracerProvider?._delegate?._tracerProvider?.activeSpanProcessor ??
      tracerProvider?._delegate?._tracerProvider?._activeSpanProcessor
    );
  }

  private static _getProcessorManager(processor: any): any {
    if (!processor || typeof processor.getProcessorManager !== "function") {
      return null;
    }

    try {
      return processor.getProcessorManager();
    } catch {
      return null;
    }
  }

  private static _restoreHook(
    target: any,
    originalOnEnd: ((span: ReadableSpan) => void) | null,
    wrappedOnEnd: ((span: ReadableSpan) => void) | null,
    label: string,
  ): boolean {
    if (!target || !originalOnEnd) {
      return true;
    }

    if (wrappedOnEnd && target.onEnd !== wrappedOnEnd) {
      console.warn(
        `[respan] OpenInferenceInstrumentor: ${label}.onEnd was modified externally; original handler could not be restored.`
      );
      return false;
    }

    target.onEnd = originalOnEnd;
    return true;
  }

  private static _restorePatchedProcessor(): void {
    if (
      OpenInferenceInstrumentor._restoreHook(
        OpenInferenceInstrumentor._patchedProcessor,
        OpenInferenceInstrumentor._originalProcessorOnEnd,
        OpenInferenceInstrumentor._wrappedProcessorOnEnd,
        "active span processor",
      )
    ) {
      OpenInferenceInstrumentor._patchedProcessor = null;
      OpenInferenceInstrumentor._originalProcessorOnEnd = null;
      OpenInferenceInstrumentor._wrappedProcessorOnEnd = null;
    }
  }

  private static _restorePatchedManager(): void {
    if (
      OpenInferenceInstrumentor._restoreHook(
        OpenInferenceInstrumentor._patchedProcessorManager,
        OpenInferenceInstrumentor._originalManagerOnEnd,
        OpenInferenceInstrumentor._wrappedManagerOnEnd,
        "processor manager",
      )
    ) {
      OpenInferenceInstrumentor._patchedProcessorManager = null;
      OpenInferenceInstrumentor._originalManagerOnEnd = null;
      OpenInferenceInstrumentor._wrappedManagerOnEnd = null;
    }
  }

  private static _installTranslatorHook(): void {
    const processor = OpenInferenceInstrumentor._getActiveSpanProcessor();
    if (!processor || typeof processor.onEnd !== "function") {
      return;
    }

    if (OpenInferenceInstrumentor._patchedProcessor !== processor) {
      if (OpenInferenceInstrumentor._patchedProcessor) {
        OpenInferenceInstrumentor._restorePatchedProcessor();
      }

      const originalProcessorOnEnd = processor.onEnd.bind(processor);
      const hasProcessorManager =
        OpenInferenceInstrumentor._getProcessorManager(processor) !== null;
      const wrappedProcessorOnEnd = (span: ReadableSpan) => {
        if (isOpenInferenceSpan(span)) {
          try {
            translateOpenInferenceSpan(span);
          } catch {
            // Never block export if translation hits an unexpected span shape.
          }
        }

        return originalProcessorOnEnd(
          hasProcessorManager || !isOpenInferenceSpan(span)
            ? span
            : prepareOpenInferenceSpanForExport(span),
        );
      };
      processor.onEnd = wrappedProcessorOnEnd;

      OpenInferenceInstrumentor._patchedProcessor = processor;
      OpenInferenceInstrumentor._originalProcessorOnEnd = originalProcessorOnEnd;
      OpenInferenceInstrumentor._wrappedProcessorOnEnd = wrappedProcessorOnEnd;
    }

    const manager = OpenInferenceInstrumentor._getProcessorManager(processor);
    if (!manager || typeof manager.onEnd !== "function") {
      return;
    }

    if (OpenInferenceInstrumentor._patchedProcessorManager !== manager) {
      if (OpenInferenceInstrumentor._patchedProcessorManager) {
        OpenInferenceInstrumentor._restorePatchedManager();
      }

      const originalManagerOnEnd = manager.onEnd.bind(manager);
      const wrappedManagerOnEnd = (span: ReadableSpan) =>
        originalManagerOnEnd(
          isOpenInferenceSpan(span)
            ? prepareOpenInferenceSpanForExport(span)
            : span,
        );
      manager.onEnd = wrappedManagerOnEnd;

      OpenInferenceInstrumentor._patchedProcessorManager = manager;
      OpenInferenceInstrumentor._originalManagerOnEnd = originalManagerOnEnd;
      OpenInferenceInstrumentor._wrappedManagerOnEnd = wrappedManagerOnEnd;
    }
  }

  private static _removeTranslatorHook(): void {
    OpenInferenceInstrumentor._restorePatchedManager();
    OpenInferenceInstrumentor._restorePatchedProcessor();
  }

  activate(): void {
    this._instrumentor = new this._instrumentorClass();
    const tp = trace.getTracerProvider();

    // Set tracer provider if the instrumentor supports it
    if (typeof this._instrumentor.setTracerProvider === "function") {
      this._instrumentor.setTracerProvider(tp);
    }

    // ESM manual instrumentation (e.g. Claude Agent SDK)
    if (this._sdkModule && typeof this._instrumentor.manuallyInstrument === "function") {
      this._instrumentor.manuallyInstrument(this._sdkModule);
    }
    // Standard OTEL auto-patching
    else if (typeof this._instrumentor.instrument === "function") {
      this._instrumentor.instrument({ tracerProvider: tp });
    }
    // SpanProcessor-based (e.g. pydantic-ai, strands-agents)
    else if (tp && typeof (tp as any).addSpanProcessor === "function") {
      (tp as any).addSpanProcessor(this._instrumentor);
      this._isSpanProcessor = true;
    }

    OpenInferenceInstrumentor._installTranslatorHook();
    if (!this._ownsTranslatorHook) {
      OpenInferenceInstrumentor._translatorHookRefCount += 1;
      this._ownsTranslatorHook = true;
    }

    this._isInstrumented = true;
  }

  deactivate(): void {
    if (this._isInstrumented && this._instrumentor) {
      try {
        if (this._isSpanProcessor) {
          this._instrumentor.shutdown();
        } else {
          this._instrumentor.uninstrument();
        }
      } catch {
        /* ignore */
      }
      this._isInstrumented = false;
    }

    if (this._ownsTranslatorHook) {
      OpenInferenceInstrumentor._translatorHookRefCount = Math.max(
        0,
        OpenInferenceInstrumentor._translatorHookRefCount - 1
      );
      this._ownsTranslatorHook = false;
    }

    if (OpenInferenceInstrumentor._translatorHookRefCount === 0) {
      OpenInferenceInstrumentor._removeTranslatorHook();
    }
  }
}
