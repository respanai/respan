/**
 * Respan instrumentation plugin for Arize Phoenix TypeScript helpers.
 *
 * `@arizeai/phoenix-otel` re-exports OpenInference helpers such as
 * `traceAgent`, `traceChain`, `traceTool`, `withSpan`, and `observe`.
 * Those helpers emit OpenInference spans through the current OpenTelemetry
 * tracer provider. This plugin leaves Phoenix registration alone and only
 * installs the Respan OpenInference translation hook on the active provider.
 */

import { trace } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  isOpenInferenceSpan,
  prepareOpenInferenceSpanForExport,
  translateOpenInferenceSpan,
} from "@respan/instrumentation-openinference";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

type ProcessorOnEnd = (span: ReadableSpan) => void;

export interface ArizeInstrumentorOptions {
  /**
   * Name reported to the Respan plugin registry.
   */
  name?: string;
}

export class ArizeInstrumentor {
  public readonly name: string;

  private _isInstrumented = false;
  private _ownsTranslatorHook = false;

  private static _translatorHookRefCount = 0;
  private static _patchedProcessor: any = null;
  private static _originalProcessorOnEnd: ProcessorOnEnd | null = null;
  private static _wrappedProcessorOnEnd: ProcessorOnEnd | null = null;
  private static _patchedProcessorManager: any = null;
  private static _originalManagerOnEnd: ProcessorOnEnd | null = null;
  private static _wrappedManagerOnEnd: ProcessorOnEnd | null = null;

  constructor(options: ArizeInstrumentorOptions = {}) {
    this.name = options.name ?? "arize";
  }

  activate(): void {
    if (this._isInstrumented) {
      return;
    }

    ArizeInstrumentor._installTranslatorHook();
    if (!this._ownsTranslatorHook) {
      ArizeInstrumentor._translatorHookRefCount += 1;
      this._ownsTranslatorHook = true;
    }

    this._isInstrumented = true;
  }

  deactivate(): void {
    if (!this._isInstrumented) {
      return;
    }

    if (this._ownsTranslatorHook) {
      ArizeInstrumentor._translatorHookRefCount = Math.max(
        0,
        ArizeInstrumentor._translatorHookRefCount - 1,
      );
      this._ownsTranslatorHook = false;
    }

    if (ArizeInstrumentor._translatorHookRefCount === 0) {
      ArizeInstrumentor._removeTranslatorHook();
    }

    this._isInstrumented = false;
  }

  isActive(): boolean {
    return this._isInstrumented;
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
    originalOnEnd: ProcessorOnEnd | null,
    wrappedOnEnd: ProcessorOnEnd | null,
    label: string,
  ): boolean {
    if (!target || !originalOnEnd) {
      return true;
    }

    if (wrappedOnEnd && target.onEnd !== wrappedOnEnd) {
      console.warn(
        `[respan] ArizeInstrumentor: ${label}.onEnd was modified externally; original handler could not be restored.`,
      );
      return false;
    }

    target.onEnd = originalOnEnd;
    return true;
  }

  private static _restorePatchedProcessor(): void {
    if (
      ArizeInstrumentor._restoreHook(
        ArizeInstrumentor._patchedProcessor,
        ArizeInstrumentor._originalProcessorOnEnd,
        ArizeInstrumentor._wrappedProcessorOnEnd,
        "active span processor",
      )
    ) {
      ArizeInstrumentor._patchedProcessor = null;
      ArizeInstrumentor._originalProcessorOnEnd = null;
      ArizeInstrumentor._wrappedProcessorOnEnd = null;
    }
  }

  private static _restorePatchedManager(): void {
    if (
      ArizeInstrumentor._restoreHook(
        ArizeInstrumentor._patchedProcessorManager,
        ArizeInstrumentor._originalManagerOnEnd,
        ArizeInstrumentor._wrappedManagerOnEnd,
        "processor manager",
      )
    ) {
      ArizeInstrumentor._patchedProcessorManager = null;
      ArizeInstrumentor._originalManagerOnEnd = null;
      ArizeInstrumentor._wrappedManagerOnEnd = null;
    }
  }

  private static _installTranslatorHook(): void {
    const processor = ArizeInstrumentor._getActiveSpanProcessor();
    if (!processor || typeof processor.onEnd !== "function") {
      return;
    }

    if (ArizeInstrumentor._patchedProcessor !== processor) {
      if (ArizeInstrumentor._patchedProcessor) {
        ArizeInstrumentor._restorePatchedProcessor();
      }

      const originalProcessorOnEnd = processor.onEnd as ProcessorOnEnd;
      const hasProcessorManager =
        ArizeInstrumentor._getProcessorManager(processor) !== null;
      const wrappedProcessorOnEnd = (span: ReadableSpan) => {
        if (isOpenInferenceSpan(span)) {
          try {
            translateOpenInferenceSpan(span);
            promoteWorkflowName(span);
          } catch {
            // Unexpected OpenInference span shapes must not block export.
          }
        }

        return originalProcessorOnEnd.call(
          processor,
          hasProcessorManager || !isOpenInferenceSpan(span)
            ? span
            : prepareOpenInferenceSpanForExport(span),
        );
      };
      processor.onEnd = wrappedProcessorOnEnd;

      ArizeInstrumentor._patchedProcessor = processor;
      ArizeInstrumentor._originalProcessorOnEnd = originalProcessorOnEnd;
      ArizeInstrumentor._wrappedProcessorOnEnd = wrappedProcessorOnEnd;
    }

    const manager = ArizeInstrumentor._getProcessorManager(processor);
    if (!manager || typeof manager.onEnd !== "function") {
      return;
    }

    if (ArizeInstrumentor._patchedProcessorManager !== manager) {
      if (ArizeInstrumentor._patchedProcessorManager) {
        ArizeInstrumentor._restorePatchedManager();
      }

      const originalManagerOnEnd = manager.onEnd as ProcessorOnEnd;
      const wrappedManagerOnEnd = (span: ReadableSpan) =>
        originalManagerOnEnd.call(
          manager,
          isOpenInferenceSpan(span)
            ? prepareOpenInferenceSpanForExport(span)
            : span,
        );
      manager.onEnd = wrappedManagerOnEnd;

      ArizeInstrumentor._patchedProcessorManager = manager;
      ArizeInstrumentor._originalManagerOnEnd = originalManagerOnEnd;
      ArizeInstrumentor._wrappedManagerOnEnd = wrappedManagerOnEnd;
    }
  }

  private static _removeTranslatorHook(): void {
    ArizeInstrumentor._restorePatchedManager();
    ArizeInstrumentor._restorePatchedProcessor();
  }
}

function promoteWorkflowName(span: ReadableSpan): void {
  const attrs = (span as any).attributes as Record<string, unknown> | undefined;
  if (!attrs || attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] !== undefined) {
    return;
  }

  const entityPath = attrs[SpanAttributes.TRACELOOP_ENTITY_PATH];
  if (typeof entityPath === "string" && entityPath.length > 0) {
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = entityPath;
  }
}

export { ArizeInstrumentor as ArizeInstrumentation };
