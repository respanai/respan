/**
 * Respan instrumentation plugin for the Vercel AI SDK.
 *
 * Registers a {@link VercelAITranslator} SpanProcessor that enriches Vercel AI SDK
 * OTEL spans with Traceloop/GenAI semantic conventions so they flow through the
 * unified Respan OTEL pipeline.
 *
 * AI SDK 4-6 emit OTEL spans when experimental_telemetry is enabled. AI SDK 7
 * emits them through @ai-sdk/otel, which this instrumentor registers when that
 * optional peer is installed. Both formats are translated into the canonical
 * Respan span contract.
 *
 * ```typescript
 * import { Respan } from "@respan/respan";
 * import { VercelAIInstrumentor } from "@respan/instrumentation-vercel";
 *
 * const respan = new Respan({
 *   instrumentations: [new VercelAIInstrumentor()],
 * });
 * await respan.initialize();
 * ```
 */

import { trace } from "@opentelemetry/api";
import {
  ensureAISDKTelemetry,
  releaseOwnedAISDKTelemetry,
  type OwnedAISDKTelemetryLease,
} from "./_ai_sdk_telemetry.js";
import { VercelAITranslator } from "./_translator.js";

export { VercelAITranslator } from "./_translator.js";
export {
  VERCEL_SPAN_CONFIG,
  VERCEL_PARENT_SPANS,
  VERCEL_STRUCTURAL_LLM_PARENT_SPANS,
} from "./constants/index.js";

export interface VercelAIInstrumentorOptions {
  /**
   * Automatically register @ai-sdk/otel when AI SDK 7 is detected.
   * Defaults to true. Set false when telemetry registration is managed by the app.
   */
  autoRegisterAISDKTelemetry?: boolean;
}

export class VercelAIInstrumentor {
  public readonly name = "vercel-ai";

  /** Each live TracerProvider owns one shared translator processor. */
  private static readonly _translatorProviders = new WeakMap<object, VercelAITranslator>();

  private readonly _autoRegisterAISDKTelemetry: boolean;
  private _aiSDKTelemetryLease: OwnedAISDKTelemetryLease | undefined;
  private _translator: VercelAITranslator | undefined;
  private _activationGeneration = 0;
  private _active = false;

  constructor(options: VercelAIInstrumentorOptions = {}) {
    this._autoRegisterAISDKTelemetry = options.autoRegisterAISDKTelemetry ?? true;
  }

  protected _ensureAISDKTelemetry(): ReturnType<typeof ensureAISDKTelemetry> {
    return ensureAISDKTelemetry();
  }

  async activate(): Promise<void> {
    if (this._active) {
      return;
    }

    this._active = true;
    const activationGeneration = ++this._activationGeneration;

    try {
      // Walk the TracerProvider chain to find one that supports addSpanProcessor.
      // trace.getTracerProvider() returns a ProxyTracerProvider; the real
      // NodeTracerProvider lives at ._delegate (or ._delegate._tracerProvider).
      const tp = trace.getTracerProvider() as any;
      const provider =
        (typeof tp?.addSpanProcessor === "function" && tp) ||
        (typeof tp?._delegate?.addSpanProcessor === "function" && tp._delegate) ||
        (typeof tp?._delegate?._tracerProvider?.addSpanProcessor === "function" &&
          tp._delegate._tracerProvider) ||
        null;

      if (provider) {
        let translator = VercelAIInstrumentor._translatorProviders.get(provider);
        if (!translator) {
          // NOTE: The translator is registered AFTER CompositeProcessor (which was
          // added during startTracing). This means CompositeProcessor.onEnd routes
          // spans to BatchSpanProcessor before our onEnd enriches them. This works
          // because BatchSpanProcessor holds span references and exports async —
          // our enrichment completes before the next batch flush. If
          // SimpleSpanProcessor is ever used, the translator must register first.
          translator = new VercelAITranslator({ initiallyActive: false });
          provider.addSpanProcessor(translator);
          VercelAIInstrumentor._translatorProviders.set(provider, translator);
        }

        translator.acquire();
        this._translator = translator;
      }

      if (this._autoRegisterAISDKTelemetry && !this._aiSDKTelemetryLease) {
        const registration = await this._ensureAISDKTelemetry();

        // If deactivate() (or a newer activate()) ran while optional modules were
        // loading, immediately return this acquisition. Otherwise a late
        // registration could survive after the instrumentor was deactivated.
        if (activationGeneration !== this._activationGeneration) {
          if (registration.lease) {
            releaseOwnedAISDKTelemetry(registration.lease);
          }
          return;
        }

        this._aiSDKTelemetryLease = registration.lease;
      }
    } catch (error) {
      // A failed optional-adapter activation must leave this instance retryable
      // and must return its translator ownership to the shared provider state.
      if (activationGeneration === this._activationGeneration) {
        this._active = false;
        this._activationGeneration += 1;
        if (this._translator) {
          this._translator.release();
          this._translator = undefined;
        }
        if (this._aiSDKTelemetryLease) {
          releaseOwnedAISDKTelemetry(this._aiSDKTelemetryLease);
          this._aiSDKTelemetryLease = undefined;
        }
      }
      throw error;
    }
  }
  deactivate(): void {
    this._active = false;
    this._activationGeneration += 1;

    // The provider retains its single processor until shutdown. Releasing this
    // owner's lease disables translation only after the final active owner
    // deactivates, and later activation reuses the same processor.
    if (this._translator) {
      this._translator.release();
      this._translator = undefined;
    }

    if (this._aiSDKTelemetryLease) {
      releaseOwnedAISDKTelemetry(this._aiSDKTelemetryLease);
      this._aiSDKTelemetryLease = undefined;
    }
  }
}
