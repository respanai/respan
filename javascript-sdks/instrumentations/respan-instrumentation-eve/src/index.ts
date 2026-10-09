/**
 * Respan instrumentation plugin for the Eve agent framework.
 *
 * Eve enables its vendored AI SDK OpenTelemetry integration whenever
 * `agent/instrumentation.ts` is present. This plugin registers a synchronous
 * transformer with Respan tracing so Eve turn, model, and tool spans are
 * translated before Respan filters and exports them.
 */

import {
  registerSpanTransformer,
  type RespanSpanTransformer,
  type SpanTransformerRegistration,
} from "@respan/tracing";
import { EveSpanProcessor } from "./_processor.js";
import type { SpanExporter } from "@opentelemetry/sdk-trace-base";

export { withEveLineage } from "./lineage.js";

const TRANSFORMER_KEY = "@respan/instrumentation-eve";

export class EveInstrumentor {
  public readonly name = "eve";

  private static readonly _processor = new EveSpanProcessor({
    initiallyActive: false,
  });
  private static readonly _transformer: RespanSpanTransformer = {
    onStart: (span, parentContext) =>
      EveInstrumentor._processor.onStart(span, parentContext),
    onEnd: (span) => EveInstrumentor._processor.onEnd(span),
    prepareForExport: (span) =>
      EveInstrumentor._processor.prepareForExport(span),
    dispose: () => {
      void EveInstrumentor._processor.shutdown();
    },
  };

  private _registration: SpanTransformerRegistration | null = null;
  private _active = false;

  activate(): void {
    if (this._active) {
      return;
    }

    EveInstrumentor._processor.acquire();
    try {
      this._registration = registerSpanTransformer(
        TRANSFORMER_KEY,
        EveInstrumentor._transformer,
      );
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

    this._registration?.unregister();
    this._registration = null;
    EveInstrumentor._processor.release();
    this._active = false;
  }

  isActive(): boolean {
    return this._active;
  }

  /** Use for an export-time privacy ceiling when Respan owns the provider. */
  wrapExporter(exporter: SpanExporter, spanNameStyle?: string): SpanExporter {
    return EveInstrumentor._processor.wrapExporter(exporter, spanNameStyle);
  }
}

export { EveSpanProcessor };
