import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-base";
import type { RespanSpanNameStyle } from "@respan/respan-sdk";
import { transformReadableSpanBatch } from "@respan/tracing";
import { N8nSpanProcessor } from "./_processor.js";
import { sanitizeN8nSpanForFailSafeExport } from "./_translator.js";

export interface N8nTransformingExporterOptions {
  spanNameStyle?: RespanSpanNameStyle | string;
}

/**
 * Applies export-only cleanup and semantic naming, then delegates to n8n's
 * original OTLP/protobuf exporter. It never creates or registers a provider.
 */
export class N8nTransformingExporter implements SpanExporter {
  private readonly _spanNameStyle: RespanSpanNameStyle | string | undefined;

  constructor(
    private readonly _delegate: SpanExporter,
    private readonly _processor: N8nSpanProcessor,
    options: N8nTransformingExporterOptions = {},
  ) {
    this._spanNameStyle = options.spanNameStyle ?? process.env.RESPAN_SPAN_NAME_STYLE;
  }

  export(
    spans: ReadableSpan[],
    resultCallback: Parameters<SpanExporter["export"]>[1],
  ): void {
    let exportSpans = spans;
    try {
      const prepared = spans.map((span) => this._processor.prepareForExport(span));
      exportSpans = transformReadableSpanBatch(prepared, this._spanNameStyle);
    } catch (error) {
      console.warn(
        "[respan] n8n export transformation failed; delegating a privacy-sanitized span batch",
        error,
      );
      exportSpans = spans.flatMap((span) => {
        try {
          return [sanitizeN8nSpanForFailSafeExport(span)];
        } catch (sanitizeError) {
          console.warn(
            "[respan] n8n fail-safe privacy cleanup failed; dropping the affected span",
            sanitizeError,
          );
          return [];
        }
      });
    }

    this._delegate.export(exportSpans, resultCallback);
  }

  shutdown(): Promise<void> {
    return this._delegate.shutdown();
  }

  forceFlush(): Promise<void> {
    const forceFlush = this._delegate.forceFlush;
    return forceFlush ? forceFlush.call(this._delegate) : Promise.resolve();
  }
}
