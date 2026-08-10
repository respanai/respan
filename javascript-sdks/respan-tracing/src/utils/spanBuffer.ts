import { context, trace, SpanKind, SpanContext, ROOT_CONTEXT, Context } from "@opentelemetry/api";
import { ReadableSpan, Span } from "@opentelemetry/sdk-trace-base";
import { hrTime } from "@opentelemetry/core";

/**
 * Interface for managing buffered spans.
 * 
 * SpanBuffer provides manual control over span creation and export timing.
 * Unlike automatic tracing, spans are buffered locally and only exported
 * when you explicitly call processSpans().
 * 
 * Key features:
 * - Manual span creation with attributes
 * - Local buffering (no automatic export)
 * - Transportable spans (extract and process anywhere)
 * - Context isolation
 * 
 * @example
 * ```typescript
 * const client = getClient();
 * let collectedSpans: ReadableSpan[] = [];
 * 
 * // Collect spans during execution
 * with (client.getSpanBuffer("trace-123")) {
 *   buffer.createSpan("step1", { status: "completed" });
 *   buffer.createSpan("step2", { status: "completed" });
 *   collectedSpans = buffer.getAllSpans();
 * }
 * 
 * // Process later based on business logic
 * if (shouldExport) {
 *   client.processSpans(collectedSpans);
 * }
 * ```
 */
export interface SpanBuffer {
  /**
   * Create a span in the buffer
   * @param name - Span name
   * @param attributes - Optional span attributes
   * @param kind - Optional span kind (default: INTERNAL)
   */
  createSpan(
    name: string,
    attributes?: Record<string, any>,
    kind?: SpanKind
  ): void;

  /**
   * Get all buffered spans as ReadableSpan objects
   * @returns Array of all buffered spans
   */
  getAllSpans(): ReadableSpan[];

  /**
   * Get the count of buffered spans
   * @returns Number of spans in the buffer
   */
  getSpanCount(): number;

  /**
   * Clear all buffered spans without processing them
   */
  clearSpans(): void;
}

/**
 * Context key for span buffer
 */
const SPAN_BUFFER_KEY = Symbol("respan.span_buffer");

/**
 * Implementation of SpanBuffer
 */
class SpanBufferImpl implements SpanBuffer {
  private spans: ReadableSpan[] = [];
  private readonly traceId: string;
  private readonly tracer = trace.getTracer("@respan/tracing");

  constructor(traceId: string) {
    this.traceId = traceId;
  }

  createSpan(
    name: string,
    attributes?: Record<string, any>,
    kind: SpanKind = SpanKind.INTERNAL
  ): void {
    // Create a span context for this buffered span
    const spanContext: SpanContext = {
      traceId: this.traceId,
      spanId: this.generateSpanId(),
      traceFlags: 1, // Sampled
    };

    // Create the span
    const span = this.tracer.startSpan(
      name,
      {
        kind,
        startTime: hrTime(),
      },
      ROOT_CONTEXT
    );

    // Set attributes
    if (attributes) {
      for (const [key, value] of Object.entries(attributes)) {
        span.setAttribute(key, value);
      }
    }

    // End the span immediately (we're creating historical spans)
    span.end();

    // Add to buffer
    // @ts-ignore - Access internal span for buffering
    this.spans.push(span as unknown as ReadableSpan);

    console.debug(
      `[Respan] Buffered span: ${name} (total: ${this.spans.length})`
    );
  }

  getAllSpans(): ReadableSpan[] {
    return [...this.spans];
  }

  getSpanCount(): number {
    return this.spans.length;
  }

  clearSpans(): void {
    const count = this.spans.length;
    this.spans = [];
    console.debug(`[Respan] Cleared ${count} buffered spans`);
  }

  private generateSpanId(): string {
    // Generate a random 16-character hex string for span ID
    return Array.from({ length: 16 }, () =>
      Math.floor(Math.random() * 16).toString(16)
    ).join("");
  }
}

/**
 * Manager for span buffers with context isolation.
 * 
 * This class provides:
 * - Context-isolated span buffering
 * - Manual span processing through OTEL pipeline
 * - Support for transportable spans
 */
export class SpanBufferManager {
  /**
   * Create a new span buffer with the given trace ID
   * @param traceId - Trace ID for all spans in this buffer
   * @returns A new SpanBuffer instance
   */
  createBuffer(traceId: string): SpanBuffer {
    return new SpanBufferImpl(traceId);
  }

  /**
   * Process spans through the OpenTelemetry processor pipeline.
   * This sends the spans to all configured processors.
   * 
   * @param spans - Array of ReadableSpan objects or a SpanBuffer
   * @returns Promise<boolean> - True if processing succeeded
   * 
   * @example
   * ```typescript
   * const manager = new SpanBufferManager();
   * const buffer = manager.createBuffer("trace-123");
   * buffer.createSpan("task", { result: "success" });
   * 
   * const spans = buffer.getAllSpans();
   * await manager.processSpans(spans);
   * ```
   */
  async processSpans(spans: ReadableSpan[] | SpanBuffer): Promise<boolean> {
    try {
      const spanArray = Array.isArray(spans) ? spans : spans.getAllSpans();

      if (spanArray.length === 0) {
        console.debug("[Respan] No spans to process");
        return true;
      }

      console.debug(`[Respan] Processing ${spanArray.length} buffered spans`);

      // Get the SDK instance to access processors
      const { getClient } = await import("./tracing.js");
      const sdk = getClient();

      if (!sdk) {
        console.warn("[Respan] SDK not initialized, cannot process spans");
        return false;
      }

      // Access the tracer provider and processors
      // @ts-ignore - Access internal SDK structure
      const tracerProvider = sdk._tracerProvider;
      if (!tracerProvider) {
        console.warn("[Respan] TracerProvider not found");
        return false;
      }

      // Get active span processor. OTEL 2.x renamed the internal field to
      // `_activeSpanProcessor` (underscore) and wraps in a ProxyTracerProvider,
      // so check both names at the top level and under `_delegate`.
      const tp = tracerProvider as any;
      const activeSpanProcessor =
        tp.activeSpanProcessor ??
        tp._activeSpanProcessor ??
        tp._delegate?.activeSpanProcessor ??
        tp._delegate?._activeSpanProcessor;
      if (!activeSpanProcessor) {
        console.warn("[Respan] No active span processor found");
        return false;
      }

      // Process each span through the pipeline
      for (const span of spanArray) {
        activeSpanProcessor.onEnd(span);
      }

      console.debug("[Respan] Successfully processed buffered spans");
      return true;
    } catch (error) {
      console.error("[Respan] Error processing spans:", error);
      return false;
    }
  }
}

/**
 * Global span buffer manager instance
 */
let _bufferManager: SpanBufferManager | undefined;

/**
 * Get the span buffer manager instance
 */
export function getSpanBufferManager(): SpanBufferManager {
  if (!_bufferManager) {
    _bufferManager = new SpanBufferManager();
  }
  return _bufferManager;
}


