import { Context } from "@opentelemetry/api";
import { 
  SpanProcessor, 
  ReadableSpan, 
  BatchSpanProcessor,
  SpanExporter
} from "@opentelemetry/sdk-trace-base";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import { getEntityPath } from "../utils/context.js";

/**
 * Custom span processor that implements intelligent span filtering:
 * 1. User-decorated spans (withEntity) become root spans (no parent hierarchy)
 * 2. Auto-instrumentation spans within withEntity context get entityPath and are preserved
 * 3. Pure auto-instrumentation spans (no context) are filtered out
 * 
 * This eliminates noise while preserving useful spans like OpenAI/HTTP calls within user workflows.
 */
export class RespanFilteringSpanProcessor implements SpanProcessor {
  private readonly _spanProcessor: SpanProcessor;
  private readonly _postprocessCallback?: (span: ReadableSpan) => void;

  constructor(
    exporter: SpanExporter,
    postprocessCallback?: (span: ReadableSpan) => void
  ) {
    this._spanProcessor = new BatchSpanProcessor(exporter);
    this._postprocessCallback = postprocessCallback;
  }

  onStart(span: ReadableSpan, parentContext: Context): void {
    // Check if this span is being created within an entity context
    // If so, add the entityPath attribute so it gets preserved by our filtering
    const entityPath = getEntityPath(parentContext);
    if (entityPath && !span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND]) {
      // This is an auto-instrumentation span within an entity context
      // Add the entityPath attribute so it doesn't get filtered out
      console.debug(
        `[Respan Debug] Adding entityPath to auto-instrumentation span: ${span.name} (entityPath: ${entityPath})`
      );
      
      // We need to cast to any to set attributes during onStart
      (span as any).setAttribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entityPath);
    }
    
    // Always call onStart for all spans to maintain proper span lifecycle
    this._spanProcessor.onStart(span as any, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    const spanKind = span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND];
    const entityPath = span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH];
    
    // Apply postprocess callback if provided
    if (this._postprocessCallback) {
      try {
        this._postprocessCallback(span);
      } catch (error) {
        console.error("[Respan] Error in span postprocess callback:", error);
      }
    }
    
    if (spanKind) {
      // This is a user-decorated span (withWorkflow, withTask, etc.) - make it a root span
      console.debug(
        `[Respan Debug] Sending user-decorated span as root: ${span.name} (kind: ${spanKind})`
      );

      // Create a wrapper that makes the span appear as a root span
      const rootSpan = Object.create(Object.getPrototypeOf(span));
      Object.assign(rootSpan, span);
      
      // Override the parent to make it a root span. OTEL 2.x reads
      // parentSpanContext on the wire (parentSpanId is ignored), so clear both.
      Object.defineProperty(rootSpan, 'parentSpanId', {
        value: undefined,
        writable: false,
        configurable: true,
        enumerable: true
      });
      Object.defineProperty(rootSpan, 'parentSpanContext', {
        value: undefined,
        writable: false,
        configurable: true,
        enumerable: true
      });

      this._spanProcessor.onEnd(rootSpan);
    } else if (entityPath && entityPath !== "") {
      // This span doesn't have a kind but has entityPath - it's a child span within a withEntity context
      // Keep it as a normal child span (preserve parent relationships)
      console.debug(
        `[Respan Debug] Sending child span within entity context: ${span.name} (entityPath: ${entityPath})`
      );
      
      this._spanProcessor.onEnd(span);
    } else {
      // This span has neither kind nor entityPath - it's pure auto-instrumentation noise
      console.debug(
        `[Respan Debug] Filtering out auto-instrumentation span: ${span.name} (no TRACELOOP_SPAN_KIND or entityPath)`
      );
    }
  }

  shutdown(): Promise<void> {
    return this._spanProcessor.shutdown();
  }

  forceFlush(): Promise<void> {
    return this._spanProcessor.forceFlush();
  }
} 