import { Context, Span } from "@opentelemetry/api";
import {
  SpanProcessor,
  ReadableSpan,
} from "@opentelemetry/sdk-trace-base";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  RESPAN_SPAN_ATTRIBUTES_MAP,
  RespanSpanAttributes,
} from "@respan/respan-sdk";
import { MultiProcessorManager } from "./manager.js";
import { getEntityPath, getPropagatedAttributes } from "../utils/context.js";
import { LOG_PREFIX, LOG_PREFIX_DEBUG, LOG_PREFIX_ERROR } from "../constants/index.js";
import { mergeCanonicalMetadataAttributes } from "../utils/metadata.js";
import {
  acquireSpanTransformerHost,
  releaseSpanTransformerHost,
  runSpanTransformersOnEnd,
  runSpanTransformersOnStart,
} from "./transformers.js";

// ── LLM span name patterns ────────────────────────────────────────────────
// Recognized span name substrings from auto-instrumentation libraries.
// Used to identify LLM calls for processing vs. filtering.
const LLM_SPAN_NAME_PATTERNS = [
  "anthropic.messages",
  "openai.chat",
  "chat.completions",
] as const;

// ── Composite processor ────────────────────────────────────────────────────

/**
 * Composite processor that combines filtering with multi-processor routing.
 *
 * Flow:
 * 1. Filter spans (keep only user-decorated spans and their children)
 * 2. Apply postprocess callback if configured
 * 3. Route filtered spans to appropriate processors
 *
 * This ensures only meaningful spans are routed to processors.
 */
export class RespanCompositeProcessor implements SpanProcessor {
  private readonly _processorManager: MultiProcessorManager;
  private readonly _postprocessCallback?: (span: ReadableSpan) => void;
  private _transformerHostReleased = false;
  private _shutdownPromise?: Promise<void>;

  constructor(
    processorManager: MultiProcessorManager,
    postprocessCallback?: (span: ReadableSpan) => void
  ) {
    this._processorManager = processorManager;
    this._postprocessCallback = postprocessCallback;
    acquireSpanTransformerHost();
  }

  onStart(span: Span, parentContext: Context): void {
    runSpanTransformersOnStart(span, parentContext);
    const readableSpan = span as unknown as ReadableSpan;

    // Check if this span is being created within an entity context
    // If so, add the entityPath attribute so it gets preserved by our filtering
    const entityPath = getEntityPath(parentContext);
    if (
      entityPath &&
      !readableSpan.attributes[SpanAttributes.TRACELOOP_SPAN_KIND]
    ) {
      // This is an auto-instrumentation span within an entity context
      // Add the entityPath attribute so it doesn't get filtered out
      console.debug(
        `[Respan Debug] Adding entityPath to auto-instrumentation span: ${readableSpan.name} (entityPath: ${entityPath})`
      );

      span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entityPath);
    }

    // Apply propagated attributes (customer_identifier, thread_identifier, etc.)
    const propagated = getPropagatedAttributes(parentContext);
    if (propagated) {
      for (const [key, value] of Object.entries(propagated)) {
        if (value === undefined || value === null) continue;
        const attrKey = RESPAN_SPAN_ATTRIBUTES_MAP[key];
        if (!attrKey) continue;

        if (key === "metadata") {
          span.setAttribute(
            attrKey,
            mergeCanonicalMetadataAttributes(
              readableSpan.attributes as Record<string, unknown>,
              value,
            ),
          );
        } else if (key === "prompt" && typeof value === "object") {
          span.setAttribute(attrKey, JSON.stringify(value));
        } else {
          span.setAttribute(attrKey, value as any);
        }
      }
    }

    // Forward to processor manager
    this._processorManager.onStart(readableSpan, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    span = runSpanTransformersOnEnd(span);

    // Strip OTEL/infrastructure attributes that pollute metadata on the backend.
    const attrs = (span as any).attributes;
    if (attrs) {
      for (const key of Object.keys(attrs)) {
        if (
          key.startsWith("otel.scope.") ||
          key.startsWith("next.") ||
          key.startsWith("http.") ||
          key.startsWith("net.") ||
          key === "service.name"
        ) {
          delete attrs[key];
        }
      }
    }

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

    // Check if this is an LLM instrumentation span (OpenAI, Anthropic, etc.)
    // These have gen_ai.* or llm.* attributes, or recognized span name patterns
    const isLLMSpan =
      span.attributes[RespanSpanAttributes.GEN_AI_SYSTEM] !== undefined ||
      span.attributes[RespanSpanAttributes.LLM_SYSTEM] !== undefined ||
      span.attributes[RespanSpanAttributes.GEN_AI_REQUEST_MODEL] !== undefined ||
      LLM_SPAN_NAME_PATTERNS.some((pattern) => span.name.includes(pattern));

    // Note: traceloop.entity.name and traceloop.entity.path are kept — they may be
    // needed by the backend (e.g. for logBatchResults, OpenAI Agents). Individual
    // instrumentation translators (Vercel, OpenInference) strip them if not needed.

    // Filter: only process spans that are user-decorated, within entity context, or LLM calls
    if (spanKind) {
      const kindStr = String(spanKind).toLowerCase();

      if (kindStr === "workflow") {
        // Workflow spans are promoted to root (clear parentSpanId).
        // This ensures the workflow is the top-level span in the trace.
        console.debug(
          `[Respan Debug] Processing workflow span as root: ${span.name}`
        );

        const rootSpan = Object.create(Object.getPrototypeOf(span));
        Object.assign(rootSpan, span);
        Object.defineProperty(rootSpan, 'parentSpanId', {
          value: undefined,
          writable: false,
          configurable: true,
          enumerable: true
        });
        // OTEL 2.x reads parentSpanContext (parentSpanId is ignored on the wire)
        Object.defineProperty(rootSpan, 'parentSpanContext', {
          value: undefined,
          writable: false,
          configurable: true,
          enumerable: true
        });
        this._processorManager.onEnd(rootSpan);
      } else {
        // Task, tool, agent spans keep their parent — preserving the hierarchy.
        console.debug(
          `[Respan Debug] Processing decorated span: ${span.name} (kind: ${spanKind})`
        );
        this._processorManager.onEnd(span);
      }
    } else if (entityPath && entityPath !== "") {
      // This span doesn't have a kind but has entityPath - it's a child span within a withEntity context.
      // Filter out HTTP/fetch noise (Next.js auto-instrumentation) — these are just the
      // underlying network calls for LLM requests that are already captured by ai.* spans.
      if (span.name.startsWith("fetch ") || span.name.startsWith("start response")) {
        console.debug(
          `[Respan Debug] Filtering out HTTP noise within entity context: ${span.name}`
        );
        return;
      }

      // Keep it as a normal child span (preserve parent relationships)
      console.debug(
        `[Respan Debug] Processing child span within entity context: ${span.name} (entityPath: ${entityPath})`
      );

      // Route to processors
      this._processorManager.onEnd(span);
    } else if (isLLMSpan) {
      // This is an LLM instrumentation span - keep it!
      console.debug(
        `[Respan Debug] Processing LLM instrumentation span: ${span.name}`
      );

      // Route to processors
      this._processorManager.onEnd(span);
    } else if (span.attributes[RespanSpanAttributes.OPENINFERENCE_SPAN_KIND] !== undefined) {
      // OpenInference span — already enriched by OpenInferenceTranslator processor
      console.debug(
        `[Respan Debug] Processing OpenInference span: ${span.name} (kind: ${span.attributes[RespanSpanAttributes.OPENINFERENCE_SPAN_KIND]})`
      );
      this._processorManager.onEnd(span);
    } else if (span.attributes[RespanSpanAttributes.RESPAN_LOG_TYPE] !== undefined) {
      // Enriched Respan span (from an instrumentation plugin)
      console.debug(
        `[Respan Debug] Processing enriched Respan span: ${span.name}`
      );
      this._processorManager.onEnd(span);
    } else {
      // This span has none of the above - it's pure auto-instrumentation noise (HTTP calls, etc.)
      console.debug(
        `[Respan Debug] Filtering out auto-instrumentation span: ${span.name}`
      );
    }
  }

  async shutdown(): Promise<void> {
    if (!this._shutdownPromise) {
      this._shutdownPromise = (async () => {
        try {
          await this._processorManager.shutdown();
        } finally {
          if (!this._transformerHostReleased) {
            this._transformerHostReleased = true;
            releaseSpanTransformerHost();
          }
        }
      })();
    }

    await this._shutdownPromise;
  }

  async forceFlush(): Promise<void> {
    await this._processorManager.forceFlush();
  }

  /**
   * Get the entity path from context
   */
  // Removed - now using imported getEntityPath function

  /**
   * Get the processor manager (for adding new processors)
   */
  public getProcessorManager(): MultiProcessorManager {
    return this._processorManager;
  }
}
