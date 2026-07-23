/**
 * Translate Vercel AI SDK spans → Traceloop/OpenLLMetry format.
 *
 * The Vercel AI SDK emits OTEL spans with its own attribute schema (ai.model.id,
 * ai.prompt.messages, ai.response.text, etc.). This SpanProcessor enriches those
 * spans with the Traceloop/GenAI semantic conventions the Respan backend expects.
 *
 * Two-phase enrichment:
 * - onStart(): Sets RESPAN_LOG_TYPE so the span passes CompositeProcessor filtering
 * - onEnd():   Full attribute translation (model, messages, tokens, metadata, etc.)
 */

import type { Context } from "@opentelemetry/api";
import type { ReadableSpan, Span, SpanProcessor } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_AGENT_ID,
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_TOOL_NAME,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes as TraceloopSpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  VERCEL_PARENT_SPANS,
  VERCEL_SPAN_CONFIG,
  VERCEL_STRUCTURAL_LLM_PARENT_SPANS,
} from "./constants/index.js";
import {
  formatCompletionOutput,
  formatPromptInput,
  formatToolInput,
  formatToolOutput,
  parseToolChoice,
  parseToolsValue,
} from "./_translator/messages.js";
import {
  AI_AGENT_ID,
  AI_MODEL_ID,
  AI_PREFIX,
  AI_TELEMETRY_METADATA_PREFIX,
  AI_TOOL_CALL_NAME,
  formatEmbeddingInput,
  formatEmbeddingOutput,
  instrumentationScopeName,
  isModernVercelAISpanName,
  isVercelAISpan,
  isVercelAIScope,
  setMetadata,
  resolveLogType,
  safeJsonStr,
  setDefault,
} from "./_translator/shared.js";
import { enrichMetadata, enrichModel, enrichPerformanceMetrics, enrichSystem, enrichTokens, stripRedundantAttrs } from "./_translator/span-enrichment.js";

/**
 * SpanProcessor that translates Vercel AI SDK attributes to Traceloop/OpenLLMetry.
 *
 * Phase 1 (onStart): Sets RESPAN_LOG_TYPE so CompositeProcessor lets the span through.
 * Phase 2 (onEnd):   Full attribute enrichment — model, messages, tokens, metadata,
 *                     tools, performance metrics, environment, etc.
 */
export class VercelAITranslator implements SpanProcessor {
  private _ownerCount: number;
  private readonly _inFlightSpans = new WeakSet<object>();
  /** Open structural wrapper spans: spanId → its own parentSpanId. */
  private readonly _openStructuralSpans = new Map<string, string | undefined>();

  constructor({ initiallyActive = true }: { initiallyActive?: boolean } = {}) {
    this._ownerCount = initiallyActive ? 1 : 0;
  }

  acquire(): void {
    this._ownerCount += 1;
  }

  release(): void {
    this._ownerCount = Math.max(0, this._ownerCount - 1);
  }

  onStart(span: Span, _parentContext: Context): void {
    if (this._ownerCount === 0) {
      return;
    }

    const writableSpan = span as any;
    const name: string = writableSpan.name ?? "";
    const scopeName = instrumentationScopeName(writableSpan);
    if (!name.startsWith(AI_PREFIX) && !isModernVercelAISpanName(name) && !isVercelAIScope(scopeName)) {
      return;
    }

    this._inFlightSpans.add(span as object);

    const spanId: string | undefined =
      typeof writableSpan.spanContext === "function"
        ? writableSpan.spanContext()?.spanId
        : undefined;
    const parentSpanId: string | undefined = writableSpan.parentSpanId;

    // If the parent chain starts inside an open structural wrapper, stamp the
    // export-time parent (the wrapper's own parent) so the exporter can drop
    // the wrapper per-span, immune to export-batch boundaries. "" marks a
    // wrapper that was itself a root span — the child is promoted to root.
    if (parentSpanId && this._openStructuralSpans.has(parentSpanId)) {
      let exportParent: string | undefined = parentSpanId;
      const seen = new Set<string>();
      while (
        exportParent &&
        this._openStructuralSpans.has(exportParent) &&
        !seen.has(exportParent)
      ) {
        seen.add(exportParent);
        exportParent = this._openStructuralSpans.get(exportParent);
      }
      writableSpan.setAttribute(
        RespanSpanAttributes.RESPAN_INTERNAL_EXPORT_PARENT,
        exportParent ?? ""
      );
    }

    if (VERCEL_STRUCTURAL_LLM_PARENT_SPANS.has(name)) {
      // Structural wrapper: the .doGenerate/.doStream child carries the real
      // model/input/output. The semantic export style drops it (children are
      // reparented); the legacy style still exports it untouched.
      writableSpan.setAttribute(RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN, true);
      if (spanId) {
        this._openStructuralSpans.set(spanId, parentSpanId);
      }
    }

    const config = VERCEL_SPAN_CONFIG[name];
    if (config) {
      writableSpan.setAttribute(RespanSpanAttributes.RESPAN_LOG_TYPE, config.logType);
      return;
    }

    const parentLogType = VERCEL_PARENT_SPANS[name];
    if (parentLogType !== undefined) {
      writableSpan.setAttribute(RespanSpanAttributes.RESPAN_LOG_TYPE, parentLogType);
      return;
    }

    writableSpan.setAttribute(RespanSpanAttributes.RESPAN_LOG_TYPE, RespanLogType.TASK);
  }

  onEnd(span: ReadableSpan): void {
    const endedSpanId = span.spanContext?.().spanId;
    if (endedSpanId) {
      this._openStructuralSpans.delete(endedSpanId);
    }

    const startedWhileActive = this._inFlightSpans.delete(span as object);
    if (this._ownerCount === 0 && !startedWhileActive) {
      return;
    }

    const attrs = (span as any).attributes as Record<string, any> | undefined;
    if (!attrs || !isVercelAISpan(span)) {
      return;
    }

    const name = span.name;
    const config = VERCEL_SPAN_CONFIG[name];
    const parentLogType = VERCEL_PARENT_SPANS[name];
    const logType = resolveLogType(name, attrs);

    // Embedding spans (span-contract.md): input = embedded text, output = the
    // embedding vector(s) — captured, not dropped (debuggable RAG data; size is
    // handled by storage tiering, not by deleting it here). Vercel's synthetic
    // ai.usage.tokens is intentionally NOT surfaced as a token count; it's
    // stripped. Extract up front, before any early-return or metadata move, so
    // both the parent (ai.embed) and child (ai.embed.doEmbed) spans are covered.
    if (
      logType === RespanLogType.EMBEDDING ||
      config?.logType === RespanLogType.EMBEDDING ||
      parentLogType === RespanLogType.EMBEDDING
    ) {
      const embInput = formatEmbeddingInput(attrs);
      if (embInput) setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT, embInput);
      const embOutput = formatEmbeddingOutput(attrs);
      if (embOutput) setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT, embOutput);
    }

    const entityName =
      logType === RespanLogType.AGENT
        ? attrs[ATTR_GEN_AI_AGENT_NAME] ??
          attrs[ATTR_GEN_AI_AGENT_ID] ??
          attrs["ai.agent.name"] ??
          attrs[AI_AGENT_ID] ??
          attrs[AI_TELEMETRY_METADATA_PREFIX + "agent_name"] ??
          name
        : name;

    enrichMetadata(attrs);
    delete attrs[TraceloopSpanAttributes.TRACELOOP_SPAN_KIND];

    attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = logType;
    setDefault(
      attrs,
      TraceloopSpanAttributes.TRACELOOP_ENTITY_NAME,
      String(entityName),
    );
    setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_PATH, "");

    if (config) {
      // Do NOT set traceloop.span.kind for auto-emitted Vercel SDK spans.
      // In the Respan composite processor `traceloop.span.kind` is reserved
      // for user-decorated spans (withWorkflow / withTask / withAgent) and
      // setting it on auto spans (a) flattens the parent/child tree and
      // (b) causes LLM detail spans (doGenerate / doStream) to be classified
      // as "task" instead of LLM in the backend. The respan.entity.log_type
      // attribute (set above) carries the correct type for ingestion.
      // Matches the patterns in respan-instrumentation-openinference (see
      // _translator.ts:500) and respan-instrumentation-openai-agents
      // (see _otel_emitter.ts:398).

      if (config.isLLM) {
        setDefault(attrs, TraceloopSpanAttributes.LLM_REQUEST_TYPE, RespanLogType.CHAT);

        enrichSystem(attrs);
        enrichModel(attrs, attrs[AI_MODEL_ID]);

        const input = formatPromptInput(attrs);
        if (input) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT, input);
        }

        const output = formatCompletionOutput(attrs);
        if (output) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT, output);
        }

        enrichTokens(attrs);

        const toolsValue = parseToolsValue(attrs);
        if (toolsValue) {
          attrs[TraceloopSpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJsonStr(toolsValue);
        }

        const toolChoice = parseToolChoice(attrs);
        if (toolChoice) {
          setMetadata(attrs, "tool_choice", toolChoice);
        }

        enrichPerformanceMetrics(attrs, name);
      }

      if (config.logType === RespanLogType.EMBEDDING || logType === RespanLogType.EMBEDDING) {
        // input/output/tokens are mapped in the up-front embedding block.
        setDefault(attrs, TraceloopSpanAttributes.LLM_REQUEST_TYPE, RespanLogType.EMBEDDING);
        enrichSystem(attrs);
        enrichModel(attrs, attrs[AI_MODEL_ID]);
      }

      if (config.logType === RespanLogType.TOOL || logType === RespanLogType.TOOL) {
        setToolSpanNameHint(attrs, name);

        const toolInput = formatToolInput(attrs);
        if (toolInput) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT, toolInput);
        }

        const toolOutput = formatToolOutput(attrs);
        if (toolOutput) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT, toolOutput);
        }
      }

    } else {
      if (logType === RespanLogType.TEXT) {
        enrichSystem(attrs);
        enrichModel(attrs, attrs[AI_MODEL_ID]);

        enrichTokens(attrs);

        setDefault(attrs, TraceloopSpanAttributes.LLM_REQUEST_TYPE, RespanLogType.CHAT);

        const input = formatPromptInput(attrs);
        if (input) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT, input);
        }

        const output = formatCompletionOutput(attrs);
        if (output) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT, output);
        }

        const toolsValue = parseToolsValue(attrs);
        if (toolsValue) {
          attrs[TraceloopSpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJsonStr(toolsValue);
        }

        const toolChoice = parseToolChoice(attrs);
        if (toolChoice) {
          setMetadata(attrs, "tool_choice", toolChoice);
        }

        enrichPerformanceMetrics(attrs, name);
      }

      if (logType === RespanLogType.EMBEDDING) {
        // input/output/tokens are mapped in the up-front embedding block.
        setDefault(attrs, TraceloopSpanAttributes.LLM_REQUEST_TYPE, RespanLogType.EMBEDDING);
        enrichSystem(attrs);
        enrichModel(attrs, attrs[AI_MODEL_ID]);
      }

      if (logType === RespanLogType.TOOL) {
        setToolSpanNameHint(attrs, name);

        const toolInput = formatToolInput(attrs);
        if (toolInput) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT, toolInput);
        }

        const toolOutput = formatToolOutput(attrs);
        if (toolOutput) {
          setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT, toolOutput);
        }
      }
    }

    stripRedundantAttrs(attrs, logType);
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }

  shutdown(): Promise<void> {
    this._ownerCount = 0;
    this._openStructuralSpans.clear();
    return Promise.resolve();
  }
}

/**
 * Semantic-name hint for tool spans. The exporter derives the "tool" prefix
 * from the log type, but the detail must be the tool's own name — the entity
 * name on Vercel tool spans is the raw span name (e.g. "ai.toolCall").
 */
function setToolSpanNameHint(attrs: Record<string, any>, spanName: string): void {
  setDefault(attrs, RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_KIND, "tool");
  setDefault(
    attrs,
    RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_DETAIL,
    attrs[AI_TOOL_CALL_NAME] ?? attrs[ATTR_GEN_AI_TOOL_NAME] ?? spanName.split(".").at(-1)
  );
}
