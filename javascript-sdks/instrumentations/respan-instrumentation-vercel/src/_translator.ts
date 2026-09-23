/**
 * Translate Vercel AI SDK spans → Traceloop/OpenLLMetry format.
 *
 * The Vercel AI SDK emits OTEL spans with its own attribute schema (ai.model.id,
 * ai.prompt.messages, ai.response.text, etc.). This transformer enriches those
 * spans with the Traceloop/GenAI semantic conventions the Respan backend expects.
 *
 * Two-phase enrichment:
 * - onStart(): Sets RESPAN_LOG_TYPE so the span passes CompositeProcessor filtering
 * - onEnd():   Full attribute translation (model, messages, tokens, metadata, etc.)
 */

import type { Context } from "@opentelemetry/api";
import type { ReadableSpan, Span } from "@opentelemetry/sdk-trace-base";
import { shouldSendTraces, type RespanSpanTransformer } from "@respan/tracing";
import { stripContentAttributes } from "./_translator/content.js";
import {
  ATTR_GEN_AI_AGENT_ID,
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_TOOL_NAME,
  GEN_AI_OPERATION_NAME_VALUE_EMBEDDINGS,
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
  AI_DOCUMENTS,
  AI_EVALUATION_STATE,
  AI_EVALUATION_QUESTIONS,
  AI_EVALUATION_ANSWERS,
  AI_RANKING,
  parseJsonish,
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
  modernOperationName,
  setMetadata,
  resolveLogType,
  safeJsonStr,
  setDefault,
} from "./_translator/shared.js";
import { enrichMetadata, enrichModel, enrichPerformanceMetrics, enrichSystem, enrichTokens, stripRedundantAttrs } from "./_translator/span-enrichment.js";

/**
 * Span transformer that translates Vercel AI SDK attributes to Traceloop/OpenLLMetry.
 *
 * Phase 1 (onStart): Sets RESPAN_LOG_TYPE so CompositeProcessor lets the span through.
 * Phase 2 (onEnd):   Full attribute enrichment — model, messages, tokens, metadata,
 *                     tools, performance metrics, environment, etc.
 */
export class VercelAITranslator implements RespanSpanTransformer {
  /** Open structural wrappers, isolated by trace even if span IDs are reused. */
  private readonly _contentDisabled = new WeakSet<object>();

  private readonly _openStructuralSpans = new Map<string, OpenStructuralSpan>();

  /**
   * AI SDK 7 emits the embedding operation and its provider request with the
   * same modern span name. Keep the open operation as a candidate; only turn
   * it into a structural wrapper if a matching direct child actually starts.
   */
  private readonly _openEmbeddingCandidates = new Map<string, OpenEmbeddingCandidate>();

  onStart(span: Span, _parentContext: Context): void {
    const writableSpan = span as any;
    const name: string = writableSpan.name ?? "";
    const scopeName = instrumentationScopeName(writableSpan);
    if (!name.startsWith(AI_PREFIX) && !isModernVercelAISpanName(name) && !isVercelAIScope(scopeName)) {
      return;
    }

    if (!shouldSendTraces()) {
      this._contentDisabled.add(writableSpan);
      if (writableSpan.attributes) stripContentAttributes(writableSpan.attributes);
    }

    const identity = spanIdentity(writableSpan);
    const spanKey = correlationKey(identity.traceId, identity.spanId);
    const parentKey = correlationKey(
      identity.parentTraceId ?? identity.traceId,
      identity.parentSpanId,
    );

    // Modern AI SDK 7 gives an embedding wrapper and its direct provider span
    // the same `embeddings <model>` name. Once the direct child proves that the
    // open candidate is structural, mark only that parent for semantic drop.
    // The child itself remains canonical (and may become structural later only
    // if it, in turn, receives a matching direct child).
    const attrs = writableSpan.attributes as Record<string, any> | undefined;
    if (parentKey && isModernEmbeddingSpan(name, attrs)) {
      const candidate = this._openEmbeddingCandidates.get(parentKey);
      if (candidate && candidate.name === name) {
        candidate.span.setAttribute(
          RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN,
          true,
        );
        this._openStructuralSpans.set(parentKey, {
          parentSpanId: candidate.parentSpanId,
          parentTraceId: candidate.parentTraceId,
        });
      }
    }

    // If the parent chain starts inside an open structural wrapper, stamp the
    // export-time parent (the wrapper's own parent) so the exporter can drop
    // the wrapper per-span, immune to export-batch boundaries. "" marks a
    // wrapper that was itself a root span — the child is promoted to root.
    if (parentKey && this._openStructuralSpans.has(parentKey)) {
      let exportParentKey: string | undefined = parentKey;
      let exportParentSpanId: string | undefined = identity.parentSpanId;
      const seen = new Set<string>();
      while (
        exportParentKey &&
        this._openStructuralSpans.has(exportParentKey) &&
        !seen.has(exportParentKey)
      ) {
        seen.add(exportParentKey);
        const structural = this._openStructuralSpans.get(exportParentKey)!;
        exportParentSpanId = structural.parentSpanId;
        exportParentKey = correlationKey(
          structural.parentTraceId ?? identity.traceId,
          structural.parentSpanId,
        );
      }
      writableSpan.setAttribute(
        RespanSpanAttributes.RESPAN_INTERNAL_EXPORT_PARENT,
        exportParentSpanId ?? ""
      );
    }

    if (VERCEL_STRUCTURAL_LLM_PARENT_SPANS.has(name)) {
      // Structural wrapper: the .doGenerate/.doStream child carries the real
      // model/input/output. The semantic export style drops it (children are
      // reparented); the legacy style still exports it untouched.
      writableSpan.setAttribute(RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN, true);
      if (spanKey) {
        this._openStructuralSpans.set(spanKey, {
          parentSpanId: identity.parentSpanId,
          parentTraceId: identity.parentTraceId,
        });
      }
    }

    if (spanKey && isModernEmbeddingSpan(name, attrs)) {
      this._openEmbeddingCandidates.set(spanKey, {
        name,
        span: writableSpan,
        parentSpanId: identity.parentSpanId,
        parentTraceId: identity.parentTraceId,
      });
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
    const identity = spanIdentity(span);
    const endedSpanKey = correlationKey(identity.traceId, identity.spanId);
    if (endedSpanKey) {
      this._openStructuralSpans.delete(endedSpanKey);
      this._openEmbeddingCandidates.delete(endedSpanKey);
    }

    const attrs = (span as any).attributes as Record<string, any> | undefined;
    if (!attrs || !isVercelAISpan(span)) {
      return;
    }

    const name = span.name;
    const config = VERCEL_SPAN_CONFIG[name];
    const parentLogType = VERCEL_PARENT_SPANS[name];
    const logType = resolveLogType(name, attrs);
    const captureContent = shouldSendTraces() && !this._contentDisabled.has(span);
    this._contentDisabled.delete(span);
    if (!captureContent) stripContentAttributes(attrs);

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

    if (modernOperationName(name, attrs) === "rerank") {
      const documents = attrs[AI_DOCUMENTS];
      const ranking = attrs[AI_RANKING];
      if (documents !== undefined) {
        setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT,
          safeJsonStr({ documents: Array.isArray(documents) ? documents.map(parseJsonish) : parseJsonish(documents) }));
      }
      if (ranking !== undefined) {
        setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT,
          safeJsonStr(Array.isArray(ranking) ? ranking.map(parseJsonish) : parseJsonish(ranking)));
      }
    }

    if (modernOperationName(name, attrs) === "evaluate") {
      const state = attrs[AI_EVALUATION_STATE];
      const questions = attrs[AI_EVALUATION_QUESTIONS];
      if (state !== undefined || questions !== undefined) {
        setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT,
          safeJsonStr({ state: parseJsonish(state), questions: parseJsonish(questions) }));
      }
      if (attrs[AI_EVALUATION_ANSWERS] !== undefined) {
        setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT,
          safeJsonStr(parseJsonish(attrs[AI_EVALUATION_ANSWERS])));
      }
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

    // AI SDK 7 emits invoke_agent as a structural agent root. Agent spans carry
    // common canonical I/O, but never chat/model/token fields themselves; the
    // child chat spans own those LLM-specific attributes.
    if (logType === RespanLogType.AGENT) {
      const input = formatPromptInput(attrs);
      if (input) {
        setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT, input);
      }

      const output = formatCompletionOutput(attrs);
      if (output) {
        setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT, output);
      }
    }

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
    if (!captureContent) stripContentAttributes(attrs);
  }

  /** Clear structural correlation state after the registry drains this lease. */
  dispose(): void {
    this._openStructuralSpans.clear();
    this._openEmbeddingCandidates.clear();
  }

}

interface OpenStructuralSpan {
  parentSpanId?: string;
  parentTraceId?: string;
}

interface OpenEmbeddingCandidate extends OpenStructuralSpan {
  name: string;
  span: Span;
}

interface SpanIdentity {
  traceId?: string;
  spanId?: string;
  parentTraceId?: string;
  parentSpanId?: string;
}

function spanIdentity(span: any): SpanIdentity {
  const context =
    typeof span.spanContext === "function" ? span.spanContext() : undefined;
  const parentContext = span.parentSpanContext;
  const parentSpanId = span.parentSpanId ?? parentContext?.spanId;
  return {
    traceId: context?.traceId,
    spanId: context?.spanId,
    parentTraceId: parentContext?.traceId ?? (parentSpanId ? context?.traceId : undefined),
    parentSpanId,
  };
}

function correlationKey(traceId?: string, spanId?: string): string | undefined {
  return spanId ? `${traceId ?? "<unknown-trace>"}:${spanId}` : undefined;
}

function isModernEmbeddingSpan(
  name: string,
  attrs: Record<string, any> | undefined,
): boolean {
  return modernOperationName(name, attrs ?? {}) === GEN_AI_OPERATION_NAME_VALUE_EMBEDDINGS;
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
