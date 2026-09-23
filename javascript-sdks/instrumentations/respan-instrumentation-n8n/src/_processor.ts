import { trace, type Context, type Span } from "@opentelemetry/api";
import type { ReadableSpan, SpanProcessor } from "@opentelemetry/sdk-trace-base";
import { ATTR_GEN_AI_TOOL_CALL_ID } from "@opentelemetry/semantic-conventions/incubating";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  classifyN8nSpan,
  enrichEndedN8nSpan,
  enrichLiveN8nSpan,
  isN8nAiSdkToolSpan,
  isN8nStructuralLlmWrapper,
  prepareN8nSpanForExport,
  sanitizeN8nSpanForFailSafeExport,
  workflowNameFromSpan,
  type N8nSpanLike,
} from "./_translator.js";

/**
 * Enriches n8n's native spans before the host BatchSpanProcessor sees them.
 *
 * The live phase only adds canonical attributes. Raw n8n/AI SDK attributes
 * are removed from an export-only clone in prepareForExport(), so any other
 * host processors continue to see n8n's original span data.
 */
export class N8nSpanProcessor implements SpanProcessor {
  private readonly _workflowNamesByTraceId = new Map<string, string>();
  private readonly _openLlmWrapperCandidates = new Map<string, OpenStructuralSpan>();
  private readonly _openStructuralSpans = new Map<string, OpenStructuralSpan>();
  private readonly _openAiToolSpans = new Map<string, OpenAiToolSpan>();
  private readonly _aiToolSpanKeysByCall = new Map<string, string>();

  onStart(span: Span, parentContext: Context): void {
    try {
      const spanLike = span as Span & N8nSpanLike;
      const kind = classifyN8nSpan(spanLike);
      if (!kind) return;

      const identity = spanIdentity(spanLike, parentContext);
      const traceId = identity.traceId;
      const spanKey = correlationKey(traceId, identity.spanId);
      const parentKey = correlationKey(identity.parentTraceId ?? traceId, identity.parentSpanId);

      const wrapperCandidate = parentKey
        ? this._openLlmWrapperCandidates.get(parentKey)
        : undefined;
      if (wrapperCandidate && kind === "llm" && !isN8nStructuralLlmWrapper(spanLike)) {
        wrapperCandidate.span.setAttribute(
          RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN,
          true,
        );
        this._openStructuralSpans.set(parentKey!, wrapperCandidate);
        span.setAttribute(
          RespanSpanAttributes.RESPAN_INTERNAL_EXPORT_PARENT,
          wrapperCandidate.parentSpanId ?? "",
        );
      } else if (parentKey && this._openStructuralSpans.has(parentKey)) {
        span.setAttribute(
          RespanSpanAttributes.RESPAN_INTERNAL_EXPORT_PARENT,
          this._resolveExportParent(parentKey, traceId) ?? "",
        );
      }

      if (isN8nStructuralLlmWrapper(spanLike)) {
        if (spanKey) {
          this._openLlmWrapperCandidates.set(spanKey, {
            span,
            parentSpanId: identity.parentSpanId,
            parentTraceId: identity.parentTraceId,
          });
        }
      }

      if (isN8nAiSdkToolSpan(spanLike) && spanKey) {
        const callId = stringAttribute(spanLike.attributes, "ai.toolCall.id");
        this._openAiToolSpans.set(spanKey, {
          span,
          parentSpanId: identity.parentSpanId,
          parentTraceId: identity.parentTraceId,
          callId,
        });
        if (callId) this._aiToolSpanKeysByCall.set(callKey(traceId, callId), spanKey);
      } else if (kind === "tool") {
        const callId = stringAttribute(spanLike.attributes, ATTR_GEN_AI_TOOL_CALL_ID);
        const candidateKey =
          (parentKey && this._openAiToolSpans.has(parentKey) ? parentKey : undefined) ??
          (callId ? this._aiToolSpanKeysByCall.get(callKey(traceId, callId)) : undefined);
        const candidate = candidateKey ? this._openAiToolSpans.get(candidateKey) : undefined;
        if (candidate) {
          candidate.span.setAttribute(RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN, true);
          const candidateParentKey = correlationKey(
            candidate.parentTraceId ?? traceId,
            candidate.parentSpanId,
          );
          span.setAttribute(
            RespanSpanAttributes.RESPAN_INTERNAL_EXPORT_PARENT,
            candidateParentKey && this._openStructuralSpans.has(candidateParentKey)
              ? this._resolveExportParent(candidateParentKey, traceId) ?? ""
              : candidate.parentSpanId ?? "",
          );
        }
      }

      if (kind === "workflow") {
        const workflowName = workflowNameFromSpan(spanLike);
        if (workflowName) this._workflowNamesByTraceId.set(traceId, workflowName);
      }

      enrichLiveN8nSpan(spanLike, this._workflowNamesByTraceId.get(traceId));
    } catch (error) {
      warnTranslationFailure("onStart", error);
    }
  }

  onEnd(span: ReadableSpan): void {
    try {
      const traceId = span.spanContext().traceId;
      const spanKey = correlationKey(traceId, span.spanContext().spanId);
      const kind = classifyN8nSpan(span as unknown as N8nSpanLike);
      if (!kind) return;

      enrichEndedN8nSpan(span, this._workflowNamesByTraceId.get(traceId));
      if (spanKey) {
        this._openLlmWrapperCandidates.delete(spanKey);
        this._openStructuralSpans.delete(spanKey);
        const candidate = this._openAiToolSpans.get(spanKey);
        this._openAiToolSpans.delete(spanKey);
        if (candidate?.callId) {
          this._aiToolSpanKeysByCall.delete(callKey(traceId, candidate.callId));
        }
      }
      if (kind === "workflow") this._workflowNamesByTraceId.delete(traceId);
    } catch (error) {
      warnTranslationFailure("onEnd", error);
    }
  }

  prepareForExport(span: ReadableSpan): ReadableSpan {
    try {
      return prepareN8nSpanForExport(span);
    } catch (error) {
      warnTranslationFailure("prepareForExport", error);
      return sanitizeN8nSpanForFailSafeExport(span);
    }
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }

  shutdown(): Promise<void> {
    this._workflowNamesByTraceId.clear();
    this._openLlmWrapperCandidates.clear();
    this._openStructuralSpans.clear();
    this._openAiToolSpans.clear();
    this._aiToolSpanKeysByCall.clear();
    return Promise.resolve();
  }

  private _resolveExportParent(structuralKey: string, traceId: string): string | undefined {
    let currentKey: string | undefined = structuralKey;
    let parentSpanId: string | undefined;
    const seen = new Set<string>();
    while (currentKey && !seen.has(currentKey)) {
      seen.add(currentKey);
      const structural = this._openStructuralSpans.get(currentKey);
      if (!structural) break;
      parentSpanId = structural.parentSpanId;
      currentKey = correlationKey(structural.parentTraceId ?? traceId, structural.parentSpanId);
    }
    return parentSpanId;
  }
}

interface OpenStructuralSpan {
  span: Span;
  parentSpanId?: string;
  parentTraceId?: string;
}

interface OpenAiToolSpan extends OpenStructuralSpan {
  span: Span;
  callId?: string;
}

interface SpanIdentity {
  traceId: string;
  spanId: string;
  parentTraceId?: string;
  parentSpanId?: string;
}

function spanIdentity(span: Span & N8nSpanLike, parentContext: Context): SpanIdentity {
  const current = span.spanContext();
  const recordedParent = (span as Span & {
    parentSpanContext?: { traceId?: string; spanId?: string };
    parentSpanId?: string;
  }).parentSpanContext;
  const contextParent = parentContext ? trace.getSpanContext(parentContext) : undefined;
  const parentSpanId =
    (span as Span & { parentSpanId?: string }).parentSpanId ??
    recordedParent?.spanId ??
    contextParent?.spanId;
  return {
    traceId: current.traceId,
    spanId: current.spanId,
    parentTraceId:
      recordedParent?.traceId ?? contextParent?.traceId ?? (parentSpanId ? current.traceId : undefined),
    parentSpanId,
  };
}

function correlationKey(traceId?: string, spanId?: string): string | undefined {
  return spanId ? `${traceId ?? "<unknown-trace>"}:${spanId}` : undefined;
}

function callKey(traceId: string, callId: string): string {
  return `${traceId}:${callId}`;
}

function stringAttribute(
  attrs: Readonly<Record<string, unknown>>,
  key: string,
): string | undefined {
  const value = attrs[key];
  if (value === undefined || value === null) return undefined;
  const text = String(value).trim();
  return text || undefined;
}

function warnTranslationFailure(phase: string, error: unknown): void {
  console.warn(
    `[respan] n8n span translation failed during ${phase}; exporting the original span`,
    error,
  );
}
