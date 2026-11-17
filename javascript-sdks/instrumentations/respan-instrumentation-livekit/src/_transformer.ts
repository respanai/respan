import type { Context, Span } from "@opentelemetry/api";
import {
  ATTR_GEN_AI_INPUT_MESSAGES,
  ATTR_GEN_AI_OUTPUT_MESSAGES,
  ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
  ATTR_GEN_AI_TOOL_DEFINITIONS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import type { RespanSpanTransformer } from "@respan/tracing";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  LIVEKIT_INTERNAL_ACTIVITY_SPAN_NAMES,
  LIVEKIT_LOG_METHOD_TS_TRACING,
  LIVEKIT_LOG_TYPE_BY_SPAN_NAME,
  LIVEKIT_SPAN_NAME_ATTRIBUTE,
  LIVEKIT_SPAN_NAMES,
  LK_LLM_METRICS,
  LK_PROVIDER_REQUEST_IDS,
  OFF_CONTRACT_ALIAS_KEYS,
} from "./_constants.js";
import {
  data,
  capture,
  guard,
  filterAttributes,
  policyFor,
  snapshot,
  type CaptureOptions,
} from "./_privacy.js";
import {
  getSpanAttributes,
  isLiveKitSpan,
  stripLiveKitRawAttributes,
  translateLiveKitSpan,
  type MutableAttributes,
} from "./_translator.js";
import { VoiceTurns } from "./_voice.js";

const LANGFUSE_COMPLETION_START_TIME =
  "langfuse.observation.completion_start_time";
type TransformerReadableSpan = Parameters<
  NonNullable<RespanSpanTransformer["onEnd"]>
>[0];
type CorrelatableSpan = Span | TransformerReadableSpan;

interface OpenLLMNode {
  readonly span: Span;
  readonly spanId: string;
  ended: boolean;
}

/**
 * Correlates LiveKit's logical `llm_node` with its transport-level
 * `llm_request` child before normal span translation.
 */
export class LiveKitSpanTransformer implements RespanSpanTransformer {
  private readonly _openLLMNodes = new Map<string, OpenLLMNode>();
  private readonly _correlatedRequests = new Map<string, OpenLLMNode>();
  private readonly _owners = new Set<CaptureOptions>();
  private _exportAttributes = new WeakMap<object, MutableAttributes>();
  private readonly _voice = new VoiceTurns();

  acquire(options: CaptureOptions): () => void {
    this._owners.add(options);
    return () => {
      this._owners.delete(options);
    };
  }

  onStart(span: Span, parentContext: Context): void {
    const attrs = getSpanAttributes(span);
    const spanName = getSpanName(span, attrs);
    if (
      !isLiveKitSpan(spanName, attrs) &&
      data(data(span, "instrumentationScope"), "name") !== "livekit-agents"
    ) {
      return;
    }

    const policy = capture(span, parentContext, this._owners);
    guard(span, policy);
    this._voice.start(getSpanIdentity(span), getParentKey(span), spanName);
    if (LIVEKIT_INTERNAL_ACTIVITY_SPAN_NAMES.has(spanName)) return;
    stampStartAttributes(span, getSpanAttributes(span), spanName);

    const identity = getSpanIdentity(span);
    if (spanName === LIVEKIT_SPAN_NAMES.LLM_NODE && identity) {
      this._openLLMNodes.set(identity.key, {
        span,
        spanId: identity.spanId,
        ended: false,
      });
      return;
    }

    const parentKey = getParentKey(span);
    if (spanName === LIVEKIT_SPAN_NAMES.LLM_REQUEST && identity && parentKey) {
      const parentNode = this._openLLMNodes.get(parentKey);
      if (parentNode) {
        this._correlatedRequests.set(identity.key, parentNode);
        // This request is transport structure beneath the logical LLM node.
        // Semantic export drops it; its retry-attempt children are reparented.
        span.setAttribute(RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN, true);
        span.setAttribute(
          RespanSpanAttributes.RESPAN_LOG_TYPE,
          RespanLogType.TASK,
        );
      }
      return;
    }

    if (spanName === LIVEKIT_SPAN_NAMES.LLM_REQUEST_RUN && parentKey) {
      const parentNode = this._correlatedRequests.get(parentKey);
      if (parentNode) {
        span.setAttribute(
          RespanSpanAttributes.RESPAN_INTERNAL_EXPORT_PARENT,
          parentNode.spanId,
        );
      }
    }
  }

  onEnd(span: TransformerReadableSpan): void {
    this._translateOnEnd(span);
    const exported = this._exportAttributes.get(span);
    if (exported) {
      this._voice.end(getSpanIdentity(span), span.attributes, exported);
    }
  }

  private _translateOnEnd(span: TransformerReadableSpan): void {
    const policy = policyFor(span);
    if (!policy) return;
    const attrs = filterAttributes(span.attributes, policy);
    this._exportAttributes.set(span, attrs);
    const spanName = getSpanName(span, attrs);
    const identity = getSpanIdentity(span);

    if (LIVEKIT_INTERNAL_ACTIVITY_SPAN_NAMES.has(spanName)) {
      stripLiveKitRawAttributes(attrs);
      for (const key of Object.keys(attrs))
        if (key.startsWith("gen_ai.") || key.startsWith("llm."))
          delete attrs[key];
      return;
    }

    if (spanName === LIVEKIT_SPAN_NAMES.LLM_NODE) {
      const node = identity ? this._openLLMNodes.get(identity.key) : undefined;
      if (node) {
        node.ended = true;
        this._openLLMNodes.delete(identity!.key);
      }
      translateLiveKitSpan(span as unknown as Span, {
        attributes: attrs,
        spanName,
        writeToSpan: false,
      });
      return;
    }

    if (spanName === LIVEKIT_SPAN_NAMES.LLM_REQUEST && identity) {
      const parentNode = this._correlatedRequests.get(identity.key);
      this._correlatedRequests.delete(identity.key);

      if (parentNode && !parentNode.ended) {
        mergeRequestIntoNode(parentNode, attrs);
        translateCorrelatedRequest(attrs, spanName);
        return;
      }

      // Unexpected end order: preserve the request as a standalone chat
      // instead of dropping the only complete LLM record.
      delete attrs[RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN];
    }

    if (spanName === LIVEKIT_SPAN_NAMES.LLM_REQUEST_RUN) {
      const parentKey = getParentKey(span);
      const parentNode = parentKey
        ? this._correlatedRequests.get(parentKey)
        : undefined;
      if (parentNode && !parentNode.ended) {
        copyAttributeToNode(
          parentNode,
          LK_PROVIDER_REQUEST_IDS,
          attrs[LK_PROVIDER_REQUEST_IDS],
        );
      }
    }

    translateLiveKitSpan(span as unknown as Span, {
      attributes: attrs,
      spanName,
      writeToSpan: false,
    });
  }

  prepareForExport(span: TransformerReadableSpan): TransformerReadableSpan {
    const attrs = this._exportAttributes.get(span);
    if (!attrs) return span;
    this._exportAttributes.delete(span);
    const copy = Object.create(Object.getPrototypeOf(span));
    Object.assign(copy, span);
    Object.defineProperty(copy, "attributes", {
      value: attrs,
      enumerable: true,
      configurable: true,
    });
    Object.defineProperty(copy, "events", {
      value: snapshot(span.events) ?? [],
      enumerable: true,
      configurable: true,
    });
    Object.defineProperty(copy, "status", {
      value: snapshot(span.status) ?? { code: 0 },
      enumerable: true,
      configurable: true,
    });
    const policy = policyFor(span);
    if (policy) guard(copy, policy);
    return copy;
  }

  /** Clear correlation state after the registry drains this lease. */
  dispose(): void {
    this._openLLMNodes.clear();
    this._correlatedRequests.clear();
    this._exportAttributes = new WeakMap();
    this._voice.dispose();
  }
}

function stampStartAttributes(
  span: Span,
  attrs: MutableAttributes,
  spanName: string,
): void {
  const logType = LIVEKIT_LOG_TYPE_BY_SPAN_NAME[spanName] ?? RespanLogType.TASK;
  for (const [key, value] of [
    [LIVEKIT_SPAN_NAME_ATTRIBUTE, spanName],
    [RespanSpanAttributes.RESPAN_LOG_METHOD, LIVEKIT_LOG_METHOD_TS_TRACING],
    [RespanSpanAttributes.RESPAN_LOG_TYPE, logType],
  ] as const) {
    attrs[key] = value;
    span.setAttribute(key, value);
  }
}

function translateCorrelatedRequest(
  attrs: MutableAttributes,
  spanName: string,
): void {
  attrs[LIVEKIT_SPAN_NAME_ATTRIBUTE] = spanName;
  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] = LIVEKIT_LOG_METHOD_TS_TRACING;
  attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = RespanLogType.TASK;
  attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] = "livekit.llm_request";
  attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] = spanName;
  delete attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT];
  delete attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT];

  for (const key of Object.keys(attrs)) {
    if (key.startsWith("gen_ai.") || key.startsWith("llm.")) {
      delete attrs[key];
    }
  }
  for (const key of OFF_CONTRACT_ALIAS_KEYS) {
    delete attrs[key];
  }
  stripLiveKitRawAttributes(attrs);
}

function mergeRequestIntoNode(
  node: OpenLLMNode,
  requestAttrs: MutableAttributes,
): void {
  const metrics = parseRecord(requestAttrs[LK_LLM_METRICS]);
  const metadata = parseRecord(metrics?.metadata);

  copyPreferredStringToNode(
    node,
    SpanAttributes.LLM_REQUEST_MODEL,
    preferredString(
      requestAttrs[SpanAttributes.LLM_REQUEST_MODEL],
      metadata?.modelName,
    ),
  );
  copyPreferredStringToNode(
    node,
    ATTR_GEN_AI_PROVIDER_NAME,
    preferredString(
      requestAttrs[ATTR_GEN_AI_PROVIDER_NAME],
      metadata?.modelProvider,
    ),
  );

  copyAttributeToNode(
    node,
    ATTR_GEN_AI_USAGE_INPUT_TOKENS,
    requestAttrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] ?? metrics?.promptTokens,
  );
  copyAttributeToNode(
    node,
    ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
    requestAttrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] ?? metrics?.completionTokens,
  );
  copyAttributeToNode(
    node,
    SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
    requestAttrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] ?? metrics?.totalTokens,
  );
  copyAttributeToNode(
    node,
    ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    requestAttrs[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] ??
      metrics?.promptCachedTokens,
  );

  for (const key of [
    ATTR_GEN_AI_INPUT_MESSAGES,
    ATTR_GEN_AI_OUTPUT_MESSAGES,
    ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
    ATTR_GEN_AI_TOOL_DEFINITIONS,
    LK_LLM_METRICS,
    LK_PROVIDER_REQUEST_IDS,
    LANGFUSE_COMPLETION_START_TIME,
  ]) {
    copyAttributeToNode(node, key, requestAttrs[key]);
  }
}

function copyPreferredStringToNode(
  node: OpenLLMNode,
  key: string,
  candidate: unknown,
): void {
  if (typeof candidate !== "string" || candidate.length === 0) {
    return;
  }
  const nodeAttrs = getSpanAttributes(node.span);
  const existing = nodeAttrs[key];
  if (isConcreteString(existing) && !isConcreteString(candidate)) {
    return;
  }
  if (isConcreteString(existing) && isConcreteString(candidate)) {
    return;
  }
  copyAttributeToNode(node, key, candidate);
}

function copyAttributeToNode(
  node: OpenLLMNode,
  key: string,
  value: unknown,
): void {
  if (value === undefined || value === null) {
    return;
  }
  const attrs = getSpanAttributes(node.span);
  attrs[key] = value;
  node.span.setAttribute(key, value as any);
}

function preferredString(primary: unknown, fallback: unknown): unknown {
  if (isConcreteString(primary)) {
    return primary;
  }
  if (isConcreteString(fallback)) {
    return fallback;
  }
  return primary ?? fallback;
}

function isConcreteString(value: unknown): boolean {
  return (
    typeof value === "string" &&
    value.trim().length > 0 &&
    value.trim().toLowerCase() !== "unknown"
  );
}

function parseRecord(value: unknown): Record<string, unknown> | undefined {
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return undefined;
    }
  }
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function getSpanName(span: CorrelatableSpan, attrs: MutableAttributes): string {
  const explicit = attrs[LIVEKIT_SPAN_NAME_ATTRIBUTE];
  return typeof explicit === "string" && explicit.length > 0
    ? explicit
    : String((span as { name?: string }).name ?? "livekit.span");
}

function getSpanIdentity(
  span: CorrelatableSpan,
): { key: string; spanId: string; traceId: string } | undefined {
  const spanContext = span.spanContext?.();
  if (!spanContext?.traceId || !spanContext.spanId) {
    return undefined;
  }
  return {
    key: identityKey(spanContext.traceId, spanContext.spanId),
    spanId: spanContext.spanId,
    traceId: spanContext.traceId,
  };
}

function getParentKey(span: CorrelatableSpan): string | undefined {
  const parent = (
    span as Span & { parentSpanContext?: { traceId?: string; spanId?: string } }
  ).parentSpanContext;
  if (!parent?.spanId) {
    return undefined;
  }
  const traceId = parent.traceId ?? span.spanContext?.().traceId;
  return traceId ? identityKey(traceId, parent.spanId) : undefined;
}

function identityKey(traceId: string, spanId: string): string {
  return `${traceId}:${spanId}`;
}
