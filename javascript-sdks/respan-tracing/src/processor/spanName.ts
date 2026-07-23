import type { ExportResult } from "@opentelemetry/core";
import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-base";
import {
  SpanAttributes,
  TraceloopSpanKindValues,
} from "@traceloop/ai-semantic-conventions";
import {
  RespanLogType,
  RespanSpanAttributes,
  type RespanSpanNameStyle,
} from "@respan/respan-sdk";

type SpanAttrs = Record<string, unknown>;

const INTERNAL_KIND_ATTR = RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_KIND;
const INTERNAL_DETAIL_ATTR = RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_DETAIL;
const INTERNAL_DROP_ATTR = RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN;
const INTERNAL_EXPORT_PARENT_ATTR = RespanSpanAttributes.RESPAN_INTERNAL_EXPORT_PARENT;

const INTERNAL_SPAN_NAME_ATTRS = [
  INTERNAL_KIND_ATTR,
  INTERNAL_DETAIL_ATTR,
  INTERNAL_DROP_ATTR,
  INTERNAL_EXPORT_PARENT_ATTR,
] as const;

const SUFFIXED_OPERATIONS = new Set(["agent", "tool", "handoff", "llm"]);

// Operation/structural tokens that must not survive as a semantic-name suffix
// (e.g. "handoff.task" or "llm.doGenerate" — the suffix carries no identity).
const GENERIC_DETAIL_TOKENS = new Set([
  "agent",
  "chat",
  "completion",
  "completions",
  "doembed",
  "dogenerate",
  "dostream",
  "embedding",
  "generate",
  "generation",
  "guardrail",
  "handoff",
  "llm",
  "response",
  "responses",
  "task",
  "text",
  "tool",
  "workflow",
]);

export function resolveSpanNameStyle(
  value?: RespanSpanNameStyle | string
): RespanSpanNameStyle {
  // Normalize like the Python resolver (.strip().lower()) so the same env
  // value means the same thing in a mixed-language deployment.
  const normalized =
    value === undefined || value === null ? "" : String(value).trim().toLowerCase();
  return normalized === "legacy" ? "legacy" : "semantic";
}

export function transformReadableSpanName(
  span: ReadableSpan,
  style: RespanSpanNameStyle | string | undefined
): ReadableSpan {
  const resolvedStyle = resolveSpanNameStyle(style);
  const attrs = span.attributes as SpanAttrs;
  const attributes = stripInternalSemanticNameAttrs(attrs);
  const name =
    resolvedStyle === "semantic" ? semanticSpanNameForSpan(span) : span.name;

  // The export-parent attr distinguishes "absent" (leave the parent alone)
  // from "" (the dropped wrapper was a root span — promote the child to root).
  let reparent: string | null | undefined;
  if (resolvedStyle === "semantic") {
    const raw = attrs[INTERNAL_EXPORT_PARENT_ATTR];
    if (raw !== undefined && raw !== null) {
      const text = String(raw).trim();
      reparent = text === "" ? null : text;
    }
  }

  if (
    name === span.name &&
    attributes === span.attributes &&
    reparent === undefined
  ) {
    return span;
  }

  return cloneReadableSpan(span, name, attributes, reparent);
}

export function semanticSpanNameForSpan(span: ReadableSpan): string {
  const attrs = span.attributes as SpanAttrs;
  const operation = resolveOperation(attrs, span.name);
  // Spans with no recognizable operation keep their original name — the
  // semantic style renames known operations, it never destroys pass-through
  // span names (matches the Python exporter).
  if (!operation) {
    return span.name;
  }
  const detail = resolveDetail(attrs, span.name, operation);

  const hasInternalHint =
    attrs[INTERNAL_KIND_ATTR] !== undefined || attrs[INTERNAL_DETAIL_ATTR] !== undefined;

  if (!SUFFIXED_OPERATIONS.has(operation)) {
    return operation;
  }

  if (operation === "llm") {
    return detail ? `${operation}.${detail}` : operation;
  }

  if (!hasInternalHint && span.name.startsWith(`${operation}.`)) {
    const existingDetail = span.name.slice(operation.length + 1);
    if (existingDetail && !GENERIC_DETAIL_TOKENS.has(existingDetail.toLowerCase())) {
      return span.name;
    }
  }

  if (!detail || detail.toLowerCase() === operation) {
    return operation;
  }

  return `${operation}.${detail}`;
}

export function transformReadableSpanBatch(
  spans: ReadableSpan[],
  style: RespanSpanNameStyle | string | undefined
): ReadableSpan[] {
  const resolvedStyle = resolveSpanNameStyle(style);

  // Dropping and reparenting are per-span decisions driven entirely by
  // attributes the owning instrumentation stamped at span start
  // (respan.internal.drop_span on the wrapper, export_parent_span_id on its
  // children) — no cross-span state, so batch boundaries cannot break trees.
  // Legacy style preserves the emitted tree exactly.
  if (resolvedStyle === "legacy") {
    return spans.map((span) => transformReadableSpanName(span, resolvedStyle));
  }

  return spans.flatMap((span) => {
    const attrs = span.attributes as SpanAttrs;
    if (attrs[INTERNAL_DROP_ATTR] === true || attrs[INTERNAL_DROP_ATTR] === "true") {
      return [];
    }
    return [transformReadableSpanName(span, resolvedStyle)];
  });
}

export class SpanNameTransformingExporter implements SpanExporter {
  constructor(
    private readonly delegate: SpanExporter,
    private readonly style: RespanSpanNameStyle
  ) {}

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void
  ): void {
    this.delegate.export(
      transformReadableSpanBatch(spans, this.style),
      resultCallback
    );
  }

  shutdown(): Promise<void> {
    return this.delegate.shutdown();
  }

  forceFlush(): Promise<void> {
    const maybeFlush = (this.delegate as { forceFlush?: () => Promise<void> })
      .forceFlush;
    return maybeFlush ? maybeFlush.call(this.delegate) : Promise.resolve();
  }
}

function stripInternalSemanticNameAttrs(attrs: SpanAttrs): SpanAttrs {
  let next: SpanAttrs | undefined;

  for (const key of INTERNAL_SPAN_NAME_ATTRS) {
    if (attrs[key] !== undefined) {
      next ??= { ...attrs };
      delete next[key];
    }
  }

  return next ?? attrs;
}

function cloneReadableSpan(
  span: ReadableSpan,
  name: string,
  attributes: SpanAttrs,
  // string = new parent id; null = promote to root; undefined = leave as-is
  reparent?: string | null
): ReadableSpan {
  const clone = Object.create(Object.getPrototypeOf(span));
  Object.assign(clone, span);
  Object.defineProperty(clone, "name", {
    value: name,
    enumerable: true,
    configurable: true,
  });
  Object.defineProperty(clone, "attributes", {
    value: attributes,
    enumerable: true,
    configurable: true,
  });
  if (reparent !== undefined) {
    // OTel SDK 1.x exposes parentSpanId on ReadableSpan; revisit for SDK 2.x
    // (parentSpanContext) when the workspace upgrades.
    Object.defineProperty(clone, "parentSpanId", {
      value: reparent === null ? undefined : reparent,
      enumerable: true,
      configurable: true,
    });
  }
  return clone as ReadableSpan;
}

function resolveOperation(attrs: SpanAttrs, spanName: string): string | undefined {
  // An explicit instrumentation hint always wins — even an unrecognized value
  // is used (lowercased + sanitized, prefixes are lowercase by contract),
  // since it states intent.
  const hintedKind = stringAttr(attrs, INTERNAL_KIND_ATTR);
  if (hintedKind) {
    return mapOperation(hintedKind) ?? sanitizeNamePart(hintedKind.toLowerCase(), "span");
  }

  // Log type before span kind, matching the Python exporter's priority.
  const logType = stringAttr(attrs, RespanSpanAttributes.RESPAN_LOG_TYPE);
  if (logType) {
    const mapped = mapOperation(logType);
    if (mapped) return mapped;
  }

  const tlKind = stringAttr(attrs, SpanAttributes.TRACELOOP_SPAN_KIND);
  if (tlKind) {
    const mapped = mapOperation(tlKind);
    if (mapped) return mapped;
  }

  return inferOperationFromName(spanName);
}

function resolveDetail(
  attrs: SpanAttrs,
  spanName: string,
  operation: string
): string {
  if (operation === "llm") {
    const model = resolveLlmModel(attrs);
    return model ? sanitizeNamePart(model, "") : "";
  }

  const hintedDetail = stringAttr(attrs, INTERNAL_DETAIL_ATTR);
  if (hintedDetail) {
    return sanitizeNamePart(hintedDetail, "");
  }

  const entityName = stringAttr(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME);
  if (entityName) {
    return sanitizeNamePart(entityName, "");
  }

  const rawDetail = detailFromRawName(spanName, operation);
  if (rawDetail && GENERIC_DETAIL_TOKENS.has(rawDetail.toLowerCase())) {
    return "";
  }
  return sanitizeNamePart(rawDetail, "");
}

function resolveLlmModel(attrs: SpanAttrs): string | undefined {
  return firstStringAttr(attrs, [
    RespanSpanAttributes.GEN_AI_REQUEST_MODEL,
    RespanSpanAttributes.OPENINFERENCE_LLM_MODEL_NAME,
    "llm.model_name",
    "model",
    SpanAttributes.LLM_REQUEST_MODEL,
  ]);
}

function firstStringAttr(attrs: SpanAttrs, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = stringAttr(attrs, key);
    if (value) return value;
  }
  return undefined;
}

function stringAttr(attrs: SpanAttrs, key: string): string | undefined {
  const value = attrs[key];
  if (value === undefined || value === null) return undefined;
  const text = String(value).trim();
  return text || undefined;
}

/** Map a kind/log-type token to a semantic operation; undefined when unknown. */
function mapOperation(value: string): string | undefined {
  const normalized = value.toLowerCase();

  switch (normalized) {
    case TraceloopSpanKindValues.WORKFLOW:
    case RespanLogType.WORKFLOW:
      return "workflow";
    case TraceloopSpanKindValues.AGENT:
    case RespanLogType.AGENT:
      return "agent";
    case TraceloopSpanKindValues.TASK:
    case RespanLogType.TASK:
      return "task";
    case TraceloopSpanKindValues.TOOL:
    case RespanLogType.TOOL:
      return "tool";
    case RespanLogType.FUNCTION:
      return "tool";
    case RespanLogType.HANDOFF:
      return "handoff";
    case RespanLogType.GUARDRAIL:
      return "guardrail";
    case RespanLogType.EMBEDDING:
    case "embed":
      return "embedding";
    case RespanLogType.TRANSCRIPTION:
      return "transcribe";
    case RespanLogType.SPEECH:
      return "speech";
    case RespanLogType.CHAT:
    case RespanLogType.TEXT:
    case RespanLogType.RESPONSE:
    case RespanLogType.GENERATION:
    case "completion":
    case "completions":
    case "generate":
    case "llm":
      return "llm";
    default:
      // Unknown values (including "custom"/"unknown" log types) resolve to no
      // operation so the caller preserves the original span name.
      return undefined;
  }
}

function inferOperationFromName(spanName: string): string | undefined {
  const suffix = spanName.split(".").at(-1);
  if (suffix && GENERIC_DETAIL_TOKENS.has(suffix.toLowerCase())) {
    return mapOperation(suffix);
  }

  return undefined;
}

function detailFromRawName(spanName: string, operation: string): string {
  if (spanName.endsWith(`.${operation}`)) {
    return spanName.slice(0, -(operation.length + 1));
  }

  if (spanName.startsWith(`${operation}.`)) {
    return spanName.slice(operation.length + 1);
  }

  if (operation === "handoff") {
    return spanName.replace(/^handoff\s*[:.-]?\s*/i, "");
  }

  return spanName;
}

function sanitizeNamePart(value: string, fallback: string): string {
  // \p{L}\p{N} keeps unicode letters/digits so non-ASCII agent/tool names
  // survive — parity with Python's unicode \w.
  const sanitized = value
    .trim()
    .replace(/\s*(?:→|->)\s*/g, "_")
    .replace(/[^\p{L}\p{N}_.-]+/gu, "_")
    .replace(/^[_\-.]+|[_\-.]+$/g, "");

  return sanitized || fallback;
}
