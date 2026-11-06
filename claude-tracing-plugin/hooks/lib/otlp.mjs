// OTLP JSON encoding and the attribute names of the Respan span contract
// (contribution/span-contract.md). A hooks module can import only its own files,
// so these names live here instead of coming from @respan/respan-sdk.

export const ATTR = {
  LOG_TYPE: 'respan.entity.log_type',
  LOG_METHOD: 'respan.entity.log_method',
  METADATA_PREFIX: 'respan.metadata.',
  SESSION: 'respan.sessions.session_identifier',
  THREAD: 'respan.threads.thread_identifier',
  CUSTOMER: 'respan.customer_params.customer_identifier',
  WORKFLOW: 'traceloop.workflow.name',
  ENTITY_NAME: 'traceloop.entity.name',
  ENTITY_PATH: 'traceloop.entity.path',
  ENTITY_INPUT: 'traceloop.entity.input',
  ENTITY_OUTPUT: 'traceloop.entity.output',
  LLM_REQUEST_TYPE: 'llm.request.type',
  LLM_TOTAL_TOKENS: 'llm.usage.total_tokens',
  SYSTEM: 'gen_ai.system',
  REQUEST_MODEL: 'gen_ai.request.model',
  RESPONSE_MODEL: 'gen_ai.response.model',
  INPUT_TOKENS: 'gen_ai.usage.input_tokens',
  OUTPUT_TOKENS: 'gen_ai.usage.output_tokens',
  PROMPT_TOKENS: 'gen_ai.usage.prompt_tokens',
  COMPLETION_TOKENS: 'gen_ai.usage.completion_tokens',
  CACHE_READ_TOKENS: 'gen_ai.usage.cache_read_input_tokens',
  CACHE_CREATION_TOKENS: 'gen_ai.usage.cache_creation_input_tokens',
  PROMPT_PREFIX: 'gen_ai.prompt.',
  COMPLETION_PREFIX: 'gen_ai.completion.',
  TOOL_CALL_ID: 'gen_ai.tool.call.id',
  TIME_TO_FIRST_CHUNK: 'gen_ai.response.time_to_first_chunk',
};

export const LOG_METHOD_TS_TRACING = 'ts_tracing';

const SPAN_KIND_INTERNAL = 1;
const SPAN_KIND_CLIENT = 3;
const STATUS_OK = 1;
const STATUS_ERROR = 2;

const CONTENT_KEY_PREFIXES = [
  ATTR.PROMPT_PREFIX,
  ATTR.COMPLETION_PREFIX,
  ATTR.ENTITY_INPUT,
  ATTR.ENTITY_OUTPUT,
];

const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const MASK_64 = 0xffffffffffffffffn;

/** A stable 16-hex span ID, so a retried export of the same span stays one span. */
export function spanIdFrom(seed) {
  let hash = FNV_OFFSET;
  for (let i = 0; i < seed.length; i += 1) {
    hash = ((hash ^ BigInt(seed.charCodeAt(i))) * FNV_PRIME) & MASK_64;
  }
  return hash.toString(16).padStart(16, '0');
}

/** A 32-hex trace ID: Claude Code's turn IDs are UUIDs, which already have that shape. */
export function traceIdFrom(turnId) {
  const hex = String(turnId).replace(/-/g, '').toLowerCase();
  if (/^[0-9a-f]{32}$/.test(hex)) return hex;
  return spanIdFrom(`${turnId}:high`) + spanIdFrom(`${turnId}:low`);
}

/** Marks a number to send as an OTLP double even when it happens to be whole. */
export class Double {
  constructor(value) {
    this.value = value;
  }
}

function toOtlpValue(value) {
  if (value instanceof Double) return { doubleValue: value.value };
  if (typeof value === 'string') return { stringValue: value };
  if (typeof value === 'boolean') return { boolValue: value };
  if (typeof value === 'number') {
    return Number.isInteger(value) ? { intValue: String(value) } : { doubleValue: value };
  }
  return { stringValue: JSON.stringify(value) };
}

function toOtlpAttributes(attributes) {
  const result = [];
  for (const [key, value] of Object.entries(attributes)) {
    if (value !== null && value !== undefined) result.push({ key, value: toOtlpValue(value) });
  }
  return result;
}

function msToNanos(ms) {
  return String(BigInt(Math.round(ms)) * 1_000_000n);
}

export function buildSpan({ traceId, spanId, parentSpanId, name, isLlm, startMs, endMs, attributes, error }) {
  const span = {
    traceId,
    spanId,
    name,
    kind: isLlm ? SPAN_KIND_CLIENT : SPAN_KIND_INTERNAL,
    startTimeUnixNano: msToNanos(startMs),
    endTimeUnixNano: msToNanos(Math.max(startMs, endMs)),
    attributes: toOtlpAttributes(attributes),
    status: error ? { code: STATUS_ERROR, message: error } : { code: STATUS_OK },
  };
  if (parentSpanId) span.parentSpanId = parentSpanId;
  return span;
}

function withoutContent(span) {
  const attributes = span.attributes.filter(
    (attribute) => !CONTENT_KEY_PREFIXES.some((prefix) => attribute.key.startsWith(prefix)),
  );
  attributes.push({ key: `${ATTR.METADATA_PREFIX}content_dropped`, value: { boolValue: true } });
  return { ...span, attributes };
}

/**
 * Group spans into OTLP JSON bodies under `maxBodyChars`. A span too large for a body on
 * its own keeps its timing and usage and loses its content.
 */
export function buildBodies(spans, { maxBodyChars, serviceVersion }) {
  const groups = [];
  let group = [];
  let groupChars = 0;
  for (const original of spans) {
    let span = original;
    let chars = JSON.stringify(span).length;
    if (chars > maxBodyChars) {
      span = withoutContent(span);
      chars = JSON.stringify(span).length;
    }
    if (group.length > 0 && groupChars + chars > maxBodyChars) {
      groups.push(group);
      group = [];
      groupChars = 0;
    }
    group.push(span);
    groupChars += chars;
  }
  if (group.length > 0) groups.push(group);

  return groups.map((scopeSpans) => JSON.stringify({
    resourceSpans: [{
      resource: {
        attributes: toOtlpAttributes({ 'service.name': 'respan-claude-code', 'service.version': serviceVersion }),
      },
      scopeSpans: [{ scope: { name: 'respan-claude-code' }, spans: scopeSpans }],
    }],
  }));
}
