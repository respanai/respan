import { context, trace } from "@opentelemetry/api";
import type { SpanContext } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import {
  RESPAN_SPAN_ATTRIBUTES_MAP,
  RespanLogType,
  RespanSpanAttributes,
} from "@respan/respan-sdk";
import {
  buildReadableSpan,
  ensureTraceId,
  getPropagatedAttributes,
  injectSpan,
} from "@respan/tracing";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

const PACKAGE_VERSION = "0.1.0";
const INSTRUMENTATION_NAME = "@respan/instrumentation-braintrust";
const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";

const BRAINTRUST_IS_MERGE_FIELD = "_is_merge";
const BRAINTRUST_MERGE_PATHS_FIELD = "_merge_paths";
const BRAINTRUST_CREATED_FIELD = "created";
const BRAINTRUST_SPAN_ID_FIELD = "span_id";
const BRAINTRUST_ROOT_SPAN_ID_FIELD = "root_span_id";
const BRAINTRUST_SPAN_PARENTS_FIELD = "span_parents";
const BRAINTRUST_PARENT_ID_FIELD = "_parent_id";

const GEN_AI_PROMPT_ROLE = (index: number) => `${ATTR_GEN_AI_PROMPT}.${index}.role`;
const GEN_AI_PROMPT_CONTENT = (index: number) => `${ATTR_GEN_AI_PROMPT}.${index}.content`;
const GEN_AI_PROMPT_TOOL_CALLS = (index: number) => `${ATTR_GEN_AI_PROMPT}.${index}.tool_calls`;
const GEN_AI_COMPLETION_PREFIX = `${ATTR_GEN_AI_COMPLETION}.0`;
const GEN_AI_COMPLETION_ROLE = `${GEN_AI_COMPLETION_PREFIX}.role`;
const GEN_AI_COMPLETION_CONTENT = `${GEN_AI_COMPLETION_PREFIX}.content`;
const GEN_AI_COMPLETION_TOOL_CALLS = `${GEN_AI_COMPLETION_PREFIX}.tool_calls`;
const LLM_USAGE_CACHE_READ_INPUT_TOKENS = "llm.usage.cache_read_input_tokens";
const STATUS_CODE_ATTR = "status_code";
const ERROR_MESSAGE_ATTR = "error.message";

const MERGE_ROW_KEY_FIELDS = [
  "org_id",
  "project_id",
  "experiment_id",
  "dataset_id",
  "prompt_session_id",
  "log_id",
  "id",
];
const MERGE_ROW_SKIP_FIELDS = [
  BRAINTRUST_CREATED_FIELD,
  BRAINTRUST_SPAN_ID_FIELD,
  BRAINTRUST_ROOT_SPAN_ID_FIELD,
  BRAINTRUST_SPAN_PARENTS_FIELD,
  BRAINTRUST_PARENT_ID_FIELD,
];
const SET_UNION_FIELDS = new Set(["tags"]);

type BraintrustRecord = Record<string, any>;
type PropagatedAttributes = Record<string, any>;
type BraintrustLazyValue<T = BraintrustRecord> =
  | T
  | (() => T | Promise<T>)
  | {
      get?: () => T | Promise<T>;
      getSync?: () => { resolved: boolean; value?: T };
    };

interface BraintrustStateLike {
  setOverrideBgLogger(logger: BraintrustBackgroundLoggerLike | null): void;
}

interface BraintrustBackgroundLoggerLike {
  log(items: BraintrustLazyValue[]): void;
  flush(): Promise<void>;
  pendingFlushBytes(): number;
  flushBackpressureBytes(): number;
  setMaskingFunction(maskingFunction: ((value: unknown) => unknown) | null): void;
}

interface CapturedLazyRecord {
  item: BraintrustLazyValue;
  activeSpanContext?: SpanContext;
  propagatedAttributes?: PropagatedAttributes;
}

export interface BraintrustExportOptions {
  activeSpanContext?: SpanContext;
  propagatedAttributes?: PropagatedAttributes;
}

export interface BraintrustInstrumentorOptions {
  includeBraintrustRawMetadata?: boolean;
}

export class BraintrustInstrumentor {
  public readonly name = "braintrust";

  private _enabled = true;
  private _state: BraintrustStateLike | null = null;
  private _bridge: RespanBraintrustBackgroundLogger | null = null;
  private readonly _traceIdMap = new Map<string, string>();
  private readonly _options: Required<BraintrustInstrumentorOptions>;

  constructor(options: BraintrustInstrumentorOptions = {}) {
    this._options = {
      includeBraintrustRawMetadata: options.includeBraintrustRawMetadata ?? false,
    };
  }

  async activate(): Promise<void> {
    this._enabled = true;

    const braintrust = await import("braintrust");
    const getGlobalState = (braintrust as any)._internalGetGlobalState;
    const state = typeof getGlobalState === "function" ? getGlobalState() : null;
    if (!state || typeof state.setOverrideBgLogger !== "function") {
      throw new Error("Braintrust global state does not support background logger overrides");
    }

    this._bridge = new RespanBraintrustBackgroundLogger(this);
    this._state = state;
    state.setOverrideBgLogger(this._bridge);
  }

  async deactivate(): Promise<void> {
    this._enabled = false;
    this._traceIdMap.clear();

    if (this._state) {
      this._state.setOverrideBgLogger(null);
    }
    this._bridge = null;
    this._state = null;
  }

  async flush(): Promise<void> {
    await this._bridge?.flush();
  }

  exportRecord(record: BraintrustRecord, options: BraintrustExportOptions = {}): boolean {
    if (!this._enabled || !isBraintrustSpanRecord(record)) {
      return false;
    }

    const readableSpan = this._buildReadableSpan(record, options);
    return injectSpan(readableSpan);
  }

  exportRecords(records: BraintrustRecord[], options: BraintrustExportOptions = {}): number {
    let exported = 0;
    for (const record of mergeBraintrustRows(records)) {
      if (this.exportRecord(record, options)) {
        exported += 1;
      }
    }
    return exported;
  }

  private _buildReadableSpan(
    record: BraintrustRecord,
    options: BraintrustExportOptions,
  ): ReadableSpan {
    const capturedActiveSpanContext = options.activeSpanContext ?? trace.getSpan(context.active())?.spanContext();
    const activeTraceId = capturedActiveSpanContext?.traceId;
    const activeSpanId = capturedActiveSpanContext?.spanId;
    const braintrustRootId = String(
      record[BRAINTRUST_ROOT_SPAN_ID_FIELD] ??
        record[BRAINTRUST_SPAN_ID_FIELD] ??
        record.id,
    );
    const traceId = this._resolveTraceId(braintrustRootId, activeTraceId);
    const spanId = String(record[BRAINTRUST_SPAN_ID_FIELD] ?? record.id);
    const parentId = resolveParentId(record, activeTraceId === traceId ? activeSpanId : undefined);
    const errorMessage = extractErrorMessage(record.error);
    const attrs = buildBraintrustAttributes(record, this._options);

    const workflowPropagatedAttributes = options.propagatedAttributes ?? (getPropagatedAttributes() as PropagatedAttributes | undefined);
    addWorkflowName(attrs, workflowPropagatedAttributes);
    mergeCapturedRespanAttributes(attrs, options.propagatedAttributes);

    if (errorMessage) {
      attrs[ERROR_MESSAGE_ATTR] = errorMessage;
      attrs[STATUS_CODE_ATTR] = 500;
    }

    const readableSpan = buildReadableSpan({
      name: resolveEntityName(record),
      traceId,
      spanId,
      parentId,
      startTimeIso: resolveStartTimeIso(record),
      endTimeIso: resolveEndTimeIso(record),
      attributes: attrs,
      statusCode: errorMessage ? 500 : 200,
      errorMessage,
      mergePropagated: options.propagatedAttributes === undefined,
    }) as ReadableSpan & {
      instrumentationScope?: { name: string; version?: string };
    };

    readableSpan.instrumentationScope = {
      name: INSTRUMENTATION_NAME,
      version: PACKAGE_VERSION,
    };
    return readableSpan;
  }

  private _resolveTraceId(braintrustRootId: string, activeTraceId?: string): string {
    const existingTraceId = this._traceIdMap.get(braintrustRootId);
    if (existingTraceId) {
      return existingTraceId;
    }

    const resolvedTraceId = activeTraceId ? ensureTraceId(activeTraceId) : ensureTraceId(braintrustRootId);
    this._traceIdMap.set(braintrustRootId, resolvedTraceId);
    return resolvedTraceId;
  }
}

export { BraintrustInstrumentor as RespanBraintrustInstrumentor };

class RespanBraintrustBackgroundLogger implements BraintrustBackgroundLoggerLike {
  private readonly _instrumentor: BraintrustInstrumentor;
  private _items: CapturedLazyRecord[] = [];
  private _maskingFunction: ((value: unknown) => unknown) | null = null;

  constructor(instrumentor: BraintrustInstrumentor) {
    this._instrumentor = instrumentor;
  }

  log(items: BraintrustLazyValue[]): void {
    const activeSpanContext = trace.getSpan(context.active())?.spanContext();
    const propagatedAttributes = getPropagatedAttributes() as PropagatedAttributes | undefined;

    for (const item of items) {
      this._items.push({
        item,
        activeSpanContext,
        propagatedAttributes,
      });
    }
  }

  async flush(): Promise<void> {
    const captured = this._items;
    this._items = [];

    const grouped = new Map<string, Array<{ record: BraintrustRecord; capture: CapturedLazyRecord }>>();
    for (const capture of captured) {
      const record = await resolveLazyValue(capture.item);
      if (!record || typeof record !== "object" || Array.isArray(record)) {
        continue;
      }

      const maskedRecord = this._applyMasking(record as BraintrustRecord);
      const key = generateMergedRowKey(maskedRecord);
      const group = grouped.get(key) ?? [];
      group.push({ record: maskedRecord, capture });
      grouped.set(key, group);
    }

    for (const group of grouped.values()) {
      const mergedRecords = mergeBraintrustRows(group.map((item) => item.record));
      for (const mergedRecord of mergedRecords) {
        const capture = group[group.length - 1]?.capture;
        this._instrumentor.exportRecord(mergedRecord, {
          activeSpanContext: capture?.activeSpanContext,
          propagatedAttributes: capture?.propagatedAttributes ?? {},
        });
      }
    }
  }

  pendingFlushBytes(): number {
    return this._items.length;
  }

  flushBackpressureBytes(): number {
    return Number.POSITIVE_INFINITY;
  }

  setMaskingFunction(maskingFunction: ((value: unknown) => unknown) | null): void {
    this._maskingFunction = maskingFunction;
  }

  private _applyMasking(record: BraintrustRecord): BraintrustRecord {
    if (!this._maskingFunction) {
      return record;
    }
    const masked = this._maskingFunction(record);
    return masked && typeof masked === "object" && !Array.isArray(masked)
      ? masked as BraintrustRecord
      : record;
  }
}

function buildBraintrustAttributes(
  record: BraintrustRecord,
  options: Required<BraintrustInstrumentorOptions>,
): Record<string, unknown> {
  const logType = resolveLogType(record);
  const attrs: Record<string, unknown> = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: resolveEntityName(record),
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: resolveEntityName(record),
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType,
    [metadataKey("braintrust_row_id")]: String(record.id),
  };

  const spanType = getBraintrustSpanType(record);
  if (spanType) {
    attrs[metadataKey("braintrust_span_type")] = spanType;
  }

  if (record[BRAINTRUST_SPAN_ID_FIELD]) {
    attrs[metadataKey("braintrust_span_id")] = String(record[BRAINTRUST_SPAN_ID_FIELD]);
  }
  if (record[BRAINTRUST_ROOT_SPAN_ID_FIELD]) {
    attrs[metadataKey("braintrust_root_span_id")] = String(record[BRAINTRUST_ROOT_SPAN_ID_FIELD]);
  }
  if (Array.isArray(record.tags) && record.tags.length > 0) {
    attrs[metadataKey("braintrust_tags")] = safeJson(record.tags);
  }
  if (record.scores !== undefined) {
    attrs[metadataKey("braintrust_scores")] = safeJson(record.scores);
  }
  if (record.expected !== undefined) {
    attrs[metadataKey("braintrust_expected")] = safeJson(record.expected);
  }
  if (record.context !== undefined) {
    attrs[metadataKey("braintrust_context")] = safeJson(record.context);
  }
  if (record.metrics !== undefined) {
    attrs[metadataKey("braintrust_metrics")] = safeJson(record.metrics);
  }
  if (options.includeBraintrustRawMetadata && record.span_attributes !== undefined) {
    attrs[metadataKey("braintrust_span_attributes")] = safeJson(record.span_attributes);
  }

  mergeMetadata(attrs, record.metadata);
  addInputOutput(attrs, record, logType);
  addModelAttributes(attrs, record);
  addUsageAttributes(attrs, record.metrics);
  addToolDefinitions(attrs, record);
  addCompletionToolCalls(attrs, collectToolCalls(record.output));

  return attrs;
}

function addInputOutput(
  attrs: Record<string, unknown>,
  record: BraintrustRecord,
  logType: string,
): void {
  if (record.input !== undefined) {
    const messages = normalizeMessages(record.input);
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = messages
      ? safeJson(messages)
      : safeJson(record.input);

    if (isLlmLogType(logType) && messages) {
      addPromptMessageAttributes(attrs, messages);
    }
  }

  if (record.output === undefined) {
    return;
  }

  if (isLlmLogType(logType)) {
    const outputMessage = normalizeAssistantOutput(record.output);
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(outputMessage);
    attrs[GEN_AI_COMPLETION_ROLE] = "assistant";
    attrs[GEN_AI_COMPLETION_CONTENT] = String(outputMessage.content ?? "");
    if (outputMessage.tool_calls) {
      attrs[GEN_AI_COMPLETION_TOOL_CALLS] = safeJson(outputMessage.tool_calls);
    }
    return;
  }

  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] =
    typeof record.output === "string" ? record.output : safeJson(record.output);
}

function addModelAttributes(
  attrs: Record<string, unknown>,
  record: BraintrustRecord,
): void {
  const spanAttrs = asRecord(record.span_attributes);
  const metadata = asRecord(record.metadata);
  const metrics = asRecord(record.metrics);
  const rawModel = firstDefined(
    spanAttrs.model,
    spanAttrs.model_name,
    spanAttrs.modelName,
    metadata.model,
    metadata.model_name,
    metrics.model,
  );
  const rawProvider = firstDefined(
    spanAttrs.provider,
    spanAttrs.system,
    metadata.provider,
    metadata.system,
  );

  if (rawModel !== undefined) {
    const parsed = parseModelInfo(String(rawModel), rawProvider ? String(rawProvider) : undefined);
    attrs[ATTR_GEN_AI_REQUEST_MODEL] = parsed.model;
    if (parsed.provider) {
      attrs[ATTR_GEN_AI_SYSTEM] = parsed.provider;
    }
  } else if (rawProvider !== undefined) {
    attrs[ATTR_GEN_AI_SYSTEM] = normalizeProvider(String(rawProvider));
  }
}

function addUsageAttributes(
  attrs: Record<string, unknown>,
  metrics: unknown,
): void {
  const metricRecord = asRecord(metrics);
  const inputTokens = coerceInteger(firstDefined(
    metricRecord.prompt_tokens,
    metricRecord.input_tokens,
    metricRecord.promptTokens,
    metricRecord.inputTokens,
  ));
  const outputTokens = coerceInteger(firstDefined(
    metricRecord.completion_tokens,
    metricRecord.output_tokens,
    metricRecord.completionTokens,
    metricRecord.outputTokens,
  ));
  const totalTokens = coerceInteger(firstDefined(
    metricRecord.tokens,
    metricRecord.total_tokens,
    metricRecord.totalTokens,
  ));
  const cacheReadTokens = coerceInteger(firstDefined(
    metricRecord.cache_read_input_tokens,
    metricRecord.cacheReadInputTokens,
  ));

  if (inputTokens !== null) {
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = inputTokens;
    attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = inputTokens;
  }
  if (outputTokens !== null) {
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = outputTokens;
    attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = outputTokens;
  }
  if (inputTokens !== null || outputTokens !== null) {
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = (inputTokens ?? 0) + (outputTokens ?? 0);
  } else if (totalTokens !== null) {
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = totalTokens;
  }
  if (cacheReadTokens !== null) {
    attrs[LLM_USAGE_CACHE_READ_INPUT_TOKENS] = cacheReadTokens;
  }
}

function addToolDefinitions(
  attrs: Record<string, unknown>,
  record: BraintrustRecord,
): void {
  const candidates = [
    asRecord(record.span_attributes).tools,
    asRecord(record.span_attributes).functions,
    asRecord(record.metadata).tools,
    asRecord(record.metadata).functions,
    asRecord(record.input).tools,
    asRecord(record.input).functions,
  ];
  const tools = candidates.flatMap((candidate) => normalizeToolDefinitions(candidate));
  if (tools.length === 0) {
    return;
  }

  attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(dedupeObjects(tools));
}

function addCompletionToolCalls(
  attrs: Record<string, unknown>,
  toolCalls: Record<string, unknown>[],
): void {
  if (toolCalls.length === 0) {
    return;
  }

  attrs[GEN_AI_COMPLETION_TOOL_CALLS] = safeJson(dedupeObjects(toolCalls));
}

function addPromptMessageAttributes(
  attrs: Record<string, unknown>,
  messages: Array<Record<string, unknown>>,
): void {
  messages.forEach((message, index) => {
    attrs[GEN_AI_PROMPT_ROLE(index)] = String(message.role ?? "user");
    attrs[GEN_AI_PROMPT_CONTENT(index)] = normalizeContent(message.content);
    if (message.tool_calls !== undefined) {
      attrs[GEN_AI_PROMPT_TOOL_CALLS(index)] = safeJson(message.tool_calls);
    }
  });
  attrs[SpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;
}

function resolveLogType(record: BraintrustRecord): string {
  const spanType = getBraintrustSpanType(record);
  switch (spanType) {
    case "llm":
      return RespanLogType.CHAT;
    case "tool":
    case "function":
      return RespanLogType.TOOL;
    case "score":
    case "classifier":
    case "review":
      return RespanLogType.GUARDRAIL;
    case "task":
    case "eval":
    case "automation":
    case "facet":
    case "preprocessor":
      return RespanLogType.TASK;
    default:
      return record.log_type ?? RespanLogType.TASK;
  }
}

function getBraintrustSpanType(record: BraintrustRecord): string | undefined {
  const spanAttrs = asRecord(record.span_attributes);
  return stringValue(firstDefined(
    spanAttrs.type,
    spanAttrs.span_type,
    spanAttrs.kind,
    asRecord(record.metadata).span_type,
  ));
}

function resolveEntityName(record: BraintrustRecord): string {
  const spanAttrs = asRecord(record.span_attributes);
  return stringValue(firstDefined(
    spanAttrs.name,
    spanAttrs.title,
    asRecord(record.metadata).name,
    asRecord(record.metadata).span_name,
    record.name,
  )) ?? `braintrust.${getBraintrustSpanType(record) ?? "span"}`;
}

function resolveParentId(record: BraintrustRecord, fallbackParentId?: string): string | undefined {
  const parents = record[BRAINTRUST_SPAN_PARENTS_FIELD];
  if (Array.isArray(parents) && parents.length > 0) {
    return String(parents[parents.length - 1]);
  }
  if (record[BRAINTRUST_PARENT_ID_FIELD]) {
    return String(record[BRAINTRUST_PARENT_ID_FIELD]);
  }
  return fallbackParentId;
}

function resolveStartTimeIso(record: BraintrustRecord): string | undefined {
  const metrics = asRecord(record.metrics);
  return toIsoString(firstDefined(
    metrics.start,
    metrics.start_time,
    metrics.startTime,
    record[BRAINTRUST_CREATED_FIELD],
  ));
}

function resolveEndTimeIso(record: BraintrustRecord): string | undefined {
  const metrics = asRecord(record.metrics);
  return toIsoString(firstDefined(
    metrics.end,
    metrics.end_time,
    metrics.endTime,
    metrics.finish,
    record[BRAINTRUST_CREATED_FIELD],
  ));
}

function mergeBraintrustRows(rows: BraintrustRecord[]): BraintrustRecord[] {
  for (const row of rows) {
    if (row.id === undefined) {
      throw new Error("Braintrust row is missing an id");
    }
  }

  const rowGroups = new Map<string, BraintrustRecord>();
  for (const row of rows.map((item) => deepCloneRecord(item))) {
    const key = generateMergedRowKey(row);
    const existingRow = rowGroups.get(key);
    if (existingRow !== undefined && row[BRAINTRUST_IS_MERGE_FIELD]) {
      const skipFields = popMergeRowSkipFields(existingRow);
      const preserveNoMerge = !existingRow[BRAINTRUST_IS_MERGE_FIELD];
      mergeDicts(existingRow, row);
      restoreMergeRowSkipFields(existingRow, skipFields);
      if (preserveNoMerge) {
        delete existingRow[BRAINTRUST_IS_MERGE_FIELD];
      }
    } else {
      rowGroups.set(key, row);
    }
  }

  return [...rowGroups.values()];
}

function generateMergedRowKey(row: BraintrustRecord): string {
  return JSON.stringify(MERGE_ROW_KEY_FIELDS.map((key) => row[key]));
}

function popMergeRowSkipFields(row: BraintrustRecord): BraintrustRecord {
  const popped: BraintrustRecord = {};
  for (const field of MERGE_ROW_SKIP_FIELDS) {
    if (field in row) {
      popped[field] = row[field];
      delete row[field];
    }
  }
  return popped;
}

function restoreMergeRowSkipFields(row: BraintrustRecord, skipFields: BraintrustRecord): void {
  for (const field of MERGE_ROW_SKIP_FIELDS) {
    delete row[field];
    if (field in skipFields) {
      row[field] = skipFields[field];
    }
  }
}

function mergeDicts(
  mergeInto: BraintrustRecord,
  mergeFrom: BraintrustRecord,
  path: string[] = [],
  mergePaths?: Set<string>,
): BraintrustRecord {
  const serializedMergePaths = mergePaths ?? new Set(
    (mergeFrom[BRAINTRUST_MERGE_PATHS_FIELD] ?? []).map((item: string[]) => JSON.stringify(item)),
  );

  for (const [key, mergeFromValue] of Object.entries(mergeFrom)) {
    const fullPath = [...path, key];
    const fullPathSerialized = JSON.stringify(fullPath);
    const mergeIntoValue = mergeInto[key];
    const isSetUnionField = path.length === 0 && SET_UNION_FIELDS.has(key) && !serializedMergePaths.has(fullPathSerialized);

    if (isSetUnionField && Array.isArray(mergeIntoValue) && Array.isArray(mergeFromValue)) {
      mergeInto[key] = dedupeValues([...mergeIntoValue, ...mergeFromValue]);
    } else if (
      isPlainObject(mergeIntoValue) &&
      isPlainObject(mergeFromValue) &&
      !serializedMergePaths.has(fullPathSerialized)
    ) {
      mergeDicts(mergeIntoValue as BraintrustRecord, mergeFromValue as BraintrustRecord, fullPath, serializedMergePaths);
    } else if (mergeFromValue !== undefined) {
      mergeInto[key] = mergeFromValue;
    }
  }
  return mergeInto;
}

function isBraintrustSpanRecord(record: BraintrustRecord): boolean {
  return Boolean(
    record.id &&
      (
        record[BRAINTRUST_SPAN_ID_FIELD] ||
        record[BRAINTRUST_ROOT_SPAN_ID_FIELD] ||
        record.span_attributes ||
        record.input !== undefined ||
        record.output !== undefined ||
        record.error !== undefined
      ),
  );
}

async function resolveLazyValue(item: BraintrustLazyValue): Promise<BraintrustRecord | undefined> {
  if (typeof item === "function") {
    return await item();
  }
  if (item && typeof item === "object" && "get" in item && typeof item.get === "function") {
    return await item.get();
  }
  if (item && typeof item === "object") {
    return item as BraintrustRecord;
  }
  return undefined;
}

function addWorkflowName(
  attrs: Record<string, unknown>,
  propagatedAttributes?: PropagatedAttributes,
): void {
  const metadata = asRecord(propagatedAttributes?.metadata);
  const workflowName = stringValue(firstDefined(
    metadata.workflow_name,
    propagatedAttributes?.trace_group_identifier,
    propagatedAttributes?.group_identifier,
  ));
  if (workflowName && attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] === undefined) {
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflowName;
  }
}

function mergeCapturedRespanAttributes(
  attrs: Record<string, unknown>,
  propagatedAttributes?: PropagatedAttributes,
): void {
  if (!propagatedAttributes) {
    return;
  }

  for (const [key, value] of Object.entries(propagatedAttributes)) {
    if (value === undefined || value === null) {
      continue;
    }
    const attrKey = RESPAN_SPAN_ATTRIBUTES_MAP[key];
    if (!attrKey || attrs[attrKey] !== undefined) {
      continue;
    }

    if (key === "metadata" && typeof value === "object" && !Array.isArray(value)) {
      mergeMetadata(attrs, value);
    } else if (key === "prompt" && typeof value === "object") {
      attrs[attrKey] = safeJson(value);
    } else {
      attrs[attrKey] = value;
    }
  }
}

function mergeMetadata(attrs: Record<string, unknown>, metadata: unknown): void {
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    return;
  }

  for (const [key, value] of Object.entries(metadata as Record<string, unknown>)) {
    if (value === undefined || value === null) {
      continue;
    }
    const attrKey = metadataKey(key);
    if (attrs[attrKey] === undefined) {
      attrs[attrKey] = typeof value === "string" ? value : safeJson(value);
    }
  }
}

function normalizeMessages(value: unknown): Array<Record<string, unknown>> | null {
  const payload = unwrapKnownPayload(value);
  if (typeof payload === "string") {
    return [{ role: "user", content: payload }];
  }
  if (Array.isArray(payload)) {
    const messages = payload.flatMap((item) => normalizeMessage(item));
    return messages.length > 0 ? messages : null;
  }

  const messages = normalizeMessage(payload);
  return messages.length > 0 ? messages : null;
}

function normalizeMessage(value: unknown): Array<Record<string, unknown>> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return [];
  }

  const record = value as Record<string, unknown>;
  if (typeof record.role === "string") {
    const normalized: Record<string, unknown> = {
      role: record.role,
      content: normalizeContent(record.content),
    };
    const toolCalls = normalizeToolCalls(record.tool_calls ?? record.toolCalls);
    if (toolCalls.length > 0) {
      normalized.tool_calls = toolCalls;
    }
    return [normalized];
  }

  for (const key of ["messages", "input", "prompt"]) {
    if (record[key] !== undefined) {
      return normalizeMessages(record[key]) ?? [];
    }
  }

  return [];
}

function normalizeAssistantOutput(value: unknown): Record<string, unknown> {
  const payload = unwrapKnownPayload(value);
  const message: Record<string, unknown> = {
    role: "assistant",
    content: extractText(payload) || (typeof payload === "string" ? payload : safeJson(payload)),
  };
  const toolCalls = collectToolCalls(payload);
  if (toolCalls.length > 0) {
    message.tool_calls = toolCalls;
  }
  return message;
}

function normalizeToolDefinitions(value: unknown): Record<string, unknown>[] {
  if (!value) {
    return [];
  }
  const values = Array.isArray(value) ? value : [value];
  return values.flatMap((item) => {
    if (typeof item === "string") {
      return [{ type: "function", function: { name: item } }];
    }
    if (!item || typeof item !== "object") {
      return [];
    }

    const record = item as Record<string, unknown>;
    if (record.type === "function" && isPlainObject(record.function)) {
      return [record as Record<string, unknown>];
    }

    const name = stringValue(firstDefined(record.name, record.toolName, record.functionName));
    if (!name) {
      return [];
    }

    const definition: Record<string, unknown> = {
      type: "function",
      function: {
        name,
      },
    };
    const fn = definition.function as Record<string, unknown>;
    if (record.description) {
      fn.description = String(record.description);
    }
    if (record.parameters ?? record.input_schema ?? record.schema) {
      fn.parameters = record.parameters ?? record.input_schema ?? record.schema;
    }
    return [definition];
  });
}

function collectToolCalls(value: unknown): Record<string, unknown>[] {
  const calls: Record<string, unknown>[] = [];

  const visit = (item: unknown): void => {
    if (!item || typeof item !== "object") {
      return;
    }
    if (Array.isArray(item)) {
      item.forEach(visit);
      return;
    }

    const record = item as Record<string, unknown>;
    const directCalls = record.tool_calls ?? record.toolCalls;
    if (Array.isArray(directCalls)) {
      directCalls.forEach((call) => calls.push(...normalizeToolCalls(call)));
    }

    if (record.toolCallId || record.tool_call_id || record.id || record.name || record.toolName) {
      calls.push(...normalizeToolCalls(record));
    }

    for (const key of ["message", "response", "output", "choices", "content"]) {
      if (record[key] !== undefined && record[key] !== item) {
        visit(record[key]);
      }
    }
  };

  visit(value);
  return dedupeObjects(calls);
}

function normalizeToolCalls(value: unknown): Record<string, unknown>[] {
  if (!value) {
    return [];
  }
  const values = Array.isArray(value) ? value : [value];
  return values.flatMap((item) => {
    if (!item || typeof item !== "object") {
      return [];
    }
    const record = item as Record<string, unknown>;
    const rawFunction = isPlainObject(record.function) ? record.function as Record<string, unknown> : {};
    const name = stringValue(firstDefined(
      rawFunction.name,
      record.name,
      record.toolName,
      record.functionName,
    ));
    if (!name) {
      return [];
    }

    const args = firstDefined(
      rawFunction.arguments,
      record.arguments,
      record.args,
      record.input,
    );

    return [{
      id: String(firstDefined(record.id, record.toolCallId, record.tool_call_id, `call_${name}`)),
      type: "function",
      function: {
        name,
        arguments: typeof args === "string" ? args : safeJson(args ?? {}),
      },
    }];
  });
}

function unwrapKnownPayload(value: unknown): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return value;
  }
  const record = value as Record<string, unknown>;
  for (const key of ["messages", "input", "prompt", "output", "response", "message"]) {
    if (record[key] !== undefined) {
      return record[key];
    }
  }
  return value;
}

function extractText(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => extractText(item)).filter(Boolean).join("");
  }
  if (!value || typeof value !== "object") {
    return value === undefined || value === null ? "" : String(value);
  }

  const record = value as Record<string, unknown>;
  for (const key of ["text", "content", "message", "response", "output"]) {
    if (record[key] !== undefined) {
      const text = extractText(record[key]);
      if (text) {
        return text;
      }
    }
  }
  return "";
}

function normalizeContent(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  const text = extractText(value);
  return text || safeJson(value);
}

function extractErrorMessage(error: unknown): string | undefined {
  if (!error) {
    return undefined;
  }
  if (typeof error === "string") {
    return error;
  }
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === "object") {
    const record = error as Record<string, unknown>;
    return stringValue(firstDefined(record.message, record.error, record.name)) ?? safeJson(error);
  }
  return String(error);
}

function parseModelInfo(rawModel: string, provider?: string): { provider?: string; model: string } {
  if (rawModel.includes("/")) {
    const [rawProvider, ...modelParts] = rawModel.split("/");
    return {
      provider: normalizeProvider(provider ?? rawProvider),
      model: modelParts.join("/") || rawModel,
    };
  }
  return {
    provider: provider ? normalizeProvider(provider) : inferProviderFromModel(rawModel),
    model: rawModel,
  };
}

function inferProviderFromModel(model: string): string | undefined {
  const lower = model.toLowerCase();
  if (lower.startsWith("gpt-") || lower.startsWith("o1") || lower.startsWith("o3") || lower.startsWith("o4")) {
    return "openai";
  }
  if (lower.startsWith("claude-")) {
    return "anthropic";
  }
  if (lower.startsWith("gemini-")) {
    return "google";
  }
  return undefined;
}

function normalizeProvider(provider: string): string {
  return provider.toLowerCase().replace(/^@/, "").replace(/[^a-z0-9._-]/g, "_");
}

function isLlmLogType(logType: string): boolean {
  return logType === RespanLogType.CHAT || logType === RespanLogType.TEXT;
}

function toIsoString(value: unknown): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (typeof value === "number") {
    const ms = value < 10_000_000_000 ? value * 1000 : value;
    return new Date(ms).toISOString();
  }
  if (typeof value === "string") {
    const numeric = Number(value);
    if (!Number.isNaN(numeric) && value.trim() !== "") {
      return toIsoString(numeric);
    }
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? undefined : date.toISOString();
  }
  return undefined;
}

function coerceInteger(value: unknown): number | null {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.trunc(numeric) : null;
}

function firstDefined(...values: unknown[]): unknown {
  return values.find((value) => value !== undefined && value !== null);
}

function stringValue(value: unknown): string | undefined {
  return value === undefined || value === null ? undefined : String(value);
}

function asRecord(value: unknown): Record<string, any> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, any>
    : {};
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function metadataKey(key: string): string {
  return `${RespanSpanAttributes.RESPAN_METADATA}.${key}`;
}

function dedupeValues(values: unknown[]): unknown[] {
  const seen = new Set<string>();
  const out: unknown[] = [];
  for (const value of values) {
    const key = typeof value === "object" ? safeJson(value) : String(value);
    if (!seen.has(key)) {
      seen.add(key);
      out.push(value);
    }
  }
  return out;
}

function dedupeObjects<T extends Record<string, unknown>>(values: T[]): T[] {
  return dedupeValues(values) as T[];
}

function deepCloneRecord(record: BraintrustRecord): BraintrustRecord {
  return JSON.parse(JSON.stringify(record));
}
