import { context, trace, TraceFlags } from "@opentelemetry/api";
import { hrTime } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  RESPAN_SPAN_ATTRIBUTES_MAP,
  RespanSpanAttributes,
} from "@respan/respan-sdk";
import { buildReadableSpan, getPropagatedAttributes, injectSpan } from "@respan/tracing";
import {
  buildDifySpanAttributes,
  difySpanName,
  safeJson,
  statusCodeFrom,
  type AttributeRecord,
  type DifyRequestOptionsLike,
} from "./_translator.js";

const PACKAGE_VERSION = "0.1.0";
const INSTRUMENTATION_SCOPE = "@respan/instrumentation-dify";

export interface DifyCallState {
  emitted: boolean;
  includeContent: boolean;
  parentId?: string;
  propagatedAttributes?: Record<string, unknown>;
  request: DifyRequestOptionsLike;
  startTime: [number, number];
  traceId?: string;
}

export function createDifyCallState(
  request: DifyRequestOptionsLike,
  includeContent: boolean,
): DifyCallState {
  const activeOtelContext = context.active();
  const activeContext = trace.getSpan(activeOtelContext)?.spanContext();
  const propagated = getPropagatedAttributes(activeOtelContext);
  return {
    emitted: false,
    includeContent,
    parentId: activeContext?.spanId,
    propagatedAttributes: snapshotRecord(propagated),
    request,
    startTime: hrTime(),
    traceId: activeContext?.traceId,
  };
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value && typeof value === "object" && !Array.isArray(value));

const metadataRecord = (value: unknown): Record<string, unknown> => {
  if (isRecord(value)) return value;
  if (typeof value !== "string") return {};
  try {
    const parsed = JSON.parse(value);
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
};

const snapshotRecord = (value: unknown): Record<string, unknown> | undefined => {
  if (!isRecord(value)) return undefined;
  try {
    const parsed = JSON.parse(safeJson(value));
    return isRecord(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
};

const otelSafeAttribute = (value: unknown): string | number | boolean | Array<string | number | boolean> => {
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (Array.isArray(value) && value.length > 0) {
    const valueType = typeof value[0];
    if (
      ["string", "number", "boolean"].includes(valueType) &&
      value.every((item) => typeof item === valueType)
    ) {
      return value as Array<string | number | boolean>;
    }
  }
  return safeJson(value);
};

export function mergeDifyPropagatedAttributes(
  attributes: AttributeRecord,
  propagated: Record<string, unknown> | undefined,
): AttributeRecord {
  if (!propagated) return attributes;
  const merged: AttributeRecord = { ...attributes };
  const difyMetadata = metadataRecord(
    merged[RespanSpanAttributes.RESPAN_METADATA],
  );
  const propagatedMetadata: Record<string, unknown> = {};
  const metadataPrefix = `${RespanSpanAttributes.RESPAN_METADATA}.`;

  for (const [key, value] of Object.entries(propagated)) {
    if (value === undefined || value === null) continue;
    if (key === "metadata" || key === RespanSpanAttributes.RESPAN_METADATA) {
      Object.assign(propagatedMetadata, metadataRecord(value));
      continue;
    }
    if (key.startsWith(metadataPrefix)) {
      const metadataKey = key.slice(metadataPrefix.length);
      if (metadataKey) propagatedMetadata[metadataKey] = value;
      continue;
    }
    const attributeKey = RESPAN_SPAN_ATTRIBUTES_MAP[key] ??
      (key.startsWith("respan.") ? key : undefined);
    if (!attributeKey || merged[attributeKey] !== undefined) continue;
    merged[attributeKey] = otelSafeAttribute(value);
  }

  if (Object.keys(propagatedMetadata).length > 0 || Object.keys(difyMetadata).length > 0) {
    merged[RespanSpanAttributes.RESPAN_METADATA] = safeJson({
      ...propagatedMetadata,
      ...difyMetadata,
    });
  }
  return merged;
}

export function mergeDifyPropagatedMetadata(
  attributes: AttributeRecord,
  propagated: { metadata?: unknown } | undefined = getPropagatedAttributes(),
): AttributeRecord {
  return mergeDifyPropagatedAttributes(
    attributes,
    propagated as Record<string, unknown> | undefined,
  );
}

export function emitDifyCall(
  state: DifyCallState,
  options: { response?: unknown; streamEvents?: unknown[]; error?: unknown } = {},
): void {
  if (state.emitted) return;
  state.emitted = true;
  try {
    const errorMessage = options.error instanceof Error
      ? options.error.message
      : options.error === undefined
        ? undefined
        : String(options.error);
    const attributes = mergeDifyPropagatedAttributes(
      buildDifySpanAttributes({
        request: state.request,
        response: options.response,
        streamEvents: options.streamEvents,
        error: options.error,
        includeContent: state.includeContent,
      }),
      state.propagatedAttributes,
    );
    const span = buildReadableSpan({
      name: difySpanName(String(state.request.path ?? "")),
      traceId: state.traceId,
      parentId: state.parentId,
      startTimeHr: state.startTime,
      endTimeHr: hrTime(),
      attributes,
      statusCode: statusCodeFrom(options.response, options.error),
      errorMessage,
      mergePropagated: false,
    }) as ReadableSpan & {
      instrumentationScope?: { name: string; version?: string };
      spanContext: () => ReturnType<ReadableSpan["spanContext"]>;
    };
    const originalSpanContext = span.spanContext.bind(span);
    span.spanContext = () => ({
      ...originalSpanContext(),
      traceFlags: TraceFlags.SAMPLED,
    });
    span.instrumentationScope = {
      name: INSTRUMENTATION_SCOPE,
      version: PACKAGE_VERSION,
    };
    injectSpan(span);
  } catch {
    // Instrumentation must never alter Dify application behavior.
  }
}
