import {
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  GEN_AI_OPERATION_NAME_VALUE_INVOKE_AGENT,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  EVE_ATTRIBUTE_PREFIX,
  EVE_CHANNEL_KIND,
  EVE_ENVIRONMENT,
  EVE_RETRY_REASON,
  EVE_SCOPE_NAME,
  EVE_SESSION_ID,
  EVE_STEP_INDEX,
  EVE_TURN_ID,
  EVE_TURN_SEQUENCE,
  EVE_VERSION,
} from "../constants/eve.js";
import {
  EVE_RESPAN_BRIDGE_RUNTIME_CONTEXT_KEY,
  EVE_RESPAN_BRIDGE_RUNTIME_CONTEXT_PREFIX,
  EVE_RESPAN_LINEAGE_PARENT_CALL_ID_ATTRIBUTE,
  EVE_RESPAN_LINEAGE_PARENT_SESSION_ID_ATTRIBUTE,
  EVE_RESPAN_LINEAGE_PARENT_TURN_ID_ATTRIBUTE,
  EVE_RESPAN_LINEAGE_PARENT_TURN_SEQUENCE_ATTRIBUTE,
  EVE_RESPAN_LINEAGE_ROOT_SESSION_ID_ATTRIBUTE,
} from "../constants/lineage.js";
import {
  AI_AGENT_ID,
  AI_SETTINGS_CONTEXT_PREFIX,
  AI_TELEMETRY_FUNCTION_ID,
  setDefault,
  setMetadata,
  type SpanAttributes,
} from "./shared.js";

export function resolveEveAgentName(
  attrs: SpanAttributes,
  fallback = "eve",
): string {
  return String(
    attrs[AI_TELEMETRY_FUNCTION_ID] ??
      attrs[ATTR_GEN_AI_AGENT_NAME] ??
      attrs[AI_AGENT_ID] ??
      fallback,
  );
}

export function enrichEveAttributes(
  attrs: SpanAttributes,
  span: { readonly name: string; readonly scopeName?: string },
): void {
  const sessionId = readEveAttribute(attrs, EVE_SESSION_ID);
  const rootSessionId = readNonEmptyAttribute(
    attrs,
    EVE_RESPAN_LINEAGE_ROOT_SESSION_ID_ATTRIBUTE,
  );
  if (sessionId !== undefined && sessionId !== null && sessionId !== "") {
    const normalizedSessionId = String(sessionId);
    setDefault(
      attrs,
      RespanSpanAttributes.RESPAN_SESSION_ID,
      normalizedSessionId,
    );
    setDefault(
      attrs,
      RespanSpanAttributes.RESPAN_THREADS_ID,
      normalizedSessionId,
    );
    setDefault(
      attrs,
      RespanSpanAttributes.RESPAN_TRACE_GROUP_ID,
      rootSessionId ?? normalizedSessionId,
    );
  }

  const environment = readEveAttribute(attrs, EVE_ENVIRONMENT);
  if (environment !== undefined && environment !== null && environment !== "") {
    setDefault(
      attrs,
      RespanSpanAttributes.RESPAN_ENVIRONMENT,
      String(environment),
    );
  }

  const eveMetadata: Record<string, unknown> = {};
  addMetadataValue(
    eveMetadata,
    "version",
    readEveAttribute(attrs, EVE_VERSION),
  );
  addMetadataValue(eveMetadata, "environment", environment);
  addMetadataValue(
    eveMetadata,
    "turn_id",
    readEveAttribute(attrs, EVE_TURN_ID),
  );
  addNumericMetadataValue(
    eveMetadata,
    "turn_sequence",
    readEveAttribute(attrs, EVE_TURN_SEQUENCE),
  );
  addNumericMetadataValue(
    eveMetadata,
    "step_index",
    readEveAttribute(attrs, EVE_STEP_INDEX),
  );
  addMetadataValue(
    eveMetadata,
    "channel_kind",
    readEveAttribute(attrs, EVE_CHANNEL_KIND),
  );
  addMetadataValue(
    eveMetadata,
    "retry_reason",
    readEveAttribute(attrs, EVE_RETRY_REASON),
  );
  addMetadataValue(eveMetadata, "root_session_id", rootSessionId);

  const parentMetadata = buildParentMetadata(attrs);
  if (parentMetadata !== undefined) {
    eveMetadata.parent = parentMetadata;
  }

  const runtimeContext = collectAuthoredRuntimeContext(attrs);
  if (Object.keys(runtimeContext).length > 0) {
    eveMetadata.runtime_context = runtimeContext;
  }

  if (
    span.scopeName === EVE_SCOPE_NAME &&
    span.name.startsWith("invoke_agent ") &&
    attrs[ATTR_GEN_AI_OPERATION_NAME] ===
      GEN_AI_OPERATION_NAME_VALUE_INVOKE_AGENT
  ) {
    const usage: Record<string, number> = {};
    addNumericUsage(
      usage,
      "input_tokens",
      attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS],
    );
    addNumericUsage(
      usage,
      "output_tokens",
      attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS],
    );
    addNumericUsage(
      usage,
      "cache_read_input_tokens",
      attrs[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS],
    );
    addNumericUsage(
      usage,
      "cache_creation_input_tokens",
      attrs[ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS],
    );

    const subagent: Record<string, unknown> = {
      name: resolveEveAgentName(attrs),
    };
    if (Object.keys(usage).length > 0) {
      subagent.usage = usage;
    }
    eveMetadata.subagent = subagent;
  }

  if (Object.keys(eveMetadata).length > 0) {
    setMetadata(attrs, "eve", eveMetadata);
  }
}

function readEveAttribute(
  attrs: SpanAttributes,
  key: string,
): unknown {
  return attrs[key] ?? attrs[AI_SETTINGS_CONTEXT_PREFIX + key];
}

/**
 * Preserve user-authored `step.started` context before the generic `ai.*`
 * cleanup runs. Eve reserves every `eve.*` runtime-context key for framework
 * fields, so those stay exclusively owned by the explicit mappings above.
 *
 * AI SDK flattens nested context objects into dotted attribute names. Keeping
 * the suffix as-is avoids guessing whether a dot came from an authored key or
 * from that flattening step (both are valid in Eve authored context).
 */
function collectAuthoredRuntimeContext(
  attrs: SpanAttributes,
): Record<string, unknown> {
  const runtimeContext: Record<string, unknown> = {};

  for (const [attributeKey, value] of Object.entries(attrs)) {
    if (!attributeKey.startsWith(AI_SETTINGS_CONTEXT_PREFIX)) {
      continue;
    }

    const contextKey = attributeKey.slice(AI_SETTINGS_CONTEXT_PREFIX.length);
    if (
      contextKey === "" ||
      contextKey.startsWith(EVE_ATTRIBUTE_PREFIX) ||
      contextKey === EVE_RESPAN_BRIDGE_RUNTIME_CONTEXT_KEY ||
      contextKey.startsWith(EVE_RESPAN_BRIDGE_RUNTIME_CONTEXT_PREFIX)
    ) {
      continue;
    }

    const jsonValue = toJsonSafeValue(value);
    if (jsonValue !== undefined) {
      runtimeContext[contextKey] = jsonValue;
    }
  }

  return runtimeContext;
}

function buildParentMetadata(
  attrs: SpanAttributes,
): Record<string, unknown> | undefined {
  const parent: Record<string, unknown> = {};
  addMetadataValue(
    parent,
    "session_id",
    readNonEmptyAttribute(
      attrs,
      EVE_RESPAN_LINEAGE_PARENT_SESSION_ID_ATTRIBUTE,
    ),
  );
  addMetadataValue(
    parent,
    "call_id",
    readNonEmptyAttribute(attrs, EVE_RESPAN_LINEAGE_PARENT_CALL_ID_ATTRIBUTE),
  );

  const turn: Record<string, unknown> = {};
  addMetadataValue(
    turn,
    "id",
    readNonEmptyAttribute(attrs, EVE_RESPAN_LINEAGE_PARENT_TURN_ID_ATTRIBUTE),
  );
  addNumericMetadataValue(
    turn,
    "sequence",
    attrs[EVE_RESPAN_LINEAGE_PARENT_TURN_SEQUENCE_ATTRIBUTE],
  );
  if (Object.keys(turn).length > 0) {
    parent.turn = turn;
  }

  return Object.keys(parent).length > 0 ? parent : undefined;
}

function readNonEmptyAttribute(
  attrs: SpanAttributes,
  key: string,
): string | undefined {
  const value = attrs[key];
  return value === undefined || value === null || value === ""
    ? undefined
    : String(value);
}

function toJsonSafeValue(value: unknown): unknown | undefined {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "boolean"
  ) {
    return value;
  }

  if (typeof value === "number") {
    return Number.isFinite(value) ? value : undefined;
  }

  if (typeof value !== "object") {
    return undefined;
  }

  try {
    const serialized = JSON.stringify(value);
    return serialized === undefined ? undefined : JSON.parse(serialized);
  } catch {
    // Authored instrumentation must never be able to break span export.
    return undefined;
  }
}

function addMetadataValue(
  metadata: Record<string, unknown>,
  key: string,
  value: unknown,
): void {
  if (value !== undefined && value !== null && value !== "") {
    metadata[key] = String(value);
  }
}

function addNumericMetadataValue(
  metadata: Record<string, unknown>,
  key: string,
  value: unknown,
): void {
  if (value === undefined || value === null || value === "") {
    return;
  }
  const parsed = Number(value);
  metadata[key] = Number.isFinite(parsed) ? parsed : String(value);
}

function addNumericUsage(
  usage: Record<string, number>,
  key: string,
  value: unknown,
): void {
  if (value === undefined || value === null || value === "") {
    return;
  }
  const parsed = Number(value);
  if (Number.isFinite(parsed)) {
    usage[key] = parsed;
  }
}
