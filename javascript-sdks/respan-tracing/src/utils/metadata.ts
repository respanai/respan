import { RespanSpanAttributes } from "@respan/respan-sdk";

type MetadataRecord = Record<string, unknown>;

/** Merge canonical metadata JSON without emitting respan.metadata.* aliases. */
export function mergeCanonicalMetadata(
  existing: unknown,
  propagated: unknown,
): string {
  return safeJson({
    ...metadataRecord(propagated),
    ...metadataRecord(existing),
  });
}

/**
 * Merge propagated metadata with metadata already present on a span.
 *
 * Older instrumentations may still emit `respan.metadata.<key>` aliases.
 * Fold those aliases into the canonical JSON attribute so explicit span
 * metadata keeps precedence and the exported span has one metadata shape.
 */
export function mergeCanonicalMetadataAttributes(
  attributes: MetadataRecord,
  propagated: unknown,
): string {
  const metadataKey = RespanSpanAttributes.RESPAN_METADATA;
  const aliasPrefix = `${metadataKey}.`;
  const aliases: MetadataRecord = {};

  for (const key of Object.keys(attributes)) {
    if (!key.startsWith(aliasPrefix)) continue;
    aliases[key.slice(aliasPrefix.length)] = attributes[key];
    delete attributes[key];
  }

  const merged = safeJson({
    ...metadataRecord(propagated),
    ...aliases,
    ...metadataRecord(attributes[metadataKey]),
  });
  attributes[metadataKey] = merged;
  return merged;
}

function metadataRecord(value: unknown): MetadataRecord {
  if (isRecord(value)) return { ...value };
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value) as unknown;
      return isRecord(parsed) ? { ...parsed } : { value: parsed };
    } catch {
      return { value };
    }
  }
  return value === undefined || value === null ? {} : { value };
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return JSON.stringify({ value: String(value) });
  }
}

function isRecord(value: unknown): value is MetadataRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
