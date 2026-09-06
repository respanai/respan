const MAX_DEPTH = 16;
const SENSITIVE_MARKERS = new Set([
  "apikey",
  "apitoken",
  "authorization",
  "authtoken",
  "bearertoken",
  "cookie",
  "credential",
  "password",
  "privatekey",
  "refreshtoken",
  "secret",
  "sessiontoken",
  "accesstoken",
  "xapikey",
]);

export function isSensitiveKey(key: PropertyKey): boolean {
  const compact = String(key).toLowerCase().replace(/[^a-z0-9]/g, "");
  if (compact === "token") return true;
  return [...SENSITIVE_MARKERS].some((marker) => compact.includes(marker));
}

export function toSerializable(
  value: unknown,
  depth = 0,
  seen: WeakSet<object> = new WeakSet(),
): unknown {
  if (
    value === null ||
    value === undefined ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value ?? null;
  }
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "symbol" || typeof value === "function") {
    return { type: typeof value };
  }
  if (depth > MAX_DEPTH) {
    return { type: typeName(value), truncated: true };
  }
  if (value instanceof Date) return value.toISOString();
  if (value instanceof Error) {
    return { name: value.name, message: value.message };
  }
  if (typeof value !== "object") return String(value);
  if (seen.has(value)) return { type: typeName(value), recursive: true };
  seen.add(value);
  try {
    if (Array.isArray(value)) {
      return value.map((item) => toSerializable(item, depth + 1, seen));
    }
    if (value instanceof Map) {
      return Object.fromEntries(
        [...value.entries()].map(([key, item]) => [
          String(key),
          isSensitiveKey(key)
            ? "<redacted>"
            : toSerializable(item, depth + 1, seen),
        ]),
      );
    }
    if (value instanceof Set) {
      return [...value].map((item) => toSerializable(item, depth + 1, seen));
    }
    const record = value as Record<string, unknown>;
    if (typeof record.toJSON === "function") {
      try {
        return toSerializable(record.toJSON(), depth + 1, seen);
      } catch {
        // Fall through to enumerable fields.
      }
    }
    return Object.fromEntries(
      Object.entries(record)
        .filter(([, item]) => typeof item !== "function")
        .map(([key, item]) => [
          key,
          isSensitiveKey(key)
            ? "<redacted>"
            : toSerializable(item, depth + 1, seen),
        ]),
    );
  } finally {
    seen.delete(value);
  }
}

export function safeJson(value: unknown): string {
  return JSON.stringify(toSerializable(value));
}

export function typeName(value: unknown): string {
  if (value && typeof value === "object") {
    return (value as { constructor?: { name?: string } }).constructor?.name ?? "object";
  }
  return typeof value;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

export function valueAt(value: unknown, key: string): unknown {
  return isRecord(value) ? value[key] : undefined;
}
