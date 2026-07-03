import { INPUT_KEY, MODEL_KEY, REPO_KEY } from "./_constants.js";

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function getObjectEntries(value: object): Array<[string, unknown]> {
  const entries = Object.entries(value);
  if (entries.length > 0) {
    return entries;
  }

  return Object.getOwnPropertyNames(value)
    .filter((key) => key !== "stack")
    .map((key) => [key, (value as Record<string, unknown>)[key]]);
}

export function serializeValue(
  value: unknown,
  seen: WeakSet<object> = new WeakSet<object>(),
): unknown {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }

  if (typeof value === "bigint") {
    return value.toString();
  }

  if (typeof value === "undefined") {
    return null;
  }

  if (typeof value === "function") {
    return `[Function ${(value as Function).name || "anonymous"}]`;
  }

  if (value instanceof Error) {
    return {
      name: value.name,
      message: value.message,
    };
  }

  if (value instanceof URL) {
    return value.href;
  }

  if (typeof Blob !== "undefined" && value instanceof Blob) {
    return {
      type: value.type,
      size: value.size,
    };
  }

  if (Array.isArray(value)) {
    return value.map((item) => serializeValue(item, seen));
  }

  if (typeof value === "object") {
    if (seen.has(value)) {
      return "[Circular]";
    }
    seen.add(value);

    const toJSON = (value as { toJSON?: () => unknown }).toJSON;
    if (typeof toJSON === "function") {
      try {
        return serializeValue(toJSON.call(value), seen);
      } catch {
        return String(value);
      }
    }

    const serialized: JsonRecord = {};
    for (const [key, item] of getObjectEntries(value)) {
      serialized[key] = serializeValue(item, seen);
    }
    return serialized;
  }

  return String(value);
}

export function safeJsonStringify(value: unknown): string {
  try {
    return JSON.stringify(serializeValue(value));
  } catch {
    return String(value);
  }
}

function getField(value: unknown, fieldName: string): unknown {
  if (!isRecord(value)) {
    return undefined;
  }
  return value[fieldName];
}

export function normalizeCallInput(
  methodName: string,
  args: unknown[],
): Record<string, unknown> {
  const payload: Record<string, unknown> = { method: methodName };

  if (args.length > 0) {
    payload.args = serializeValue(args);
  }

  return payload;
}

export function extractModel(args: unknown[]): string | undefined {
  const firstArg = args[0];
  const model = getField(firstArg, MODEL_KEY);
  return typeof model === "string" && model.length > 0 ? model : undefined;
}

export function extractPrimaryInput(
  methodName: string,
  args: unknown[],
): unknown {
  const firstArg = args[0];
  const input = getField(firstArg, INPUT_KEY);
  if (input !== undefined) {
    return input;
  }

  const repo = getField(firstArg, REPO_KEY);
  if (repo !== undefined) {
    return repo;
  }

  if (typeof firstArg === "string" || firstArg instanceof URL) {
    return firstArg;
  }

  if (methodName === "scan") {
    return repo;
  }

  return undefined;
}

export function getAttr(value: unknown, fieldName: string): unknown {
  return getField(value, fieldName);
}
