import { types } from "node:util";
import { data, snapshot } from "./_privacy.js";

export function serializeValue(value: unknown): unknown {
  if (value && typeof value === "object" && !types.isProxy(value)) {
    if (value instanceof URL)
      return Object.getOwnPropertyDescriptor(URL.prototype, "href")!.get!.call(
        value,
      );
    if (typeof Blob !== "undefined" && value instanceof Blob)
      return {
        type: Object.getOwnPropertyDescriptor(
          Blob.prototype,
          "type",
        )!.get!.call(value),
        size: Object.getOwnPropertyDescriptor(
          Blob.prototype,
          "size",
        )!.get!.call(value),
      };
  }
  return snapshot(value);
}
export function safeJsonStringify(value: unknown): string {
  try {
    return JSON.stringify(serializeValue(value)) ?? "null";
  } catch {
    return "null";
  }
}
export function normalizeCallInput(
  methodName: string,
  args: unknown[],
): Record<string, unknown> {
  const fields =
    methodName === "guard"
      ? ["input", "model", "systemPrompt", "fallbackModel", "chunkSize"]
      : methodName === "redact"
        ? ["input", "model", "entities", "rewrite", "fallbackModel"]
        : methodName === "scan"
          ? ["repo", "branch", "model", "fallbackModel"]
          : [];
  const options: Record<string, unknown> = {};
  for (const field of fields) {
    const value = data(args[0], field);
    if (value !== undefined) options[field] = serializeValue(value);
  }
  return methodName === "guard"
    ? { method: methodName, arguments: options }
    : { name: `superagent.${methodName}`, arguments: options };
}
export function extractModel(args: unknown[]): string | undefined {
  const value = data(args[0], "model");
  return typeof value === "string" && value.length ? value : undefined;
}
export function extractPrimaryInput(
  methodName: string,
  args: unknown[],
): unknown {
  return data(args[0], methodName === "scan" ? "repo" : "input");
}
export const getAttr = data;
