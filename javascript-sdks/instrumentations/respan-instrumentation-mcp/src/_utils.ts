import { types } from "node:util";
import { own } from "./_privacy.js";
import type { McpClientMethod, OperationConfig } from "./_constants.js";

/** Snapshot data properties without invoking application getters or toJSON. */
export function toSerializableValue(
  value: unknown,
  _depth = 0,
  seen = new WeakSet<object>(),
): unknown {
  if (
    value === null ||
    value === undefined ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  )
    return value;
  if (typeof value === "bigint") return value.toString();
  if (typeof value !== "object" || types.isProxy(value)) return undefined;
  if (seen.has(value)) return "[Circular]";
  seen.add(value);
  try {
    if (value instanceof Date) return Date.prototype.toISOString.call(value);
    if (Array.isArray(value)) {
      const result: unknown[] = [];
      for (let i = 0; i < value.length; i++)
        result.push(toSerializableValue(own(value, String(i)), 0, seen));
      return result;
    }
    const result: Record<string, unknown> = {};
    for (const key of Object.keys(value)) {
      const item = toSerializableValue(own(value, key), 0, seen);
      if (item !== undefined)
        Object.defineProperty(result, key, {
          value: item,
          enumerable: true,
          configurable: true,
          writable: true,
        });
    }
    return result;
  } finally {
    seen.delete(value);
  }
}
export function safeJson(value: unknown): string {
  try {
    return JSON.stringify(toSerializableValue(value)) ?? "null";
  } catch {
    return '"[Unserializable]"';
  }
}
function firstObjectArg(args: unknown[]): Record<string, unknown> {
  const arg = args[0];
  return arg && typeof arg === "object" && !Array.isArray(arg)
    ? (arg as Record<string, unknown>)
    : {};
}
export function resolveClientEntityName(
  methodName: McpClientMethod,
  args: unknown[],
  config: OperationConfig,
): string {
  const params = firstObjectArg(args);
  const name = own(params, methodName === "readResource" ? "uri" : "name");
  return ["callTool", "getPrompt", "readResource"].includes(methodName) &&
    typeof name === "string" &&
    name
    ? name
    : config.entityName;
}
export function resolveClientSpanName(
  methodName: McpClientMethod,
  entityName: string,
  config: OperationConfig,
): string {
  return methodName === "callTool"
    ? `mcp.tool.${entityName}`
    : methodName === "getPrompt"
      ? `mcp.prompt.${entityName}`
      : config.spanName;
}
export function clientMethodInput(
  methodName: McpClientMethod,
  args: unknown[],
): Record<string, unknown> {
  if (methodName === "connect")
    return { method: methodName, transport: "Transport" };
  const params = firstObjectArg(args);
  if (methodName === "callTool" || methodName === "getPrompt")
    return { name: own(params, "name"), arguments: own(params, "arguments") };
  if (methodName === "readResource") return { uri: own(params, "uri") };
  return { method: methodName, params };
}
export function errorOutput(error: unknown): Record<string, string> {
  const message = own(error, "message");
  const name = own(error, "name");
  return {
    error: typeof name === "string" ? name : "Error",
    message:
      typeof message === "string"
        ? message
        : typeof error === "string"
          ? error
          : "MCP operation failed",
  };
}
export function findLastFunctionIndex(args: unknown[]): number {
  for (let i = args.length - 1; i >= 0; i--)
    if (typeof args[i] === "function") return i;
  return -1;
}
