/** Turn Playwright call parameters and results into the JSON Respan renders. */

import { CREDENTIAL_KEY, MASKED_TEXT, OUTPUT_TEXT_LIMIT, TRUNCATED_SUFFIX } from "./_constants.js";
import { imagePart, isImage } from "./_shared.js";

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
type Row = Record<string, any>;
const MAX_DEPTH = 8;
/** Read-only accessors Playwright objects answer locally (no browser round trip). */
const DESCRIBED = ["url", "status", "method", "suggestedFilename"] as const;

function text(value: string): string {
  return value.length <= OUTPUT_TEXT_LIMIT ? value : value.slice(0, OUTPUT_TEXT_LIMIT) + TRUNCATED_SUFFIX.replace("{size}", String(value.length));
}

/** A Playwright object (Page, Response, ElementHandle, ...), from what it keeps locally. */
function describe(value: Row): Json {
  const described: Row = { type: value._type };
  for (const name of DESCRIBED) {
    const accessor = value[name];
    if (typeof accessor !== "function" || accessor.length || accessor.constructor?.name === "AsyncFunction") continue;
    try {
      const result = accessor.call(value);
      if (typeof result === "string" || typeof result === "number") described[name] = result;
    } catch {
      // A closed page or a disposed object: leave the field out.
    }
  }
  return described;
}

function isPlaywrightObject(value: Row): boolean {
  return typeof value._guid === "string" && typeof value._type === "string";
}

/** A JSON-safe copy: binary becomes its length, Playwright objects a description. */
export function jsonable(value: unknown, depth = 0): Json {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") return text(value);
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : String(value);
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "function") return { type: "function", name: value.name || "anonymous" };
  if (value instanceof Uint8Array) return { type: "bytes", length: value.byteLength };
  if (typeof value !== "object") return { type: typeof value };
  if (value instanceof RegExp) return String(value);
  if (value instanceof Date) return value.toISOString();
  if (value instanceof Error) return { type: value.name, message: text(value.message) };
  if (isPlaywrightObject(value as Row)) return describe(value as Row);
  // Results carry channels; each points at its Playwright object.
  const owner = (value as Row)._object;
  if (owner && typeof owner === "object" && isPlaywrightObject(owner)) return describe(owner);
  if (depth >= MAX_DEPTH) return { type: "object", truncated: true };
  if (Array.isArray(value)) return value.map((item) => jsonable(item, depth + 1));
  const out: Record<string, Json> = {};
  for (const [key, item] of Object.entries(value)) {
    if (item === undefined) continue;
    out[key] = CREDENTIAL_KEY.test(key) && typeof item === "string" ? MASKED_TEXT : jsonable(item, depth + 1);
  }
  return out;
}

/**
 * A value in Playwright's wire format for JS values (what `evaluate` returns):
 * `{n: 1}`, `{s: "x"}`, `{a: [...]}`, `{o: [{k, v}]}`, ...
 */
export function deserialize(value: any, depth = 0): unknown {
  if (!value || typeof value !== "object" || depth > MAX_DEPTH * 4) return value;
  if ("n" in value) return value.n;
  if ("s" in value) return value.s;
  if ("b" in value) return value.b;
  if ("v" in value) return value.v === "undefined" || value.v === "null" ? null : value.v;
  if ("d" in value) return value.d;
  if ("u" in value) return value.u;
  if ("bi" in value) return value.bi;
  if ("r" in value) return `/${value.r.p}/${value.r.f}`;
  if ("e" in value) return { type: value.e.n, message: value.e.m };
  if ("a" in value) return value.a.map((item: unknown) => deserialize(item, depth + 1));
  if ("o" in value) return Object.fromEntries(value.o.map(({ k, v }: Row) => [k, deserialize(v, depth + 1)]));
  if ("h" in value) return { type: "JSHandle" };
  if ("ref" in value) return { type: "ref" };
  return value;
}

/** The call's parameters as sent to the browser, made readable. */
export function callInput(params: Row | undefined): Row {
  const input: Row = {};
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value === undefined) continue;
    // `evaluate` arguments travel serialized; show the value the caller passed.
    input[key] = key === "arg" && value && typeof value === "object" && "value" in value ? jsonable(deserialize(value.value)) : jsonable(value);
  }
  return input;
}

/** The call's result, made readable; screenshot bytes become an inline JPEG image part. */
export function callOutput(result: unknown, keepImages: boolean, evaluated: boolean): unknown {
  if (result === null || result === undefined) return undefined;
  if (typeof result !== "object") return jsonable(result);
  const row = result as Row;
  const binary = row.binary;
  if (binary instanceof Uint8Array && isImage(binary)) {
    return keepImages ? { screenshot: imagePart(binary) } : { screenshot: { type: "bytes", length: binary.byteLength } };
  }
  const keys = Object.keys(row);
  // Most results are `{value}` / `{response}` / `{element}`: unwrap the single field.
  const single = keys.length === 1 && !isPlaywrightObject(row) && !row._object ? row[keys[0]] : result;
  if (single === undefined || single === null) return undefined;
  return jsonable(evaluated && keys[0] === "value" ? deserialize(single) : single);
}
