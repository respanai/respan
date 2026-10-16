/** Turn Puppeteer call arguments and results into the JSON Respan renders. */

import { CREDENTIAL_KEY, EVALUATE_PARAM, MASKED_TEXT, OUTPUT_TEXT_LIMIT, TRUNCATED_SUFFIX } from "./_constants.js";
import { imagePart, isImage } from "./_shared.js";

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
type Row = Record<string, any>;
type Describe = (value: any) => Row;
const MAX_DEPTH = 8;
/** Parameter names that collect the rest of the arguments (`select(selector, ...values)`). */
const REST_PARAMS = new Set(["args", "values", "cookies", "urls", "paths"]);

/** Puppeteer classes, most specific first, with what each one answers locally (no browser round trip). */
const known: Array<[Function, Describe]> = [];

function read<T>(fn: () => T): T | undefined {
  try {
    return fn();
  } catch {
    return undefined; // a closed page or a disposed object: leave the field out
  }
}

/** Register the Puppeteer module's public classes, for describing call results. */
export function registerClasses(module: Row): void {
  known.length = 0;
  const add = (name: string, describe: Describe) => {
    if (typeof module[name] === "function") known.push([module[name], (v) => ({ type: name, ...describe(v) })]);
  };
  add("Page", (v) => ({ url: read(() => v.url()) }));
  add("Frame", (v) => ({ url: read(() => v.url()), name: read(() => v.name()) || undefined }));
  add("HTTPResponse", (v) => ({ url: read(() => v.url()), status: read(() => v.status()) }));
  add("HTTPRequest", (v) => ({ url: read(() => v.url()), method: read(() => v.method()), resourceType: read(() => v.resourceType()) }));
  add("ElementHandle", () => ({}));
  add("JSHandle", () => ({}));
  add("Locator", () => ({}));
  add("BrowserContext", () => ({}));
  add("Browser", (v) => ({ connected: read(() => v.connected) }));
  add("Dialog", (v) => ({ dialogType: read(() => v.type()), message: read(() => v.message()) }));
  add("ScreenRecorder", () => ({}));
  add("ScreenRecording", () => ({}));
}

function text(value: string): string {
  return value.length <= OUTPUT_TEXT_LIMIT ? value : value.slice(0, OUTPUT_TEXT_LIMIT) + TRUNCATED_SUFFIX.replace("{size}", String(value.length));
}

function describe(value: object): Row | undefined {
  for (const [ctor, describer] of known) {
    if (value instanceof ctor) {
      const described = describer(value);
      for (const key of Object.keys(described)) if (described[key] === undefined) delete described[key];
      return described;
    }
  }
  return undefined;
}

/** A JSON-safe copy: binary becomes its length, Puppeteer objects a description. */
export function jsonable(value: unknown, depth = 0, credentials: RegExp = CREDENTIAL_KEY): Json {
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
  const described = describe(value);
  if (described) return described as Json;
  if (depth >= MAX_DEPTH) return { type: "object", truncated: true };
  if (Array.isArray(value)) return value.map((item) => jsonable(item, depth + 1, credentials));
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) return { type: prototype?.constructor?.name || "object" };
  const out: Record<string, Json> = {};
  for (const [key, item] of Object.entries(value)) {
    if (item === undefined || typeof item === "function") continue;
    out[key] = credentials.test(key) && typeof item === "string" ? MASKED_TEXT : jsonable(item, depth + 1, credentials);
  }
  return out;
}

/** The call's positional arguments under their parameter names; page functions as their source. */
export function callInput(params: readonly string[], args: readonly unknown[], extra?: Row, credentials: RegExp = CREDENTIAL_KEY): Row {
  const input: Row = { ...extra };
  params.forEach((name, index) => {
    if (REST_PARAMS.has(name) && index === params.length - 1) {
      if (args.length > index) input[name] = jsonable(args.slice(index), 0, credentials);
      return;
    }
    const value = args[index];
    if (value === undefined) return;
    input[name] = name === EVALUATE_PARAM && typeof value === "function" ? text(value.toString()) : jsonable(value, 0, credentials);
  });
  return input;
}

/** Bytes of an image result (`screenshot()` returns bytes, or base64 with `encoding: "base64"`). */
function imageBytes(result: unknown): Uint8Array | undefined {
  if (result instanceof Uint8Array) return isImage(result) ? result : undefined;
  if (typeof result === "string" && result.length > 16 && /^[A-Za-z0-9+/=]+$/.test(result.slice(0, 64))) {
    const bytes = Buffer.from(result, "base64");
    return isImage(bytes) ? bytes : undefined;
  }
  return undefined;
}

/** The call's result, made readable; screenshot bytes become an inline JPEG image part. */
export function callOutput(result: unknown, keepImages: boolean, isScreenshot: boolean, credentials: RegExp = CREDENTIAL_KEY): unknown {
  if (result === null || result === undefined) return undefined;
  const image = isScreenshot ? imageBytes(result) : undefined;
  if (image) return keepImages ? { screenshot: imagePart(image) } : { screenshot: { type: "bytes", length: image.byteLength } };
  return jsonable(result, 0, credentials);
}
