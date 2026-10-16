/**
 * Span helpers every Respan browser instrumentation needs: flat customer
 * metadata, secret masking, `Type: message` errors, compact screenshots.
 *
 * @respan/instrumentation-stagehand carries its own copy of the first three;
 * they belong in @respan/tracing once its published build and its source agree.
 */
import { createRequire } from "node:module";
import { SpanStatusCode, type Attributes, type Span } from "@opentelemetry/api";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  ERROR_MESSAGE_LIMIT,
  IMAGE_PLACEHOLDER,
  MAX_EXPORTED_IMAGE_BASE64,
  PLAYWRIGHT_UTILS_BUNDLE,
  SCREENSHOT_JPEG_QUALITY,
  SCREENSHOT_MAX_WIDTH,
  SECRET_PLACEHOLDER,
} from "./_constants.js";

type Row = Record<string, any>;
/** [secret, placeholder] pairs, longest secret first. */
export type Replacements = ReadonlyArray<readonly [string, string]>;

export function safely(fn: () => void): void {
  try {
    fn();
  } catch {
    // Telemetry must never break the customer's browser flow.
  }
}

/**
 * Customer metadata as one flat `respan.metadata.<key>` attribute per key, the
 * shape Respan ingestion reads. Some @respan/tracing versions fold propagated
 * metadata into a single JSON `respan.metadata` attribute when the span starts,
 * which ingestion drops; expand it before the span is exported.
 */
export function flattenMetadata(span: unknown): void {
  const attributes = (span as any)?.attributes as Record<string, unknown> | undefined;
  const canonical = attributes?.[RespanSpanAttributes.RESPAN_METADATA];
  if (typeof canonical !== "string") return;
  let parsed: unknown;
  try {
    parsed = JSON.parse(canonical);
  } catch {
    return;
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return;
  for (const [key, value] of Object.entries(parsed as Row)) {
    const flat = `${RespanSpanAttributes.RESPAN_METADATA}.${key}`;
    if (attributes![flat] !== undefined || value === undefined || value === null) continue;
    // Written directly: an ended span ignores setAttribute.
    attributes![flat] = typeof value === "string" || typeof value === "number" || typeof value === "boolean" ? value : JSON.stringify(value);
  }
  delete attributes![RespanSpanAttributes.RESPAN_METADATA];
}

/**
 * `{name: value}` as replacements: each value (and its URL-encoded forms, since
 * typed text shows up in URLs) becomes `<secret>name</secret>`.
 */
export function secretReplacements(secrets: Record<string, unknown>): Replacements {
  const pairs = new Map<string, string>();
  for (const [name, value] of Object.entries(secrets ?? {})) {
    if (typeof value !== "string" || !value) continue;
    const placeholder = SECRET_PLACEHOLDER.replace("{name}", name);
    const encoded = encodeURIComponent(value);
    for (const variant of [value, encoded, encoded.replace(/%20/g, "+")]) pairs.set(variant, placeholder);
  }
  return [...pairs].sort((a, b) => b[0].length - a[0].length);
}

export function mask(text: string, replacements: Replacements): string {
  for (const [secret, placeholder] of replacements) text = text.split(secret).join(placeholder);
  return text;
}

/** Mask every string attribute, the status message and event attributes of a span. */
export function maskSpan(span: Span, replacements: Replacements): void {
  if (!replacements.length) return;
  const readable = span as any;
  for (const [key, value] of Object.entries((readable.attributes ?? {}) as Attributes)) {
    if (typeof value === "string") {
      const masked = mask(value, replacements);
      if (masked !== value) span.setAttribute(key, masked);
    }
  }
  const status = readable.status;
  if (typeof status?.message === "string") span.setStatus({ code: status.code, message: mask(status.message, replacements) });
  for (const event of (readable.events ?? []) as Array<{ attributes?: Attributes }>) {
    for (const [key, value] of Object.entries(event.attributes ?? {})) {
      if (typeof value === "string") (event.attributes as Row)[key] = mask(value, replacements);
    }
  }
}

/** ERROR status whose description is what Respan shows as the error message. */
export function setError(span: Span, summary: string, detail: unknown, withMessage: boolean): void {
  const message = withMessage ? String(detail ?? "").trim() : "";
  span.setStatus({ code: SpanStatusCode.ERROR, message: message ? `${summary}: ${message.slice(0, ERROR_MESSAGE_LIMIT)}` : summary });
}

/** `Type: message` status plus an exception event; without content, the type only. */
export function recordError(span: Span, error: unknown, withMessage: boolean): void {
  const type = String((error as any)?.name || typeof error);
  const message = String((error as any)?.message ?? error);
  setError(span, type, message, withMessage);
  span.addEvent(
    "exception",
    withMessage
      ? { "exception.type": type, "exception.message": message, "exception.stacktrace": String((error as any)?.stack ?? "") }
      : { "exception.type": type },
  );
}

// -- screenshots --------------------------------------------------------------

interface Codecs {
  PNG: { sync: { read(data: Buffer): { width: number; height: number; data: Buffer } } };
  jpegjs: {
    decode(data: Buffer, options?: Row): { width: number; height: number; data: Buffer | Uint8Array };
    encode(image: { width: number; height: number; data: Buffer | Uint8Array }, quality: number): { data: Buffer };
  };
}

let codecs: Codecs | null | undefined;

function loadCodecs(): Codecs | null {
  if (codecs !== undefined) return codecs;
  try {
    // playwright-core already loaded this bundle; it ships PNG and JPEG codecs.
    const bundle = createRequire(import.meta.url)(PLAYWRIGHT_UTILS_BUNDLE);
    codecs = bundle?.PNG?.sync?.read && bundle?.jpegjs?.encode ? (bundle as Codecs) : null;
  } catch {
    codecs = null;
  }
  return codecs;
}

export function isImage(data: Uint8Array): boolean {
  return isPng(data) || isJpeg(data);
}

function isPng(data: Uint8Array): boolean {
  return data.length > 8 && data[0] === 0x89 && data[1] === 0x50 && data[2] === 0x4e && data[3] === 0x47;
}

function isJpeg(data: Uint8Array): boolean {
  return data.length > 3 && data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff;
}

/** RGBA pixels scaled down to `width` by averaging each source block. */
function downscale(image: { width: number; height: number; data: Uint8Array }, width: number) {
  const height = Math.max(1, Math.round((image.height * width) / image.width));
  const out = Buffer.alloc(width * height * 4);
  const xScale = image.width / width;
  const yScale = image.height / height;
  for (let y = 0; y < height; y++) {
    const y0 = Math.floor(y * yScale);
    const y1 = Math.max(y0 + 1, Math.floor((y + 1) * yScale));
    for (let x = 0; x < width; x++) {
      const x0 = Math.floor(x * xScale);
      const x1 = Math.max(x0 + 1, Math.floor((x + 1) * xScale));
      let r = 0, g = 0, b = 0, n = 0;
      for (let sy = y0; sy < y1; sy++) {
        for (let sx = x0; sx < x1; sx++) {
          const i = (sy * image.width + sx) * 4;
          r += image.data[i];
          g += image.data[i + 1];
          b += image.data[i + 2];
          n++;
        }
      }
      const o = (y * width + x) * 4;
      out[o] = r / n;
      out[o + 1] = g / n;
      out[o + 2] = b / n;
      out[o + 3] = 255;
    }
  }
  return { width, height, data: out };
}

/**
 * The image as a compact JPEG (at most 1280 px wide): full-page PNG screenshots
 * are megabytes of base64 and large export batches time out. Returns the input
 * when the codecs are unavailable, the data is unreadable, or JPEG is not smaller.
 */
export function compactImage(data: Uint8Array): { data: Uint8Array; mime: string } {
  const original = { data, mime: isPng(data) ? "image/png" : "image/jpeg" };
  const codec = loadCodecs();
  if (!codec) return original;
  try {
    const buffer = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
    let image = isPng(data) ? codec.PNG.sync.read(buffer) : codec.jpegjs.decode(buffer, { useTArray: true, formatAsRGBA: true });
    if (isJpeg(data) && image.width <= SCREENSHOT_MAX_WIDTH) return original;
    if (image.width > SCREENSHOT_MAX_WIDTH) image = downscale(image, SCREENSHOT_MAX_WIDTH);
    const jpeg = codec.jpegjs.encode(image, SCREENSHOT_JPEG_QUALITY).data;
    return jpeg.length < data.length ? { data: jpeg, mime: "image/jpeg" } : original;
  } catch {
    return original;
  }
}

/** A data-URI image part (Respan's ingest lifts it into media storage), or a summary when too large. */
export function imagePart(data: Uint8Array): Row {
  const { data: compact, mime } = compactImage(data);
  const base64 = Buffer.from(compact).toString("base64");
  if (base64.length <= MAX_EXPORTED_IMAGE_BASE64) return { type: "image_url", image_url: { url: `data:${mime};base64,${base64}` } };
  return { type: "text", text: IMAGE_PLACEHOLDER.replace("{mime}", mime).replace("{size}", String(base64.length)) };
}
