/**
 * Constants used throughout the Respan tracing SDK
 */
import { RespanSpanAttributes } from "@respan/respan-sdk";

/**
 * The name of this package, used as the instrumentation library name
 * This should match the package.json name field
 */
export const RESPAN_PACKAGE_NAME = "@respan/tracing" as const;

/**
 * Log prefixes for consistent console output across the SDK.
 */
export const LOG_PREFIX = "[Respan]" as const;
export const LOG_PREFIX_DEBUG = "[Respan Debug]" as const;
export const LOG_PREFIX_ERROR = "[Respan Error]" as const;
export const LOG_PREFIX_WARN = "[Respan Warning]" as const;

/**
 * Bare backend column key for time to first token, in seconds. Mirrors
 * RespanSpanAttributes.TIME_TO_FIRST_TOKEN in @respan/respan-sdk; restated here
 * so this package works on its released @respan/respan-sdk floor.
 */
export const TIME_TO_FIRST_TOKEN_ATTR = "time_to_first_token" as const;

/**
 * Build a fully-qualified metadata attribute key.
 * Single source of truth for the `respan.metadata.<key>` pattern
 * used in span attributes across the SDK.
 */
export function metadataAttributeKey(key: string): string {
  return `${RespanSpanAttributes.RESPAN_METADATA}.${key}`;
}
