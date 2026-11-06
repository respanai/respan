import { Span } from "@opentelemetry/api";
import { LOG_PREFIX_WARN, TIME_TO_FIRST_TOKEN_ATTR } from "../constants/index.js";

/** respan_params alias for time to first token, as documented for the API. */
const TTFT_PARAM_KEY = "ttft";

/**
 * Returns the value as finite, non-negative seconds, or undefined if it is not one.
 * No unit conversion: the value is already in seconds.
 */
const timeToFirstTokenSeconds = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : undefined;

/**
 * Sets the bare `time_to_first_token` span attribute (seconds) from a
 * respan_params value. It is not in RESPAN_SPAN_ATTRIBUTES_MAP because it must
 * not propagate to child spans. An invalid value is dropped with a warning.
 */
export const setTimeToFirstTokenAttribute = (
  span: Span,
  value: unknown
): void => {
  if (value === undefined || value === null) {
    return;
  }
  const seconds = timeToFirstTokenSeconds(value);
  if (seconds === undefined) {
    console.warn(
      `${LOG_PREFIX_WARN} Ignoring ${TIME_TO_FIRST_TOKEN_ATTR}=${String(value)}: expected a finite, non-negative number of seconds`
    );
    return;
  }
  span.setAttribute(TIME_TO_FIRST_TOKEN_ATTR, seconds);
};

/**
 * Reads time to first token from snake_case respan_params. `ttft` wins when
 * both keys are given, matching the backend's alias handling.
 */
export const timeToFirstTokenParam = (params: Record<string, unknown>): unknown =>
  TTFT_PARAM_KEY in params
    ? params[TTFT_PARAM_KEY]
    : params[TIME_TO_FIRST_TOKEN_ATTR];
