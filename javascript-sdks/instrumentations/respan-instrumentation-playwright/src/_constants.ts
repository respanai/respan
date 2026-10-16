/** Names owned by the Playwright instrumentation. */

export const INSTRUMENTATION_NAME = "playwright";
export const TRACER_NAME = "@respan/instrumentation-playwright";
export const PLAYWRIGHT_MODULE = "playwright-core";
/** playwright-core's bundled image codecs (pngjs, jpeg-js), used to compact screenshots. */
export const PLAYWRIGHT_UTILS_BUNDLE = "playwright-core/lib/utilsBundle";
export const TRACING_MODULE = "@respan/tracing";
export const METADATA_TRANSFORMER_KEY = "@respan/instrumentation-playwright.flat-metadata";

// Span attributes specific to browser automation (the same names as the Python
// package). Everything else uses the OpenLLMetry names Respan already parses.
export const ATTR_API_NAME = "playwright.api.name";
export const ATTR_PAGE_URL = "playwright.page.url";
export const ATTR_PAGE_TITLE = "playwright.page.title";

/** screenshotOn values: a page screenshot after the call, for a flip-book of the run. */
export const SCREENSHOT_MODES = ["never", "navigation", "action"] as const;
export type ScreenshotOn = (typeof SCREENSHOT_MODES)[number];

// Protocol methods (what Playwright sends to the browser), not public API names:
// `locator.fill` and `page.fill` both send `fill`, `keyboard.type` sends `keyboardType`.
/** Calls that load a document: the span also gets the page title. */
export const NAVIGATION_METHODS = new Set(["goto", "reload", "goBack", "goForward", "setContent"]);
/** Calls that change the page (plus navigations), for screenshotOn: "action". */
export const ACTION_METHODS = new Set([
  ...NAVIGATION_METHODS,
  "blur",
  "check",
  "click",
  "dblclick",
  "dispatchEvent",
  "dragAndDrop",
  "fill",
  "focus",
  "hover",
  "keyboardInsertText",
  "keyboardPress",
  "keyboardType",
  "mouseClick",
  "mouseMove",
  "mouseWheel",
  "press",
  "scrollIntoViewIfNeeded",
  "selectOption",
  "selectText",
  "setInputFiles",
  "tap",
  "type",
  "uncheck",
]);
/** Calls whose parameter is text typed into the page: masked when it lands in a password field. */
export const TEXT_ENTRY_PARAMS: Record<string, string> = {
  fill: "value",
  type: "text",
  keyboardType: "text",
  keyboardInsertText: "text",
};
/** Calls whose result is a serialized JS value (`page.evaluate`, `locator.evaluateAll`, ...). */
export const EVALUATE_METHODS = new Set(["evaluateExpression", "evalOnSelector", "evalOnSelectorAll"]);
export const SCREENSHOT_METHOD = "screenshot";
/** `expect(...)` matchers (@playwright/test) send this; it reports a mismatch instead of throwing. */
export const EXPECT_METHOD = "expect";
export const EXPECT_FAILURE = "AssertionError";
/** Registrations whose callbacks Playwright later fires from a browser event. */
export const CALLBACK_REGISTRATIONS = new Set([
  "setNetworkInterceptionPatterns",
  "setWebSocketInterceptionPatterns",
  "exposeBinding",
  "registerLocatorHandler",
]);
/** Events that fire those callbacks (route handlers, exposed functions, locator handlers). */
export const CALLBACK_EVENTS = new Set(["route", "webSocketRoute", "bindingCall", "locatorHandlerTriggered"]);
/** Event-listener registration methods on Playwright objects (`page.on`, `once`, `off`, ...). */
export const LISTENER_ADD_METHODS = ["on", "addListener", "prependListener", "once", "prependOnceListener"] as const;
export const LISTENER_REMOVE_METHODS = ["off", "removeListener"] as const;

export const MASKED_TEXT = "<masked>";
export const SECRET_PLACEHOLDER = "<secret>{name}</secret>";
/** Text typed into a password field is masked everywhere afterwards in the trace as this name. */
export const LEARNED_SECRET_NAME = "password";
export const MAX_LEARNED_SECRETS = 32;
/** Shorter typed text is masked in its own call only: replacing it everywhere would garble the trace. */
export const MIN_LEARNED_SECRET_LENGTH = 4;
/** Traces whose learned passwords are remembered (oldest dropped first). */
export const MAX_LEARNED_TRACES = 256;
/** Option keys whose values are credentials (`httpCredentials.password`, `proxy.password`). */
export const CREDENTIAL_KEY = /^(password|passphrase|secret|token|apiKey|api_key)$/i;

export const OUTPUT_TEXT_LIMIT = 4000;
export const TRUNCATED_SUFFIX = "... [truncated, {size} chars]";
export const ERROR_MESSAGE_LIMIT = 1000;
export const PROBE_TIMEOUT_MS = 5000;

export const SCREENSHOT_MAX_WIDTH = 1280;
export const SCREENSHOT_JPEG_QUALITY = 70;
/** Images larger than this (base64 chars) after compaction are summarised instead of exported. */
export const MAX_EXPORTED_IMAGE_BASE64 = 2_000_000;
export const IMAGE_PLACEHOLDER = "[image omitted: {mime}, {size} base64 chars]";
