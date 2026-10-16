/** Names owned by the Stagehand instrumentation. */

export const INSTRUMENTATION_NAME = "stagehand";
export const TRACER_NAME = "@respan/instrumentation-stagehand";
export const STAGEHAND_MODULE = "@browserbasehq/stagehand";

export const SPAN_SESSION = "stagehand.session";
export const SPAN_LLM = "stagehand.llm";
/** act / observe / extract spans are `stagehand.<operation>`. */
export const OPERATION_PREFIX = "stagehand.";
export const OPERATIONS = ["act", "observe", "extract"] as const;
export type Operation = (typeof OPERATIONS)[number];

/** Browser actions traced as `tool` spans: `page.goto`, `locator.click`, ... */
export const PAGE_ACTIONS = [
  "goto",
  "reload",
  "goBack",
  "goForward",
  "click",
  "hover",
  "scroll",
  "dragAndDrop",
  "type",
  "keyPress",
  "evaluate",
  "screenshot",
  "pdf",
  "waitForSelector",
  "setViewportSize",
  "close",
] as const;
export const LOCATOR_ACTIONS = [
  "click",
  "hover",
  "fill",
  "type",
  "selectOption",
  "setInputFiles",
  "scrollTo",
  "sendClickEvent",
] as const;
export const CONTEXT_ACTIONS = ["newPage", "setActivePage"] as const;

/** Display-name detail: Respan shows the session as "agent.stagehand". */
export const SESSION_NAME_DETAIL = "stagehand";

// Span attributes specific to Stagehand. Everything else uses the
// OpenLLMetry / gen_ai names the Respan backend already parses.
export const ATTR_SESSION_ID = "stagehand.session.id";
export const ATTR_PAGE_URL = "stagehand.page.url";
export const ATTR_ACTION_ID = "stagehand.action.id";
export const ATTR_CACHE_STATUS = "stagehand.cache.status";
export const ATTR_USAGE_SOURCE = "stagehand.usage.source";
/** The chat span's usage came from the operation result, not a client `generate` call. */
export const USAGE_SOURCE_OPERATION = "operation";

export const GEN_AI_USAGE_REASONING_TOKENS = "gen_ai.usage.reasoning_tokens";
export const LLM_USAGE_CACHE_READ_INPUT_TOKENS = "llm.usage.cache_read_input_tokens";
export const GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read_input_tokens";

/** Screenshot attached to each `act` span: JPEG, CSS pixels, short timeout. */
export const SCREENSHOT_OPTIONS = { type: "jpeg", quality: 70, scale: "css", timeout: 3000 } as const;
export const SCREENSHOT_DEADLINE_MS = 4000;
/** Page screenshots larger than this are summarised instead of exported. */
export const MAX_EXPORTED_IMAGE_BASE64 = 2_000_000;

export const IMAGE_PLACEHOLDER = "[image omitted: {mime}, {size} base64 chars]";
export const REDACTED = "[redacted]";
export const ERROR_MESSAGE_LIMIT = 1000;
/** Text typed into a field whose selector looks like a credential is never exported. */
export const SENSITIVE_SELECTOR = /pass(word|code|phrase)?|secret|token|otp|cvv|cvc|card.?num|ssn|api.?key/i;
/** A field described by the page (type, name, id, autocomplete, aria-label) that holds a credential. */
export const SENSITIVE_FIELD = /type=password|pass(word|code|phrase)?|secret|token|otp|one-time-code|cvv|cvc|cc-number|cc-csc|card.?num|ssn|api.?key|\bpin\b/i;
