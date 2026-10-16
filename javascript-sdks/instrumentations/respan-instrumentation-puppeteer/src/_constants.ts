/** Names owned by the Puppeteer instrumentation. */

export const INSTRUMENTATION_NAME = "puppeteer";
export const TRACER_NAME = "@respan/instrumentation-puppeteer";
export const PUPPETEER_MODULE = "puppeteer-core";
/** The WebDriver BiDi implementation, which Puppeteer loads lazily for `protocol: "webDriverBiDi"`. */
export const PUPPETEER_BIDI_MODULE = "puppeteer-core/internal/bidi/bidi.js";
export const PNG_MODULE = "pngjs";
export const JPEG_MODULE = "jpeg-js";
export const TRACING_MODULE = "@respan/tracing";
export const METADATA_TRANSFORMER_KEY = "@respan/instrumentation-puppeteer.flat-metadata";

// Span attributes specific to browser automation (the same shape as the
// Playwright packages). Everything else uses the OpenLLMetry names Respan parses.
export const ATTR_API_NAME = "puppeteer.api.name";
export const ATTR_PAGE_URL = "puppeteer.page.url";
export const ATTR_PAGE_TITLE = "puppeteer.page.title";
/** Where `page.screencast()` / `page.record()` writes its video. */
export const ATTR_RECORDING_PATH = "puppeteer.recording.path";

/** screenshotOn values: a page screenshot after the call, for a flip-book of the run. */
export const SCREENSHOT_MODES = ["never", "navigation", "action"] as const;
export type ScreenshotOn = (typeof SCREENSHOT_MODES)[number];

/**
 * The public calls that become spans, per Puppeteer class, with the names of
 * their positional parameters (for the span input). The api/ classes are
 * abstract; a method is wrapped on whichever of the abstract class and its CDP
 * and BiDi implementations defines it.
 */
export const TRACED_METHODS: Record<string, Record<string, readonly string[]>> = {
  page: {
    goto: ["url", "options"],
    reload: ["options"],
    goBack: ["options"],
    goForward: ["options"],
    setContent: ["html", "options"],
    waitForNavigation: ["options"],
    waitForSelector: ["selector", "options"],
    waitForFunction: ["pageFunction", "options", "args"],
    waitForNetworkIdle: ["options"],
    content: [],
    title: [],
    evaluate: ["pageFunction", "args"],
    evaluateHandle: ["pageFunction", "args"],
    $: ["selector"],
    $$: ["selector", "options"],
    $eval: ["selector", "pageFunction", "args"],
    $$eval: ["selector", "pageFunction", "args"],
    click: ["selector", "options"],
    focus: ["selector"],
    hover: ["selector"],
    select: ["selector", "values"],
    tap: ["selector"],
    type: ["selector", "text", "options"],
    screenshot: ["options"],
    pdf: ["options"],
    screencast: ["options"],
    record: ["options"],
    emulate: ["device"],
    setViewport: ["viewport"],
    setUserAgent: ["userAgent"],
    setExtraHTTPHeaders: ["headers"],
    setRequestInterception: ["value"],
    setCookie: ["cookies"],
    cookies: ["urls"],
    deleteCookie: ["cookies"],
    authenticate: ["credentials"],
    addScriptTag: ["options"],
    addStyleTag: ["options"],
    exposeFunction: ["name", "function"],
    evaluateOnNewDocument: ["pageFunction", "args"],
    bringToFront: [],
    waitForFileChooser: ["options"],
    close: ["options"],
  },
  frame: {
    goto: ["url", "options"],
    setContent: ["html", "options"],
    waitForNavigation: ["options"],
    waitForSelector: ["selector", "options"],
    waitForFunction: ["pageFunction", "options", "args"],
    content: [],
    title: [],
    evaluate: ["pageFunction", "args"],
    evaluateHandle: ["pageFunction", "args"],
    $: ["selector"],
    $$: ["selector", "options"],
    $eval: ["selector", "pageFunction", "args"],
    $$eval: ["selector", "pageFunction", "args"],
    click: ["selector", "options"],
    focus: ["selector"],
    hover: ["selector"],
    select: ["selector", "values"],
    tap: ["selector"],
    type: ["selector", "text", "options"],
    addScriptTag: ["options"],
    addStyleTag: ["options"],
  },
  elementHandle: {
    click: ["options"],
    hover: ["options"],
    focus: [],
    tap: [],
    type: ["text", "options"],
    press: ["key", "options"],
    select: ["values"],
    uploadFile: ["paths"],
    screenshot: ["options"],
    evaluate: ["pageFunction", "args"],
    $: ["selector"],
    $$: ["selector", "options"],
    $eval: ["selector", "pageFunction", "args"],
    $$eval: ["selector", "pageFunction", "args"],
    waitForSelector: ["selector", "options"],
    drag: ["target"],
    dragAndDrop: ["target", "options"],
    scrollIntoView: [],
    boundingBox: [],
    isVisible: [],
    isHidden: [],
    contentFrame: [],
  },
  locator: {
    click: ["options"],
    fill: ["value", "options"],
    hover: ["options"],
    scroll: ["options"],
    wait: ["options"],
    waitHandle: ["options"],
  },
  keyboard: {
    type: ["text", "options"],
    press: ["key", "options"],
    sendCharacter: ["char"],
    down: ["key", "options"],
    up: ["key"],
  },
  mouse: {
    click: ["x", "y", "options"],
    move: ["x", "y", "options"],
    down: ["options"],
    up: ["options"],
    wheel: ["options"],
    dragAndDrop: ["start", "target", "options"],
  },
  request: {
    continue: ["overrides", "priority"],
    respond: ["response", "priority"],
    abort: ["errorCode", "priority"],
  },
  dialog: {
    accept: ["promptText"],
    dismiss: [],
  },
  browser: {
    newPage: ["options"],
    createBrowserContext: ["options"],
    pages: [],
    close: [],
    disconnect: [],
  },
  browserContext: {
    newPage: ["options"],
    close: [],
  },
  screenRecorder: {
    stop: [],
  },
  puppeteer: {
    launch: ["options"],
    connect: ["options"],
  },
};

/** Calls that load a document: the span also gets the page title. */
export const NAVIGATION_METHODS = new Set(["goto", "reload", "goBack", "goForward", "setContent", "waitForNavigation"]);
/** Calls that change the page (plus navigations), for screenshotOn: "action". */
export const ACTION_METHODS = new Set([
  ...NAVIGATION_METHODS,
  "click",
  "dragAndDrop",
  "fill",
  "focus",
  "hover",
  "press",
  "scroll",
  "select",
  "tap",
  "type",
  "uploadFile",
]);
/** Calls whose parameter is text typed into the page: masked when it lands in a password field. */
export const TEXT_ENTRY_PARAMS: Record<string, string> = {
  type: "text",
  fill: "value",
  sendCharacter: "char",
};
/** Calls on an element handle that act on that element, during a text-entry call. */
export const TEXT_TARGET_METHODS = new Set(["type", "focus", "press", "select"]);
/** Calls whose `pageFunction` parameter is code. */
export const EVALUATE_PARAM = "pageFunction";
export const SCREENSHOT_METHOD = "screenshot";
export const RECORDING_METHODS = new Set(["screencast", "record"]);
/** Calls that register a callback Puppeteer later fires from a browser event: [method, argument index]. */
export const CALLBACK_ARGUMENTS: Record<string, number> = { exposeFunction: 1 };
/** Event-listener registration methods on Puppeteer emitters (`page.on`, `once`, `off`, ...). */
export const LISTENER_ADD_METHODS = ["on", "once", "addListener"] as const;
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
/** Keys whose values are credentials (`authenticate({password})`, `setExtraHTTPHeaders({authorization})`). */
export const CREDENTIAL_KEY = /^(password|passphrase|secret|token|apiKey|api_key|authorization|proxy-authorization|cookie|set-cookie|x-api-key)$/i;
/** Calls whose arguments or results are cookies: their values are credentials too. */
export const COOKIE_METHODS = new Set(["setCookie", "cookies", "deleteCookie"]);
export const COOKIE_CREDENTIAL_KEY = /^(value|password|passphrase|secret|token|apiKey|api_key)$/i;

export const OUTPUT_TEXT_LIMIT = 4000;
export const TRUNCATED_SUFFIX = "... [truncated, {size} chars]";
export const ERROR_MESSAGE_LIMIT = 1000;
export const PROBE_TIMEOUT_MS = 5000;

export const SCREENSHOT_MAX_WIDTH = 1280;
export const SCREENSHOT_JPEG_QUALITY = 70;
/** Images larger than this (base64 chars) after compaction are summarised instead of exported. */
export const MAX_EXPORTED_IMAGE_BASE64 = 2_000_000;
export const IMAGE_PLACEHOLDER = "[image omitted: {mime}, {size} base64 chars]";
