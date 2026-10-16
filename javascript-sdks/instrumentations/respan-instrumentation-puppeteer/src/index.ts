/**
 * Respan tracing for Puppeteer (`puppeteer`, `puppeteer-core`).
 *
 * Puppeteer is not an agent: it is what an agent (or a test, or a script)
 * drives. So this instrumentation adds no root of its own. Each public call
 * (`page.goto`, `locator.click`, `elementHandle.type`, `keyboard.press`, ...)
 * becomes a `tool` span under whatever span is active when it runs - a Respan
 * workflow, an LLM agent's span - and calls made with no active span are not
 * traced at all.
 *
 * Seams (Puppeteer's own public classes, patched from outside):
 *
 * - a curated list of public methods on the abstract `api/` classes (`Page`,
 *   `Frame`, `ElementHandle`, `Locator`, `Keyboard`, `Mouse`, `HTTPRequest`,
 *   `Dialog`, `Browser`, `BrowserContext`) and on their CDP and WebDriver BiDi
 *   implementations wherever those define the method. Puppeteer has no single
 *   API-call boundary (`page.click` calls `frame.click`, which calls
 *   `elementHandle.click`, which calls `mouse.click`), so a reentrancy guard in
 *   the OpenTelemetry context makes only the outermost call a span;
 * - `BrowserLauncher.launch` / `Puppeteer.connect`, for the browser's start;
 * - event-listener registration (`page.on`, `browser.on`, ...) and
 *   `exposeFunction`: Puppeteer fires listeners and exposed functions from its
 *   connection, in the context the browser was launched in. They run in the
 *   context they were registered in instead, so calls inside them nest under
 *   the caller's span and keep the customer's identifier and metadata.
 */
import {
  context,
  createContextKey,
  isSpanContextValid,
  trace,
  type Context,
  type Span,
} from "@opentelemetry/api";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT, SpanAttributes, TraceloopSpanKindValues } from "@traceloop/ai-semantic-conventions";
import {
  ACTION_METHODS,
  ATTR_API_NAME,
  ATTR_PAGE_TITLE,
  ATTR_PAGE_URL,
  ATTR_RECORDING_PATH,
  CALLBACK_ARGUMENTS,
  COOKIE_CREDENTIAL_KEY,
  COOKIE_METHODS,
  CREDENTIAL_KEY,
  INSTRUMENTATION_NAME,
  LEARNED_SECRET_NAME,
  LISTENER_ADD_METHODS,
  LISTENER_REMOVE_METHODS,
  MASKED_TEXT,
  MAX_LEARNED_SECRETS,
  MAX_LEARNED_TRACES,
  METADATA_TRANSFORMER_KEY,
  MIN_LEARNED_SECRET_LENGTH,
  NAVIGATION_METHODS,
  PROBE_TIMEOUT_MS,
  PUPPETEER_BIDI_MODULE,
  PUPPETEER_MODULE,
  RECORDING_METHODS,
  SCREENSHOT_JPEG_QUALITY,
  SCREENSHOT_METHOD,
  SCREENSHOT_MODES,
  TEXT_ENTRY_PARAMS,
  TEXT_TARGET_METHODS,
  TRACED_METHODS,
  TRACER_NAME,
  TRACING_MODULE,
  type ScreenshotOn,
} from "./_constants.js";
import { callInput, callOutput, jsonable, registerClasses } from "./_serialization.js";
import {
  flattenMetadata,
  imagePart,
  maskSpan,
  recordError,
  safely,
  secretReplacements,
  type Replacements,
} from "./_shared.js";

type Fn = (...args: any[]) => any;
type Row = Record<string, any>;

export type { ScreenshotOn } from "./_constants.js";

export interface PuppeteerInstrumentorOptions {
  /**
   * `false` keeps span names, timings, errors (type only) and customer
   * context, and drops arguments, results, page URLs and titles, and
   * screenshots. `RESPAN_TRACE_CONTENT=false` or `TRACELOOP_TRACE_CONTENT=false`
   * do the same.
   */
  captureContent?: boolean;
  /** `false` drops all images; screenshot calls report only their byte size. */
  captureScreenshots?: boolean;
  /**
   * `"navigation"`: a page screenshot after each navigation. `"action"`: also
   * after each click, type, fill, press, ... Each one is an extra browser
   * round trip. Default `"never"`.
   */
  screenshotOn?: ScreenshotOn;
  /**
   * The Puppeteer module whose classes to trace (`import * as pptr from
   * "puppeteer-core"`). Defaults to `puppeteer-core`, which `puppeteer` loads;
   * pass yours if it resolves to a different copy.
   */
  puppeteer?: unknown;
}

/** One traced public call. */
interface Call {
  span: Span;
  /** The caller's context: parent, customer attributes, declared secrets. */
  ctx: Context;
  content: boolean;
  label: string;
  method: string;
  params: readonly string[];
  args: unknown[];
  /** The object the call was made on. */
  self: any;
  result?: unknown;
  /** For text entry: whether the text went into a password field (undefined = not seen). */
  isPassword?: boolean;
  /** A password check that could not run. */
  isPasswordUnknown?: boolean;
  /** Elements already checked for this call (page.type reaches elementHandle.type, then its focus). */
  checked?: WeakSet<object>;
  pageUrl?: string;
  title?: string;
  screenshot?: Row;
}

/** The traced call in progress: set while it runs, so the calls it makes are not spans. */
const CALL_KEY = createContextKey("respan.puppeteer.call");
const SECRETS_KEY = createContextKey("respan.puppeteer.secrets");
const BOUND = Symbol.for("respan.puppeteer.bound_listener");

const IS_PASSWORD_ONE = (e: Element) => (e.getAttribute("type") || "").toLowerCase() === "password";
// Typed text goes to the focused element; inside an iframe we cannot see which one.
const IS_PASSWORD_FOCUSED = () => {
  const e = document.activeElement;
  return !!e && (e.tagName === "IFRAME" || e.tagName === "FRAME" || (e.getAttribute("type") || "").toLowerCase() === "password");
};

function declaredSecrets(ctx: Context): Replacements {
  return (ctx.getValue(SECRETS_KEY) as Replacements | undefined) ?? [];
}

/**
 * Run `fn` with these values (`{name: value}`) masked out of every Puppeteer
 * span it causes: each is exported as `<secret>name</secret>`, wherever it shows
 * up (typed text, URLs, `evaluate` arguments, error messages).
 */
export function withSecrets<T>(secrets: Record<string, string>, fn: () => T): T {
  const ctx = context.active();
  const merged = [...declaredSecrets(ctx), ...secretReplacements(secrets)].sort((a, b) => b[0].length - a[0].length);
  return context.with(ctx.setValue(SECRETS_KEY, merged), fn);
}

function hasParent(ctx: Context): boolean {
  const parent = trace.getSpanContext(ctx);
  return !!parent && isSpanContextValid(parent);
}

function isThenable(value: unknown): value is PromiseLike<unknown> {
  return !!value && typeof (value as any).then === "function";
}

function withDeadline<T>(promise: Promise<T>): Promise<T | undefined> {
  let timer: NodeJS.Timeout | undefined;
  const deadline = new Promise<undefined>((resolve) => {
    timer = setTimeout(() => resolve(undefined), PROBE_TIMEOUT_MS);
    timer.unref?.();
  });
  return Promise.race([promise.catch(() => undefined), deadline]).finally(() => clearTimeout(timer));
}

/** Run a probe; failures and slow probes give `undefined`. */
async function probe<T>(fn: () => Promise<T>): Promise<T | undefined> {
  try {
    return await withDeadline(fn());
  } catch {
    return undefined;
  }
}

class Runtime {
  active = true;
  owners = 1;
  readonly restores: Array<() => void> = [];
  /** Text typed into password fields, per trace: masked from every later span of that trace. */
  readonly learned = new Map<string, Map<string, string>>();
  /** page.keyboard / page.mouse -> the page, for page URL and the focused-field check. */
  readonly inputs = new WeakMap<object, any>();
  /** Locator -> what it was created from (`page.locator(selector)`), and its page. */
  readonly locators = new WeakMap<object, { locator: unknown; page?: any }>();
  /** emitter -> listener -> the listener bound to its registration context. */
  readonly bindings = new WeakMap<object, WeakMap<Function, Fn>>();
  transformer?: { unregister(): void };
  classes: Row = {};

  constructor(
    readonly captureContent: boolean,
    readonly captureScreenshots: boolean,
    readonly screenshotOn: ScreenshotOn,
  ) {}

  get tracer() {
    // Resolved per call so a provider registered after activation is used.
    return trace.getTracer(TRACER_NAME);
  }

  contentAllowed(ctx: Context): boolean {
    return (
      this.captureContent &&
      process.env.RESPAN_TRACE_CONTENT !== "false" &&
      process.env.TRACELOOP_TRACE_CONTENT !== "false" &&
      ctx.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT) !== false
    );
  }

  patch(owner: any, key: string, wrap: (original: Fn) => Fn): void {
    if (!owner || !Object.prototype.hasOwnProperty.call(owner, key)) return;
    const descriptor = Object.getOwnPropertyDescriptor(owner, key);
    const original = descriptor?.value;
    if (typeof original !== "function") return; // accessors and fields stay as they are
    const wrapper = wrap(original);
    owner[key] = wrapper;
    this.restores.push(() => {
      if (owner[key] === wrapper) owner[key] = original;
    });
  }

  restore(): void {
    for (const restore of this.restores.splice(0).reverse()) restore();
  }

  // -- one span per outermost public call -----------------------------------------

  wrapMethod(label: string, method: string, original: Fn): Fn {
    const runtime = this;
    const params = TRACED_METHODS[label][method];
    return function (this: unknown, ...args: unknown[]) {
      if (!runtime.active) return original.apply(this, args);
      const ctx = context.active();
      const outer = ctx.getValue(CALL_KEY) as Call | undefined;
      if (outer) return runtime.inner(outer, label, method, this, original, args);
      if (!hasParent(ctx)) return original.apply(this, args);
      return runtime.traced(label, method, params, this, original, args, ctx);
    };
  }

  /** A call made by a traced call: not a span; the element it acts on may hold typed text. */
  inner(outer: Call, label: string, method: string, self: any, original: Fn, args: unknown[]): unknown {
    if (outer.content && TEXT_ENTRY_PARAMS[outer.method] && label === "elementHandle" && TEXT_TARGET_METHODS.has(method)) {
      return this.checkElement(outer, self).then(() => original.apply(self, args));
    }
    return original.apply(self, args);
  }

  traced(label: string, method: string, params: readonly string[], self: any, original: Fn, args: unknown[], ctx: Context): unknown {
    const name = `${label}.${method}`;
    const span = this.tracer.startSpan(
      name,
      {
        attributes: {
          [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.TOOL,
          [SpanAttributes.TRACELOOP_SPAN_KIND]: TraceloopSpanKindValues.TOOL,
          [SpanAttributes.TRACELOOP_ENTITY_NAME]: name,
          [ATTR_API_NAME]: name,
        },
      },
      ctx,
    );
    const call: Call = { span, ctx, content: this.contentAllowed(ctx), label, method, params, args, self };
    safely(() => this.rememberPage(self));
    const callbackIndex = CALLBACK_ARGUMENTS[method];
    if (callbackIndex !== undefined && typeof args[callbackIndex] === "function") {
      // Fired later from the connection; runs in the caller's context, as a listener would.
      args = [...args];
      args[callbackIndex] = this.bindTo(args[callbackIndex] as Fn, ctx);
    }
    const callCtx = trace.setSpan(ctx, span).setValue(CALL_KEY, call);
    const textEntry = call.content && TEXT_ENTRY_PARAMS[method] !== undefined;
    let result: unknown;
    try {
      result = context.with(callCtx, () => {
        if (textEntry && (label === "elementHandle" || label === "keyboard")) {
          return this.checkTarget(call, self).then(() => original.apply(self, args));
        }
        return original.apply(self, args);
      });
    } catch (error) {
      this.finish(call, error);
      throw error;
    }
    if (!isThenable(result)) {
      call.result = result;
      this.finish(call);
      return result;
    }
    return context.with(callCtx, async () => {
      let value: unknown;
      try {
        value = await result;
      } catch (error) {
        this.finish(call, error);
        throw error;
      }
      call.result = value;
      try {
        await withDeadline(this.after(call, self));
      } finally {
        this.finish(call);
      }
      return value;
    });
  }

  finish(call: Call, error?: unknown): void {
    try {
      this.record(call, error);
    } catch {
      // Telemetry must never break the customer's browser flow.
    } finally {
      const traceId = call.span.spanContext().traceId;
      safely(() => flattenMetadata(call.span));
      safely(() => maskSpan(call.span, this.replacements(call.ctx, traceId)));
      call.span.end();
    }
  }

  record(call: Call, error: unknown): void {
    const { span, method, label } = call;
    safely(() => this.rememberPage(call.result));
    if (call.content) {
      const description = label === "locator" ? this.locators.get(call.self) : undefined;
      const credentials = COOKIE_METHODS.has(method) ? COOKIE_CREDENTIAL_KEY : CREDENTIAL_KEY;
      const input = callInput(call.params, call.args, description ? { locator: description.locator } : undefined, credentials);
      const textParam = TEXT_ENTRY_PARAMS[method];
      if (textParam && typeof input[textParam] === "string" && input[textParam]) {
        // Fail closed: a failed call, or a check that could not run or never saw the field, masks the text.
        if (error || call.isPassword !== false || call.isPasswordUnknown) {
          this.learn(span.spanContext().traceId, String(call.args[call.params.indexOf(textParam)]));
          input[textParam] = MASKED_TEXT;
        }
      }
      if (Object.keys(input).length) span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, JSON.stringify(input));
      if (call.pageUrl) span.setAttribute(ATTR_PAGE_URL, call.pageUrl);
      if (call.title) span.setAttribute(ATTR_PAGE_TITLE, call.title);
      if (RECORDING_METHODS.has(method) && typeof (call.args[0] as Row | undefined)?.path === "string") {
        span.setAttribute(ATTR_RECORDING_PATH, (call.args[0] as Row).path);
      }
      if (!error) {
        let output = callOutput(call.result, this.captureScreenshots, method === SCREENSHOT_METHOD, credentials);
        if (call.screenshot) output = { ...(output !== undefined ? { result: output } : {}), screenshot: call.screenshot };
        if (output !== undefined) span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, JSON.stringify(output));
      }
    }
    if (error) recordError(span, error, call.content);
  }

  replacements(ctx: Context, traceId: string): Replacements {
    const learned = this.learned.get(traceId);
    const declared = declaredSecrets(ctx);
    if (!learned?.size) return declared;
    return [...declared, ...learned].sort((a, b) => b[0].length - a[0].length);
  }

  learn(traceId: string, value: string): void {
    if (value.length < MIN_LEARNED_SECRET_LENGTH) return;
    let learned = this.learned.get(traceId);
    if (!learned) {
      learned = new Map();
      this.learned.set(traceId, learned);
      if (this.learned.size > MAX_LEARNED_TRACES) this.learned.delete(this.learned.keys().next().value!);
    }
    for (const [secret, placeholder] of secretReplacements({ [LEARNED_SECRET_NAME]: value })) {
      if (learned.size < MAX_LEARNED_SECRETS) learned.set(secret, placeholder);
    }
  }

  // -- page context, probes ------------------------------------------------------

  isPage(value: unknown): boolean {
    return typeof this.classes.Page === "function" && value instanceof this.classes.Page;
  }

  /** Remember a page's keyboard and mouse, so calls on them know their page. */
  rememberPage(value: any): void {
    if (!this.isPage(value)) return;
    for (const input of [value.keyboard, value.mouse]) if (input && typeof input === "object") this.inputs.set(input, value);
  }

  /** The page a call acts on, from what Puppeteer keeps locally. */
  ownerPage(call: Call, self: any): any {
    const { label } = call;
    if (label === "page") return self;
    if (label === "frame") return self.page?.();
    if (label === "elementHandle") return self.frame?.page?.();
    if (label === "keyboard" || label === "mouse") return this.inputs.get(self);
    if (label === "locator") return this.locators.get(self)?.page;
    if (label === "request") return self.frame?.()?.page?.();
    return undefined;
  }

  /** Before text entry on an element or the keyboard: is the target a password field? */
  async checkTarget(call: Call, self: any): Promise<void> {
    if (call.label === "elementHandle") return this.checkElement(call, self);
    const page = this.inputs.get(self);
    const result = page ? await probe(() => page.evaluate(IS_PASSWORD_FOCUSED)) : undefined;
    this.note(call, result);
  }

  async checkElement(call: Call, handle: any): Promise<void> {
    if (call.checked?.has(handle)) return;
    (call.checked ??= new WeakSet()).add(handle);
    this.note(call, await probe(() => handle.evaluate(IS_PASSWORD_ONE)));
  }

  note(call: Call, isPassword: unknown): void {
    if (typeof isPassword !== "boolean") call.isPasswordUnknown = true;
    else call.isPassword = call.isPassword || isPassword;
  }

  /** Post-call probes (page URL, title, screenshot), as calls of this call, before it returns. */
  async after(call: Call, self: any): Promise<void> {
    if (!call.content) return;
    const page = this.ownerPage(call, self);
    if (!page || page.isClosed?.()) return;
    call.pageUrl = page.url();
    const { label, method } = call;
    if ((label === "page" || label === "frame") && NAVIGATION_METHODS.has(method)) {
      const title = await probe(() => page.title());
      if (typeof title === "string" && title) call.title = title;
    }
    if (this.wantsScreenshot(method)) {
      const image = await probe(() => page.screenshot({ type: "jpeg", quality: SCREENSHOT_JPEG_QUALITY }));
      if (image instanceof Uint8Array) call.screenshot = imagePart(image);
    }
  }

  wantsScreenshot(method: string): boolean {
    if (!this.captureScreenshots || this.screenshotOn === "never" || method === SCREENSHOT_METHOD) return false;
    return (this.screenshotOn === "navigation" ? NAVIGATION_METHODS : ACTION_METHODS).has(method);
  }

  // -- locators --------------------------------------------------------------------

  /** `page.locator(selector)` and friends: remember what the locator was made from. */
  wrapLocatorFactory(original: Fn): Fn {
    const runtime = this;
    return function (this: any, input: unknown, ...rest: unknown[]) {
      const locator = original.call(this, input, ...rest);
      if (runtime.active && locator && typeof locator === "object") {
        safely(() => {
          const page = runtime.isPage(this) ? this : this.page?.() ?? this.frame?.page?.();
          runtime.locators.set(locator, { locator: typeof input === "function" ? input.toString() : jsonable(input), page });
        });
      }
      return locator;
    };
  }

  /** Clones (`setTimeout`, `filter`, `map`, ...) copy their source's options: and its description. */
  wrapCopyOptions(original: Fn): Fn {
    const runtime = this;
    return function (this: any, source: any, ...rest: unknown[]) {
      const description = source && runtime.locators.get(source);
      if (description && !runtime.locators.has(this)) runtime.locators.set(this, description);
      return original.call(this, source, ...rest);
    };
  }

  // -- callbacks ---------------------------------------------------------------------

  /** `fn`, running in `ctx` (without the guard of the call that registered it). */
  bindTo(fn: Fn, ctx: Context): Fn {
    const bound = function (this: unknown, ...args: unknown[]) {
      return context.with(ctx, () => fn.apply(this, args));
    };
    (bound as any)[BOUND] = true;
    return bound;
  }

  /**
   * A listener that runs in the context it was registered in. Bindings are kept
   * per emitter, so one handler function shared by several pages runs in each
   * page's own registration context (the first registration on an emitter wins).
   */
  bind(listener: unknown, emitter: unknown): unknown {
    if (!this.active || typeof listener !== "function" || (listener as any)[BOUND] || !emitter || typeof emitter !== "object") return listener;
    const existing = this.bindings.get(emitter)?.get(listener);
    if (existing) return existing;
    const registered = context.active();
    // Puppeteer's own listeners (registered inside a call, or with no span active) stay as they are.
    if (registered.getValue(CALL_KEY) || !hasParent(registered)) return listener;
    const bound = this.bindTo(listener as Fn, registered);
    let byListener = this.bindings.get(emitter);
    if (!byListener) this.bindings.set(emitter, (byListener = new WeakMap()));
    byListener.set(listener, bound);
    return bound;
  }

  wrapAddListener(original: Fn): Fn {
    const runtime = this;
    return function (this: unknown, event: unknown, listener: unknown, ...rest: unknown[]) {
      return original.call(this, event, runtime.bind(listener, this), ...rest);
    };
  }

  wrapRemoveListener(original: Fn): Fn {
    const runtime = this;
    return function (this: any, event: unknown, listener: any, ...rest: unknown[]) {
      const bound = typeof listener === "function" && this && typeof this === "object" ? runtime.bindings.get(this)?.get(listener) : undefined;
      return original.call(this, event, bound ?? listener, ...rest);
    };
  }
}

/** The classes a label's methods live on: the abstract api/ class first, then its implementations. */
function classesFor(module: Row, bidi: Row): Record<string, Function[]> {
  const pick = (...candidates: unknown[]) => candidates.filter((c): c is Function => typeof c === "function");
  return {
    page: pick(module.Page, module.CdpPage, bidi.BidiPage),
    frame: pick(module.Frame, module.CdpFrame, bidi.BidiFrame),
    elementHandle: pick(module.ElementHandle, module.CdpElementHandle, bidi.BidiElementHandle),
    locator: pick(module.Locator),
    keyboard: pick(module.Keyboard, module.CdpKeyboard, bidi.BidiKeyboard),
    mouse: pick(module.Mouse, module.CdpMouse, bidi.BidiMouse),
    request: pick(module.HTTPRequest, module.CdpHTTPRequest, bidi.BidiHTTPRequest),
    dialog: pick(module.Dialog, module.CdpDialog, bidi.BidiDialog),
    browser: pick(module.Browser, module.CdpBrowser, bidi.BidiBrowser),
    browserContext: pick(module.BrowserContext, module.CdpBrowserContext, bidi.BidiBrowserContext),
    screenRecorder: pick(module.ScreenRecorder, module.ScreenRecording, module.CdpScreenRecording, bidi.BidiScreenRecording),
    puppeteer: pick(module.BrowserLauncher, module.ChromeLauncher, module.FirefoxLauncher, module.Puppeteer),
  };
}

let shared: { runtime: Runtime } | undefined;

async function registerMetadataTransformer(runtime: Runtime): Promise<void> {
  try {
    const tracing: any = await import(TRACING_MODULE);
    runtime.transformer = tracing.registerSpanTransformer?.(METADATA_TRANSFORMER_KEY, { onEnd: flattenMetadata });
  } catch {
    // No @respan/tracing, or no Respan pipeline yet: our own spans are still flattened.
  }
}

/** Trace Puppeteer calls under the active span. */
export class PuppeteerInstrumentor {
  public readonly name = INSTRUMENTATION_NAME;
  private readonly options: PuppeteerInstrumentorOptions;
  private runtime?: Runtime;

  constructor(options: PuppeteerInstrumentorOptions = {}) {
    if (options.screenshotOn !== undefined && !SCREENSHOT_MODES.includes(options.screenshotOn)) {
      throw new Error(`screenshotOn must be one of ${SCREENSHOT_MODES.join(", ")}`);
    }
    this.options = options;
  }

  async activate(): Promise<void> {
    if (this.runtime) return;
    const { captureContent = true, captureScreenshots = true, screenshotOn = "never" } = this.options;
    if (shared) {
      const { runtime } = shared;
      if (runtime.captureContent !== captureContent || runtime.captureScreenshots !== captureScreenshots || runtime.screenshotOn !== screenshotOn) {
        throw new Error("Puppeteer instrumentation is already active with different settings");
      }
      runtime.owners += 1;
      this.runtime = runtime;
      return;
    }
    let module: any = this.options.puppeteer;
    if (!module) {
      try {
        module = await import(PUPPETEER_MODULE);
      } catch {
        return; // Puppeteer is not installed: nothing to trace.
      }
    }
    if (typeof module?.Page !== "function" || typeof module?.Locator !== "function" || typeof module?.EventEmitter !== "function") {
      module = module?.default && typeof module.default.Page === "function" ? module.default : module;
    }
    if (typeof module?.Page !== "function" || typeof module?.ElementHandle !== "function" || typeof module?.EventEmitter !== "function") {
      throw new Error("PuppeteerInstrumentor: unsupported Puppeteer version (needs puppeteer-core 25).");
    }
    let bidi: Row = {};
    try {
      // Loaded here so BiDi pages (`protocol: "webDriverBiDi"`) are traced too; Puppeteer imports it lazily.
      bidi = await import(PUPPETEER_BIDI_MODULE);
    } catch {
      // An install without the BiDi implementation: CDP only.
    }
    const runtime = new Runtime(captureContent, captureScreenshots, screenshotOn);
    runtime.classes = module;
    registerClasses(module);
    try {
      for (const [label, classes] of Object.entries(classesFor(module, bidi))) {
        for (const cls of classes) {
          for (const method of Object.keys(TRACED_METHODS[label])) {
            runtime.patch(cls.prototype, method, (original) => runtime.wrapMethod(label, method, original));
          }
        }
      }
      for (const cls of [module.Page, module.Frame]) runtime.patch(cls.prototype, "locator", (original) => runtime.wrapLocatorFactory(original));
      runtime.patch(module.ElementHandle.prototype, "asLocator", (original) => runtime.wrapLocatorFactory(original));
      runtime.patch(module.Locator?.prototype, "copyOptions", (original) => runtime.wrapCopyOptions(original));
      // Puppeteer's emitters, and Page's own on/off (it wraps `request` listeners before registering them).
      for (const proto of [module.EventEmitter.prototype, module.Page.prototype]) {
        for (const method of LISTENER_ADD_METHODS) runtime.patch(proto, method, (original) => runtime.wrapAddListener(original));
        for (const method of LISTENER_REMOVE_METHODS) runtime.patch(proto, method, (original) => runtime.wrapRemoveListener(original));
      }
    } catch (error) {
      runtime.restore();
      throw error;
    }
    shared = { runtime };
    this.runtime = runtime;
    await registerMetadataTransformer(runtime);
  }

  deactivate(): void {
    const runtime = this.runtime;
    this.runtime = undefined;
    if (!runtime) return;
    runtime.owners -= 1;
    if (runtime.owners > 0) return;
    runtime.active = false;
    runtime.restore();
    safely(() => runtime.transformer?.unregister());
    if (shared?.runtime === runtime) shared = undefined;
  }

  isActive(): boolean {
    return Boolean(this.runtime);
  }
}
