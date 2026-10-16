/**
 * Respan tracing for Playwright (`playwright`, `playwright-core`, `@playwright/test`).
 *
 * Playwright is not an agent: it is what an agent (or a test, or a script)
 * drives. So this instrumentation adds no root of its own. Each public call
 * that reaches the browser (`page.goto`, `locator.click`, `expect(...).toHaveText`,
 * ...) becomes a `tool` span under whatever span is active when it runs - a
 * Respan workflow, an LLM agent's span - and calls made with no active span are
 * not traced at all.
 *
 * Seams (playwright-core's own, used from outside):
 *
 * - the client instrumentation listener (`onApiCallBegin` / `onApiCallEnd`),
 *   the hook `@playwright/test` uses for its steps: one call per public API
 *   call, named the way Playwright's trace viewer names it, internal calls
 *   excluded;
 * - `Connection.sendMessageToServer`: the call's object, protocol parameters and
 *   result, and the place to run post-call probes (page title, password-field
 *   check, screenshots) before the call returns, as internal calls that
 *   Playwright's own trace does not show;
 * - event-listener registration (`page.on`, ...) and `Connection.dispatch`:
 *   Playwright fires callbacks (`page.on`, `page.route`, `exposeFunction`, ...)
 *   from its connection, in the context `chromium.launch()` ran in. Callbacks
 *   run in the context they were registered in instead, so calls inside them
 *   nest under the caller's span and keep the customer's identifier and metadata.
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
  CALLBACK_EVENTS,
  CALLBACK_REGISTRATIONS,
  EVALUATE_METHODS,
  EXPECT_FAILURE,
  EXPECT_METHOD,
  INSTRUMENTATION_NAME,
  LEARNED_SECRET_NAME,
  LISTENER_ADD_METHODS,
  LISTENER_REMOVE_METHODS,
  MASKED_TEXT,
  MAX_LEARNED_SECRETS,
  MAX_LEARNED_TRACES,
  MIN_LEARNED_SECRET_LENGTH,
  METADATA_TRANSFORMER_KEY,
  NAVIGATION_METHODS,
  PLAYWRIGHT_MODULE,
  PROBE_TIMEOUT_MS,
  SCREENSHOT_JPEG_QUALITY,
  SCREENSHOT_METHOD,
  SCREENSHOT_MODES,
  TEXT_ENTRY_PARAMS,
  TRACER_NAME,
  TRACING_MODULE,
  type ScreenshotOn,
} from "./_constants.js";
import { callInput, callOutput } from "./_serialization.js";
import {
  flattenMetadata,
  imagePart,
  maskSpan,
  recordError,
  safely,
  secretReplacements,
  setError,
  type Replacements,
} from "./_shared.js";

type Fn = (...args: any[]) => any;
type Row = Record<string, any>;

export type { ScreenshotOn } from "./_constants.js";

export interface PlaywrightInstrumentorOptions {
  /**
   * `false` keeps span names, timings, errors (type only) and customer
   * context, and drops parameters, results, page URLs and titles, and
   * screenshots. `RESPAN_TRACE_CONTENT=false` or `TRACELOOP_TRACE_CONTENT=false`
   * do the same.
   */
  captureContent?: boolean;
  /** `false` drops all images; screenshot calls report only their byte size. */
  captureScreenshots?: boolean;
  /**
   * `"navigation"`: a page screenshot after each navigation. `"action"`: also
   * after each click, fill, check, press, ... Each one is an extra browser
   * round trip. Default `"never"`.
   */
  screenshotOn?: ScreenshotOn;
  /**
   * The Playwright module whose calls to trace (`import * as pw from
   * "playwright"`, or `@playwright/test`). Defaults to `playwright-core`, which
   * `playwright` and `@playwright/test` load; pass yours if it resolves to a
   * different copy.
   */
  playwright?: unknown;
}

/** What one public call records between `onApiCallBegin` and `onApiCallEnd`. */
interface Call {
  span: Span;
  /** The caller's context: parent, customer attributes, declared secrets. */
  ctx: Context;
  content: boolean;
  method?: string;
  params?: Row;
  object?: any;
  result?: unknown;
  /** The call's first message failed (an `expect` mismatch is reported this way, then caught). */
  sendError?: unknown;
  /** For text entry: whether the text went into a password field (undefined = unknown). */
  isPassword?: boolean;
  pageUrl?: string;
  title?: string;
  screenshot?: Row;
}

const SECRETS_KEY = createContextKey("respan.playwright.secrets");
const BOUND = Symbol.for("respan.playwright.bound_listener");
const BOUND_BY = Symbol.for("respan.playwright.bound_by");

const IS_PASSWORD_ANY = (elements: Element[]) => elements.some((e) => (e.getAttribute("type") || "").toLowerCase() === "password");
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
 * Run `fn` with these values (`{name: value}`) masked out of every Playwright
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

/** The Page a call acts on, from fields Playwright keeps locally. */
function ownerPage(object: any): any {
  for (let i = 0; i < 4 && object; i++) {
    if (object._type === "Page") return object;
    object = object._page ?? object._frame;
  }
  return undefined;
}

/** `expect(locator).not.toHaveText(...)` -> `expect.not.toHaveText`. */
function expectName(params: Row | undefined): string | undefined {
  if (typeof params?.expression !== "string") return undefined;
  const matcher = params.expression.replace(/\.(\w)/g, (_: string, c: string) => c.toUpperCase());
  return `expect.${params.isNot ? "not." : ""}${matcher}`;
}

function withDeadline<T>(promise: Promise<T>): Promise<T | undefined> {
  let timer: NodeJS.Timeout | undefined;
  const deadline = new Promise<undefined>((resolve) => {
    timer = setTimeout(() => resolve(undefined), PROBE_TIMEOUT_MS);
    timer.unref?.();
  });
  return Promise.race([promise.catch(() => undefined), deadline]).finally(() => clearTimeout(timer));
}

class Runtime {
  active = true;
  owners = 1;
  readonly restores: Array<() => void> = [];
  readonly calls = new WeakMap<object, Call>();
  /** Calls in flight, for results Playwright receives outside the call's first message. */
  readonly open = new Set<Call>();
  /** Context a page / browser context last registered a callback in (route, exposeFunction, ...). */
  readonly registrations = new WeakMap<object, Context>();
  /** Text typed into password fields, per trace: masked from every later span of that trace. */
  readonly learned = new Map<string, Map<string, string>>();
  readonly listener = {
    onApiCallBegin: (zone: Row, data?: Row) => safely(() => this.begin(zone, data)),
    onApiCallEnd: (zone: Row) => safely(() => this.end(zone)),
  };
  transformer?: { unregister(): void };

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
    const original = owner[key];
    if (typeof original !== "function") return;
    const wrapper = wrap(original);
    owner[key] = wrapper;
    this.restores.push(() => {
      if (owner[key] === wrapper) owner[key] = original;
    });
  }

  restore(): void {
    for (const restore of this.restores.splice(0).reverse()) restore();
  }

  // -- one span per public call ------------------------------------------------

  begin(zone: Row, data?: Row): void {
    const ctx = context.active();
    if (!this.active || !hasParent(ctx)) return;
    const name = String(zone.apiName || data?.method || "playwright");
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
    const call: Call = { span, ctx, content: this.contentAllowed(ctx), method: data?.method, params: data?.params };
    this.calls.set(zone, call);
    // The call's first message to the browser carries the zone's stack frames.
    if (zone.frames && typeof zone.frames === "object") this.calls.set(zone.frames, call);
    this.open.add(call);
  }

  end(zone: Row): void {
    const call = this.calls.get(zone);
    if (!call) return;
    this.calls.delete(zone);
    this.open.delete(call);
    try {
      this.record(call, zone.error);
    } finally {
      const traceId = call.span.spanContext().traceId;
      safely(() => flattenMetadata(call.span));
      safely(() => maskSpan(call.span, this.replacements(call.ctx, traceId)));
      call.span.end();
    }
  }

  record(call: Call, error: unknown): void {
    const { span, method = "", params } = call;
    if (method === EXPECT_METHOD) {
      const name = expectName(params);
      if (name) {
        span.updateName(name);
        span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name);
        span.setAttribute(ATTR_API_NAME, name);
      }
    }
    if (call.content) {
      const input = callInput(params);
      const textParam = TEXT_ENTRY_PARAMS[method];
      if (textParam && typeof params?.[textParam] === "string" && params[textParam]) {
        // Fail closed: a failed call, or a check that could not run, masks the text.
        if (error || call.isPassword !== false) {
          this.learn(span.spanContext().traceId, params[textParam]);
          input[textParam] = MASKED_TEXT;
        }
      }
      if (Object.keys(input).length) span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, JSON.stringify(input));
      if (call.pageUrl) span.setAttribute(ATTR_PAGE_URL, call.pageUrl);
      if (call.title) span.setAttribute(ATTR_PAGE_TITLE, call.title);
      if (!error) {
        let output = callOutput(call.result, this.captureScreenshots, EVALUATE_METHODS.has(method));
        if (call.screenshot) output = { ...(output !== undefined ? { result: output } : {}), screenshot: call.screenshot };
        if (output !== undefined) span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, JSON.stringify(output));
      }
    }
    if (error) {
      recordError(span, error, call.content);
    } else if (method === EXPECT_METHOD && call.sendError) {
      // A mismatch fails the browser call; Playwright catches it and the matcher throws later, outside Playwright.
      setError(span, EXPECT_FAILURE, (call.sendError as Error)?.message, call.content);
    }
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

  // -- the call's messages to the browser --------------------------------------

  wrapSend(original: Fn): Fn {
    const runtime = this;
    return async function sendMessageToServer(this: unknown, object: any, method: string, params: Row, ...rest: any[]) {
      if (!runtime.active) return original.call(this, object, method, params, ...rest);
      if (CALLBACK_REGISTRATIONS.has(method) && object) runtime.registrations.set(object, context.active());
      // The call's first message carries its API zone (or a copy of it) as options.
      const frames = rest[0]?.frames;
      const call = frames && typeof frames === "object" ? runtime.calls.get(frames) : undefined;
      if (!call) {
        const result = await original.call(this, object, method, params, ...rest);
        if (method === SCREENSHOT_METHOD) safely(() => runtime.adoptScreenshot(result));
        return result;
      }
      runtime.calls.delete(frames);
      call.method = method;
      call.params = params;
      call.object = object;
      let result: unknown;
      try {
        result = await original.call(this, object, method, params, ...rest);
      } catch (error) {
        call.sendError = error;
        throw error;
      }
      call.result = result;
      await withDeadline(runtime.after(call));
      return result;
    };
  }

  /** `locator.screenshot()` gets its image from a later message than its first: attach it. */
  adoptScreenshot(result: any): void {
    if (!(result?.binary instanceof Uint8Array)) return;
    const ctx = context.active();
    for (const call of this.open) {
      const name = (call.span as any).name as string;
      if (call.ctx === ctx && name.endsWith(`.${SCREENSHOT_METHOD}`) && !(call.result as Row | undefined)?.binary) {
        call.result = result;
        return;
      }
    }
  }

  /** Post-call probes, as internal calls, before the caller's call returns. */
  async after(call: Call): Promise<void> {
    if (!call.content) return;
    const { object, method = "", params } = call;
    if (TEXT_ENTRY_PARAMS[method]) call.isPassword = await this.passwordProbe(object, method, params);
    const page = ownerPage(object);
    if (!page || page.isClosed?.()) return;
    call.pageUrl = page.url();
    if (NAVIGATION_METHODS.has(method)) {
      const title = await internal(page, () => page.title());
      if (typeof title === "string" && title) call.title = title;
    }
    if (this.wantsScreenshot(method)) {
      const image = await internal(page, () =>
        page.screenshot({ type: "jpeg", quality: SCREENSHOT_JPEG_QUALITY, timeout: PROBE_TIMEOUT_MS }),
      );
      if (image instanceof Uint8Array) call.screenshot = imagePart(image);
    }
  }

  wantsScreenshot(method: string): boolean {
    if (!this.captureScreenshots || this.screenshotOn === "never" || method === SCREENSHOT_METHOD) return false;
    return (this.screenshotOn === "navigation" ? NAVIGATION_METHODS : ACTION_METHODS).has(method);
  }

  async passwordProbe(object: any, method: string, params: Row | undefined): Promise<boolean | undefined> {
    let probe: (() => Promise<unknown>) | undefined;
    if (object?._type === "Frame" && typeof params?.selector === "string") probe = () => object.$$eval(params.selector, IS_PASSWORD_ANY);
    else if (object?._type === "ElementHandle") probe = () => object.evaluate(IS_PASSWORD_ONE);
    else if (object?._type === "Page" && method.startsWith("keyboard")) probe = () => object.mainFrame().evaluate(IS_PASSWORD_FOCUSED);
    if (!probe) return undefined;
    const result = await internal(object, probe);
    return typeof result === "boolean" ? result : undefined;
  }

  // -- callbacks ---------------------------------------------------------------

  /** A listener that runs in the context it was registered in (first registration wins). */
  bind(listener: unknown, emitter: any): unknown {
    if (!this.active || typeof listener !== "function" || (listener as any)[BOUND]) return listener;
    // Playwright's own channel listeners stay as they are.
    if (emitter?._object) return listener;
    const existing = (listener as any)[BOUND_BY];
    if (existing) return existing;
    const registered = context.active();
    const bound = function (this: unknown, ...args: unknown[]) {
      return context.with(registered, () => (listener as Fn).apply(this, args));
    };
    (bound as any)[BOUND] = true;
    try {
      Object.defineProperty(listener, BOUND_BY, { value: bound, configurable: true });
    } catch {
      return listener;
    }
    return bound;
  }

  wrapAddListener(original: Fn): Fn {
    const runtime = this;
    return function (this: unknown, event: unknown, listener: unknown, ...rest: unknown[]) {
      return original.call(this, event, runtime.bind(listener, this), ...rest);
    };
  }

  wrapRemoveListener(original: Fn): Fn {
    return function (this: unknown, event: unknown, listener: any, ...rest: unknown[]) {
      return original.call(this, event, (typeof listener === "function" && listener[BOUND_BY]) || listener, ...rest);
    };
  }

  wrapDispatch(original: Fn): Fn {
    const runtime = this;
    return function dispatch(this: any, message: Row) {
      if (runtime.active && !message?.id && CALLBACK_EVENTS.has(message?.method)) {
        const target = this._objects?.get?.(message.guid);
        const registered = target ? runtime.registrations.get(target) : undefined;
        if (registered) return context.with(registered, () => original.call(this, message));
      }
      return original.call(this, message);
    };
  }
}

/** Run a probe as an internal call: hidden from Playwright's own trace and from ours. */
async function internal<T>(owner: any, probe: () => Promise<T>): Promise<T | undefined> {
  const target = owner?._wrapApiCall ? owner : owner?._page ?? owner?._frame;
  try {
    return await withDeadline(target?._wrapApiCall ? target._wrapApiCall(probe, { internal: true }) : probe());
  } catch {
    return undefined;
  }
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

/** Trace Playwright API calls under the active span. */
export class PlaywrightInstrumentor {
  public readonly name = INSTRUMENTATION_NAME;
  private readonly options: PlaywrightInstrumentorOptions;
  private runtime?: Runtime;

  constructor(options: PlaywrightInstrumentorOptions = {}) {
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
        throw new Error("Playwright instrumentation is already active with different settings");
      }
      runtime.owners += 1;
      this.runtime = runtime;
      return;
    }
    let module: any = this.options.playwright;
    if (!module) {
      try {
        module = await import(PLAYWRIGHT_MODULE);
      } catch {
        return; // Playwright is not installed: nothing to trace.
      }
    }
    const browserType = module?.chromium ?? module?.default?.chromium;
    const instrumentation = browserType?._instrumentation;
    const connection = browserType?._connection;
    const channelOwner = browserType && Object.getPrototypeOf(Object.getPrototypeOf(browserType));
    if (typeof instrumentation?.addListener !== "function" || !connection || typeof channelOwner?._wrapApiCall !== "function") {
      throw new Error("PlaywrightInstrumentor: unsupported Playwright version (needs playwright-core >= 1.55).");
    }
    const runtime = new Runtime(captureContent, captureScreenshots, screenshotOn);
    try {
      const connectionProto = Object.getPrototypeOf(connection);
      runtime.patch(connectionProto, "sendMessageToServer", (original) => runtime.wrapSend(original));
      runtime.patch(connectionProto, "dispatch", (original) => runtime.wrapDispatch(original));
      // Playwright objects are event emitters (ChannelOwner, and the EventEmitter under it).
      for (const proto of [channelOwner, Object.getPrototypeOf(channelOwner)]) {
        for (const method of LISTENER_ADD_METHODS) runtime.patch(proto, method, (original) => runtime.wrapAddListener(original));
        for (const method of LISTENER_REMOVE_METHODS) runtime.patch(proto, method, (original) => runtime.wrapRemoveListener(original));
      }
      instrumentation.addListener(runtime.listener);
      runtime.restores.push(() => instrumentation.removeListener(runtime.listener));
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
