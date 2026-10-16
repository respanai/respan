/**
 * Respan tracing for Stagehand v4 (`@browserbasehq/stagehand`).
 *
 * Stagehand v4 runs its browser logic inside a Chrome extension and talks to
 * it over JSON-RPC. This instrumentation wraps the public seams from outside:
 *
 * - `Stagehand.create` ... `close()`        -> `agent` span (the session)
 * - `act` / `observe` / `extract`           -> `task` span per operation
 * - the client model's `generate` callback  -> `chat` span per LLM call
 * - `page.*` / `locator.*` browser actions  -> `tool` spans
 *
 * Stagehand's own OpenTelemetry RPC spans are left out of Respan exports; the
 * spans above carry the same structure with Respan's attributes.
 */
import { randomUUID } from "node:crypto";
import {
  context,
  createContextKey,
  trace,
  SpanStatusCode,
  type Attributes,
  type Context,
  type Span,
} from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT, SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  ATTR_ACTION_ID,
  ATTR_CACHE_STATUS,
  ATTR_PAGE_URL,
  ATTR_SESSION_ID,
  ATTR_USAGE_SOURCE,
  CONTEXT_ACTIONS,
  ERROR_MESSAGE_LIMIT,
  GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  GEN_AI_USAGE_REASONING_TOKENS,
  INSTRUMENTATION_NAME,
  LLM_USAGE_CACHE_READ_INPUT_TOKENS,
  LOCATOR_ACTIONS,
  OPERATIONS,
  OPERATION_PREFIX,
  PAGE_ACTIONS,
  REDACTED,
  SCREENSHOT_DEADLINE_MS,
  SCREENSHOT_OPTIONS,
  SENSITIVE_FIELD,
  SENSITIVE_SELECTOR,
  SESSION_NAME_DETAIL,
  SPAN_LLM,
  SPAN_SESSION,
  STAGEHAND_MODULE,
  TRACER_NAME,
  USAGE_SOURCE_OPERATION,
  type Operation,
} from "./_constants.js";
import {
  chatMessages,
  completion,
  contentText,
  functions,
  imagePart,
  json,
  jsonable,
} from "./_serialization.js";

type Fn = (...args: any[]) => any;
type Row = Record<string, any>;

export interface StagehandInstrumentorOptions {
  /**
   * `false` keeps span structure, timings, models, operation names and token
   * counts but drops instructions, prompts, completions, results, page URLs,
   * typed text and exception messages. `RESPAN_TRACE_CONTENT=false` or
   * `TRACELOOP_TRACE_CONTENT=false` do the same.
   */
  captureContent?: boolean;
  /** `false` skips the screenshot after each `act` and drops images from prompts. */
  captureScreenshots?: boolean;
  /**
   * Model name reported on `chat` spans for a client model
   * (`Stagehand.create({ model: { generate } })`), which tells Stagehand
   * nothing about the model it calls. Needed for cost. `withModelName()` sets
   * it per model instead.
   */
  clientModelName?: string;
  /** The `@browserbasehq/stagehand` module to patch; imported when omitted. */
  stagehandModule?: StagehandModule;
}

export interface StagehandModule {
  Stagehand: any;
  Page?: any;
  Locator?: any;
  BrowserContext?: any;
}

interface Session {
  id: string;
  span: Span;
  ctx: Context;
  runtime: Runtime;
  modelName?: string;
  /** Secret value -> `%name%` placeholder, from `variables` passed to act/observe. */
  secrets: Map<string, string>;
  ops: Set<Op>;
  operations: number;
  failed: number;
  usage: Row;
  closed: boolean;
}

interface Op {
  kind: Operation;
  span: Span;
  ctx: Context;
  session: Session;
  llmCalls: number;
  /** Text of the instruction as it appears inside a JSON-serialised prompt. */
  needle?: string;
}

const OP_KEY = createContextKey("respan.stagehand.operation");
/** Set while the instrumentation itself drives the browser (act screenshots). */
const INTERNAL_KEY = createContextKey("respan.stagehand.internal");
// Same Symbol.for keys @respan/tracing uses for its entity path: clearing them
// keeps Stagehand's native RPC spans out of exports inside a Respan workflow.
const ENTITY_NAME_KEY = createContextKey(SpanAttributes.TRACELOOP_ENTITY_NAME);
const WORKFLOW_NAME_KEY = createContextKey(SpanAttributes.TRACELOOP_WORKFLOW_NAME);

const sessionsByStagehand = new WeakMap<object, Session>();
const sessionsByRpc = new WeakMap<object, Session>();

function isZod(value: unknown): boolean {
  return typeof (value as any)?.safeParse === "function";
}

function nonNegative(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : undefined;
}

function detached(ctx: Context): Context {
  return ctx.deleteValue(ENTITY_NAME_KEY).deleteValue(WORKFLOW_NAME_KEY);
}

class Runtime {
  active = true;
  owners = 0;
  readonly restores: Array<() => void> = [];

  constructor(
    readonly captureContent: boolean,
    readonly captureScreenshots: boolean,
    readonly clientModelName: string | undefined,
  ) {}

  get tracer() {
    // Resolved per call so a provider registered after activation is used.
    return trace.getTracer(TRACER_NAME);
  }

  contentAllowed(ctx: Context = context.active()): boolean {
    return (
      this.captureContent &&
      process.env.RESPAN_TRACE_CONTENT !== "false" &&
      process.env.TRACELOOP_TRACE_CONTENT !== "false" &&
      ctx.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT) !== false
    );
  }

  patch(owner: any, key: string, wrap: (original: Fn) => Fn): void {
    const original = owner?.[key];
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

  startSpan(session: Session | undefined, name: string, logType: RespanLogType, detail: string, parent: Context, extra: Attributes = {}): Span {
    const attributes: Attributes = {
      [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType,
      [SpanAttributes.TRACELOOP_ENTITY_NAME]: name,
      [RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_DETAIL]: detail,
      ...extra,
    };
    if (session) {
      // Defaults: Respan's span processor applies the customer's propagated
      // thread / session / trace group after this, so those win.
      attributes[ATTR_SESSION_ID] = session.id;
      attributes[RespanSpanAttributes.RESPAN_THREADS_ID] = session.id;
      attributes[RespanSpanAttributes.RESPAN_SESSION_ID] = session.id;
      attributes[RespanSpanAttributes.RESPAN_TRACE_GROUP_ID] = session.id;
    }
    return this.tracer.startSpan(name, { attributes }, parent);
  }

  // -- session ----------------------------------------------------------------

  wrapCreate(original: Fn): Fn {
    const runtime = this;
    return async function create(this: unknown, input: any, ...rest: unknown[]) {
      if (!runtime.active) return original.call(this, input, ...rest);
      const parent = context.active();
      const browserSession = input?.browser?.sessionId;
      const id = typeof browserSession === "string" && browserSession ? browserSession : randomUUID();
      const model = input?.model;
      const isClientModel = typeof model?.generate === "function";
      const modelName = isClientModel ? (model.generate[CLIENT_MODEL_NAME] ?? runtime.clientModelName) : typeof model?.modelName === "string" ? model.modelName : undefined;
      const span = runtime.startSpan(undefined, SPAN_SESSION, RespanLogType.AGENT, SESSION_NAME_DETAIL, parent);
      const session: Session = {
        id,
        span,
        ctx: trace.setSpan(parent, span),
        runtime,
        modelName,
        secrets: new Map(),
        ops: new Set(),
        operations: 0,
        failed: 0,
        usage: {},
        closed: false,
      };
      for (const [key, value] of [
        [ATTR_SESSION_ID, id],
        [RespanSpanAttributes.RESPAN_THREADS_ID, id],
        [RespanSpanAttributes.RESPAN_SESSION_ID, id],
        [RespanSpanAttributes.RESPAN_TRACE_GROUP_ID, id],
      ] as const) {
        // The session span is started before the session exists; customer
        // values copied at start win over these defaults.
        if ((span as any).attributes?.[key] === undefined) span.setAttribute(key, value);
      }
      safely(() => runtime.sessionStart(session, input, isClientModel, parent));
      const createInput = isClientModel ? { ...input, model: { ...model, generate: runtime.wrapGenerate(session, model.generate) } } : input;
      let stagehand: any;
      try {
        stagehand = await context.with(detached(session.ctx), () => original.call(this, createInput, ...rest));
      } catch (error) {
        runtime.fail(session, span, error, parent);
        runtime.endSpan(session, span);
        throw error;
      }
      safely(() => runtime.register(session, stagehand));
      return stagehand;
    };
  }

  sessionStart(session: Session, input: any, isClientModel: boolean, parent: Context): void {
    const settings: Row = {
      model: isClientModel ? (session.modelName ?? "client") : input?.model?.modelName,
      browser: input?.browser?.provider,
      selfHeal: input?.selfHeal,
      cache: input?.cache,
      domSettleTimeoutMs: input?.domSettleTimeoutMs,
    };
    if (this.contentAllowed(parent) && typeof input?.systemPrompt === "string") settings.systemPrompt = input.systemPrompt;
    session.span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json(settings));
  }

  register(session: Session, stagehand: any): void {
    sessionsByStagehand.set(stagehand, session);
    // Pages and locators hold the session's RPC client, not the Stagehand.
    if (stagehand?.rpcClient) sessionsByRpc.set(stagehand.rpcClient, session);
  }

  wrapClose(original: Fn): Fn {
    const runtime = this;
    return async function close(this: any, ...args: unknown[]) {
      const session = sessionsByStagehand.get(this);
      if (!session || session.closed) return original.apply(this, args);
      try {
        return await context.with(detached(session.ctx), () => original.apply(this, args));
      } catch (error) {
        runtime.fail(session, session.span, error, session.ctx);
        throw error;
      } finally {
        if (!session.closed) {
          session.closed = true;
          safely(() =>
            session.span.setAttribute(
              SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
              json({ operations: session.operations, failed_operations: session.failed, usage: session.usage }),
            ),
          );
          runtime.endSpan(session, session.span);
        }
      }
    };
  }

  // -- act / observe / extract -------------------------------------------------

  wrapOperation(kind: Operation, original: Fn): Fn {
    const runtime = this;
    return async function operation(this: any, ...args: any[]) {
      const session = sessionsByStagehand.get(this);
      if (!runtime.active || !session || session.closed) return original.apply(this, args);
      const caller = context.active();
      const parent = trace.setSpan(caller, session.span);
      const options = (kind === "extract" && isZod(args[1]) ? args[2] : args[1]) as Row | undefined;
      safely(() => runtime.addSecrets(session, options?.variables));
      const span = runtime.startSpan(session, `${OPERATION_PREFIX}${kind}`, RespanLogType.TASK, kind, parent);
      const op: Op = { kind, span, session, ctx: trace.setSpan(parent, span), llmCalls: 0, needle: needleOf(args[0]) };
      session.ops.add(op);
      session.operations += 1;
      safely(() => runtime.operationStart(op, args, options, caller));
      try {
        const result = await context.with(detached(op.ctx).setValue(OP_KEY, op), () => original.apply(this, args));
        await safelyAsync(() => runtime.operationEnd(op, this, result, options, caller));
        return result;
      } catch (error) {
        session.failed += 1;
        runtime.fail(session, span, error, caller);
        throw error;
      } finally {
        session.ops.delete(op);
        runtime.endSpan(session, span);
      }
    };
  }

  addSecrets(session: Session, variables: unknown): void {
    if (!variables || typeof variables !== "object") return;
    for (const [name, raw] of Object.entries(variables as Row)) {
      const value = raw && typeof raw === "object" ? raw.value : raw;
      const text = value === undefined || value === null ? "" : String(value);
      // Booleans and tiny values would mask unrelated text.
      if (text.length >= 3 && typeof value !== "boolean") session.secrets.set(text, `%${name}%`);
    }
  }

  operationStart(op: Op, args: any[], options: Row | undefined, caller: Context): void {
    const page = options?.page;
    const settings: Row = {};
    if (options) {
      for (const [key, value] of Object.entries(options)) {
        if (key === "page" || value === undefined) continue;
        if (key === "variables") settings.variables = Object.keys(value ?? {});
        else if (key === "model") settings.model = typeof value?.modelName === "string" ? value.modelName : "client";
        else settings[key] = jsonable(value);
      }
    }
    if (!this.contentAllowed(caller)) {
      op.span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json({ options: settings }));
      return;
    }
    const input: Row = { instruction: jsonable(args[0]), options: settings };
    if (op.kind === "extract" && isZod(args[1])) input.schema = schemaJson(args[1]);
    if (typeof page?.ref?.url === "string") {
      input.url = page.ref.url;
      op.span.setAttribute(ATTR_PAGE_URL, page.ref.url);
    }
    op.span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json(input));
  }

  async operationEnd(op: Op, stagehand: any, result: any, options: Row | undefined, caller: Context): Promise<void> {
    const { span, session } = op;
    const metadata = result?.metadata ?? {};
    const usage = metadata.usage ?? {};
    for (const key of ["inputTokens", "outputTokens", "reasoningTokens", "cachedInputTokens", "inferenceTimeMs"]) {
      const count = nonNegative(usage[key]);
      if (count !== undefined) session.usage[key] = (session.usage[key] ?? 0) + count;
    }
    if (typeof metadata.cache?.status === "string") span.setAttribute(ATTR_CACHE_STATUS, metadata.cache.status);
    if (typeof metadata.actionId === "string") span.setAttribute(ATTR_ACTION_ID, metadata.actionId);
    // A built-in model runs inside the browser extension, out of reach; the
    // operation's aggregate usage is all Stagehand reports. Record it as one
    // content-free chat span so tokens and cost still show up.
    if (op.llmCalls === 0 && ((nonNegative(usage.inputTokens) ?? 0) > 0 || (nonNegative(usage.outputTokens) ?? 0) > 0)) {
      this.usageOnlyChat(op, usage, options);
    }
    const failed = op.kind === "act" && result?.data?.success === false;
    if (failed) {
      session.failed += 1;
      setError(span, "act reported failure", result?.data?.message, this.contentAllowed(caller), session);
    }
    const output: Row = { metadata: jsonable(metadata) };
    if (this.contentAllowed(caller)) {
      output.data = jsonable(result?.data);
      if (op.kind === "act" && this.captureScreenshots) {
        const shot = await this.screenshot(op, stagehand, options?.page);
        if (shot) output.screenshot = shot;
      }
    }
    span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, JSON.stringify(output));
  }

  usageOnlyChat(op: Op, usage: Row, options: Row | undefined): void {
    const modelName = typeof options?.model?.modelName === "string" ? options.model.modelName : op.session.modelName;
    const span = this.startSpan(op.session, SPAN_LLM, RespanLogType.CHAT, modelName ?? "llm", op.ctx, {
      [SpanAttributes.LLM_REQUEST_TYPE]: RespanLogType.CHAT,
      [ATTR_USAGE_SOURCE]: USAGE_SOURCE_OPERATION,
      ...(modelName ? { [SpanAttributes.LLM_REQUEST_MODEL]: modelName } : {}),
    });
    setUsage(span, usage);
    span.end();
  }

  async screenshot(op: Op, stagehand: any, page: any): Promise<Row | undefined> {
    const internal = detached(op.ctx).setValue(INTERNAL_KEY, true);
    const take = context.with(internal, async () => {
      const target = page ?? (await stagehand?.browser?.context?.activePage());
      if (!target) return undefined;
      const data: Uint8Array = await target.screenshot(SCREENSHOT_OPTIONS);
      return imagePart(Buffer.from(data).toString("base64"), "image/jpeg", true);
    });
    let timer: NodeJS.Timeout | undefined;
    const deadline = new Promise<undefined>((resolve) => {
      timer = setTimeout(() => resolve(undefined), SCREENSHOT_DEADLINE_MS);
      timer.unref?.();
    });
    try {
      return await Promise.race([take.catch(() => undefined), deadline]);
    } finally {
      clearTimeout(timer);
    }
  }

  // -- LLM -----------------------------------------------------------------------

  wrapGenerate(session: Session, generate: Fn): Fn {
    const runtime = this;
    return async function traced(this: unknown, params: any, ...rest: unknown[]) {
      if (!runtime.active || session.closed) return generate.call(this, params, ...rest);
      const op = findOperation(session, params);
      const parent = op?.ctx ?? session.ctx;
      if (op) op.llmCalls += 1;
      const span = runtime.startSpan(session, SPAN_LLM, RespanLogType.CHAT, session.modelName ?? "llm", parent, {
        [SpanAttributes.LLM_REQUEST_TYPE]: RespanLogType.CHAT,
        ...(session.modelName ? { [SpanAttributes.LLM_REQUEST_MODEL]: session.modelName } : {}),
      });
      safely(() => runtime.llmStart(span, params, parent));
      try {
        // Provider instrumentations honour OTel suppression; without it the
        // call this span already carries would be logged a second time.
        const result = await context.with(suppressTracing(trace.setSpan(detached(parent), span)), () => generate.call(this, params, ...rest));
        safely(() => runtime.llmEnd(span, result, parent));
        return result;
      } catch (error) {
        runtime.fail(session, span, error, parent);
        throw error;
      } finally {
        runtime.endSpan(session, span);
      }
    };
  }

  llmStart(span: Span, params: any, parent: Context): void {
    const tools = functions(params?.tools);
    if (!this.contentAllowed(parent)) return;
    const messages = chatMessages(params ?? {}, this.captureScreenshots);
    messages.forEach((message, index) => {
      const prefix = `${SpanAttributes.LLM_PROMPTS}.${index}`;
      span.setAttribute(`${prefix}.role`, message.role);
      span.setAttribute(`${prefix}.content`, contentText(message.content));
      if (message.tool_calls) span.setAttribute(`${prefix}.tool_calls`, JSON.stringify(message.tool_calls));
    });
    const input: Row = { messages };
    if (params?.responseFormat) input.response_format = jsonable(params.responseFormat);
    if (tools.length) {
      input.tools = tools;
      span.setAttribute(SpanAttributes.LLM_REQUEST_FUNCTIONS, JSON.stringify(tools));
    }
    span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, JSON.stringify(input));
  }

  llmEnd(span: Span, result: any, parent: Context): void {
    setUsage(span, result?.usage);
    if (!this.contentAllowed(parent) || !result) return;
    const message = completion(result);
    const prefix = `${SpanAttributes.LLM_COMPLETIONS}.0`;
    span.setAttribute(`${prefix}.role`, message.role);
    span.setAttribute(`${prefix}.content`, contentText(message.content));
    if (message.tool_calls) span.setAttribute(`${prefix}.tool_calls`, JSON.stringify(message.tool_calls));
    span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, JSON.stringify(message));
  }

  // -- browser actions -------------------------------------------------------------

  wrapAction(owner: "page" | "locator" | "context", method: string, original: Fn): Fn {
    const runtime = this;
    const name = `${owner}.${method}`;
    return async function action(this: any, ...args: any[]) {
      const session = this?.rpcClient ? sessionsByRpc.get(this.rpcClient) : undefined;
      const caller = context.active();
      if (!runtime.active || !session || session.closed || caller.getValue(INTERNAL_KEY)) return original.apply(this, args);
      const op = caller.getValue(OP_KEY) as Op | undefined;
      const parent = trace.setSpan(caller, op?.span ?? session.span);
      const span = runtime.startSpan(session, name, RespanLogType.TOOL, `${owner}_${method}`, parent);
      const allowed = runtime.contentAllowed(caller);
      // Nested public calls (e.g. a locator helper using its page) are part of this action.
      const internal = detached(trace.setSpan(parent, span)).setValue(INTERNAL_KEY, true);
      if (allowed) await safelyAsync(() => runtime.actionStart(span, owner, method, this, args, internal));
      try {
        const result = await context.with(internal, () => original.apply(this, args));
        if (allowed) safely(() => runtime.actionEnd(span, owner, method, this, args, result));
        return result;
      } catch (error) {
        runtime.fail(session, span, error, caller);
        throw error;
      } finally {
        runtime.endSpan(session, span);
      }
    };
  }

  async actionStart(span: Span, owner: string, method: string, target: any, args: any[], internal: Context): Promise<void> {
    const input = actionInput(owner, method, target, args);
    if (TYPED_TEXT.has(`${owner}.${method}`) && input.text !== REDACTED && (await context.with(internal, () => this.typesIntoSecret(owner, target)))) {
      input.text = REDACTED;
    }
    span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json(input));
  }

  /**
   * Whether text typed by this action lands in a credential field. Selectors
   * from observe() are XPaths that say nothing about the field, so the page is
   * asked for the target's type / name / id / autocomplete. Unknown = secret.
   */
  async typesIntoSecret(owner: string, target: any): Promise<boolean> {
    const Page = shared?.module.Page;
    const pageId = owner === "locator" ? target?.descriptor?.pageId : target?.pageId;
    if (typeof Page !== "function" || typeof pageId !== "string") return true;
    const selector = owner === "locator" ? target?.descriptor?.selector : null;
    try {
      const described = await new Page(target.rpcClient, { pageId }).evaluate(describeField(selector));
      return typeof described !== "string" || described === "" || SENSITIVE_FIELD.test(described);
    } catch {
      return true;
    }
  }

  actionEnd(span: Span, owner: string, method: string, target: any, args: any[], result: any): void {
    const url = owner === "page" ? target?.ref?.url : result?.ref?.url;
    if (typeof url === "string") span.setAttribute(ATTR_PAGE_URL, url);
    let output: unknown;
    if (method === "goto" || method === "reload" || method === "goBack" || method === "goForward") {
      output = { url, status: typeof result?.status === "function" ? result.status() : null };
    } else if (method === "screenshot" && result instanceof Uint8Array) {
      output = imagePart(Buffer.from(result).toString("base64"), args?.[0]?.type === "jpeg" ? "image/jpeg" : "image/png", this.captureScreenshots);
    } else if (method === "newPage") {
      output = { pageId: result?.pageId, url };
    } else if (result !== undefined && !(result instanceof Uint8Array)) {
      output = result;
    }
    if (output !== undefined) span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json(output));
  }

  // -- shared ------------------------------------------------------------------------

  fail(session: Session, span: Span, error: unknown, ctx: Context): void {
    const type = (error as any)?.name || typeof error;
    const allowed = this.contentAllowed(ctx);
    setError(span, String(type), (error as any)?.message ?? error, allowed, session);
    // Provider and validation errors echo prompts; only the type goes out without content.
    span.addEvent("exception", allowed ? { "exception.type": String(type), "exception.message": mask(session, String((error as any)?.message ?? error)) } : { "exception.type": String(type) });
  }

  endSpan(session: Session, span: Span): void {
    safely(() => flattenMetadata(span));
    if (session.secrets.size) safely(() => maskSpan(session, span));
    span.end();
  }
}

function needleOf(instruction: unknown): string | undefined {
  const text = typeof instruction === "string" ? instruction : (instruction as Row | undefined)?.description;
  return typeof text === "string" && text.trim() ? JSON.stringify(text.trim()).slice(1, -1) : undefined;
}

/**
 * The operation an `llm.generate` call belongs to. The extension's callback
 * carries no trace context, so with several operations running at once on one
 * session the prompt is matched against each operation's instruction.
 */
function findOperation(session: Session, params: any): Op | undefined {
  const ops = [...session.ops];
  if (ops.length <= 1) return ops[0];
  const prompt = JSON.stringify([params?.systemPrompt, params?.messages]);
  const matches = ops.filter((op) => op.needle && prompt.includes(op.needle));
  return (matches.length ? matches : ops).pop();
}

const TYPED_TEXT = new Set(["locator.fill", "locator.type", "page.type"]);

/** Page-side expression describing the field a selector (or, for null, the focus) points at. */
function describeField(selector: string | null): string {
  return `(() => {
    const selector = ${JSON.stringify(selector)};
    try {
      const el = selector === null
        ? document.activeElement
        : selector.startsWith("xpath=")
          ? document.evaluate(selector.slice(6), document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue
          : document.querySelector(selector);
      if (!el) return "";
      return ["type=" + (el.type || ""), el.name, el.id, el.autocomplete, el.getAttribute("aria-label")].join(" ");
    } catch (error) {
      return "";
    }
  })()`;
}

function actionInput(owner: string, method: string, target: any, args: any[]): Row {
  const input: Row = {};
  if (owner === "locator") input.selector = target?.descriptor?.selector;
  if (owner === "page" && typeof target?.ref?.url === "string") input.url = target.ref.url;
  const selector = String(input.selector ?? "");
  const sensitive = SENSITIVE_SELECTOR.test(selector);
  switch (`${owner}.${method}`) {
    case "locator.fill":
    case "locator.type":
    case "page.type":
      // Typed text is often a credential; a field that looks like one is never exported.
      input.text = sensitive ? REDACTED : args[0];
      break;
    case "locator.setInputFiles":
      input.files = jsonable(args[0]);
      break;
    default:
      input.arguments = args.map((arg) => (arg && typeof arg === "object" && arg.descriptor ? { selector: arg.descriptor.selector } : arg));
  }
  if (owner === "context" && method === "setActivePage") input.arguments = [args[0]?.pageId];
  return input;
}

function schemaJson(schema: any): unknown {
  try {
    const toJSONSchema = schema?.toJSONSchema ?? schema?.constructor?.prototype?.toJSONSchema;
    return typeof toJSONSchema === "function" ? jsonable(toJSONSchema.call(schema)) : { type: "zod" };
  } catch {
    return { type: "zod" };
  }
}

function setUsage(span: Span, usage: any): void {
  if (!usage || typeof usage !== "object") return;
  const input = nonNegative(usage.inputTokens);
  const output = nonNegative(usage.outputTokens);
  const total = nonNegative(usage.totalTokens) ?? (input !== undefined && output !== undefined ? input + output : undefined);
  const cached = nonNegative(usage.cachedInputTokens);
  const reasoning = nonNegative(usage.reasoningTokens);
  if (input !== undefined) span.setAttribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input);
  if (output !== undefined) span.setAttribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output);
  if (total !== undefined) span.setAttribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total);
  if (cached) {
    span.setAttribute(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cached);
    span.setAttribute(LLM_USAGE_CACHE_READ_INPUT_TOKENS, cached);
  }
  if (reasoning) span.setAttribute(GEN_AI_USAGE_REASONING_TOKENS, reasoning);
}

/**
 * Customer metadata as one flat `respan.metadata.<key>` attribute per key, the
 * shape Respan ingestion reads. Some @respan/tracing versions fold propagated
 * metadata into a single JSON `respan.metadata` attribute when the span starts,
 * which ingestion drops; expand it before the span ends.
 */
function flattenMetadata(span: Span): void {
  const attributes = (span as any).attributes as Record<string, unknown> | undefined;
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
    if (attributes![flat] === undefined && value !== undefined && value !== null)
      span.setAttribute(flat, typeof value === "string" || typeof value === "number" || typeof value === "boolean" ? value : JSON.stringify(value));
  }
  delete attributes![RespanSpanAttributes.RESPAN_METADATA];
}

function mask(session: Session, text: string): string {
  for (const [secret, placeholder] of session.secrets) text = text.split(secret).join(placeholder);
  return text;
}

function maskSpan(session: Session, span: Span): void {
  const attributes = (span as any).attributes as Attributes | undefined;
  for (const [key, value] of Object.entries(attributes ?? {})) {
    if (typeof value === "string") {
      const masked = mask(session, value);
      if (masked !== value) span.setAttribute(key, masked);
    }
  }
  const status = (span as any).status;
  if (typeof status?.message === "string") span.setStatus({ code: status.code, message: mask(session, status.message) });
  for (const event of ((span as any).events ?? []) as Array<{ attributes?: Attributes }>) {
    for (const [key, value] of Object.entries(event.attributes ?? {})) {
      if (typeof value === "string") (event.attributes as Row)[key] = mask(session, value);
    }
  }
}

/** ERROR status whose description is what Respan shows as the error message. */
function setError(span: Span, summary: string, detail: unknown, withMessage: boolean, session: Session): void {
  const message = withMessage ? mask(session, String(detail ?? "")).trim() : "";
  span.setStatus({ code: SpanStatusCode.ERROR, message: message ? `${summary}: ${message.slice(0, ERROR_MESSAGE_LIMIT)}` : summary });
}

function safely(fn: () => void): void {
  try {
    fn();
  } catch {
    // Telemetry must never break the customer's browser flow.
  }
}

async function safelyAsync(fn: () => Promise<void>): Promise<void> {
  try {
    await fn();
  } catch {
    // Telemetry must never break the customer's browser flow.
  }
}

const CLIENT_MODEL_NAME = Symbol.for("respan.stagehand.client_model_name");

/**
 * Name the model a client model's `generate` calls, so its chat spans report
 * it (and Respan can price them). Returns `generate` itself; Stagehand is
 * unaffected. Takes precedence over the instrumentor's `clientModelName`.
 *
 * ```ts
 * Stagehand.create({ browser, model: { generate: withModelName(generate, "gpt-4.1-mini") } });
 * ```
 */
export function withModelName<T extends Fn>(generate: T, modelName: string): T {
  (generate as any)[CLIENT_MODEL_NAME] = modelName;
  return generate;
}

let shared: { runtime: Runtime; module: StagehandModule } | undefined;

/** Trace Stagehand v4 sessions: operations, LLM calls and browser actions. */
export class StagehandInstrumentor {
  public readonly name = INSTRUMENTATION_NAME;
  private readonly options: StagehandInstrumentorOptions;
  private runtime?: Runtime;

  constructor(options: StagehandInstrumentorOptions = {}) {
    this.options = options;
  }

  async activate(): Promise<void> {
    if (this.runtime) return;
    const { captureContent = true, captureScreenshots = true, clientModelName } = this.options;
    if (shared) {
      const { runtime } = shared;
      if (runtime.captureContent !== captureContent || runtime.captureScreenshots !== captureScreenshots || runtime.clientModelName !== clientModelName) {
        throw new Error("Stagehand instrumentation is already active with different settings");
      }
      runtime.owners += 1;
      this.runtime = runtime;
      return;
    }
    let module = this.options.stagehandModule;
    if (!module) {
      try {
        module = (await import(STAGEHAND_MODULE)) as StagehandModule;
      } catch {
        return; // Stagehand is not installed: nothing to trace.
      }
    }
    const { Stagehand, Page, Locator, BrowserContext } = module;
    if (typeof Stagehand?.create !== "function") throw new Error("StagehandInstrumentor requires Stagehand v4 (Stagehand.create).");
    const runtime = new Runtime(captureContent, captureScreenshots, clientModelName);
    try {
      runtime.patch(Stagehand, "create", (original) => runtime.wrapCreate(original));
      for (const kind of OPERATIONS) runtime.patch(Stagehand.prototype, kind, (original) => runtime.wrapOperation(kind, original));
      runtime.patch(Stagehand.prototype, "close", (original) => runtime.wrapClose(original));
      for (const method of PAGE_ACTIONS) runtime.patch(Page?.prototype, method, (original) => runtime.wrapAction("page", method, original));
      for (const method of LOCATOR_ACTIONS) runtime.patch(Locator?.prototype, method, (original) => runtime.wrapAction("locator", method, original));
      for (const method of CONTEXT_ACTIONS) runtime.patch(BrowserContext?.prototype, method, (original) => runtime.wrapAction("context", method, original));
    } catch (error) {
      runtime.restore();
      throw error;
    }
    runtime.owners = 1;
    shared = { runtime, module };
    this.runtime = runtime;
  }

  deactivate(): void {
    const runtime = this.runtime;
    this.runtime = undefined;
    if (!runtime) return;
    runtime.owners -= 1;
    if (runtime.owners > 0) return;
    runtime.active = false;
    runtime.restore();
    if (shared?.runtime === runtime) shared = undefined;
  }

  isActive(): boolean {
    return Boolean(this.runtime);
  }
}
