/** Native Respan instrumentation for exa-js. */

import {
  SpanKind,
  SpanStatusCode,
  context,
  trace,
  type Context,
  type Span,
} from "@opentelemetry/api";
import { ATTR_ERROR_MESSAGE } from "@opentelemetry/semantic-conventions/incubating";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { AsyncLocalStorage } from "node:async_hooks";
import {
  EXA_INSTRUMENTATION_NAME,
  EXA_INSTRUMENTATION_SCOPE,
  PACKAGE_VERSION,
  STATUS_CODE_ATTR,
  type OperationConfig,
} from "./_constants.js";
import { safeJson, typeName } from "./_serialization.js";
import {
  CANONICAL_ATTRS,
  buildStartAttributes,
  buildSuccessAttributes,
  streamResult,
} from "./_translator.js";

type AnyFunction = (...args: unknown[]) => unknown;
type PatchableConstructor = { prototype?: Record<string, unknown> };
type PatchTarget = {
  methods: Record<string, OperationConfig>;
  prototype: Record<string, unknown>;
};

interface ExaSDKModule {
  default?: PatchableConstructor;
  Exa?: PatchableConstructor;
  AgentRunsClient?: PatchableConstructor;
  AgentBetaRunsClient?: PatchableConstructor;
  AgentRunEventsClient?: PatchableConstructor;
  AgentBetaRunEventsClient?: PatchableConstructor;
  ResearchClient?: PatchableConstructor;
}

export interface ExaInstrumentorOptions {
  captureContent?: boolean;
  sdkModule?: ExaSDKModule;
}

interface ClassSpec {
  exportName: keyof ExaSDKModule;
  methods: Record<string, OperationConfig>;
}

interface AppliedPatch {
  owner: Record<string, unknown>;
  name: string;
  original: AnyFunction;
  replacement: AnyFunction;
}

function op(
  entityName: string,
  family: OperationConfig["family"],
  operation: string,
  options: Partial<OperationConfig> = {},
): OperationConfig {
  return { entityName, family, operation, ...options };
}

const CORE_METHODS: Record<string, OperationConfig> = {
  search: op("search", "tool", "search"),
  streamSearch: op("search", "tool", "streamSearch", {
    alwaysStreaming: true,
  }),
  searchAndContents: op(
    "search_and_contents",
    "tool",
    "searchAndContents",
  ),
  getContents: op("get_contents", "tool", "getContents"),
  findSimilar: op("find_similar", "tool", "findSimilar"),
  findSimilarAndContents: op(
    "find_similar_and_contents",
    "tool",
    "findSimilarAndContents",
  ),
  answer: op("answer", "chat", "answer"),
  streamAnswer: op("answer", "chat", "streamAnswer", {
    alwaysStreaming: true,
  }),
};

const AGENT_METHODS: Record<string, OperationConfig> = {
  create: op("run", "agent", "agent.runs.create", {
    streamFlag: "stream",
  }),
  get: op("run.get", "task", "agent.runs.get"),
  list: op("run.list", "task", "agent.runs.list"),
  cancel: op("run.cancel", "task", "agent.runs.cancel"),
  stop: op("run.stop", "task", "agent.runs.stop"),
  delete: op("run.delete", "task", "agent.runs.delete"),
  pollUntilFinished: op(
    "run",
    "agent",
    "agent.runs.pollUntilFinished",
  ),
  createAndWait: op("run", "agent", "agent.runs.createAndWait"),
};

const RESEARCH_METHODS: Record<string, OperationConfig> = {
  create: op("research", "agent", "research.create", {
    legacyResearch: true,
  }),
  get: op("research.get", "task", "research.get", {
    streamFlag: "stream",
    streamFamily: "agent",
    legacyResearch: true,
  }),
  list: op("research.list", "task", "research.list", {
    legacyResearch: true,
  }),
  pollUntilFinished: op(
    "research",
    "agent",
    "research.pollUntilFinished",
    { legacyResearch: true },
  ),
};

const CLASS_SPECS: ClassSpec[] = [
  { exportName: "Exa", methods: CORE_METHODS },
  { exportName: "AgentRunsClient", methods: AGENT_METHODS },
  { exportName: "AgentBetaRunsClient", methods: AGENT_METHODS },
  {
    exportName: "AgentRunEventsClient",
    methods: {
      list: op("run.events.list", "task", "agent.runs.events.list"),
    },
  },
  {
    exportName: "AgentBetaRunEventsClient",
    methods: {
      list: op("run.events.list", "task", "agent.runs.events.list"),
    },
  },
  { exportName: "ResearchClient", methods: RESEARCH_METHODS },
];

function nestedTargets(sdk: ExaSDKModule): PatchTarget[] {
  const ExaConstructor = sdk.Exa ?? sdk.default;
  if (typeof ExaConstructor !== "function") return [];
  let client: Record<string, unknown>;
  try {
    const ConstructableExa = ExaConstructor as unknown as new (
      apiKey: string,
    ) => Record<string, unknown>;
    client = new ConstructableExa("respan-instrumentation-discovery");
  } catch {
    return [];
  }

  const agent = isPlainRecord(client.agent) ? client.agent : undefined;
  const runs = agent && isPlainRecord(agent.runs) ? agent.runs : undefined;
  const beta = isPlainRecord(client.beta) ? client.beta : undefined;
  const betaAgent = beta && isPlainRecord(beta.agent) ? beta.agent : undefined;
  const betaRuns =
    betaAgent && isPlainRecord(betaAgent.runs) ? betaAgent.runs : undefined;
  const research = isPlainRecord(client.research) ? client.research : undefined;
  const candidates: Array<[
    Record<string, unknown> | undefined,
    Record<string, OperationConfig>,
  ]> = [
    [runs, AGENT_METHODS],
    [runs && isPlainRecord(runs.events) ? runs.events : undefined, {
      list: op("run.events.list", "task", "agent.runs.events.list"),
    }],
    [betaRuns, AGENT_METHODS],
    [betaRuns && isPlainRecord(betaRuns.events) ? betaRuns.events : undefined, {
      list: op("run.events.list", "task", "agent.runs.events.list"),
    }],
    [research, RESEARCH_METHODS],
  ];
  const targets: PatchTarget[] = [];
  for (const [instance, methods] of candidates) {
    const prototype = instance && Object.getPrototypeOf(instance);
    if (prototype && typeof prototype === "object") {
      targets.push({ prototype: prototype as Record<string, unknown>, methods });
    }
  }
  return targets;
}

const SHARED_STATE: {
  activeCount: number;
  captureContent?: boolean;
  enabled: boolean;
  patches: AppliedPatch[];
  sdkModule?: ExaSDKModule;
} = {
  activeCount: 0,
  enabled: false,
  patches: [],
};

const ACTIVE_DEPTH = new AsyncLocalStorage<number>();

function withinPatchedCall<T>(fn: () => T): T {
  return ACTIVE_DEPTH.run((ACTIVE_DEPTH.getStore() ?? 0) + 1, fn);
}

function captureFromEnvironment(): boolean {
  return !new Set(["0", "false", "no", "off"]).has(
    (process.env.TRACELOOP_TRACE_CONTENT ?? "true").trim().toLowerCase(),
  );
}

function hasParent(activeContext: Context): boolean {
  const spanContext = trace.getSpan(activeContext)?.spanContext();
  return Boolean(spanContext && trace.isSpanContextValid(spanContext));
}

function normalizeInput(
  config: OperationConfig,
  args: unknown[],
): Record<string, unknown> {
  const first = args[0];
  const second = args[1];
  if (["search", "streamSearch", "searchAndContents", "answer", "streamAnswer"].includes(config.operation)) {
    return {
      query: first,
      ...(isPlainRecord(second) ? second : {}),
    };
  }
  if (["getContents"].includes(config.operation)) {
    return {
      urls: first,
      ...(isPlainRecord(second) ? second : {}),
    };
  }
  if (["findSimilar", "findSimilarAndContents"].includes(config.operation)) {
    return {
      url: first,
      ...(isPlainRecord(second) ? second : {}),
    };
  }
  if (
    config.operation === "agent.runs.create" ||
    config.operation === "agent.runs.createAndWait" ||
    config.operation === "research.create"
  ) {
    const options = isPlainRecord(second) ? second : {};
    return isPlainRecord(first)
      ? { ...first, ...options }
      : { input: first, ...options };
  }
  if (config.operation === "agent.runs.list" || config.operation === "research.list") {
    return isPlainRecord(first) ? { ...first } : { input: first };
  }
  if (config.operation.startsWith("agent.runs.")) {
    return {
      runId: first,
      ...(isPlainRecord(second) ? second : {}),
    };
  }
  if (config.operation.startsWith("research.")) {
    return {
      researchId: first,
      ...(isPlainRecord(second) ? second : {}),
    };
  }
  return { args };
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function isStreaming(
  config: OperationConfig,
  input: Record<string, unknown>,
): boolean {
  if (config.alwaysStreaming) return true;
  return config.streamFlag ? Boolean(input[config.streamFlag]) : false;
}

function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  return Boolean(
    value &&
      (typeof value === "object" || typeof value === "function") &&
      typeof (value as PromiseLike<unknown>).then === "function",
  );
}

function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return Boolean(
    value &&
      typeof value === "object" &&
      typeof (value as AsyncIterable<unknown>)[Symbol.asyncIterator] === "function",
  );
}

function setAttributes(span: Span, attributes: Record<string, string | number | boolean>): void {
  for (const [key, value] of Object.entries(attributes)) {
    span.setAttribute(key, value);
  }
}

function metadataObject(value: unknown): Record<string, unknown> {
  if (isPlainRecord(value)) return value;
  if (typeof value !== "string") return {};
  try {
    const parsed: unknown = JSON.parse(value);
    return isPlainRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function preserveExistingMetadata(
  span: Span,
  attributes: Record<string, string | number | boolean>,
): void {
  const key = RespanSpanAttributes.RESPAN_METADATA;
  const update = attributes[key];
  if (update === undefined) return;
  const readable = span as unknown as {
    attributes?: Record<string, unknown>;
  };
  attributes[key] = JSON.stringify({
    ...metadataObject(readable.attributes?.[key]),
    ...metadataObject(update),
  });
}

function finishSuccess(params: {
  span: Span;
  config: OperationConfig;
  input: Record<string, unknown>;
  result: unknown;
  streaming: boolean;
  streamCompleted?: boolean;
}): void {
  const attributes = buildSuccessAttributes({
    config: params.config,
    input: params.input,
    result: params.result,
    captureContent: Boolean(SHARED_STATE.captureContent),
    streaming: params.streaming,
    streamCompleted: params.streamCompleted,
  });
  preserveExistingMetadata(params.span, attributes);
  setAttributes(params.span, attributes);
  params.span.setAttribute(STATUS_CODE_ATTR, 200);
  params.span.setStatus({ code: SpanStatusCode.OK });
  params.span.end();
}

function finishError(span: Span, error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  if (error instanceof Error) span.recordException(error);
  span.setAttribute(ATTR_ERROR_MESSAGE, message);
  span.setAttribute(STATUS_CODE_ATTR, errorStatusCode(error));
  if (SHARED_STATE.captureContent) {
    span.setAttribute(
      CANONICAL_ATTRS.entityOutput,
      safeJson({ error: typeName(error), message }),
    );
  }
  span.setStatus({ code: SpanStatusCode.ERROR, message });
  span.end();
}

function errorStatusCode(error: unknown): number {
  const candidate = isPlainRecord(error) ? error : undefined;
  const response = candidate && isPlainRecord(candidate.response)
    ? candidate.response
    : undefined;
  for (const value of [
    candidate?.statusCode,
    candidate?.status_code,
    candidate?.status,
    candidate?.code,
    response?.statusCode,
    response?.status_code,
    response?.status,
  ]) {
    const statusCode = typeof value === "number" ? value : Number(value);
    if (Number.isInteger(statusCode) && statusCode >= 400 && statusCode <= 599) {
      return statusCode;
    }
  }
  const match = String(error).match(
    /\b(?:status(?:\s+code)?|http)\D{0,16}([45]\d{2})\b/i,
  );
  return match ? Number(match[1]) : 500;
}

function wrapAsyncIterable(params: {
  source: AsyncIterable<unknown>;
  spanContext: Context;
  span: Span;
  config: OperationConfig;
  input: Record<string, unknown>;
}): AsyncGenerator<unknown> {
  const iterator = params.source[Symbol.asyncIterator]();
  return (async function* tracedExaStream() {
    const chunks: unknown[] = [];
    let completed = false;
    let failed = false;
    try {
      while (true) {
        const next = await context.with(params.spanContext, () =>
          withinPatchedCall(() => iterator.next()),
        );
        if (next.done) {
          completed = true;
          break;
        }
        chunks.push(next.value);
        yield next.value;
      }
    } catch (error) {
      failed = true;
      finishError(params.span, error);
      throw error;
    } finally {
      if (!completed && !failed && typeof iterator.return === "function") {
        try {
          await context.with(params.spanContext, () =>
            withinPatchedCall(() => iterator.return?.()),
          );
        } catch (error) {
          failed = true;
          finishError(params.span, error);
          throw error;
        }
      }
      if (!failed) {
        finishSuccess({
          span: params.span,
          config: params.config,
          input: params.input,
          result: streamResult(chunks),
          streaming: true,
          streamCompleted: completed,
        });
      }
    }
  })();
}

function makeWrapper(original: AnyFunction, config: OperationConfig): AnyFunction {
  return function instrumentedExaMethod(
    this: unknown,
    ...args: unknown[]
  ): unknown {
    if (!SHARED_STATE.enabled || (ACTIVE_DEPTH.getStore() ?? 0) > 0) {
      return original.apply(this, args);
    }
    const input = normalizeInput(config, args);
    const streaming = isStreaming(config, input);
    const activeContext = context.active();
    const tracer = trace.getTracer(EXA_INSTRUMENTATION_SCOPE, PACKAGE_VERSION);
    const span = tracer.startSpan(
      config.entityName,
      {
        kind: SpanKind.CLIENT,
        attributes: buildStartAttributes({
          config,
          input,
          captureContent: Boolean(SHARED_STATE.captureContent),
          streaming,
          hasParent: hasParent(activeContext),
        }),
      },
      activeContext,
    );
    const spanContext = trace.setSpan(activeContext, span);

    let result: unknown;
    try {
      result = context.with(spanContext, () =>
        withinPatchedCall(() => original.apply(this, args)),
      );
    } catch (error) {
      finishError(span, error);
      throw error;
    }

    const handleResolved = (value: unknown): unknown => {
      if (streaming) {
        if (!isAsyncIterable(value)) {
          finishError(
            span,
            new TypeError(
              `${config.operation} returned non-async-iterable ${typeName(value)}`,
            ),
          );
          return value;
        }
        return wrapAsyncIterable({
          source: value,
          spanContext,
          span,
          config,
          input,
        });
      }
      finishSuccess({ span, config, input, result: value, streaming: false });
      return value;
    };

    if (isPromiseLike(result)) {
      return Promise.resolve(result).then(handleResolved, (error) => {
        finishError(span, error);
        throw error;
      });
    }
    return handleResolved(result);
  };
}

export class ExaInstrumentor {
  public readonly name = EXA_INSTRUMENTATION_NAME;

  private readonly captureContent: boolean;
  private readonly sdkModule?: ExaSDKModule;
  private active = false;

  constructor(options: ExaInstrumentorOptions = {}) {
    this.captureContent = options.captureContent ?? captureFromEnvironment();
    this.sdkModule = options.sdkModule;
  }

  async activate(): Promise<void> {
    if (this.active) return;
    if (SHARED_STATE.activeCount > 0) {
      if (SHARED_STATE.captureContent !== this.captureContent) {
        throw new Error(
          "all active ExaInstrumentor instances must use the same captureContent setting",
        );
      }
      SHARED_STATE.activeCount += 1;
      this.active = true;
      return;
    }

    const sdk =
      this.sdkModule ??
      ((await import("exa-js")) as unknown as ExaSDKModule);
    if (!sdk.Exa && sdk.default) sdk.Exa = sdk.default;
    const applied: AppliedPatch[] = [];
    const targets: PatchTarget[] = [];
    for (const spec of CLASS_SPECS) {
      const prototype = sdk[spec.exportName]?.prototype;
      if (prototype) targets.push({ prototype, methods: spec.methods });
    }
    targets.push(...nestedTargets(sdk));
    const seen = new WeakMap<Record<string, unknown>, Set<string>>();
    try {
      for (const target of targets) {
        let methodNames = seen.get(target.prototype);
        if (!methodNames) {
          methodNames = new Set<string>();
          seen.set(target.prototype, methodNames);
        }
        for (const [methodName, config] of Object.entries(target.methods)) {
          if (methodNames.has(methodName)) continue;
          methodNames.add(methodName);
          const prototype = target.prototype;
          if (!Object.prototype.hasOwnProperty.call(prototype, methodName)) continue;
          const original = prototype[methodName];
          if (typeof original !== "function") continue;
          const replacement = makeWrapper(original as AnyFunction, config);
          prototype[methodName] = replacement;
          applied.push({
            owner: prototype,
            name: methodName,
            original: original as AnyFunction,
            replacement,
          });
        }
      }
    } catch (error) {
      for (const patch of applied.reverse()) {
        if (patch.owner[patch.name] === patch.replacement) {
          patch.owner[patch.name] = patch.original;
        }
      }
      throw error;
    }

    if (applied.length === 0) {
      throw new Error("ExaInstrumentor found no supported exa-js methods");
    }
    SHARED_STATE.patches = applied;
    SHARED_STATE.sdkModule = sdk;
    SHARED_STATE.captureContent = this.captureContent;
    SHARED_STATE.enabled = true;
    SHARED_STATE.activeCount = 1;
    this.active = true;
  }

  deactivate(): void {
    if (!this.active) return;
    this.active = false;
    SHARED_STATE.activeCount = Math.max(0, SHARED_STATE.activeCount - 1);
    if (SHARED_STATE.activeCount > 0) return;
    SHARED_STATE.enabled = false;
    for (const patch of [...SHARED_STATE.patches].reverse()) {
      if (patch.owner[patch.name] === patch.replacement) {
        patch.owner[patch.name] = patch.original;
      }
    }
    SHARED_STATE.patches = [];
    SHARED_STATE.sdkModule = undefined;
    SHARED_STATE.captureContent = undefined;
  }

  isActive(): boolean {
    return this.active;
  }
}
