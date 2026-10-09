import { inspect, types } from "node:util";
import { Message } from "beeai-framework/backend/message";
import {
  context,
  createContextKey,
  trace,
  SpanStatusCode,
  type Context,
  type Span,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import {
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";

type Data = Record<string, any>;
type Method = (...args: any[]) => any;
interface PatchEntry {
  prototype: Data;
  name: string;
  original: PropertyDescriptor;
  wrapped: Method;
}
interface Patch {
  owners: number;
  contentDisabledOwners: number;
  methods: PatchEntry[];
}
interface Capture {
  span: Span;
  allow: boolean;
  type: RespanLogType;
  definitions: Data[];
}
const patches = new WeakMap<object, Patch>();
const captureKey = createContextKey("respan.beeai.native.capture");
const contentVetoKey = createContextKey("respan.beeai.native.content-veto");
const observedContentVetoes = new WeakSet<object>();
interface ReadableGuard {
  denied: boolean;
  attributes: Data;
  events: any[];
  status: Data;
  refresh: () => void;
}
const readableGuards = new WeakMap<object, ReadableGuard>();
const lmSuppressionKey = createContextKey(
  "suppress_language_model_instrumentation",
);

// Only inspect data descriptors. Telemetry must never call application getters,
// toJSON, iterators, schema factories, or output/string conversion methods.
function own(value: unknown, key: string): any {
  if (
    (typeof value !== "object" || value === null) &&
    typeof value !== "function"
  )
    return undefined;
  if (types.isProxy(value)) return undefined;
  const descriptor = Object.getOwnPropertyDescriptor(value, key);
  return descriptor && "value" in descriptor ? descriptor.value : undefined;
}
function snapshot(
  value: unknown,
  ancestors = new WeakSet<object>(),
  sdkContent = false,
): any {
  if (value === null || ["string", "boolean"].includes(typeof value))
    return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "bigint") return value.toString();
  if (typeof value !== "object") return undefined;
  if (types.isProxy(value)) {
    if (!sdkContent) return "[Proxy]";
    // Node's inspect uses native proxy details without executing proxy traps.
    // Probe only the target shape (no payload inspection or string conversion).
    const shape = inspect(value, {
      showProxy: true,
      showHidden: true,
      customInspect: false,
      getters: false,
      depth: 1,
      maxArrayLength: 0,
      maxStringLength: 0,
      compact: true,
      breakLength: Infinity,
    });
    if (
      !shape.startsWith("Proxy [ [") ||
      !shape.endsWith(", { set: [Function], deleteProperty: [Function] } ]")
    )
      return "[Proxy]";
  }
  if (ancestors.has(value)) return "[Circular]";
  ancestors.add(value);
  try {
    if (types.isDate(value)) {
      const time = Date.prototype.getTime.call(value);
      return Number.isFinite(time)
        ? Date.prototype.toISOString.call(value)
        : null;
    }
    if (value instanceof URL) return URL.prototype.toString.call(value);
    if (types.isMap(value)) {
      const output: unknown[] = [];
      const iterator = Map.prototype.entries.call(value);
      for (let entry = iterator.next(); !entry.done; entry = iterator.next())
        output.push(snapshot(entry.value, ancestors));
      return { entries: output };
    }
    if (types.isSet(value)) {
      const output: unknown[] = [];
      const iterator = Set.prototype.values.call(value);
      for (let entry = iterator.next(); !entry.done; entry = iterator.next())
        output.push(snapshot(entry.value, ancestors));
      return { values: output };
    }
    if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
      const output: unknown[] = [];
      const descriptors = Object.getOwnPropertyDescriptors(value);
      for (const key of Object.keys(descriptors)) {
        if (/^\d+$/.test(key) && "value" in descriptors[key])
          output.push(snapshot(descriptors[key].value, ancestors));
      }
      return output;
    }
    const output: any = Array.isArray(value) ? [] : Object.create(null);
    const descriptors = Object.getOwnPropertyDescriptors(value);
    for (const key of Object.keys(descriptors)) {
      const descriptor = descriptors[key];
      if (!descriptor.enumerable || !("value" in descriptor)) continue;
      // BeeAI Message.content is a framework-owned watchArray proxy. Its
      // released handler has only set/delete traps; retain these native values.
      // The trap-free shape probe above rejects caller proxy backing arrays.
      const nativeContent = key === "content" && value instanceof Message;
      const item = snapshot(descriptor.value, ancestors, nativeContent);
      if (item !== undefined)
        Object.defineProperty(output, key, {
          value: item,
          enumerable: true,
          configurable: true,
          writable: true,
        });
    }
    return output;
  } finally {
    ancestors.delete(value);
  }
}
function json(value: unknown): string | undefined {
  const safe = snapshot(value);
  return safe === undefined ? undefined : JSON.stringify(safe);
}
function allowed(ctx: Context): boolean {
  const parent = trace.getSpan(ctx);
  const guard = parent && readableGuards.get(parent);
  guard?.refresh();
  if (
    parent &&
    (guard?.denied ||
      own(
        guard?.attributes ?? own(parent, "attributes"),
        "allow_trace_content",
      ) === false)
  )
    observedContentVetoes.add(parent);
  return (
    process.env.RESPAN_TRACE_CONTENT !== "false" &&
    process.env.TRACELOOP_TRACE_CONTENT !== "false" &&
    !(parent && observedContentVetoes.has(parent)) &&
    ctx.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT) !== false &&
    ctx.getValue(contentVetoKey) !== true
  );
}
function suppressed(ctx: Context, type: RespanLogType): boolean {
  return (
    isTracingSuppressed(ctx) ||
    ([RespanLogType.CHAT, RespanLogType.EMBEDDING].includes(type) &&
      ctx.getValue(lmSuppressionKey) === true)
  );
}
function attempt(fn: () => void): void {
  try {
    fn();
  } catch {
    /* Telemetry is observational. */
  }
}
function setJson(span: Span, key: string, value: unknown): void {
  const serialized = json(value);
  if (serialized !== undefined) span.setAttribute(key, serialized);
}
function className(instance: unknown): string | undefined {
  const prototype =
    instance && typeof instance === "object"
      ? Object.getPrototypeOf(instance)
      : undefined;
  const name = own(own(prototype, "constructor"), "name");
  return typeof name === "string" ? name : undefined;
}
function identity(instance: Data): { model?: string; provider?: string } {
  const model = own(instance, "model");
  const modelId = own(instance, "modelId") ?? own(model, "modelId");
  const provider = own(model, "provider");
  const namespace = own(own(instance, "emitter"), "namespace");
  const emitterProvider = Array.isArray(namespace)
    ? own(namespace, "1")
    : undefined;
  return {
    ...(typeof modelId === "string" && modelId ? { model: modelId } : {}),
    ...(typeof provider === "string"
      ? { provider: provider.split(".")[0].toLowerCase() }
      : typeof emitterProvider === "string"
        ? { provider: emitterProvider.toLowerCase() }
        : {}),
  };
}
function messages(value: unknown): Data[] {
  const input = snapshot(value);
  if (!Array.isArray(input)) return [];
  return input.map((message) => {
    if (!message || typeof message !== "object") return message;
    const normalized: Data = {};
    if (message.role !== undefined) normalized.role = message.role;
    if (message.id !== undefined) normalized.id = message.id;
    if (typeof message.content === "string") {
      normalized.content = message.content;
      return normalized;
    }
    const blocks = Array.isArray(message.content) ? message.content : [];
    const calls: Data[] = [];
    const results: Data[] = [];
    const other: Data[] = [];
    const text: string[] = [];
    for (const block of blocks) {
      if (!block || typeof block !== "object") {
        other.push(block);
        continue;
      }
      if (block.type === "text" && typeof block.text === "string")
        text.push(block.text);
      else if (block.type === "tool-call")
        calls.push({
          ...(block.toolCallId !== undefined ? { id: block.toolCallId } : {}),
          type: "function",
          function: {
            name: block.toolName,
            arguments: json(block.input ?? block.args),
          },
        });
      else if (block.type === "tool-result")
        results.push({
          tool_call_id: block.toolCallId,
          name: block.toolName,
          content:
            block.output && Object.hasOwn(block.output, "value")
              ? block.output.value
              : (block.output ?? block.result),
          ...(block.isError !== undefined ? { is_error: block.isError } : {}),
        });
      else other.push(block);
    }
    normalized.content = other.length ? blocks : text.join("");
    if (calls.length) normalized.tool_calls = calls;
    if (results.length) {
      // Keep every result block, including false, zero and empty values.
      normalized.content = results.length === 1 ? results[0].content : results;
      if (results.length === 1) {
        normalized.tool_call_id = results[0].tool_call_id;
        normalized.name = results[0].name;
      } else normalized.tool_results = results;
    }
    return normalized;
  });
}
function setMessages(span: Span, prefix: string, input: Data[]): void {
  input.forEach((message, index) => {
    for (const [key, value] of Object.entries(message)) {
      if (value !== undefined)
        span.setAttribute(
          `${prefix}.${index}.${key}`,
          typeof value === "string" ? value : json(value)!,
        );
    }
  });
}
function setUsage(span: Span, usage: unknown): void {
  const input = own(usage, "inputTokens") ?? own(usage, "promptTokens");
  const output = own(usage, "outputTokens") ?? own(usage, "completionTokens");
  const total = own(usage, "totalTokens");
  for (const [key, value] of [
    [ATTR_GEN_AI_USAGE_INPUT_TOKENS, input],
    [ATTR_GEN_AI_USAGE_PROMPT_TOKENS, input],
    [ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, output],
    [ATTR_GEN_AI_USAGE_COMPLETION_TOKENS, output],
    [SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total],
  ] as const)
    if (typeof value === "number" && Number.isFinite(value))
      span.setAttribute(key, value);
}
function captureInput(
  span: Span,
  type: RespanLogType,
  instance: Data,
  input: unknown,
  run: Data,
): void {
  if (type === RespanLogType.CHAT) {
    const prompt = messages(own(input, "messages"));
    const responseFormat = own(input, "responseFormat");
    setJson(
      span,
      SpanAttributes.TRACELOOP_ENTITY_INPUT,
      responseFormat === undefined
        ? prompt
        : { messages: prompt, responseFormat },
    );
    setMessages(span, ATTR_GEN_AI_PROMPT, prompt);
  } else if (type === RespanLogType.EMBEDDING)
    setJson(span, SpanAttributes.TRACELOOP_ENTITY_INPUT, own(input, "values"));
  else if (type === RespanLogType.TOOL) {
    setJson(span, SpanAttributes.TRACELOOP_ENTITY_INPUT, {
      name: own(instance, "name"),
      arguments: input,
    });
    const callId = own(own(own(run, "runContext"), "context"), "toolCallMsg");
    const id = own(callId, "toolCallId");
    if (typeof id === "string") span.setAttribute(ATTR_GEN_AI_TOOL_CALL_ID, id);
  } else if (type === RespanLogType.AGENT) {
    const memory =
      own(instance, "memory") ?? own(own(instance, "input"), "memory");
    setJson(span, SpanAttributes.TRACELOOP_ENTITY_INPUT, {
      ...snapshot(input),
      history: messages(own(memory, "messages")),
    });
  } else setJson(span, SpanAttributes.TRACELOOP_ENTITY_INPUT, input);
}
function captureOutput(span: Span, type: RespanLogType, output: unknown): void {
  if (type === RespanLogType.CHAT) {
    const completion = messages(own(output, "messages"));
    setJson(
      span,
      SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
      completion.length === 1 ? completion[0] : completion,
    );
    setMessages(span, ATTR_GEN_AI_COMPLETION, completion);
  } else if (type === RespanLogType.EMBEDDING)
    setJson(
      span,
      SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
      own(output, "embeddings"),
    );
  else if (type === RespanLogType.AGENT) {
    const result = own(output, "result");
    const completion = messages([result]);
    setJson(
      span,
      SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
      result === undefined ? output : completion[0],
    );
  } else if (type === RespanLogType.TOOL) {
    const result = own(output, "result");
    setJson(
      span,
      SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
      result === undefined ? output : result,
    );
  } else setJson(span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, output);
}
function contentKey(key: string): boolean {
  return (
    key === SpanAttributes.LLM_REQUEST_FUNCTIONS ||
    /(?:^gen_ai\.(?:prompt|completion|input|output|system_instructions|tool\.(?:call\.(?:arguments|result)|definitions|description))|^traceloop\.entity\.(?:input|output)$|exception|error\.message|(?:^|[._])(?:prompt|input|output|arguments|result|description|message|stack)(?:[._]|$))/.test(
      key,
    )
  );
}
function guardReadable(span: Span, parent: Context, denied: boolean): void {
  const state: ReadableGuard = {
    denied,
    attributes: own(span, "attributes") ?? {},
    events: own(span, "events") ?? [],
    status: own(span, "status") ?? {},
    refresh: () => {},
  };
  const refresh = () => {
    if (
      own(state.attributes, "allow_trace_content") === false ||
      !allowed(parent)
    )
      state.denied = true;
    if (state.denied) {
      observedContentVetoes.add(span);
      for (const key of Object.keys(state.attributes))
        if (contentKey(key)) delete state.attributes[key];
      state.events.length = 0;
      delete state.status.message;
    }
  };
  state.refresh = refresh;
  readableGuards.set(span, state);
  // Protect the real ReadableSpan through processor injection, replacement,
  // queued export and deactivate/reactivate. The veto lives with this span.
  for (const field of ["attributes", "events", "status"] as const)
    Object.defineProperty(span, field, {
      enumerable: true,
      configurable: false,
      get() {
        refresh();
        if (!state.denied) return state[field];
        return field === "events" ? [] : { ...state[field] };
      },
      set(value) {
        const copied = snapshot(value);
        (state as any)[field] =
          copied && typeof copied === "object"
            ? copied
            : field === "events"
              ? []
              : {};
        refresh();
      },
    });
}
function recordError(span: Span, error: unknown, allow: boolean): void {
  span.setStatus({ code: SpanStatusCode.ERROR });
  const message = own(error, "message");
  const name = own(error, "name") ?? className(error);
  span.recordException({
    name: typeof name === "string" ? name : "Error",
    ...(allow && typeof message === "string" ? { message } : {}),
  });
}

/** Modern BeeAI needs native translation: released OI still rejects 0.1.14+.
 * Wrap only each Run's lazy handler, retaining its identity, observers,
 * cancellation, middleware, callbacks, and the SDK's own return/error values.
 */
export class NativeBeeAIInstrumentor {
  private active = false;
  constructor(
    private readonly sdk: Data,
    private readonly traceContent = true,
  ) {}
  activate(): void {
    if (this.active) return;
    const base = this.sdk.BaseAgent?.prototype;
    if (!base)
      throw new Error("BeeAIInstrumentor requires BaseAgent.prototype");
    const existing = patches.get(base);
    if (existing) {
      existing.owners++;
      if (!this.traceContent) existing.contentDisabledOwners++;
      this.active = true;
      return;
    }
    const patch: Patch = {
      owners: 1,
      contentDisabledOwners: this.traceContent ? 0 : 1,
      methods: [],
    };
    const install = (
      prototype: Data,
      name: string,
      factory: (method: Method) => Method,
    ) => {
      const original = Object.getOwnPropertyDescriptor(prototype, name);
      if (typeof original?.value !== "function") return;
      const wrapped = factory(original.value);
      Object.defineProperty(prototype, name, { ...original, value: wrapped });
      patch.methods.push({ prototype, name, original, wrapped });
    };
    try {
      for (const [name, method, type] of [
        ["BaseAgent", "run", RespanLogType.AGENT],
        ["Tool", "run", RespanLogType.TOOL],
        ["ChatModel", "create", RespanLogType.CHAT],
        ["ChatModel", "createStructure", RespanLogType.TASK],
        ["EmbeddingModel", "create", RespanLogType.EMBEDDING],
      ] as const) {
        const prototype = this.sdk[name]?.prototype;
        if (!prototype) continue;
        install(
          prototype,
          method,
          (original) =>
            function (this: Data, ...args: unknown[]) {
              const instance = this;
              const observed = context.active();
              const veto =
                patch.contentDisabledOwners > 0 || !allowed(observed);
              const skip = suppressed(observed, type);
              const run = original.apply(instance, args);
              try {
                if (method === "createStructure") {
                  // Observe the already-converted schema when the real adapter calls
                  // its provider. Never execute an application schema factory twice.
                  const model = own(instance, "model");
                  if (model && !types.isProxy(model)) {
                    let prototype = Object.getPrototypeOf(model);
                    while (
                      prototype &&
                      !Object.getOwnPropertyDescriptor(prototype, "doGenerate")
                    )
                      prototype = Object.getPrototypeOf(prototype);
                    if (
                      prototype &&
                      !patch.methods.some(
                        (entry) =>
                          entry.prototype === prototype &&
                          entry.name === "doGenerate",
                      )
                    ) {
                      install(
                        prototype,
                        "doGenerate",
                        (providerMethod) =>
                          function (this: Data, ...providerArgs: unknown[]) {
                            const capture = context
                              .active()
                              .getValue(captureKey) as Capture | undefined;
                            if (
                              capture?.type === RespanLogType.TASK &&
                              capture.allow
                            )
                              attempt(() => {
                                if (!capture.span.isRecording()) return;
                                const options = providerArgs[0];
                                const schema =
                                  own(own(options, "mode"), "schema") ??
                                  own(
                                    own(own(options, "mode"), "tool"),
                                    "parameters",
                                  ) ??
                                  own(own(options, "responseFormat"), "schema");
                                if (schema !== undefined) {
                                  const serialized = own(
                                    own(capture.span, "attributes"),
                                    SpanAttributes.TRACELOOP_ENTITY_INPUT,
                                  );
                                  const input =
                                    typeof serialized === "string"
                                      ? JSON.parse(serialized)
                                      : {};
                                  input.schema = { schema: snapshot(schema) };
                                  setJson(
                                    capture.span,
                                    SpanAttributes.TRACELOOP_ENTITY_INPUT,
                                    input,
                                  );
                                }
                              });
                            return providerMethod.apply(this, providerArgs);
                          },
                      );
                    }
                  }
                }

                const descriptor = Object.getOwnPropertyDescriptor(
                  run,
                  "handler",
                );
                if (typeof descriptor?.value !== "function") return run;
                const handler = descriptor.value;
                Object.defineProperty(run, "handler", {
                  ...descriptor,
                  value: function (this: unknown, ...handlerArgs: unknown[]) {
                    const parent = context.active();
                    if (!patch.owners || skip || suppressed(parent, type))
                      return handler.apply(this, handlerArgs);
                    let span: Span;
                    let allow: boolean;
                    let active: Context;
                    try {
                      const model = identity(instance);
                      const entity =
                        type === RespanLogType.TOOL
                          ? own(instance, "name")
                          : type === RespanLogType.AGENT
                            ? className(instance)
                            : type === RespanLogType.TASK
                              ? "createStructure"
                              : model.model;
                      span = trace
                        .getTracer("@respan/instrumentation-beeai")
                        .startSpan(
                          `beeai.${type}`,
                          {
                            attributes: {
                              [RespanSpanAttributes.RESPAN_LOG_TYPE]: type,
                              [SpanAttributes.TRACELOOP_ENTITY_NAME]:
                                typeof entity === "string" ? entity : type,
                              [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
                              ...([
                                RespanLogType.CHAT,
                                RespanLogType.EMBEDDING,
                              ].includes(type)
                                ? {
                                    [SpanAttributes.LLM_REQUEST_TYPE]: type,
                                    ...(model.model
                                      ? {
                                          [ATTR_GEN_AI_REQUEST_MODEL]:
                                            model.model,
                                        }
                                      : {}),
                                    ...(model.provider
                                      ? { [ATTR_GEN_AI_SYSTEM]: model.provider }
                                      : {}),
                                  }
                                : {}),
                            },
                          },
                          parent,
                        );
                      allow =
                        !veto &&
                        allowed(parent) &&
                        allowed(trace.setSpan(parent, span)) &&
                        span.isRecording();
                      active = trace
                        .setSpan(parent, span)
                        .setValue(contentVetoKey, !allow)
                        .setValue(captureKey, {
                          span,
                          allow,
                          type,
                          definitions: [],
                        });
                    } catch {
                      return handler.apply(this, handlerArgs);
                    }
                    return context.with(active, async () => {
                      try {
                        if (allow)
                          attempt(() =>
                            captureInput(span, type, instance, args[0], run),
                          );
                        const result = await handler.apply(this, handlerArgs);
                        attempt(() => {
                          if (!span.isRecording()) return;
                          if (type === RespanLogType.CHAT)
                            setUsage(span, own(result, "usage"));
                          if (type === RespanLogType.EMBEDDING) {
                            const tokens = own(own(result, "usage"), "tokens");
                            if (
                              typeof tokens === "number" &&
                              Number.isFinite(tokens)
                            )
                              setUsage(span, { inputTokens: tokens });
                          }
                          if (allow && allowed(active) && allowed(parent))
                            captureOutput(span, type, result);
                        });
                        return result;
                      } catch (error) {
                        attempt(() =>
                          recordError(
                            span,
                            error,
                            allow && allowed(active) && allowed(parent),
                          ),
                        );
                        throw error;
                      } finally {
                        attempt(() => {
                          if (!allowed(active) || !allowed(parent)) {
                            const attrs = own(span, "attributes");
                            for (const key of Object.keys(attrs ?? {}))
                              if (
                                key.startsWith(ATTR_GEN_AI_PROMPT + ".") ||
                                key.startsWith(ATTR_GEN_AI_COMPLETION + ".") ||
                                [
                                  SpanAttributes.TRACELOOP_ENTITY_INPUT,
                                  SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                                  SpanAttributes.LLM_REQUEST_FUNCTIONS,
                                ].includes(key)
                              )
                                delete attrs[key];
                          }
                          guardReadable(
                            span,
                            parent,
                            !allow || !allowed(active) || !allowed(parent),
                          );
                          span.end();
                        });
                      }
                    });
                  },
                });
              } catch {
                /* Unsupported/custom Run remains untouched. */
              }
              return run;
            },
        );
      }
      const prototype = this.sdk.Tool?.prototype;
      if (prototype)
        install(
          prototype,
          "getInputJsonSchema",
          (original) =>
            function (this: Data, ...args: unknown[]) {
              const result = original.apply(this, args);
              const capture = context.active().getValue(captureKey) as
                Capture | undefined;
              if (
                capture?.allow &&
                capture.type === RespanLogType.CHAT &&
                capture.span.isRecording()
              ) {
                const tool = this;
                // Observe only a schema requested by the SDK itself; do not invoke
                // caller-provided schema methods a second time for instrumentation.
                void Promise.resolve(result).then(
                  (schema) =>
                    attempt(() => {
                      const definitions = capture.definitions;
                      const name = own(tool, "name");
                      const definition = {
                        name,
                        description: own(tool, "description"),
                        parameters: snapshot(schema),
                      };
                      const index = definitions.findIndex(
                        (item: Data) => item.name === name,
                      );
                      if (index >= 0) definitions[index] = definition;
                      else definitions.push(definition);
                      setJson(
                        capture.span,
                        SpanAttributes.LLM_REQUEST_FUNCTIONS,
                        definitions,
                      );
                    }),
                  () => {},
                );
              }
              return result;
            },
        );
      patches.set(base, patch);
      this.active = true;
    } catch (error) {
      for (const entry of patch.methods.reverse())
        if (own(entry.prototype, entry.name) === entry.wrapped)
          Object.defineProperty(entry.prototype, entry.name, entry.original);
      throw error;
    }
  }
  deactivate(): void {
    if (!this.active) return;
    this.active = false;
    const base = this.sdk.BaseAgent.prototype;
    const patch = patches.get(base);
    if (!patch) return;
    if (!this.traceContent) patch.contentDisabledOwners--;
    if (--patch.owners > 0) return;
    for (const entry of patch.methods)
      if (own(entry.prototype, entry.name) === entry.wrapped)
        Object.defineProperty(entry.prototype, entry.name, entry.original);
    patches.delete(base);
  }
}
