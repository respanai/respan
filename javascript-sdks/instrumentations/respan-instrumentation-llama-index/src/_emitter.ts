import {
  context,
  createContextKey,
  trace,
  TraceFlags,
  SpanStatusCode,
  type Context,
  type Span,
} from "@opentelemetry/api";
import {
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_HTTP_RESPONSE_STATUS_CODE,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  capture,
  data,
  guard,
  internalCall,
  refresh,
  snapshot,
  type CaptureOptions,
  type Policy,
} from "./_privacy.js";
import { json, messages, responseMessage, tools, usage } from "./_helpers.js";
import { INSTRUMENTATION_LIBRARY_NAME } from "./_constants.js";

export type Operation =
  "chat" | "text" | "embedding" | "workflow" | "agent" | "task" | "tool";
export interface Options extends CaptureOptions {
  workflowName?: string;
}
const toolObservers = new Set<ToolObserver>();
const iteratorObservation = createContextKey(
  "respan.llamaindex.iterator_observation",
);
interface State {
  span: Span;
  ctx: Context;
  policy: Policy;
  operation: Operation;
  input?: any;
  sampled: boolean;
  parentId?: string;
  ended: boolean;
}
function safe(fn: () => void): void {
  try {
    fn();
  } catch {
    /* Observation must not change native behavior. */
  }
}
export function start(
  operation: Operation,
  name: string,
  receiver: unknown,
  options: Options,
): State {
  const ctx = context.active();
  const policy = capture(options, ctx);
  const attrs: Record<string, any> = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: name,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: operation,
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: "ts_tracing",
  };
  if (options.workflowName)
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = options.workflowName;
  if (["chat", "text", "embedding"].includes(operation)) {
    attrs[SpanAttributes.LLM_REQUEST_TYPE] =
      operation === "text" ? "chat" : operation;
    const model = data(receiver, "model");
    if (typeof model === "string")
      attrs[SpanAttributes.LLM_REQUEST_MODEL] = model;
  }
  // The real sampler runs with safe identity fields before any payload copy.
  const span = trace
    .getTracer(INSTRUMENTATION_LIBRARY_NAME)
    .startSpan(`llamaindex.${name}`, { attributes: attrs }, ctx);
  const sampled =
    span.isRecording() &&
    !!(span.spanContext().traceFlags & TraceFlags.SAMPLED);
  policy.sampled = sampled;
  guard(span, policy);
  return {
    span,
    ctx: trace.setSpan(ctx, span),
    policy,
    operation,
    sampled,
    parentId: trace.getSpan(ctx)?.spanContext().spanId,
    ended: false,
  };
}
function attr(state: State, key: string, value: any): void {
  if (value !== undefined) state.span.setAttribute(key, value);
}
function metadata(state: State, value: any): void {
  const previous = data(
    data(state.span, "attributes"),
    RespanSpanAttributes.RESPAN_METADATA,
  );
  let inherited: any = {};
  if (typeof previous === "string") {
    try {
      inherited = JSON.parse(previous);
    } catch {}
  }
  attr(
    state,
    RespanSpanAttributes.RESPAN_METADATA,
    json({ ...inherited, ...value }),
  );
}
export function input(state: State, args: any[]): void {
  if (!state.sampled || !refresh(state.policy).inputs) return;
  const request = args[0];
  if (state.operation === "chat") {
    state.input = messages(data(request, "messages"));
    attr(state, SpanAttributes.TRACELOOP_ENTITY_INPUT, json(state.input));
    const definitions = tools(data(request, "tools"));
    attr(state, SpanAttributes.LLM_REQUEST_FUNCTIONS, json(definitions));
    const responseFormat = data(request, "responseFormat");
    if (responseFormat !== undefined)
      metadata(state, { response_format: snapshot(responseFormat) });
  } else if (state.operation === "text") {
    state.input = snapshot(data(request, "prompt"));
    attr(state, SpanAttributes.TRACELOOP_ENTITY_INPUT, json(state.input));
  } else {
    state.input = snapshot(request);
    attr(state, SpanAttributes.TRACELOOP_ENTITY_INPUT, json(state.input));
  }
}
function indexed(state: State, prefix: string, items: any[]): void {
  items.forEach((m, i) => {
    for (const key of [
      "role",
      "content",
      "tool_calls",
      "tool_call_id",
      "name",
    ]) {
      const value = data(m, key);
      if (value !== undefined)
        attr(
          state,
          `${prefix}.${i}.${key}`,
          typeof value === "string" ? value : json(value),
        );
    }
  });
}
export function finish(state: State, result?: unknown, error?: unknown): void {
  if (state.ended) return;
  state.ended = true;
  safe(() => {
    refresh(state.policy);
    if (!state.sampled) {
      state.span.setStatus({
        code: error === undefined ? SpanStatusCode.OK : SpanStatusCode.ERROR,
      });
      return;
    }
    if (state.operation === "chat" || state.operation === "text") {
      const counts = usage(result);
      const prompt = counts.prompt_tokens ?? counts.input_tokens;
      const completion = counts.completion_tokens ?? counts.output_tokens;
      attr(state, ATTR_GEN_AI_USAGE_INPUT_TOKENS, prompt);
      attr(state, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt);
      attr(state, ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, completion);
      attr(state, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion);
      attr(state, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, counts.total_tokens);
      attr(
        state,
        ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        counts.cached_tokens,
      );
      const model = data(data(result, "raw"), "model");
      if (typeof model === "string")
        attr(state, SpanAttributes.LLM_RESPONSE_MODEL, model);
    }
    // Required canonical output and usage precede optional message indices under OTel's default 128 attribute limit.
    if (error === undefined && result !== undefined && state.policy.outputs) {
      const output =
        state.operation === "chat"
          ? responseMessage(result)
          : state.operation === "text"
            ? data(result, "text")
            : result;
      attr(state, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json(output));
      if (state.operation === "chat") {
        const out = responseMessage(result);
        indexed(state, SpanAttributes.LLM_COMPLETIONS, out);
      }
    }
    if (error !== undefined) {
      const status = data(error, "status");
      if (typeof status === "number" && Number.isInteger(status))
        attr(state, ATTR_HTTP_RESPONSE_STATUS_CODE, status);
      state.span.setStatus({ code: SpanStatusCode.ERROR });
      if (state.policy.inputs && state.policy.outputs) {
        const message = data(error, "message");
        if (typeof message === "string")
          state.span.addEvent("exception", { "exception.message": message });
      }
    } else state.span.setStatus({ code: SpanStatusCode.OK });
    if (state.policy.inputs && state.operation === "chat")
      indexed(state, SpanAttributes.LLM_PROMPTS, state.input ?? []);
  });
  for (const observer of toolObservers)
    observer.finishParent(state.span.spanContext().spanId);
  state.span.end();
}
function observeIterator(state: State, source: any): any {
  const seen = new WeakSet<object>();
  const chunks: any[] = [];
  const safeUsage: any[] = [];
  const visit = (iterator: any): any => {
    if (!iterator || typeof iterator !== "object" || seen.has(iterator))
      return iterator;
    seen.add(iterator);
    for (const method of ["next", "return", "throw"]) {
      const original = iterator[method];
      if (typeof original !== "function") continue;
      try {
        Object.defineProperty(iterator, method, {
          configurable: true,
          writable: true,
          value: function (...args: any[]) {
            const outer =
              context.active().getValue(iteratorObservation) !== state;
            try {
              const result = context.with(
                state.ctx.setValue(iteratorObservation, state),
                () => original.apply(this, args),
              );
              const receive = (part: any) => {
                if (!outer) return part;
                if (method !== "next" || part.done) {
                  const message: any = {
                    role: "assistant",
                    content: chunks
                      .map((c) => data(c, "delta") ?? data(c, "text") ?? "")
                      .join(""),
                    options: {},
                  };
                  for (const c of chunks) {
                    const opts = data(c, "options");
                    if (opts) Object.assign(message.options, opts);
                  }
                  finish(
                    state,
                    state.operation === "text"
                      ? { text: message.content, raw: safeUsage }
                      : { message, raw: safeUsage },
                  );
                } else if (state.sampled) {
                  safeUsage.push({
                    raw: { usage: usage({ raw: [part.value] }) },
                  });
                  if (refresh(state.policy).outputs)
                    chunks.push(snapshot(part.value));
                }
                return part;
              };
              if (result && typeof result.then === "function")
                result
                  .then(receive, (error: unknown) => {
                    if (outer) finish(state, undefined, error);
                  })
                  .catch(() => {});
              else receive(result);
              return result;
            } catch (error) {
              if (outer) finish(state, undefined, error);
              throw error;
            }
          },
        });
      } catch {
        /* Frozen native iterators remain usable. */
      }
    }
    return iterator;
  };
  const symbol = source[Symbol.asyncIterator];
  if (typeof symbol === "function") {
    try {
      source[Symbol.asyncIterator] = function () {
        return visit(context.with(state.ctx, () => symbol.call(this)));
      };
    } catch {}
  }
  return visit(source);
}
export function observe(
  original: Function,
  receiver: any,
  args: any[],
  operation: Operation,
  name: string,
  options: Options,
): any {
  const prior: any = context.active().getValue(internalCall);
  const lane = ["chat", "text"].includes(operation) ? "llm" : operation;
  if (prior?.receiver === receiver && prior?.lane === lane)
    return original.apply(receiver, args);
  const policy = capture(options);
  if (!policy.emit) return original.apply(receiver, args);
  const state = start(operation, name, receiver, options);
  state.ctx = state.ctx.setValue(internalCall, { receiver, lane });
  safe(() => input(state, args));
  const receive = (result: any) => {
    if (result && typeof result[Symbol.asyncIterator] === "function")
      observeIterator(state, result);
    else finish(state, result);
  };
  try {
    const result = context.with(state.ctx, () =>
      original.apply(receiver, args),
    );
    if (result && typeof result.then === "function")
      result
        .then(receive, (error: unknown) => finish(state, undefined, error))
        .catch(() => {});
    else receive(result);
    return result;
  } catch (error) {
    finish(state, undefined, error);
    throw error;
  }
}
export class ToolObserver {
  private readonly pending = new Map<string, State>();
  private readonly native = new Map<object, Map<string, State[]>>();
  constructor(private readonly options: Options) {}
  onDrain?: () => void;
  hasPending(): boolean {
    return this.pending.size > 0 || this.native.size > 0;
  }
  finishParent(id: string): void {
    for (const [key, state] of this.pending)
      if (key.startsWith(`${id}:`)) {
        this.pending.delete(key);
        finish(state, undefined, {});
      }
    for (const [manager, records] of this.native) {
      for (const [key, states] of records) {
        const matches = states.filter((state) => state.parentId === id);
        const remaining = states.filter((state) => state.parentId !== id);
        if (remaining.length) records.set(key, remaining);
        else records.delete(key);
        for (const state of matches) finish(state, undefined, {});
      }
      if (!records.size) this.native.delete(manager);
    }
    if (!this.hasPending()) {
      toolObservers.delete(this);
      this.onDrain?.();
    }
  }
  event(
    name: string,
    detail: unknown,
    manager: object,
    reason?: unknown,
    options: Options = this.options,
  ): void {
    if (name !== "llm-tool-call" && name !== "llm-tool-result") {
      this.nativeEvent(name, detail, manager, reason, options);
      return;
    }
    const call = data(detail, "toolCall");
    const id = data(call, "id");
    if (typeof id !== "string") return;
    const parent = trace.getSpan(context.active())?.spanContext().spanId ?? "";
    const key = `${parent}:${id}`;
    if (name === "llm-tool-call") {
      const policy = capture(options);
      if (!policy.emit) return;
      const toolName = data(call, "name");
      const state = start(
        "tool",
        typeof toolName === "string" ? toolName : "tool",
        undefined,
        options,
      );
      attr(state, ATTR_GEN_AI_TOOL_CALL_ID, id);
      if (state.sampled && state.policy.inputs)
        attr(
          state,
          SpanAttributes.TRACELOOP_ENTITY_INPUT,
          json({ name: toolName, arguments: snapshot(data(call, "input")) }),
        );
      this.pending.set(key, state);
      toolObservers.add(this);
    } else if (name === "llm-tool-result") {
      const state = this.pending.get(key);
      if (!state) return;
      this.pending.delete(key);
      const result = data(detail, "toolResult");
      finish(
        state,
        data(result, "output"),
        data(result, "isError") === true ? {} : undefined,
      );
      if (!this.hasPending()) {
        toolObservers.delete(this);
        this.onDrain?.();
      }
    }
  }
  private nativeEvent(
    name: string,
    detail: unknown,
    manager: object,
    reason: unknown,
    options: Options,
  ): void {
    const match =
      /^(llm|query|retrieve|synthesize|chunking|node-parsing|agent)-(start|end)$/.exec(
        name,
      );
    if (!match) return;
    const [, family, phase] = match;
    const operation: Operation =
      family === "llm"
        ? "chat"
        : family === "query"
          ? "workflow"
          : family === "agent"
            ? "agent"
            : "task";
    const lane = family === "llm" ? "llm" : operation;
    if ((context.active().getValue(internalCall) as any)?.lane === lane) return;
    const step = data(detail, "startStep") ?? data(detail, "endStep");
    const id =
      data(detail, "id") ??
      data(step, "id") ??
      data(reason, "id") ??
      trace.getSpan(context.active())?.spanContext().spanId ??
      "root";
    const key = `${family}:${typeof id === "string" ? id : "root"}`;
    let records = this.native.get(manager);
    if (phase === "start") {
      if (!capture(options).emit) return;
      const state = start(operation, `llamaindex.${family}`, detail, options);
      if (state.sampled && state.policy.inputs) {
        if (operation === "chat")
          input(state, [{ messages: data(detail, "messages") }]);
        else
          input(state, [
            data(detail, "query") ??
              data(detail, "text") ??
              data(detail, "documents") ??
              data(detail, "chunks") ??
              snapshot(step),
          ]);
      }
      if (!records) this.native.set(manager, (records = new Map()));
      const states = records.get(key) ?? [];
      states.push(state);
      records.set(key, states);
      toolObservers.add(this);
    } else {
      const states = records?.get(key);
      const state = states?.shift();
      if (!state) return;
      if (!states?.length) records?.delete(key);
      if (!records?.size) this.native.delete(manager);
      const result = state.sampled
        ? (data(detail, "response") ??
          data(detail, "nodes") ??
          data(detail, "chunks") ??
          step)
        : undefined;
      finish(state, result);
      if (!this.hasPending()) {
        toolObservers.delete(this);
        this.onDrain?.();
      }
    }
  }
}
