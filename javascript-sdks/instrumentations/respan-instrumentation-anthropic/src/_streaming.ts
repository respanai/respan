import { context } from "@opentelemetry/api";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { safeJson } from "./_helpers.js";
import { data, snapshot, refresh } from "./_privacy.js";
import type { Session } from "./_span_emitter.js";
interface State {
  message: Record<string, any>;
  blocks: Map<number, any>;
  json: Map<number, string>;
  events: unknown[];
}
function update(state: State, event: any): void {
  if (event.type === "message_start") {
    state.message = { ...event.message };
    for (const [index, block] of (event.message.content ?? []).entries())
      state.blocks.set(index, block);
  }
  if (event.type === "content_block_start")
    state.blocks.set(event.index, event.content_block);
  if (event.type === "content_block_delta") {
    const block = state.blocks.get(event.index) ?? {};
    const delta = event.delta ?? {};
    if (delta.type === "input_json_delta") {
      const text =
        (state.json.get(event.index) ?? "") + (delta.partial_json ?? "");
      state.json.set(event.index, text);
      try {
        block.input = JSON.parse(text);
      } catch {
        block.input = text;
      }
    } else if (delta.type === "text_delta")
      block.text = (block.text ?? "") + (delta.text ?? "");
    else if (delta.type === "thinking_delta")
      block.thinking = (block.thinking ?? "") + (delta.thinking ?? "");
    else if (delta.type === "signature_delta")
      block.signature = (block.signature ?? "") + (delta.signature ?? "");
    else if (delta.type === "citations_delta")
      block.citations = [...(block.citations ?? []), delta.citation];
    else if (delta.type === "compaction_delta") {
      // The native SDK treats these as final replacement values, including null.
      block.content = delta.content;
      if (Object.hasOwn(delta, "encrypted_content"))
        block.encrypted_content = delta.encrypted_content;
    }
    // Unknown deltas remain in canonical metadata; never replace the block type.
    state.blocks.set(event.index, block);
  }
  if (event.type === "message_delta") {
    Object.assign(state.message, event.delta);
    if (event.usage)
      state.message.usage = { ...state.message.usage, ...event.usage };
  }
  if (
    (event.type === undefined || event.type === "completion") &&
    event.completion !== undefined
  ) {
    const prior = state.message.completion ?? "";
    Object.assign(state.message, event);
    state.message.completion = prior + event.completion;
  }
}
export function observeIterable<T extends object>(
  stream: T,
  session: Session,
  mode: "sse" | "rows" | "agent" = "sse",
): T {
  const target = stream as any;
  const state: State = {
    message: {},
    blocks: new Map(),
    json: new Map(),
    events: [],
  };
  const controller = data(target, "controller");
  let started = false;
  const abort = () => {
    if (!started) session.finish(undefined, undefined, false);
  };
  if (controller instanceof AbortController)
    controller.signal.addEventListener("abort", abort, { once: true });
  const complete = (error?: unknown, success = true) => {
    if (controller instanceof AbortController)
      controller.signal.removeEventListener("abort", abort);
    if (mode === "sse" && refresh(session.policy).outputs) {
      let metadata: Record<string, unknown> = {};
      try {
        const prior = (session.span as any).attributes;
        const json = data(prior, RespanSpanAttributes.RESPAN_METADATA);
        if (typeof json === "string") metadata = JSON.parse(json);
      } catch {}
      session.span.setAttribute(
        RespanSpanAttributes.RESPAN_METADATA,
        safeJson({ ...metadata, stream_events: state.events }),
      );
    }
    const value =
      mode === "rows"
        ? state.events
        : mode === "agent"
          ? undefined
          : {
              ...state.message,
              ...(state.blocks.size
                ? {
                    content: [...state.blocks.entries()]
                      .sort(([a], [b]) => a - b)
                      .map(([, v]) => v),
                  }
                : {}),
            };
    session.finish(value, error, success);
  };
  const factory = data(target, "iterator");
  const original =
    typeof factory === "function" ? factory : target[Symbol.asyncIterator];
  if (typeof original !== "function") {
    session.finish(stream);
    return stream;
  }
  const wrapped = function (this: any, ...args: any[]) {
    const iterator = context.with(session.ctx, () =>
      Reflect.apply(original, this, args),
    ) as any;
    for (const key of ["next", "return", "throw"]) {
      const method = iterator[key];
      if (typeof method !== "function") continue;
      Object.defineProperty(iterator, key, {
        configurable: true,
        writable: true,
        value: function (...a: any[]) {
          if (key === "next") started = true;
          let result: any;
          try {
            result = context.with(session.ctx, () =>
              Reflect.apply(method, iterator, a),
            );
          } catch (error) {
            complete(error);
            throw error;
          }
          const observe = (item: any) => {
            if (
              !item.done &&
              key === "next" &&
              refresh(session.policy).outputs
            ) {
              const copy = snapshot(item.value);
              if (mode === "sse") {
                state.events.push(copy);
                update(state, snapshot(copy));
              } else if (mode === "rows") state.events.push(copy);
            }
            if (
              (item.done && mode !== "agent") ||
              key === "return" ||
              key === "throw"
            )
              complete(key === "throw" ? a[0] : undefined, key === "next");
          };
          if (result && typeof result.then === "function")
            void result
              .then(observe, (error: unknown) => complete(error))
              .catch(() => {});
          else observe(result);
          return result;
        },
      });
    }
    return iterator;
  };
  try {
    if (typeof data(target, "iterator") === "function")
      Object.defineProperty(target, "iterator", {
        value: wrapped,
        writable: true,
        configurable: true,
      });
    else
      Object.defineProperty(target, Symbol.asyncIterator, {
        value: wrapped,
        writable: true,
        configurable: true,
      });
  } catch {}
  return stream;
}
/** Observe the SDK's lazy parser, never APIPromise.then/catch/raw helper methods. */
export function observePromise<T>(
  result: T,
  session: Session,
  streaming = false,
  rows = false,
): T {
  const target = result as any;
  const parser = data(target, "parseResponse");
  const responsePromise = data(target, "responsePromise");
  if (responsePromise instanceof Promise)
    void responsePromise
      .then(
        (props: any) => {
          session.headers(data(props, "response"));
          if (!data(target, "parsedPromise")) session.finish();
        },
        (error: unknown) => session.finish(undefined, error),
      )
      .catch(() => {});
  const success = (value: any) => {
    if (streaming || rows)
      return observeIterable(value, session, rows ? "rows" : "sse");
    session.finish(value);
    return value;
  };
  if (typeof parser === "function") {
    Object.defineProperty(target, "parseResponse", {
      configurable: true,
      writable: true,
      value: function (this: any, ...args: any[]) {
        let parsed: any;
        try {
          parsed = context.with(session.ctx, () =>
            Reflect.apply(parser, this, args),
          );
        } catch (error) {
          session.finish(undefined, error);
          throw error;
        }
        if (parsed && typeof parsed.then === "function") {
          void parsed
            .then(success, (error: unknown) => session.finish(undefined, error))
            .catch(() => {});
          return parsed;
        }
        return success(parsed);
      },
    });
  } else if (target instanceof Promise)
    void target
      .then(success, (error: unknown) => session.finish(undefined, error))
      .catch(() => {});
  else success(target);
  return result;
}
