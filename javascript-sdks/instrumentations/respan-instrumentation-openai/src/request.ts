import {
  context,
  trace,
  SpanKind,
  SpanStatusCode,
  type Span,
} from "@opentelemetry/api";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";
import {
  ATTR_ERROR_TYPE,
  ATTR_HTTP_RESPONSE_STATUS_CODE,
} from "@opentelemetry/semantic-conventions";
import {
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_HTTP_STATUS_CODE,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanSpanAttributes } from "@respan/respan-sdk";

export type Operation = "chat" | "text" | "responses" | "embedding";

// The Respan gateway returns the unique_id of the log row it wrote for a call in
// this response header. That row already carries the call's usage, cost and
// content, so the span carries the id instead (a link only): reporting usage or
// content here too would count the call twice (the server estimates tokens from
// content when usage is absent). Mirrors respan_sdk RESPAN_GATEWAY_LOG_ID.
export const RESPAN_LOG_ID_RESPONSE_HEADER = "x-respan-log-id";
export const RESPAN_GATEWAY_LOG_ID = "respan.gateway.log_id";

function headerLogId(headers: any): string | undefined {
  const value =
    typeof headers?.get === "function"
      ? headers.get(RESPAN_LOG_ID_RESPONSE_HEADER)
      : headers?.[RESPAN_LOG_ID_RESPONSE_HEADER];
  return typeof value === "string" && value ? value : undefined;
}

export function observeRequest(
  original: any,
  receiver: any,
  args: any[],
  operation: Operation,
  traceContent?: boolean,
): any {
  const params = args[0] ?? {};
  const parent = context.active();
  const content =
    parent.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT) ??
    traceContent ??
    process.env.TRACELOOP_TRACE_CONTENT !== "false";
  const span = trace.getTracer("@respan/instrumentation-openai").startSpan(
    `openai.${operation}`,
    {
      kind: SpanKind.CLIENT,
      attributes: {
        [RespanSpanAttributes.RESPAN_LOG_TYPE]:
          operation === "responses" ? "chat" : operation,
        [SpanAttributes.LLM_SYSTEM]: "openai",
        [SpanAttributes.LLM_REQUEST_TYPE]:
          operation === "embedding" ? "embedding" : "chat",
        ...(typeof params.model === "string"
          ? { [SpanAttributes.LLM_REQUEST_MODEL]: params.model }
          : {}),
        [SpanAttributes.TRACELOOP_ENTITY_NAME]: `openai.${operation}`,
        [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
      },
    },
    parent,
  );
  const active = trace.setSpan(parent, span);
  let ended = false;
  let removeAbortListener: (() => void) | undefined;
  const accumulator = new StreamAccumulator(operation);
  let gatewayLogId: string | undefined;
  const safely = (callback: () => void) => {
    try {
      callback();
    } catch {
      /* tracing must not change SDK results */
    }
  };
  const finish = (result?: any, error?: unknown) => {
    if (ended) return;
    ended = true;
    removeAbortListener?.();
    safely(() => {
      if (error !== undefined && !gatewayLogId) {
        gatewayLogId = headerLogId((error as { headers?: unknown })?.headers);
      }
      if (gatewayLogId) {
        span.setAttribute(RESPAN_GATEWAY_LOG_ID, gatewayLogId);
      } else {
        // Request attributes are set here, not eagerly: a linked span must not
        // carry the prompt.
        setRequest(span, operation, params, Boolean(content));
      }
      if (result !== undefined && !gatewayLogId) {
        if (
          typeof params.model !== "string" &&
          typeof result.model === "string"
        ) {
          span.setAttribute(SpanAttributes.LLM_REQUEST_MODEL, result.model);
        }
        setResult(span, operation, result, Boolean(content));
      }
      if (error !== undefined) {
        const exception =
          error instanceof Error ? error : new Error(String(error));
        span.recordException(exception);
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: exception.message,
        });
        span.setAttribute(ATTR_ERROR_TYPE, exception.name);
        const status = (error as { status?: unknown })?.status;
        if (typeof status === "number") {
          span.setAttribute(ATTR_HTTP_RESPONSE_STATUS_CODE, status);
          // Keep the upstream legacy HTTP key while ingestion supports both generations.
          span.setAttribute(ATTR_HTTP_STATUS_CODE, status);
        }
      }
    });
    span.end();
  };

  const seenStreams = new WeakSet<object>();
  const onResult = (result: any) => {
    if (!result || typeof result[Symbol.asyncIterator] !== "function") {
      finish(result);
      return result;
    }
    if (seenStreams.has(result)) return result;
    seenStreams.add(result);
    // Modern Stream.tee() reads iterator() directly; older releases read Symbol.asyncIterator.
    const key =
      typeof result.iterator === "function" ? "iterator" : Symbol.asyncIterator;
    const iterator = result[key];
    let pendingReads = 0;
    result[key] = function () {
      const source = iterator.call(this);
      return (async function* () {
        try {
          while (true) {
            let next;
            pendingReads += 1;
            try {
              next = await context.with(active, () => source.next());
            } finally {
              pendingReads -= 1;
            }
            if (next.done) break;
            safely(() => accumulator.add(next.value));
            yield next.value;
          }
          finish(accumulator.result());
        } catch (error) {
          finish(accumulator.result(), error);
          throw error;
        } finally {
          // Mark partial output before returning: native cleanup can abort the controller.
          finish(accumulator.result());
          if (source.return) await context.with(active, () => source.return());
        }
      })();
    };
    const signal = result.controller?.signal;
    if (signal) {
      // SDKs also abort during error cleanup. Let iterator rejection record the error first.
      let abortTimer: ReturnType<typeof setTimeout> | undefined;
      const onAbort = () => {
        if (pendingReads === 0) finish(accumulator.result());
        else abortTimer = setTimeout(() => finish(accumulator.result()), 0);
      };
      signal.addEventListener("abort", onAbort, { once: true });
      removeAbortListener = () => {
        signal.removeEventListener("abort", onAbort);
        if (abortTimer) clearTimeout(abortTimer);
      };
      if (signal.aborted) onAbort();
    }
    return result;
  };

  try {
    const promise = context.with(active, () => original.apply(receiver, args));
    // APIPromise is lazy. Observe parsing without eagerly consuming raw responses.
    const decorate = (apiPromise: any): any => {
      let parsing = false;
      const parse = apiPromise.parse;
      apiPromise.parse = function () {
        parsing = true;
        return parse.call(this).then(onResult, (error: unknown) => {
          finish(undefined, error);
          throw error;
        });
      };
      const asResponse = apiPromise.asResponse;
      apiPromise.asResponse = function () {
        return asResponse.call(this).then(
          (response: any) => {
            // withResponse() calls parse() first. Raw-only access deliberately records no body.
            if (!parsing) finish();
            return response;
          },
          (error: unknown) => {
            finish(undefined, error);
            throw error;
          },
        );
      };
      const unwrap = apiPromise._thenUnwrap;
      apiPromise._thenUnwrap = function (transform: any) {
        return decorate(unwrap.call(this, transform));
      };
      // Read the gateway's log id as soon as headers arrive: this handler is
      // registered before parse()'s, so it runs before the span finishes.
      apiPromise.responsePromise.then(
        (props: any) =>
          safely(() => {
            gatewayLogId ??= headerLogId(props?.response?.headers);
          }),
        () => undefined,
      );
      // Observe transport rejection even when callers only use a derived parsing helper.
      apiPromise.responsePromise.catch((error: unknown) =>
        finish(undefined, error),
      );
      return apiPromise;
    };
    return decorate(promise);
  } catch (error) {
    finish(undefined, error);
    throw error;
  }
}

function setRequest(
  span: Span,
  operation: Operation,
  params: any,
  content: boolean,
): void {
  for (const [key, value] of [
    [
      SpanAttributes.LLM_REQUEST_MAX_TOKENS,
      params.max_completion_tokens ??
        params.max_output_tokens ??
        params.max_tokens,
    ],
    [SpanAttributes.LLM_REQUEST_TEMPERATURE, params.temperature],
    [SpanAttributes.LLM_REQUEST_TOP_P, params.top_p],
  ] as const) {
    if (typeof value === "number") span.setAttribute(key, value);
  }
  if (!content) return;
  const input =
    operation === "chat"
      ? params.messages
      : operation === "text"
        ? params.prompt
        : params.input;
  setJSON(span, SpanAttributes.TRACELOOP_ENTITY_INPUT, input);
  if (operation === "embedding") return;
  let messages: any[];
  if (operation === "responses") {
    messages = responseMessages(params.input);
    if (params.instructions)
      messages.unshift({ role: "system", content: params.instructions });
  } else if (operation === "text") {
    messages = [{ role: "user", content: params.prompt }];
  } else {
    messages = params.messages ?? [];
  }
  messages.forEach((message, index) =>
    setMessage(span, SpanAttributes.LLM_PROMPTS, index, message),
  );
  if (params.tools ?? params.functions)
    setJSON(
      span,
      SpanAttributes.LLM_REQUEST_FUNCTIONS,
      params.tools ?? params.functions,
    );
}

function setResult(
  span: Span,
  operation: Operation,
  result: any,
  content: boolean,
): void {
  if (typeof result.model === "string")
    span.setAttribute(SpanAttributes.LLM_RESPONSE_MODEL, result.model);
  const usage = result.usage;
  if (usage) {
    const input = usage.input_tokens ?? usage.prompt_tokens;
    const output = usage.output_tokens ?? usage.completion_tokens;
    const total =
      usage.total_tokens ??
      (typeof input === "number" && typeof output === "number"
        ? input + output
        : undefined);
    for (const [key, value] of [
      [ATTR_GEN_AI_USAGE_INPUT_TOKENS, input],
      [SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input],
      [ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, output],
      [SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output],
      [SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total],
      [
        ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        usage.input_tokens_details?.cached_tokens ??
          usage.prompt_tokens_details?.cached_tokens,
      ],
    ] as const) {
      if (typeof value === "number") span.setAttribute(key, value);
    }
  }
  if (result.error || result.status === "failed") {
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: result.error?.message ?? "OpenAI response failed",
    });
  }
  if (!content) return;
  if (operation === "embedding") {
    setJSON(
      span,
      SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
      result.data?.map((item: any) => item.embedding),
    );
  } else if (operation === "responses") {
    const messages = responseMessages(result.output ?? [], true);
    messages.forEach((message, index) =>
      setMessage(span, SpanAttributes.LLM_COMPLETIONS, index, message),
    );
    setJSON(span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, result.output);
  } else {
    result.choices?.forEach((choice: any, index: number) => {
      setMessage(
        span,
        SpanAttributes.LLM_COMPLETIONS,
        choice.index ?? index,
        operation === "text"
          ? { role: "assistant", content: choice.text }
          : choice.message,
      );
    });
    setJSON(
      span,
      SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
      result.choices?.map((choice: any) =>
        operation === "text" ? choice.text : choice.message,
      ),
    );
  }
}

function setJSON(span: Span, key: string, value: any): void {
  if (value !== undefined) span.setAttribute(key, JSON.stringify(value));
}

function setMessage(
  span: Span,
  prefix: string,
  index: number,
  message: any,
): void {
  if (!message) return;
  span.setAttribute(`${prefix}.${index}.role`, message.role ?? "assistant");
  if (message.content != null) {
    span.setAttribute(
      `${prefix}.${index}.content`,
      typeof message.content === "string"
        ? message.content
        : JSON.stringify(message.content),
    );
  }
  const calls = message.tool_calls?.length
    ? message.tool_calls
    : message.function_call
      ? [{ type: "function", function: message.function_call }]
      : undefined;
  if (calls?.length) setJSON(span, `${prefix}.${index}.tool_calls`, calls);
}

function responseMessages(input: any, combineOutput = false): any[] {
  if (typeof input === "string") return [{ role: "user", content: input }];
  const messages: any[] = [];
  for (const item of input ?? []) {
    if (item.type === "function_call") {
      let assistant = combineOutput
        ? messages.find((message) => message.role === "assistant")
        : undefined;
      if (!assistant)
        messages.push(
          (assistant = { role: "assistant", content: "", tool_calls: [] }),
        );
      (assistant.tool_calls ??= []).push({
        id: item.call_id ?? item.id,
        type: "function",
        function: { name: item.name, arguments: item.arguments },
      });
    } else if (item.type === "function_call_output") {
      messages.push({ role: "tool", content: item.output });
    } else if (item.role || item.type === "message") {
      const parts = item.content;
      const text =
        Array.isArray(parts) &&
        parts.every((part: any) =>
          ["input_text", "output_text", "text"].includes(part.type),
        )
          ? parts.map((part: any) => part.text).join("")
          : parts;
      const existing = combineOutput
        ? messages.find((message) => message.role === "assistant")
        : undefined;
      if ((item.role ?? "assistant") === "assistant" && existing)
        existing.content = (existing.content ?? "") + (text ?? "");
      else messages.push({ role: item.role ?? "assistant", content: text });
    }
  }
  return messages;
}

/** Accumulates provider chunks only; never estimates usage or mutates delivered chunks. */
class StreamAccumulator {
  private value: any = { choices: [], output: [] };
  constructor(private operation: Operation) {}

  add(chunk: any): void {
    if (this.operation === "responses") {
      if (
        [
          "response.completed",
          "response.failed",
          "response.incomplete",
        ].includes(chunk.type)
      ) {
        this.value = JSON.parse(JSON.stringify(chunk.response));
      } else if (
        chunk.type === "response.created" ||
        chunk.type === "response.in_progress"
      ) {
        this.value.model = chunk.response?.model;
      } else if (
        chunk.type === "response.output_item.added" ||
        chunk.type === "response.output_item.done"
      ) {
        this.value.output[chunk.output_index] = JSON.parse(
          JSON.stringify(chunk.item),
        );
      } else if (
        chunk.type === "response.content_part.added" ||
        chunk.type === "response.content_part.done"
      ) {
        const item = this.value.output[chunk.output_index];
        if (item)
          (item.content ??= [])[chunk.content_index] = { ...chunk.part };
      } else if (chunk.type === "response.output_text.delta") {
        const item = (this.value.output[chunk.output_index] ??= {
          type: "message",
          role: "assistant",
          content: [],
        });
        const part = (item.content[chunk.content_index] ??= {
          type: "output_text",
          text: "",
        });
        part.text += chunk.delta;
      } else if (chunk.type === "response.function_call_arguments.delta") {
        const item = this.value.output[chunk.output_index];
        if (item) item.arguments = (item.arguments ?? "") + chunk.delta;
      } else if (chunk.type === "error") {
        this.value.error = chunk;
      }
      return;
    }
    if (chunk.model) this.value.model = chunk.model;
    if (chunk.usage) this.value.usage = chunk.usage;
    for (const part of chunk.choices ?? []) {
      const choice = (this.value.choices[part.index] ??= {
        index: part.index,
        text: "",
        message: { role: "assistant", content: "", tool_calls: [] },
      });
      if (part.text) choice.text += part.text;
      const delta = part.delta;
      if (!delta) continue;
      if (delta.role) choice.message.role = delta.role;
      if (delta.content) choice.message.content += delta.content;
      for (const call of delta.tool_calls ?? []) {
        const target = (choice.message.tool_calls[call.index] ??= {
          id: "",
          type: "function",
          function: { name: "", arguments: "" },
        });
        if (call.id) target.id = call.id;
        if (call.function?.name) target.function.name += call.function.name;
        if (call.function?.arguments)
          target.function.arguments += call.function.arguments;
      }
      if (delta.function_call) {
        const target = (choice.message.function_call ??= {
          name: "",
          arguments: "",
        });
        target.name += delta.function_call.name ?? "";
        target.arguments += delta.function_call.arguments ?? "";
      }
    }
  }

  result(): any {
    return {
      ...this.value,
      ...(this.value.output
        ? { output: this.value.output.filter(Boolean) }
        : {}),
      ...(this.value.choices
        ? { choices: this.value.choices.filter(Boolean) }
        : {}),
    };
  }
}
