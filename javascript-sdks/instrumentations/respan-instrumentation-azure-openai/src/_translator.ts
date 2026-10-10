import type { Span } from "@opentelemetry/api";
import { SpanStatusCode } from "@opentelemetry/api";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { snapshot } from "./_privacy.js";
import type { Operation } from "./_request.js";
export function setRequest(
  span: Span,
  operation: Operation,
  params: any,
  content: boolean,
): (() => void) | undefined {
  for (const [key, value] of [
    [
      SpanAttributes.LLM_REQUEST_MAX_TOKENS,
      params.max_completion_tokens ??
        params.max_output_tokens ??
        params.max_tokens,
    ],
    [SpanAttributes.LLM_REQUEST_TEMPERATURE, params.temperature],
    [SpanAttributes.LLM_REQUEST_TOP_P, params.top_p],
    [SpanAttributes.LLM_FREQUENCY_PENALTY, params.frequency_penalty],
    [SpanAttributes.LLM_PRESENCE_PENALTY, params.presence_penalty],
  ] as const) {
    if (typeof value === "number") span.setAttribute(key, value);
  }
  if (!content) return;
  const input =
    operation === "chat"
      ? params.messages
      : operation === "completion"
        ? params.prompt
        : params.input;
  setJSON(span, SpanAttributes.TRACELOOP_ENTITY_INPUT, input);
  if (operation === "embedding") return;
  if (params.tools ?? params.functions)
    setJSON(
      span,
      SpanAttributes.LLM_REQUEST_FUNCTIONS,
      params.tools ?? params.functions,
    );
  // Indexed attributes compete for the native OTel budget. Add them only after
  // the completed response and usage have their canonical attributes.
  return () => {
    let messages: any[];
    if (operation === "responses") {
      messages = responseMessages(params.input);
      if (params.instructions !== undefined)
        messages.unshift({ role: "system", content: params.instructions });
    } else if (operation === "completion")
      messages = [{ role: "user", content: params.prompt }];
    else messages = params.messages ?? [];
    messages.forEach((message, index) =>
      setMessage(span, SpanAttributes.LLM_PROMPTS, index, message),
    );
  };
}

export function setResult(
  span: Span,
  operation: Operation,
  result: any,
  content: boolean,
): (() => void) | undefined {
  if (typeof result.model === "string")
    span.setAttribute(SpanAttributes.LLM_RESPONSE_MODEL, result.model);
  const usage = result.usage;
  if (usage) {
    const input = usage.input_tokens ?? usage.prompt_tokens;
    const output = usage.output_tokens ?? usage.completion_tokens;
    const total = usage.total_tokens;
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
    setJSON(span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, result.output);
    return () =>
      responseMessages(result.output ?? [], true).forEach((message, index) =>
        setMessage(span, SpanAttributes.LLM_COMPLETIONS, index, message),
      );
  } else {
    setJSON(
      span,
      SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
      result.choices?.map((choice: any) =>
        operation === "completion" ? choice.text : choice.message,
      ),
    );
    return () =>
      result.choices?.forEach((choice: any, index: number) =>
        setMessage(
          span,
          SpanAttributes.LLM_COMPLETIONS,
          choice.index ?? index,
          operation === "completion"
            ? { role: "assistant", content: choice.text }
            : choice.message,
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
  if (message.content !== undefined) {
    span.setAttribute(
      `${prefix}.${index}.content`,
      typeof message.content === "string"
        ? message.content
        : JSON.stringify(message.content),
    );
  }
  if (typeof message.tool_call_id === "string")
    span.setAttribute(`${prefix}.${index}.tool_call_id`, message.tool_call_id);
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
      messages.push({
        role: "tool",
        content: item.output,
        tool_call_id: item.call_id,
      });
    } else if (item.role || item.type === "message") {
      const parts = item.content;
      const text =
        Array.isArray(parts) &&
        parts.every((part: any) =>
          ["input_text", "output_text", "completion"].includes(part.type),
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
export class StreamAccumulator {
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
        this.value = snapshot(chunk.response);
      } else if (
        chunk.type === "response.created" ||
        chunk.type === "response.in_progress"
      ) {
        this.value.model = chunk.response?.model;
      } else if (
        chunk.type === "response.output_item.added" ||
        chunk.type === "response.output_item.done"
      ) {
        this.value.output[chunk.output_index] = snapshot(chunk.item);
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
      // Preserve the SDK's additional streamed content fields, such as refusal.
      for (const key of Object.keys(delta)) {
        if (["role", "content", "tool_calls", "function_call"].includes(key))
          continue;
        choice.message[key] = mergeDelta(choice.message[key], delta[key]);
      }
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

function mergeDelta(previous: any, value: any): any {
  if (typeof value === "string")
    return (typeof previous === "string" ? previous : "") + value;
  if (Array.isArray(value))
    return [...(Array.isArray(previous) ? previous : []), ...snapshot(value)];
  if (value && typeof value === "object") {
    const result = previous && typeof previous === "object" ? previous : {};
    for (const key of Object.keys(value))
      result[key] = ["id", "type"].includes(key)
        ? snapshot(value[key])
        : mergeDelta(result[key], value[key]);
    return result;
  }
  return value;
}
