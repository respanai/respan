import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { data, snapshot } from "./_privacy.js";
import { normalizeCallInput, safeJsonStringify } from "./_serialization.js";

export type SuperagentSpanAttributeValue = string | number | boolean | string[];
export type SuperagentSpanAttributes = Record<
  string,
  SuperagentSpanAttributeValue
>;
export interface BuildSuperagentSpanAttributesOptions {
  methodName: string;
  args: unknown[];
  result?: unknown;
  error?: unknown;
  workflowName?: string;
}
export type BuildSuperagentModelSpanAttributesOptions =
  BuildSuperagentSpanAttributesOptions;
export function identity(methodName: string): SuperagentSpanAttributes {
  return {
    [RespanSpanAttributes.RESPAN_LOG_TYPE]:
      methodName === "guard" ? RespanLogType.GUARDRAIL : RespanLogType.TOOL,
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: `superagent.${methodName}`,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
    [RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_KIND]:
      methodName === "guard" ? "guardrail" : "tool",
    ...(methodName !== "guard"
      ? { [RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_DETAIL]: methodName }
      : {}),
  };
}
export function buildSuperagentSpanAttributes({
  methodName,
  args,
  result,
  workflowName,
}: BuildSuperagentSpanAttributesOptions): SuperagentSpanAttributes {
  const attrs = identity(methodName);
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJsonStringify(
    normalizeCallInput(methodName, args),
  );
  if (workflowName)
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflowName;
  if (result !== undefined)
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJsonStringify(result);
  return attrs;
}
// Retained helper export. A SafetyClient operation result is not a model response.
export function buildSuperagentModelSpanAttributes(
  _options: BuildSuperagentModelSpanAttributesOptions,
): SuperagentSpanAttributes {
  return {};
}
export function modelIdentity(
  provider: string,
  model: unknown,
): SuperagentSpanAttributes {
  return {
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.CHAT,
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: "superagent.model",
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
    [SpanAttributes.LLM_REQUEST_TYPE]: "chat",
    [ATTR_GEN_AI_SYSTEM]: provider,
    ...(typeof model === "string"
      ? { [ATTR_GEN_AI_REQUEST_MODEL]: model }
      : {}),
  };
}
export function requestAttributes(
  body: unknown,
  nativeMessages?: unknown,
): SuperagentSpanAttributes {
  const messages = Array.isArray(nativeMessages)
    ? nativeMessages
    : (data(body, "messages") ?? data(body, "input"));
  const attrs: SuperagentSpanAttributes = {
    [SpanAttributes.TRACELOOP_ENTITY_INPUT]: safeJsonStringify(
      messages ?? body,
    ),
  };
  if (Array.isArray(messages))
    messages.forEach((message, index) => {
      const role = data(message, "role"),
        content = data(message, "content");
      if (typeof role === "string")
        attrs[`${SpanAttributes.LLM_PROMPTS}.${index}.role`] = role;
      if (content !== undefined)
        attrs[`${SpanAttributes.LLM_PROMPTS}.${index}.content`] =
          typeof content === "string" ? content : safeJsonStringify(content);
    });
  const schema =
    data(body, "response_format") ?? data(data(body, "text"), "format");
  attrs[RespanSpanAttributes.RESPAN_METADATA] = safeJsonStringify({
    provider_request: body,
    ...(schema !== undefined ? { response_format: schema } : {}),
  });
  return attrs;
}
function count(usage: unknown, keys: string[]): number | undefined {
  for (const key of keys) {
    const n = data(usage, key);
    if (typeof n === "number" && Number.isFinite(n) && n >= 0) return n;
  }
  return undefined;
}
export function responseAttributes(
  raw: unknown,
  transformed: unknown,
  capture: boolean,
): SuperagentSpanAttributes {
  const attrs: SuperagentSpanAttributes = {};
  const id = data(raw, "id") ?? data(transformed, "id"),
    model = data(raw, "model");
  if (typeof id === "string") attrs[ATTR_GEN_AI_RESPONSE_ID] = id;
  if (typeof model === "string") attrs[ATTR_GEN_AI_RESPONSE_MODEL] = model;
  const usage = data(raw, "usage") ?? data(raw, "usageMetadata");
  const input = count(usage, [
    "prompt_tokens",
    "input_tokens",
    "promptTokens",
    "inputTokens",
    "promptTokenCount",
  ]);
  const output = count(usage, [
    "completion_tokens",
    "output_tokens",
    "completionTokens",
    "outputTokens",
    "candidatesTokenCount",
  ]);
  const total = count(usage, [
    "total_tokens",
    "totalTokens",
    "totalTokenCount",
  ]);
  if (input !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = input;
    attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = input;
  }
  if (output !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = output;
    attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = output;
  }
  if (total !== undefined) attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total;
  if (capture) {
    const choices = data(transformed, "choices");
    if (Array.isArray(choices)) {
      const messages = choices
        .map((c) => snapshot(data(c, "message")))
        .filter((m) => m !== undefined);
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] =
        safeJsonStringify(messages);
      messages.forEach((m, i) => {
        if (typeof m.role === "string")
          attrs[`${SpanAttributes.LLM_COMPLETIONS}.${i}.role`] = m.role;
        if (m.content !== undefined)
          attrs[`${SpanAttributes.LLM_COMPLETIONS}.${i}.content`] =
            typeof m.content === "string"
              ? m.content
              : safeJsonStringify(m.content);
      });
    }
  }
  return attrs;
}
