import {
  AI_MODEL_ID,
  AI_OPERATION_ID,
  AI_EMBEDDING,
  AI_EMBEDDINGS,
  AI_PROMPT,
  AI_PROMPT_MESSAGES,
  AI_PROMPT_TOOL_CHOICE,
  AI_PROMPT_TOOLS,
  AI_RESPONSE_OBJECT,
  AI_RESPONSE_MS_TO_FINISH,
  AI_RESPONSE_TEXT,
  AI_RESPONSE_TOOL_CALLS,
  AI_SDK,
  AI_TELEMETRY_METADATA_PREFIX,
  AI_TOOL_CALL_ARGS,
  AI_TOOL_CALL_ID,
  AI_TOOL_CALL_NAME,
  AI_TOOL_CALL_RESULT,
  CUSTOMER_EMAIL,
  CUSTOMER_ID,
  CUSTOMER_NAME,
  GEN_AI_USAGE_COST,
  GEN_AI_USAGE_GENERATION_TIME,
  GEN_AI_USAGE_INPUT_TOKENS,
  GEN_AI_USAGE_OUTPUT_TOKENS,
  GEN_AI_USAGE_TTFT,
  GEN_AI_USAGE_TYPE,
  GEN_AI_REQUEST_MODEL,
  GEN_AI_USAGE_WARNINGS,
  GEN_AI_USAGE_COMPLETION_TOKENS,
  GEN_AI_USAGE_PROMPT_TOKENS,
  SESSION_ID,
  THREAD_ID,
  TRACE_GROUP_ID,
  metadataKey,
  normalizeModel,
  setDefault,
  type SpanAttributes,
} from "./shared.js";

export function enrichMetadata(attrs: SpanAttributes): void {
  for (const [key, value] of Object.entries(attrs)) {
    if (!key.startsWith(AI_TELEMETRY_METADATA_PREFIX)) {
      continue;
    }

    const cleanKey = key.slice(AI_TELEMETRY_METADATA_PREFIX.length);
    switch (cleanKey) {
      case "customer_identifier":
        setDefault(attrs, CUSTOMER_ID, String(value));
        break;
      case "customer_email":
        setDefault(attrs, CUSTOMER_EMAIL, String(value));
        break;
      case "customer_name":
        setDefault(attrs, CUSTOMER_NAME, String(value));
        break;
      case "session_identifier":
        setDefault(attrs, SESSION_ID, String(value));
        break;
      case "thread_identifier":
        setDefault(attrs, THREAD_ID, String(value));
        break;
      case "trace_group_identifier":
        setDefault(attrs, TRACE_GROUP_ID, String(value));
        break;
      case "customer_params": {
        // customer_params is a JSON-stringified object (Vercel telemetry
        // metadata values must be flat scalars, so users serialize the object).
        // Documented shape uses `email` / `name` (matching the Customer columns
        // in the UI); accept the legacy `customer_email` / `customer_name`
        // aliases too so older integrations keep working.
        try {
          const parsed = typeof value === "string" ? JSON.parse(value) : value;
          if (parsed?.customer_identifier) setDefault(attrs, CUSTOMER_ID, parsed.customer_identifier);
          const email = parsed?.email ?? parsed?.customer_email;
          if (email) setDefault(attrs, CUSTOMER_EMAIL, email);
          const name = parsed?.name ?? parsed?.customer_name;
          if (name) setDefault(attrs, CUSTOMER_NAME, name);
        } catch {
          // Ignore malformed customer_params metadata.
        }
        break;
      }
      case "prompt_unit_price":
        setDefault(attrs, metadataKey("prompt_unit_price"), String(value));
        break;
      case "completion_unit_price":
        setDefault(attrs, metadataKey("completion_unit_price"), String(value));
        break;
      case "userId":
        setDefault(attrs, CUSTOMER_ID, String(value));
        setDefault(attrs, metadataKey(cleanKey), String(value ?? ""));
        break;
      default:
        setDefault(attrs, metadataKey(cleanKey), String(value ?? ""));
        break;
    }
  }
}

export function enrichModel(attrs: SpanAttributes, modelId: unknown): void {
  if (!modelId) {
    return;
  }

  const model = normalizeModel(String(modelId));
  setDefault(attrs, GEN_AI_REQUEST_MODEL, model);
}

export function enrichTokens(attrs: SpanAttributes): void {
  const inputTokens =
    attrs[GEN_AI_USAGE_INPUT_TOKENS] ??
    attrs["gen_ai.usage.prompt_tokens"] ??
    attrs["ai.usage.promptTokens"];
  const outputTokens =
    attrs[GEN_AI_USAGE_OUTPUT_TOKENS] ??
    attrs["gen_ai.usage.completion_tokens"] ??
    attrs["ai.usage.completionTokens"];

  if (inputTokens !== undefined) {
    const promptTokens = Number(inputTokens);
    setDefault(attrs, GEN_AI_USAGE_INPUT_TOKENS, promptTokens);
    setDefault(attrs, GEN_AI_USAGE_PROMPT_TOKENS, promptTokens);
  }
  if (outputTokens !== undefined) {
    const completionTokens = Number(outputTokens);
    setDefault(attrs, GEN_AI_USAGE_OUTPUT_TOKENS, completionTokens);
    setDefault(attrs, GEN_AI_USAGE_COMPLETION_TOKENS, completionTokens);
  }
}

export function enrichPerformanceMetrics(attrs: SpanAttributes, spanName: string): void {
  setDefault(attrs, metadataKey("stream"), String(spanName.includes("doStream")));

  const msToFinish = attrs[AI_RESPONSE_MS_TO_FINISH];
  if (msToFinish !== undefined) {
    setDefault(attrs, metadataKey("time_to_first_token"), String(Number(msToFinish) / 1000));
  }

  const cost = attrs[GEN_AI_USAGE_COST];
  if (cost !== undefined) {
    setDefault(attrs, metadataKey("cost"), String(cost));
  }

  const ttft = attrs[GEN_AI_USAGE_TTFT];
  if (ttft !== undefined) {
    setDefault(attrs, metadataKey("ttft"), String(ttft));
  }

  const generationTime = attrs[GEN_AI_USAGE_GENERATION_TIME];
  if (generationTime !== undefined) {
    setDefault(attrs, metadataKey("generation_time"), String(generationTime));
  }

  const warnings = attrs[GEN_AI_USAGE_WARNINGS];
  if (warnings !== undefined) {
    setDefault(attrs, metadataKey("warnings"), String(warnings));
  }

  const responseType = attrs[GEN_AI_USAGE_TYPE];
  if (responseType !== undefined) {
    setDefault(attrs, metadataKey("type"), String(responseType));
  }
}

const VERCEL_ATTRS_TO_STRIP = [
  AI_MODEL_ID,
  "ai.model.provider",
  "ai.response.model",
  AI_PROMPT,
  AI_PROMPT_MESSAGES,
  "ai.prompt.format",
  AI_RESPONSE_TEXT,
  AI_RESPONSE_OBJECT,
  AI_EMBEDDING,
  AI_EMBEDDINGS,
  "ai.usage.promptTokens",
  "ai.usage.completionTokens",
  "ai.usage.inputTokens",
  "ai.usage.outputTokens",
  "ai.usage.totalTokens",
  "ai.usage.reasoningTokens",
  "ai.usage.cachedInputTokens",
  "ai.usage.tokens",
  "ai.response.finishReason",
  "ai.response.id",
  "ai.response.timestamp",
  "ai.response.providerMetadata",
  AI_RESPONSE_MS_TO_FINISH,
  "ai.response.msToFirstChunk",
  "ai.response.avgOutputTokensPerSecond",
  "ai.response.avgCompletionTokensPerSecond",
  "ai.request.headers.user-agent",
  AI_PROMPT_TOOL_CHOICE,
  AI_OPERATION_ID,
  "ai.settings.maxRetries",
  "ai.settings.maxSteps",
  AI_SDK,
  "operation.name",
  AI_TOOL_CALL_ID,
  AI_TOOL_CALL_NAME,
  AI_TOOL_CALL_ARGS,
  AI_TOOL_CALL_RESULT,
  AI_RESPONSE_TOOL_CALLS,
  "gen_ai.response.finish_reasons",
  "gen_ai.response.id",
  "gen_ai.system",
  "traceloop.entity.name",
  "traceloop.entity.path",
  "service.name",
  "telemetry.sdk.language",
  "telemetry.sdk.name",
  "telemetry.sdk.version",
  "process.pid",
  "process.executable.name",
  "process.executable.path",
  "process.command_args",
  "process.runtime.version",
  "process.runtime.name",
  "process.runtime.description",
  "process.command",
  "process.owner",
  "host.name",
  "host.arch",
  "host.id",
  "otel.scope.name",
  "otel.scope.version",
  "next.span_name",
  "next.span_type",
  "http.url",
  "http.method",
  "net.peer.name",
];

export function stripRedundantAttrs(attrs: SpanAttributes): void {
  for (const key of VERCEL_ATTRS_TO_STRIP) {
    delete attrs[key];
  }

  for (const key of Object.keys(attrs)) {
    if (key.startsWith(AI_TELEMETRY_METADATA_PREFIX)) {
      delete attrs[key];
      continue;
    }

    if (key.startsWith("ai.usage.") && key.includes("Details.")) {
      delete attrs[key];
    }
  }

  if (attrs[AI_PROMPT_TOOLS] !== undefined) {
    delete attrs[AI_PROMPT_TOOLS];
  }
}
