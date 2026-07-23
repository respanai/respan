import {
  ATTR_GEN_AI_INPUT_MESSAGES,
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_OUTPUT_MESSAGES,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_TOOL_DEFINITIONS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes as TraceloopSpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  AI_RESPONSE_MS_TO_FINISH,
  AI_PREFIX,
  AI_MODEL_PROVIDER,
  AI_TELEMETRY_METADATA_PREFIX,
  AI_USAGE_CACHED_INPUT_TOKENS,
  AI_USAGE_COMPLETION_TOKENS,
  AI_USAGE_INPUT_TOKENS,
  AI_USAGE_OUTPUT_TOKENS,
  AI_USAGE_PROMPT_TOKENS,
  AI_USAGE_TOTAL_TOKENS,
  setMetadata,
  normalizeModel,
  setDefault,
  type SpanAttributes,
} from "./shared.js";

export function enrichMetadata(attrs: SpanAttributes): void {
  for (const [key, value] of Object.entries(attrs)) {
    const respanMetadataPrefix = RespanSpanAttributes.RESPAN_METADATA + ".";
    if (key.startsWith(respanMetadataPrefix)) {
      const cleanKey = key.slice(respanMetadataPrefix.length);
      if (cleanKey !== "agent_name") {
        setMetadata(attrs, cleanKey, value);
      }
      delete attrs[key];
      continue;
    }

    if (!key.startsWith(AI_TELEMETRY_METADATA_PREFIX)) {
      continue;
    }

    const cleanKey = key.slice(AI_TELEMETRY_METADATA_PREFIX.length);
    switch (cleanKey) {
      case "customer_identifier":
        setDefault(attrs, RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID, String(value));
        break;
      case "customer_email":
        setDefault(attrs, RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_EMAIL, String(value));
        break;
      case "customer_name":
        setDefault(attrs, RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_NAME, String(value));
        break;
      case "session_identifier":
        setDefault(attrs, RespanSpanAttributes.RESPAN_SESSION_ID, String(value));
        break;
      case "thread_identifier":
        setDefault(attrs, RespanSpanAttributes.RESPAN_THREADS_ID, String(value));
        break;
      case "trace_group_identifier":
        setDefault(attrs, RespanSpanAttributes.RESPAN_TRACE_GROUP_ID, String(value));
        break;
      case "customer_params": {
        // customer_params is a JSON-stringified object (Vercel telemetry
        // metadata values must be flat scalars, so users serialize the object).
        // Documented shape uses `email` / `name` (matching the Customer columns
        // in the UI); accept the legacy `customer_email` / `customer_name`
        // aliases too so older integrations keep working.
        try {
          const parsed = typeof value === "string" ? JSON.parse(value) : value;
          if (parsed?.customer_identifier) setDefault(attrs, RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID, parsed.customer_identifier);
          const email = parsed?.email ?? parsed?.customer_email;
          if (email) setDefault(attrs, RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_EMAIL, email);
          const name = parsed?.name ?? parsed?.customer_name;
          if (name) setDefault(attrs, RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_NAME, name);
        } catch {
          // Ignore malformed customer_params metadata.
        }
        break;
      }
      case "prompt_unit_price":
        setMetadata(attrs, "prompt_unit_price", String(value));
        break;
      case "completion_unit_price":
        setMetadata(attrs, "completion_unit_price", String(value));
        break;
      case "userId":
        setDefault(attrs, RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID, String(value));
        setMetadata(attrs, cleanKey, String(value ?? ""));
        break;
      case "agent_name":
        // Agent display names are carried by traceloop.entity.name.
        break;
      default:
        setMetadata(attrs, cleanKey, String(value ?? ""));
        break;
    }
  }
}

export function enrichModel(attrs: SpanAttributes, modelId: unknown): void {
  if (!modelId) {
    return;
  }

  const model = normalizeModel(String(modelId));
  setDefault(attrs, ATTR_GEN_AI_REQUEST_MODEL, model);
}

function normalizeSystem(system: unknown): string | undefined {
  if (!system) {
    return undefined;
  }

  const value = String(system).trim().toLowerCase();
  if (!value) {
    return undefined;
  }

  if (value.includes("openai")) return "openai";
  if (value.includes("anthropic")) return "anthropic";
  if (value.includes("google") || value.includes("gemini")) return "google";
  if (value.includes("bedrock")) return "bedrock";
  if (value.includes("azure")) return "azure";
  if (value.includes("mistral")) return "mistral";
  if (value.includes("cohere")) return "cohere";
  if (value.includes("groq")) return "groq";
  if (value.includes("xai")) return "xai";
  if (value.includes("deepseek")) return "deepseek";

  return value.split(/[.:/]/, 1)[0] || value;
}

export function enrichSystem(attrs: SpanAttributes): void {
  const system = normalizeSystem(attrs[ATTR_GEN_AI_SYSTEM] ?? attrs[ATTR_GEN_AI_PROVIDER_NAME] ?? attrs[AI_MODEL_PROVIDER]);
  if (system) {
    setDefault(attrs, ATTR_GEN_AI_SYSTEM, system);
  }
}

function numberAttr(value: unknown): number | undefined {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }

  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : undefined;
}

export function enrichTokens(attrs: SpanAttributes): void {
  const inputTokens =
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] ??
    attrs[TraceloopSpanAttributes.LLM_USAGE_PROMPT_TOKENS] ??
    attrs[AI_USAGE_INPUT_TOKENS] ??
    attrs[AI_USAGE_PROMPT_TOKENS];
  const outputTokens =
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] ??
    attrs[TraceloopSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] ??
    attrs[AI_USAGE_OUTPUT_TOKENS] ??
    attrs[AI_USAGE_COMPLETION_TOKENS];
  const totalTokens = attrs[TraceloopSpanAttributes.LLM_USAGE_TOTAL_TOKENS] ?? attrs[AI_USAGE_TOTAL_TOKENS];
  const cacheReadInputTokens =
    attrs[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] ??
    attrs["llm.usage.cache_read_input_tokens"] ??
    attrs[AI_USAGE_CACHED_INPUT_TOKENS];

  const promptTokens = numberAttr(inputTokens);
  const completionTokens = numberAttr(outputTokens);

  if (promptTokens !== undefined) {
    setDefault(attrs, ATTR_GEN_AI_USAGE_INPUT_TOKENS, promptTokens);
    setDefault(attrs, TraceloopSpanAttributes.LLM_USAGE_PROMPT_TOKENS, promptTokens);
  }
  if (completionTokens !== undefined) {
    setDefault(attrs, ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, completionTokens);
    setDefault(attrs, TraceloopSpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completionTokens);
  }

  const resolvedTotalTokens = numberAttr(totalTokens) ?? (
    promptTokens !== undefined && completionTokens !== undefined
      ? promptTokens + completionTokens
      : undefined
  );
  if (resolvedTotalTokens !== undefined) {
    setDefault(attrs, TraceloopSpanAttributes.LLM_USAGE_TOTAL_TOKENS, resolvedTotalTokens);
  }

  const resolvedCacheReadInputTokens = numberAttr(cacheReadInputTokens);
  if (resolvedCacheReadInputTokens !== undefined) {
    setDefault(attrs, "llm.usage.cache_read_input_tokens", resolvedCacheReadInputTokens);
  }
}

export function enrichPerformanceMetrics(attrs: SpanAttributes, spanName: string): void {
  // Streaming is a first-class promoted attribute (llm.is_streaming), not metadata.
  setDefault(attrs, "llm.is_streaming", spanName.toLowerCase().includes("stream"));

  const msToFinish = attrs[AI_RESPONSE_MS_TO_FINISH];
  if (msToFinish !== undefined) {
    setMetadata(attrs, "time_to_first_token", String(Number(msToFinish) / 1000));
  }

  const cost = attrs["gen_ai.usage.cost"];
  if (cost !== undefined) {
    setMetadata(attrs, "cost", String(cost));
  }

  const ttft = attrs["gen_ai.usage.ttft"];
  if (ttft !== undefined) {
    setMetadata(attrs, "ttft", String(ttft));
  }

  const generationTime = attrs["gen_ai.usage.generation_time"];
  if (generationTime !== undefined) {
    setMetadata(attrs, "generation_time", String(generationTime));
  }

  const warnings = attrs["gen_ai.usage.warnings"];
  if (warnings !== undefined) {
    setMetadata(attrs, "warnings", String(warnings));
  }

  const responseType = attrs["gen_ai.usage.type"];
  if (responseType !== undefined) {
    setMetadata(attrs, "type", String(responseType));
  }
}

const NON_CONTRACT_ATTRS_TO_STRIP = [
  "operation.name",
  ATTR_GEN_AI_INPUT_MESSAGES,
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_OUTPUT_MESSAGES,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_TOOL_DEFINITIONS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  "gen_ai.usage.cost",
  "gen_ai.usage.ttft",
  "gen_ai.usage.generation_time",
  "gen_ai.usage.warnings",
  "gen_ai.usage.type",
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

const OFF_CONTRACT_ALIAS_ATTRS_TO_STRIP = [
  "tools",
  "tool_calls",
  "model",
  "prompt_tokens",
  "completion_tokens",
  "total_request_tokens",
  "span_tools",
  "has_tool_calls",
  "parallel_tool_calls",
  RespanSpanAttributes.RESPAN_SPAN_TOOLS,
  RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS,
  RespanSpanAttributes.RESPAN_SPAN_HANDOFFS,
];

export function stripRedundantAttrs(
  attrs: SpanAttributes,
  logType: string,
): void {
  for (const key of NON_CONTRACT_ATTRS_TO_STRIP) {
    delete attrs[key];
  }

  for (const key of OFF_CONTRACT_ALIAS_ATTRS_TO_STRIP) {
    delete attrs[key];
  }

  const carriesLLMFields =
    logType === RespanLogType.TEXT ||
    logType === RespanLogType.EMBEDDING;
  const respanMetadataPrefix = RespanSpanAttributes.RESPAN_METADATA + ".";

  for (const key of Object.keys(attrs)) {
    if (
      key.startsWith(AI_PREFIX) ||
      key.startsWith("gen_ai.tool.") ||
      key.startsWith(respanMetadataPrefix) ||
      (!carriesLLMFields &&
        (key.startsWith("gen_ai.") || key.startsWith("llm.")))
    ) {
      delete attrs[key];
    }
  }
}
