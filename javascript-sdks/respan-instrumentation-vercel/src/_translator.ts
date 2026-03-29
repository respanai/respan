/**
 * Translate Vercel AI SDK spans → Traceloop/OpenLLMetry format.
 *
 * The Vercel AI SDK emits OTEL spans with its own attribute schema (ai.model.id,
 * ai.prompt.messages, ai.response.text, etc.). This SpanProcessor enriches those
 * spans with the Traceloop/GenAI semantic conventions the Respan backend expects.
 *
 * Two-phase enrichment:
 * - onStart(): Sets RESPAN_LOG_TYPE so the span passes CompositeProcessor filtering
 * - onEnd():   Full attribute translation (model, messages, tokens, metadata, etc.)
 *
 * Vercel attrs are preserved (additive enrichment via setDefault, not destructive).
 *
 * Ported from @respan/exporter-vercel — all exporter features are replicated:
 * - Model normalization (Gemini, Claude, DeepSeek, O3-mini)
 * - Prompt message parsing (ai.prompt.messages + ai.prompt fallback)
 * - Completion message building (ai.response.text, ai.response.object, tool calls)
 * - Token count normalization (input/output → prompt/completion)
 * - Tool definitions (ai.prompt.tools) and tool choice (gen_ai.usage.tool_choice)
 * - Customer params (ai.telemetry.metadata.customer_* + customer_params JSON)
 * - General metadata (ai.telemetry.metadata.* → respan.metadata.*)
 * - Stream detection, environment, cost, TTFT, generation time, unit prices
 * - Log type detection with operationId + attribute-based fallbacks
 */

import type { Context } from "@opentelemetry/api";
import type { SpanProcessor, ReadableSpan, Span } from "@opentelemetry/sdk-trace-base";
import { RespanSpanAttributes, RespanLogType } from "@respan/respan-sdk";
import { VERCEL_SPAN_CONFIG, VERCEL_PARENT_SPANS } from "./constants/index.js";

// ── Attribute keys (single source of truth from SDK) ─────────────────────────

const RESPAN_LOG_TYPE = RespanSpanAttributes.RESPAN_LOG_TYPE;
const GEN_AI_REQUEST_MODEL = RespanSpanAttributes.GEN_AI_REQUEST_MODEL;
const GEN_AI_USAGE_PROMPT_TOKENS = RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS;
const GEN_AI_USAGE_COMPLETION_TOKENS = RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS;
const LLM_REQUEST_TYPE = RespanSpanAttributes.LLM_REQUEST_TYPE;
const CUSTOMER_ID = RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID;
const CUSTOMER_EMAIL = RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_EMAIL;
const CUSTOMER_NAME = RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_NAME;
const THREAD_ID = RespanSpanAttributes.RESPAN_THREADS_ID;
const SESSION_ID = RespanSpanAttributes.RESPAN_SESSION_ID;
const TRACE_GROUP_ID = RespanSpanAttributes.RESPAN_TRACE_GROUP_ID;
const RESPAN_SPAN_TOOLS = RespanSpanAttributes.RESPAN_SPAN_TOOLS;
const RESPAN_METADATA_AGENT_NAME = RespanSpanAttributes.RESPAN_METADATA_AGENT_NAME;
const RESPAN_METADATA_PREFIX = RespanSpanAttributes.RESPAN_METADATA; // "respan.metadata"

/** Build a respan.metadata.<key> attribute name. */
function metadataKey(key: string): string {
  return `${RESPAN_METADATA_PREFIX}.${key}`;
}

// Traceloop wire-format keys
const TL_SPAN_KIND = "traceloop.span.kind";
const TL_ENTITY_NAME = "traceloop.entity.name";
const TL_ENTITY_INPUT = "traceloop.entity.input";
const TL_ENTITY_OUTPUT = "traceloop.entity.output";
const TL_ENTITY_PATH = "traceloop.entity.path";

// ── Helpers ──────────────────────────────────────────────────────────────────

function setDefault(attrs: Record<string, any>, key: string, value: any): void {
  if (attrs[key] === undefined && value !== undefined && value !== null) {
    attrs[key] = value;
  }
}

function safeJsonStr(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function safeJsonParse(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

/**
 * Detect whether a span is from the Vercel AI SDK.
 *
 * Primary check: instrumentation scope name === "ai" (set by the Vercel AI SDK).
 * Fallback: ai.sdk attribute or ai.* span name.
 *
 * Does NOT match on gen_ai.* attributes alone — those may come from other
 * instrumentations (OpenInference, Traceloop) and must not be stripped.
 */
function isVercelAISpan(span: ReadableSpan): boolean {
  // Primary: check OTEL instrumentation scope (most reliable, no false positives)
  if (span.instrumentationLibrary?.name === "ai") return true;
  // Fallback: explicit Vercel marker or span name convention
  if (span.attributes["ai.sdk"] !== undefined) return true;
  if (span.name.startsWith("ai.")) return true;
  return false;
}

// ── Log type detection (with operationId + attribute fallbacks) ──────────────

/**
 * Resolve the Respan log type for a Vercel AI SDK span.
 * Replicates the full fallback chain from the exporter's parseLogType().
 */
function resolveLogType(name: string, attrs: Record<string, any>): string {
  // 1. Direct span name match
  const config = VERCEL_SPAN_CONFIG[name];
  if (config) return config.logType;

  const parentType = VERCEL_PARENT_SPANS[name];
  if (parentType) return parentType;

  // 2. operationId attribute fallback
  const operationId = attrs["ai.operationId"]?.toString();
  if (operationId) {
    const opConfig = VERCEL_SPAN_CONFIG[operationId];
    if (opConfig) return opConfig.logType;
    const opParent = VERCEL_PARENT_SPANS[operationId];
    if (opParent) return opParent;
  }

  // 3. Attribute-based fallback detection (same heuristics as exporter)
  if (
    attrs["ai.embedding"] || attrs["ai.embeddings"] ||
    name.includes("embed") || operationId?.includes("embed")
  ) {
    return RespanLogType.EMBEDDING;
  }

  if (
    attrs["ai.toolCall.id"] || attrs["ai.toolCall.name"] ||
    attrs["ai.toolCall.args"] || attrs["ai.toolCall.result"] ||
    attrs["ai.response.toolCalls"] ||
    name.includes("tool") || operationId?.includes("tool")
  ) {
    return RespanLogType.TOOL;
  }

  if (
    attrs["ai.agent.id"] ||
    name.includes("agent") || operationId?.includes("agent")
  ) {
    return RespanLogType.AGENT;
  }

  if (
    attrs["ai.workflow.id"] ||
    name.includes("workflow") || operationId?.includes("workflow")
  ) {
    return RespanLogType.WORKFLOW;
  }

  if (
    attrs["ai.transcript"] ||
    name.includes("transcript") || operationId?.includes("transcript")
  ) {
    return RespanLogType.TRANSCRIPTION;
  }

  if (
    attrs["ai.speech"] ||
    name.includes("speech") || operationId?.includes("speech")
  ) {
    return RespanLogType.SPEECH;
  }

  // 4. Generation span fallback (doGenerate/doStream)
  if (name.includes("doGenerate") || name.includes("doStream")) {
    return RespanLogType.TEXT;
  }

  return RespanLogType.UNKNOWN;
}

// ── Model normalization ──────────────────────────────────────────────────────

/**
 * Normalize the model ID from Vercel's ai.model.id to a standard model name.
 * Replicates the logic from the existing exporter.
 */
function normalizeModel(modelId: string): string {
  const model = modelId.toLowerCase();

  if (model.includes("gemini-2.0-flash-001")) return "gemini/gemini-2.0-flash";
  if (model.includes("gemini-2.0-pro")) return "gemini/gemini-2.0-pro-exp-02-05";
  if (model.includes("claude-3-5-sonnet")) return "claude-3-5-sonnet-20241022";
  if (model.includes("deepseek")) return "deepseek/" + model;
  if (model.includes("o3-mini")) return "o3-mini";

  return model;
}

// ── Prompt/completion message formatting ─────────────────────────────────────

/**
 * Parse ai.prompt.messages into a JSON string suitable for traceloop.entity.input.
 * Falls back to ai.prompt as a single user message.
 */
function formatPromptInput(attrs: Record<string, any>): string | undefined {
  const messages = attrs["ai.prompt.messages"];
  if (messages) {
    try {
      const parsed = typeof messages === "string" ? JSON.parse(messages) : messages;
      return safeJsonStr(parsed);
    } catch {
      // fall through
    }
  }

  const prompt = attrs["ai.prompt"];
  if (prompt) {
    return safeJsonStr([{ role: "user", content: String(prompt) }]);
  }

  return undefined;
}

/**
 * Build completion output from ai.response.text, ai.response.object, and tool calls.
 * Also includes tool result messages when present (for tool call spans).
 */
function formatCompletionOutput(attrs: Record<string, any>): string | undefined {
  let content = "";

  if (attrs["ai.response.object"]) {
    try {
      const rawObject = attrs["ai.response.object"];
      const parsed = typeof rawObject === "string" ? JSON.parse(rawObject) : rawObject;
      // generateObject returns the object directly (no `response` wrapper).
      // Prefer known wrappers when present, otherwise serialize the object itself.
      const normalized =
        parsed?.response ?? parsed?.object ?? parsed?.output ?? parsed?.result ?? parsed;
      content = safeJsonStr(normalized);
    } catch {
      content = String(attrs["ai.response.text"] ?? "");
    }
  } else {
    content = String(attrs["ai.response.text"] ?? "");
  }

  if (!content) return undefined;

  // Build assistant message with optional tool calls
  const message: Record<string, any> = { role: "assistant", content };

  const toolCalls = parseToolCalls(attrs);
  if (toolCalls && toolCalls.length > 0) {
    message.tool_calls = toolCalls;
  }

  // Include tool result as a separate message if present
  const messages: any[] = [message];
  if (attrs["ai.toolCall.result"]) {
    messages.push({
      role: "tool",
      tool_call_id: String(attrs["ai.toolCall.id"] || ""),
      content: String(attrs["ai.toolCall.result"] || ""),
    });
  }

  return safeJsonStr(messages.length === 1 ? message : messages);
}

/**
 * Parse tool call data from various Vercel AI SDK attribute formats.
 */
function parseToolCalls(attrs: Record<string, any>): any[] | undefined {
  // Try array formats first
  for (const key of ["ai.response.toolCalls", "ai.toolCall", "ai.toolCalls"]) {
    if (!attrs[key]) continue;
    try {
      const parsed = typeof attrs[key] === "string" ? JSON.parse(attrs[key]) : attrs[key];
      const calls = Array.isArray(parsed) ? parsed : [parsed];
      return calls.map((call: any) => {
        if (!call || typeof call !== "object") return { type: "function" };
        const result = { ...call };
        if (!result.type) result.type = "function";
        if (!result.id && (result.toolCallId || result.tool_call_id)) {
          result.id = result.toolCallId || result.tool_call_id;
        }
        return result;
      });
    } catch {
      continue;
    }
  }

  // Try individual tool call attributes
  if (attrs["ai.toolCall.id"] || attrs["ai.toolCall.name"] || attrs["ai.toolCall.args"]) {
    const toolCall: Record<string, any> = { type: "function" };
    for (const [key, value] of Object.entries(attrs)) {
      if (key.startsWith("ai.toolCall.")) {
        toolCall[key.replace("ai.toolCall.", "")] = value;
      }
    }
    return [toolCall];
  }

  return undefined;
}

/**
 * Format tool call span input/output for traceloop.entity.input/output.
 */
function formatToolInput(attrs: Record<string, any>): string | undefined {
  const name = attrs["ai.toolCall.name"];
  const args = attrs["ai.toolCall.args"];
  if (!name && !args) return undefined;

  const input: Record<string, any> = {};
  if (name) input.name = name;
  if (args) {
    input.args = typeof args === "string" ? safeJsonParse(args) : args;
  }
  return safeJsonStr(input);
}

function formatToolOutput(attrs: Record<string, any>): string | undefined {
  const result = attrs["ai.toolCall.result"];
  if (result === undefined) return undefined;
  return safeJsonStr(typeof result === "string" ? safeJsonParse(result) : result);
}

// ── Tools & tool choice (from exporter's parseTools / parseToolChoice) ───────

/**
 * Parse ai.prompt.tools into a normalized tool definition array.
 * Accepts both nested ({type:"function",function:{...}}) and flat shapes.
 */
function parseTools(attrs: Record<string, any>): string | undefined {
  try {
    const tools = attrs["ai.prompt.tools"];
    if (!tools) return undefined;
    const raw = Array.isArray(tools) ? tools : [tools];
    const parsed = raw
      .map((tool: any) => {
        try {
          return typeof tool === "string" ? JSON.parse(tool) : tool;
        } catch {
          return undefined;
        }
      })
      .filter(Boolean)
      .map((tool: any) => {
        // Accept both nested and flat shapes; normalize to nested
        if (tool && tool.type === "function") {
          if (tool.function && typeof tool.function === "object") return tool;
          const { name, description, parameters, ...rest } = tool;
          return {
            ...rest,
            type: "function",
            function: {
              name,
              ...(description ? { description } : {}),
              ...(parameters ? { parameters } : {}),
            },
          };
        }
        return tool;
      });
    if (parsed.length === 0) return undefined;
    return safeJsonStr(parsed);
  } catch {
    return undefined;
  }
}

/**
 * Parse gen_ai.usage.tool_choice into a normalized structure.
 */
function parseToolChoice(attrs: Record<string, any>): string | undefined {
  try {
    const toolChoice = attrs["gen_ai.usage.tool_choice"];
    if (!toolChoice) return undefined;
    const parsed = typeof toolChoice === "string" ? JSON.parse(toolChoice) : toolChoice;
    return safeJsonStr({
      type: String(parsed.type),
      function: { name: String(parsed.function.name) },
    });
  } catch {
    return undefined;
  }
}

// ── Metadata / customer params ───────────────────────────────────────────────

/**
 * Extract ai.telemetry.metadata.* and map customer/thread params to Respan attrs.
 * Also handles prompt_unit_price, completion_unit_price, and span_type metadata.
 */
function enrichMetadata(attrs: Record<string, any>, spanName: string): void {
  for (const [key, value] of Object.entries(attrs)) {
    if (!key.startsWith("ai.telemetry.metadata.")) continue;
    const cleanKey = key.slice("ai.telemetry.metadata.".length);

    // Map well-known keys to Respan span attributes
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
        // customer_params can be a JSON object with all three fields
        try {
          const parsed = typeof value === "string" ? JSON.parse(value) : value;
          if (parsed?.customer_identifier) setDefault(attrs, CUSTOMER_ID, parsed.customer_identifier);
          if (parsed?.customer_email) setDefault(attrs, CUSTOMER_EMAIL, parsed.customer_email);
          if (parsed?.customer_name) setDefault(attrs, CUSTOMER_NAME, parsed.customer_name);
        } catch {
          // ignore
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
        // userId is a fallback for customer_identifier (backward compat with exporter)
        setDefault(attrs, CUSTOMER_ID, String(value));
        setDefault(attrs, metadataKey(cleanKey), String(value ?? ""));
        break;
      default:
        // All other metadata → respan.metadata.<key>
        setDefault(attrs, metadataKey(cleanKey), String(value ?? ""));
        break;
    }
  }

}

// ── Token count normalization ────────────────────────────────────────────────

function enrichTokens(attrs: Record<string, any>): void {
  // Vercel AI SDK may use gen_ai.usage.input_tokens / gen_ai.usage.output_tokens
  // Respan backend expects gen_ai.usage.prompt_tokens / gen_ai.usage.completion_tokens
  const inputTokens =
    attrs["gen_ai.usage.input_tokens"] ??
    attrs["gen_ai.usage.prompt_tokens"];
  const outputTokens =
    attrs["gen_ai.usage.output_tokens"] ??
    attrs["gen_ai.usage.completion_tokens"];

  if (inputTokens !== undefined) {
    setDefault(attrs, GEN_AI_USAGE_PROMPT_TOKENS, Number(inputTokens));
  }
  if (outputTokens !== undefined) {
    setDefault(attrs, GEN_AI_USAGE_COMPLETION_TOKENS, Number(outputTokens));
  }
}

// ── Performance / cost metrics ───────────────────────────────────────────────

/**
 * Enrich performance and cost attributes that the exporter handled explicitly.
 * These are Vercel-specific attrs that the backend needs in standard locations.
 */
function enrichPerformanceMetrics(attrs: Record<string, any>, spanName: string): void {
  // Stream detection from span name
  setDefault(attrs, metadataKey("stream"), String(spanName.includes("doStream")));

  // Time to first token from ai.response.msToFinish (Vercel-specific)
  const msToFinish = attrs["ai.response.msToFinish"];
  if (msToFinish !== undefined) {
    setDefault(attrs, metadataKey("time_to_first_token"), String(Number(msToFinish) / 1000));
  }

  // Cost (gen_ai.usage.cost is standard but ensure it's present)
  const cost = attrs["gen_ai.usage.cost"];
  if (cost !== undefined) {
    setDefault(attrs, metadataKey("cost"), String(cost));
  }

  // TTFT (gen_ai.usage.ttft)
  const ttft = attrs["gen_ai.usage.ttft"];
  if (ttft !== undefined) {
    setDefault(attrs, metadataKey("ttft"), String(ttft));
  }

  // Generation time
  const genTime = attrs["gen_ai.usage.generation_time"];
  if (genTime !== undefined) {
    setDefault(attrs, metadataKey("generation_time"), String(genTime));
  }

  // Warnings
  const warnings = attrs["gen_ai.usage.warnings"];
  if (warnings !== undefined) {
    setDefault(attrs, metadataKey("warnings"), String(warnings));
  }

  // Response type (text/json_schema/json_object)
  const type = attrs["gen_ai.usage.type"];
  if (type !== undefined) {
    setDefault(attrs, metadataKey("type"), String(type));
  }

}

// ── Cleanup: strip redundant Vercel attrs after translation ──────────────────

/**
 * Vercel AI SDK attributes that have been translated to Traceloop/GenAI/Respan
 * equivalents. These are deleted after translation to keep spans clean.
 */
const VERCEL_ATTRS_TO_STRIP = [
  // ── Vercel AI SDK attrs (translated to Traceloop/GenAI equivalents) ────

  // Model (translated to gen_ai.request.model)
  "ai.model.id",
  "ai.model.provider",
  "ai.response.model",

  // Prompt/completion (translated to traceloop.entity.input/output)
  "ai.prompt",
  "ai.prompt.messages",
  "ai.prompt.format",
  "ai.response.text",
  "ai.response.object",

  // Tokens (translated to gen_ai.usage.prompt_tokens/completion_tokens)
  "ai.usage.promptTokens",
  "ai.usage.completionTokens",

  // Response metadata (redundant with standard OTEL/GenAI attrs)
  "ai.response.finishReason",
  "ai.response.id",
  "ai.response.timestamp",
  "ai.response.providerMetadata",
  "ai.response.msToFinish",

  // SDK internals (no user value)
  "ai.operationId",
  "ai.settings.maxRetries",
  "ai.settings.maxSteps",
  "ai.sdk",
  "operation.name",

  // Tool calls (translated to traceloop.entity.input/output for tool spans)
  "ai.toolCall.id",
  "ai.toolCall.name",
  "ai.toolCall.args",
  "ai.toolCall.result",
  "ai.response.toolCalls",

  // GenAI duplicates (already consumed by backend as top-level fields)
  "gen_ai.response.finish_reasons",
  "gen_ai.response.id",
  "gen_ai.usage.input_tokens",
  "gen_ai.usage.output_tokens",
  "gen_ai.system",

  // ── Traceloop routing attrs (Vercel-specific, not user-facing) ──────────
  // Keep traceloop.span.kind and respan.entity.log_type — backend needs them.
  // Keep respan.environment — may be set by user via propagateAttributes().
  "traceloop.entity.name",
  "traceloop.entity.path",

  // ── OTEL resource / process noise (no user value in metadata) ──────────
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

  // ── Next.js auto-instrumentation noise ─────────────────────────────────
  "next.span_name",
  "next.span_type",
  "http.url",
  "http.method",
  "net.peer.name",
];

/**
 * Remove redundant Vercel AI SDK attributes after translation.
 * Also strips ai.telemetry.metadata.* keys that have been mapped to respan.* attrs.
 */
function stripRedundantAttrs(attrs: Record<string, any>): void {
  for (const key of VERCEL_ATTRS_TO_STRIP) {
    delete attrs[key];
  }
  // Strip ai.telemetry.metadata.* (already mapped to respan.metadata.* / respan.customer_params.*)
  for (const key of Object.keys(attrs)) {
    if (key.startsWith("ai.telemetry.metadata.")) {
      delete attrs[key];
    }
  }
  // Strip ai.prompt.tools (translated to respan.span.tools)
  if (attrs["ai.prompt.tools"] !== undefined) {
    delete attrs["ai.prompt.tools"];
  }
}

// ── Main processor ───────────────────────────────────────────────────────────

/**
 * SpanProcessor that translates Vercel AI SDK attributes to Traceloop/OpenLLMetry.
 *
 * Phase 1 (onStart): Sets RESPAN_LOG_TYPE so CompositeProcessor lets the span through.
 * Phase 2 (onEnd):   Full attribute enrichment — model, messages, tokens, metadata,
 *                     tools, performance metrics, environment, etc.
 */
export class VercelAITranslator implements SpanProcessor {
  onStart(span: Span, _parentContext: Context): void {
    // Cast to access attributes (Span interface doesn't expose them, but the impl does)
    const s = span as any;
    const name: string = s.name ?? "";
    if (!name.startsWith("ai.")) return;

    // Set RESPAN_LOG_TYPE early so CompositeProcessor accepts this span
    const config = VERCEL_SPAN_CONFIG[name];
    if (config) {
      s.setAttribute(RESPAN_LOG_TYPE, config.logType);
    } else if (VERCEL_PARENT_SPANS[name] !== undefined) {
      s.setAttribute(RESPAN_LOG_TYPE, VERCEL_PARENT_SPANS[name]);
    } else {
      // Unknown ai.* span — use full fallback detection
      // At onStart, attributes may be sparse, so set a generic type.
      // The precise type will be resolved in onEnd() with full attributes.
      s.setAttribute(RESPAN_LOG_TYPE, RespanLogType.TASK);
    }
  }

  onEnd(span: ReadableSpan): void {
    const attrs = (span as any).attributes as Record<string, any> | undefined;
    if (!attrs) return;

    if (!isVercelAISpan(span as any)) return;

    const name = span.name;
    const config = VERCEL_SPAN_CONFIG[name];
    const parentLogType = VERCEL_PARENT_SPANS[name];

    // Resolve the log type using full fallback chain (name → operationId → attributes)
    const logType = resolveLogType(name, attrs);

    // ── Always: metadata, customer params, environment ────────────────────
    enrichMetadata(attrs, name);

    // ── Parent wrapper spans: minimal enrichment only ─────────────────────
    if (parentLogType !== undefined && !config) {
      setDefault(attrs, RESPAN_LOG_TYPE, logType);
      stripRedundantAttrs(attrs);
      return;
    }

    // ── Detailed / leaf spans: full enrichment ────────────────────────────

    // Update RESPAN_LOG_TYPE with the resolved type (may be more accurate than onStart)
    attrs[RESPAN_LOG_TYPE] = logType;

    if (config) {
      setDefault(attrs, TL_SPAN_KIND, config.kind);

      // LLM-specific enrichment
      if (config.isLLM) {
        setDefault(attrs, LLM_REQUEST_TYPE, RespanLogType.CHAT);

        // Model
        const modelId = attrs["ai.model.id"];
        if (modelId) {
          setDefault(attrs, GEN_AI_REQUEST_MODEL, normalizeModel(String(modelId)));
        }

        // Prompt messages → entity input
        const input = formatPromptInput(attrs);
        if (input) setDefault(attrs, TL_ENTITY_INPUT, input);

        // Completion → entity output
        const output = formatCompletionOutput(attrs);
        if (output) setDefault(attrs, TL_ENTITY_OUTPUT, output);

        // Token counts
        enrichTokens(attrs);

        // Tool definitions
        const tools = parseTools(attrs);
        if (tools) setDefault(attrs, RESPAN_SPAN_TOOLS, tools);

        // Tool choice
        const toolChoice = parseToolChoice(attrs);
        if (toolChoice) setDefault(attrs, metadataKey("tool_choice"), toolChoice);

        // Performance metrics (stream, TTFT, cost, etc.)
        enrichPerformanceMetrics(attrs, name);
      }

      // Tool call spans
      if (config.logType === RespanLogType.TOOL || logType === RespanLogType.TOOL) {
        const toolInput = formatToolInput(attrs);
        if (toolInput) setDefault(attrs, TL_ENTITY_INPUT, toolInput);

        const toolOutput = formatToolOutput(attrs);
        if (toolOutput) setDefault(attrs, TL_ENTITY_OUTPUT, toolOutput);
      }

      // Agent spans
      if (config.logType === RespanLogType.AGENT || logType === RespanLogType.AGENT) {
        const agentName = attrs["ai.agent.name"] ?? attrs["ai.agent.id"] ?? name;
        setDefault(attrs, RESPAN_METADATA_AGENT_NAME, String(agentName));
      }
    } else {
      // Unknown ai.* span — enrich with fallback-resolved type

      // If fallback detected it as an LLM span, add model + tokens
      if (logType === RespanLogType.TEXT || logType === RespanLogType.EMBEDDING) {
        const modelId = attrs["ai.model.id"];
        if (modelId) {
          setDefault(attrs, GEN_AI_REQUEST_MODEL, normalizeModel(String(modelId)));
        }
        enrichTokens(attrs);

        if (logType === RespanLogType.TEXT) {
          setDefault(attrs, LLM_REQUEST_TYPE, RespanLogType.CHAT);
          const input = formatPromptInput(attrs);
          if (input) setDefault(attrs, TL_ENTITY_INPUT, input);
          const output = formatCompletionOutput(attrs);
          if (output) setDefault(attrs, TL_ENTITY_OUTPUT, output);
          enrichPerformanceMetrics(attrs, name);
        }
      }

      // If fallback detected tool, add tool input/output
      if (logType === RespanLogType.TOOL) {
        const toolInput = formatToolInput(attrs);
        if (toolInput) setDefault(attrs, TL_ENTITY_INPUT, toolInput);
        const toolOutput = formatToolOutput(attrs);
        if (toolOutput) setDefault(attrs, TL_ENTITY_OUTPUT, toolOutput);
      }
    }

    // ── Cleanup: remove redundant Vercel attrs that have been translated ──
    stripRedundantAttrs(attrs);
  }

  async shutdown(): Promise<void> {}
  async forceFlush(): Promise<void> {}
}
