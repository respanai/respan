/**
 * Mappings from Vercel AI SDK span names to Respan/Traceloop conventions.
 *
 * The Vercel AI SDK emits OTEL spans with names like "ai.generateText.doGenerate".
 * These mappings tell the translator which traceloop.span.kind, respan log type,
 * and whether to apply full LLM attribute enrichment.
 */

import { RespanLogType } from "@respan/respan-sdk";

export interface VercelSpanConfig {
  /** Traceloop span kind (workflow, agent, task, tool) */
  kind: string;
  /** Respan log type for backend categorization */
  logType: string;
  /** Whether this span represents an LLM call (triggers prompt/completion enrichment) */
  isLLM: boolean;
}

// ── Detailed spans (leaf nodes with actual LLM/embedding/tool data) ─────────

export const VERCEL_SPAN_CONFIG: Record<string, VercelSpanConfig> = {
  // LLM generation (detailed spans carry response data)
  "ai.generateText.doGenerate": { kind: RespanLogType.TASK, logType: RespanLogType.TEXT, isLLM: true },
  "ai.streamText.doStream":     { kind: RespanLogType.TASK, logType: RespanLogType.TEXT, isLLM: true },
  "ai.generateObject.doGenerate": { kind: RespanLogType.TASK, logType: RespanLogType.TEXT, isLLM: true },
  "ai.streamObject.doStream":   { kind: RespanLogType.TASK, logType: RespanLogType.TEXT, isLLM: true },

  // Embeddings
  "ai.embed.doEmbed":     { kind: RespanLogType.TASK, logType: RespanLogType.EMBEDDING, isLLM: false },
  "ai.embedMany.doEmbed": { kind: RespanLogType.TASK, logType: RespanLogType.EMBEDDING, isLLM: false },

  // Tool calls
  "ai.toolCall": { kind: RespanLogType.TOOL, logType: RespanLogType.TOOL, isLLM: false },

  // Agent / workflow
  "ai.agent":      { kind: RespanLogType.AGENT,    logType: RespanLogType.AGENT,    isLLM: false },
  "ai.agent.run":  { kind: RespanLogType.AGENT,    logType: RespanLogType.AGENT,    isLLM: false },
  "ai.agent.step": { kind: RespanLogType.TASK,     logType: RespanLogType.TASK,     isLLM: false },
  "ai.workflow":   { kind: RespanLogType.WORKFLOW,  logType: RespanLogType.WORKFLOW, isLLM: false },

  // Function / handoff
  "ai.function": { kind: RespanLogType.TOOL, logType: RespanLogType.TOOL, isLLM: false },
  "ai.handoff":  { kind: RespanLogType.TASK, logType: RespanLogType.TASK, isLLM: false },

  // Media
  "ai.transcript": { kind: RespanLogType.TASK, logType: RespanLogType.TEXT, isLLM: false },
  "ai.speech":     { kind: RespanLogType.TASK, logType: RespanLogType.TEXT, isLLM: false },

  // Other
  "ai.response":          { kind: RespanLogType.TASK, logType: RespanLogType.TEXT,     isLLM: true },
  "ai.stream.firstChunk": { kind: RespanLogType.TASK, logType: RespanLogType.TEXT,     isLLM: false },
};

// ── Parent wrapper spans (structural only, no LLM data) ─────────────────────

export const VERCEL_PARENT_SPANS: Record<string, string> = {
  "ai.generateText":   RespanLogType.TASK,
  "ai.streamText":     RespanLogType.TASK,
  "ai.generateObject": RespanLogType.TASK,
  "ai.streamObject":   RespanLogType.TASK,
  "ai.embed":          RespanLogType.TASK,
  "ai.embedMany":      RespanLogType.TASK,
};

// Classic-telemetry LLM wrappers whose detailed `.doGenerate`/`.doStream`
// child carries the actual model/input/output. Marked with the internal
// drop attribute so the semantic export style removes them (children are
// reparented); legacy style still exports them. Modern AI SDK 7 telemetry
// emits flat spans, so no modern names belong here.
export const VERCEL_STRUCTURAL_LLM_PARENT_SPANS = new Set([
  "ai.generateText",
  "ai.streamText",
  "ai.generateObject",
  "ai.streamObject",
]);
