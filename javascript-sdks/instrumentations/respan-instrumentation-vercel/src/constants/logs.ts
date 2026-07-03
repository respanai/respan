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
  "ai.function": { kind: RespanLogType.TOOL, logType: RespanLogType.FUNCTION, isLLM: false },
  "ai.handoff":  { kind: RespanLogType.TASK, logType: RespanLogType.HANDOFF,  isLLM: false },

  // Media
  "ai.transcript": { kind: RespanLogType.TASK, logType: RespanLogType.TRANSCRIPTION, isLLM: false },
  "ai.speech":     { kind: RespanLogType.TASK, logType: RespanLogType.SPEECH,        isLLM: false },

  // Other
  "ai.response":          { kind: RespanLogType.TASK, logType: RespanLogType.TEXT,     isLLM: true },
  "ai.stream.firstChunk": { kind: RespanLogType.TASK, logType: RespanLogType.TEXT,     isLLM: false },
};

// ── Parent wrapper spans (structural only, no LLM data) ─────────────────────

export const VERCEL_PARENT_SPANS: Record<string, string> = {
  "ai.generateText":   RespanLogType.TEXT,
  "ai.streamText":     RespanLogType.TEXT,
  "ai.generateObject": RespanLogType.TEXT,
  "ai.streamObject":   RespanLogType.TEXT,
  "ai.embed":          RespanLogType.EMBEDDING,
  "ai.embedMany":      RespanLogType.EMBEDDING,
};
