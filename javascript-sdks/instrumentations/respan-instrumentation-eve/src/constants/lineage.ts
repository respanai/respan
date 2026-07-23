/**
 * Private bridge between Eve's authored runtime context and this translator.
 *
 * Eve reserves `eve.*` for framework-owned values. The helper therefore uses
 * a package-private namespace that AI SDK 7 flattens under
 * `ai.settings.context.*`. These keys are consumed and stripped before export;
 * they are not additions to the public Respan span contract.
 */
export const EVE_RESPAN_BRIDGE_RUNTIME_CONTEXT_KEY = "__respan_eve";
export const EVE_RESPAN_BRIDGE_RUNTIME_CONTEXT_PREFIX =
  EVE_RESPAN_BRIDGE_RUNTIME_CONTEXT_KEY + ".";

export const EVE_RESPAN_LINEAGE_ROOT_SESSION_ID_ATTRIBUTE =
  "ai.settings.context.__respan_eve.lineage.rootSessionId";
export const EVE_RESPAN_LINEAGE_PARENT_SESSION_ID_ATTRIBUTE =
  "ai.settings.context.__respan_eve.lineage.sessionId";
export const EVE_RESPAN_LINEAGE_PARENT_CALL_ID_ATTRIBUTE =
  "ai.settings.context.__respan_eve.lineage.callId";
export const EVE_RESPAN_LINEAGE_PARENT_TURN_ID_ATTRIBUTE =
  "ai.settings.context.__respan_eve.lineage.turn.id";
export const EVE_RESPAN_LINEAGE_PARENT_TURN_SEQUENCE_ATTRIBUTE =
  "ai.settings.context.__respan_eve.lineage.turn.sequence";

// Package-private handoff consumed by EveProcessorWrapper before Respan export.
export const EVE_RESPAN_INTERNAL_EXPORT_TRACE_ID_ATTRIBUTE =
  "respan.instrumentation.eve.export_trace_id";
