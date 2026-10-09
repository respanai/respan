/**
 * Eve-owned raw OpenTelemetry span names and attributes.
 *
 * These are translator-local inputs, not part of the public Respan contract.
 */
export const EVE_TURN_SPAN_NAME = "ai.eve.turn";
export const EVE_SCOPE_NAME = "eve";
export const EVE_ATTRIBUTE_PREFIX = "eve.";
export const EVE_VERSION = "eve.version";
export const EVE_ENVIRONMENT = "eve.environment";
export const EVE_SESSION_ID = "eve.session.id";
export const EVE_TURN_ID = "eve.turn.id";
export const EVE_TURN_SEQUENCE = "eve.turn.sequence";
export const EVE_STEP_INDEX = "eve.step.index";
export const EVE_CHANNEL_KIND = "eve.channel.kind";
export const EVE_RETRY_REASON = "eve.retry.reason";

// Eve's native agent trace schema (0.66+). These remain vendor-local inputs.
export const EVE_AGENT_RUN_ID = "agent.run.id";
export const EVE_AGENT_FRAMEWORK_VERSION = "agent.framework.version";
export const EVE_AGENT_TURN_ID = "agent.turn.id";
export const EVE_AGENT_TURN_SEQUENCE = "agent.turn.sequence";
export const EVE_AGENT_STEP_INDEX = "agent.step.index";
export const EVE_AGENT_CHANNEL_KIND = "agent.channel.kind";
export const EVE_AGENT_PARENT_RUN_ID = "agent.parent_run.id";
export const EVE_AGENT_PARENT_CALL_ID = "agent.parent_call.id";
export const EVE_AGENT_CONTENT_INPUT = "agent.trace.content.input";
export const EVE_AGENT_CONTENT_OUTPUT = "agent.trace.content.output";
export const EVE_AGENT_NAME = "agent.name";
export const EVE_AGENT_DELIVERY_INPUT = "agent.channel.delivery.input";
// Eve emits these draft GenAI memory keys before semconv 1.43 exports them.
export const EVE_MEMORY_RECORDS = "gen_ai.memory.records";
export const EVE_MEMORY_STORE_ID = "gen_ai.memory.store.id";
export const EVE_MEMORY_RECORD_COUNT = "gen_ai.memory.record.count";

export function isEveScope(scope: string | undefined): boolean {
  return scope === EVE_SCOPE_NAME || scope === "eve.agent";
}
