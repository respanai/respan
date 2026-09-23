/**
 * n8n-owned OpenTelemetry keys.
 *
 * These stay local to the n8n translator by contract. They are not Respan
 * public span attributes and must not be promoted into @respan/respan-sdk.
 */
export const N8N_SPAN_NAMES = {
  workflow: "workflow.execute",
  node: "node.execute",
  aiGenerate: "ai.generateText",
  aiGenerateDetail: "ai.generateText.doGenerate",
  aiStream: "ai.streamText",
  aiStreamDetail: "ai.streamText.doStream",
  aiToolCall: "ai.toolCall",
} as const;

export const N8N_AI_SDK_LLM_SPAN_NAMES = new Set<string>([
  N8N_SPAN_NAMES.aiGenerate,
  N8N_SPAN_NAMES.aiGenerateDetail,
  N8N_SPAN_NAMES.aiStream,
  N8N_SPAN_NAMES.aiStreamDetail,
]);

export const N8N_AI_SDK_STRUCTURAL_LLM_SPAN_NAMES = new Set<string>([
  N8N_SPAN_NAMES.aiGenerate,
  N8N_SPAN_NAMES.aiStream,
]);

export const N8N_ATTRIBUTES = {
  instanceId: "n8n.instance.id",
  instanceRole: "n8n.instance.role",
  projectId: "n8n.project.id",
  workflowId: "n8n.workflow.id",
  workflowName: "n8n.workflow.name",
  workflowVersionId: "n8n.workflow.version_id",
  workflowNodeCount: "n8n.workflow.node_count",
  executionId: "n8n.execution.id",
  executionMode: "n8n.execution.mode",
  executionStatus: "n8n.execution.status",
  executionIsRetry: "n8n.execution.is_retry",
  executionRetryOf: "n8n.execution.retry_of",
  executionErrorType: "n8n.execution.error_type",
  nodeId: "n8n.node.id",
  nodeName: "n8n.node.name",
  nodeType: "n8n.node.type",
  nodeTypeVersion: "n8n.node.type_version",
  nodeItemsInput: "n8n.node.items.input",
  nodeItemsOutput: "n8n.node.items.output",
  nodeTerminationReason: "n8n.node.termination_reason",
  continuationReason: "n8n.continuation.reason",
} as const;

export const N8N_ATTRIBUTE_PREFIX = "n8n.";
export const N8N_AGENT_SCOPE = "@n8n/agents";
export const AI_TELEMETRY_METADATA_PREFIX = "ai.telemetry.metadata.";

export const N8N_AGENT_METADATA_KEYS = [
  "agent_id",
  "project_id",
  "thread_id",
  "source",
  "user_id",
  "model_id",
  "execution_id",
  "workflow_id",
  "node_id",
] as const;

export const N8N_MEMORY_OPERATIONS = new Set(["query_memory", "save_memory"]);

export const N8N_MEMORY_ATTRIBUTE_PREFIX = "gen_ai.memory.";

export const OFF_CONTRACT_ALIAS_KEYS = [
  "tools",
  "tool_calls",
  "model",
  "prompt_tokens",
  "completion_tokens",
  "total_request_tokens",
  "span_tools",
  "has_tool_calls",
  "parallel_tool_calls",
  "respan.span.tools",
  "respan.span.tool_calls",
  "respan.span.handoffs",
] as const;
