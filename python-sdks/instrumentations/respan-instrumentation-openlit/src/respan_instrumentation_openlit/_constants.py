"""OpenLIT-owned raw attribute names used by the contract translator."""

OPENLIT_INSTRUMENTATION_NAME = "openlit"
OPENLIT_SCOPE_PREFIX = "openlit"

# OpenLIT extensions which are not part of the upstream GenAI contract.
OPENLIT_RESPONSE_TOOL_CALLS = "gen_ai.response.tool_calls"
OPENLIT_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
OPENLIT_REQUEST_PROVIDER = "gen_ai.request.provider"
OPENLIT_TOOL_INPUT = "gen_ai.tool.input"
OPENLIT_TOOL_OUTPUT = "gen_ai.tool.output"
OPENLIT_TOOL_ARGS = "gen_ai.tool.args"
OPENLIT_WORKFLOW_INPUT = "gen_ai.workflow.input"
OPENLIT_WORKFLOW_OUTPUT = "gen_ai.workflow.output"

OPENLIT_OPERATION_LOG_TYPES = {
    "chat": "chat",
    "text_completion": "text",
    "embeddings": "embedding",
    "execute_tool": "tool",
    "invoke_agent": "agent",
    "create_agent": "agent",
    "execute_task": "task",
    "invoke_workflow": "workflow",
    "retrieval": "task",
    "vectordb": "task",
    "moderation": "guardrail",
}

OFF_CONTRACT_ALIASES = {
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "tools",
    "tool_calls",
    "span_tools",
    "has_tool_calls",
    "parallel_tool_calls",
    "respan.span.tools",
    "respan.span.tool_calls",
    "respan.span.handoffs",
}

# Official OTel GenAI fields which remain useful after OpenLIT native spans are
# normalized. OpenLIT emits a much larger gen_ai namespace containing
# vendor-only cost, metrics, framework, and legacy payload attributes.
STANDARD_GEN_AI_ATTRIBUTES = {
    "gen_ai.agent.description",
    "gen_ai.agent.id",
    "gen_ai.agent.name",
    "gen_ai.agent.version",
    "gen_ai.conversation.id",
    "gen_ai.data_source.id",
    "gen_ai.embeddings.dimension.count",
    "gen_ai.evaluation.explanation",
    "gen_ai.evaluation.name",
    "gen_ai.evaluation.score.label",
    "gen_ai.evaluation.score.value",
    "gen_ai.input.messages",
    "gen_ai.operation.name",
    "gen_ai.openai.request.response_format",
    "gen_ai.openai.request.seed",
    "gen_ai.openai.request.service_tier",
    "gen_ai.openai.response.service_tier",
    "gen_ai.openai.response.system_fingerprint",
    "gen_ai.output.messages",
    "gen_ai.output.type",
    "gen_ai.prompt",
    "gen_ai.completion",
    "gen_ai.provider.name",
    "gen_ai.request.choice.count",
    "gen_ai.request.encoding_formats",
    "gen_ai.request.frequency_penalty",
    "gen_ai.request.max_tokens",
    "gen_ai.request.model",
    "gen_ai.request.presence_penalty",
    "gen_ai.request.seed",
    "gen_ai.request.stream",
    "gen_ai.request.stop_sequences",
    "gen_ai.request.temperature",
    "gen_ai.request.top_k",
    "gen_ai.request.top_p",
    "gen_ai.response.finish_reasons",
    "gen_ai.response.id",
    "gen_ai.response.model",
    "gen_ai.response.time_to_first_chunk",
    "gen_ai.retrieval.documents",
    "gen_ai.retrieval.query.text",
    "gen_ai.system",
    "gen_ai.system_instructions",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.id",
    "gen_ai.tool.call.result",
    "gen_ai.tool.definitions",
    "gen_ai.tool.description",
    "gen_ai.tool.name",
    "gen_ai.tool.type",
    "gen_ai.token.type",
    "gen_ai.usage.cache_creation.input_tokens",
    "gen_ai.usage.cache_read.input_tokens",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.prompt_tokens",
    "gen_ai.usage.reasoning.output_tokens",
    "gen_ai.usage.completion_tokens",
    "gen_ai.usage.total_tokens",
    "gen_ai.usage.cache_creation_input_tokens",
    "gen_ai.usage.cache_read_input_tokens",
    "gen_ai.workflow.name",
}

STANDARD_DB_ATTRIBUTES = {
    "db.collection.name",
    "db.namespace",
    "db.operation.name",
    "db.query.parameter",
    "db.query.summary",
    "db.query.text",
    "db.response.returned_rows",
    "db.system.name",
}
