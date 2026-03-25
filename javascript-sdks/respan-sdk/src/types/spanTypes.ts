export enum RespanSpanAttributes {
    // Span attributes
    RESPAN_SPAN_CUSTOM_ID = "respan.span_params.custom_identifier",

    // Customer params
    RESPAN_CUSTOMER_PARAMS_ID = "respan.customer_params.customer_identifier",
    RESPAN_CUSTOMER_PARAMS_EMAIL = "respan.customer_params.email",
    RESPAN_CUSTOMER_PARAMS_NAME = "respan.customer_params.name",
    
    // Evaluation params
    RESPAN_EVALUATION_PARAMS_ID = "respan.evaluation_params.evaluation_identifier",

    // Threads
    RESPAN_THREADS_ID = "respan.threads.thread_identifier",

    // Trace
    RESPAN_TRACE_GROUP_ID = "respan.trace.trace_group_identifier",

    // Metadata
    RESPAN_METADATA = "respan.metadata",
    RESPAN_PROPERTIES = "respan.properties",

    // Prompt & environment
    RESPAN_PROMPT = "respan.prompt",
    RESPAN_ENVIRONMENT = "respan.environment",

    // Span links
    RESPAN_LINK_TIMESTAMP = "respan.link.timestamp",

    // Processor routing (used by decorators to target specific processors)
    RESPAN_PROCESSORS = "respan.processors",

    // Logging
    RESPAN_LOG_METHOD = "respan.entity.log_method",
    RESPAN_LOG_TYPE = "respan.entity.log_type",
    RESPAN_LOG_ID = "respan.entity.log_id",
    RESPAN_LOG_PARENT_ID = "respan.entity.log_parent_id",
    RESPAN_LOG_ROOT_ID = "respan.entity.log_root_id",
    RESPAN_LOG_SOURCE = "respan.entity.log_source",

    // OpenInference attributes (used for OI → Traceloop/GenAI enrichment)
    OPENINFERENCE_SPAN_KIND = "openinference.span.kind",
    OPENINFERENCE_LLM_MODEL_NAME = "llm.model_name",
    OPENINFERENCE_LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt",
    OPENINFERENCE_LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion",

    // GenAI / LLM semantic conventions (for packages that don't depend on @traceloop/ai-semantic-conventions)
    GEN_AI_SYSTEM = "gen_ai.system",
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model",
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens",
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens",
    LLM_REQUEST_TYPE = "llm.request.type",
    LLM_SYSTEM = "llm.system",
}

export const RESPAN_SPAN_ATTRIBUTES_MAP: { [key: string]: string } = {
    customer_identifier: RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID,
    customer_email: RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_EMAIL,
    customer_name: RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_NAME,
    evaluation_identifier: RespanSpanAttributes.RESPAN_EVALUATION_PARAMS_ID,
    thread_identifier: RespanSpanAttributes.RESPAN_THREADS_ID,
    custom_identifier: RespanSpanAttributes.RESPAN_SPAN_CUSTOM_ID,
    trace_group_identifier: RespanSpanAttributes.RESPAN_TRACE_GROUP_ID,
    group_identifier: RespanSpanAttributes.RESPAN_TRACE_GROUP_ID,
    metadata: RespanSpanAttributes.RESPAN_METADATA,
    properties: RespanSpanAttributes.RESPAN_PROPERTIES,
    prompt: RespanSpanAttributes.RESPAN_PROMPT,
    environment: RespanSpanAttributes.RESPAN_ENVIRONMENT,
};

// Type for valid span attribute values
export type SpanAttributeValue = string | number | boolean | Array<string | number | boolean>;
