import { z } from "zod";

// Log type definition
export const LOG_TYPE_VALUES = [
  "text",
  "chat",
  "completion",
  "response",
  "embedding",
  "transcription",
  "speech",
  "workflow",
  "task",
  "tool",
  "agent",
  "handoff",
  "guardrail",
  "function",
  "custom",
  "generation",
  "unknown",
] as const;

export type LogType = (typeof LOG_TYPE_VALUES)[number];

export const LOG_METHOD_VALUES = [
  "inference",
  "logging_api",
  "batch",
  "python_tracing",
  "ts_tracing",
] as const;

export type LogMethod = (typeof LOG_METHOD_VALUES)[number];

// Basic utility schemas
const StringOrNumberSchema = z.union([z.string(), z.number()]);
const DateTimeSchema = z.union([z.string(), z.date()]);

// Message content schemas
const ImageURLSchema = z.object({
  url: z.string(),
  detail: z.string().optional(),
});

const CacheControlSchema = z.object({
  type: z.string(),
});

const BaseContentSchema = z.object({
  type: z.string(),
});

const TextContentSchema = BaseContentSchema.extend({
  text: z.string(),
  cache_control: CacheControlSchema.optional(),
});

const ImageContentSchema = BaseContentSchema.extend({
  image_url: z.union([ImageURLSchema, z.string()]),
});

const InputImageSchema = BaseContentSchema.extend({
  file: z.string(),
  providerData: z.record(z.string(), z.any()).optional(),
});

const FileContentSchema = BaseContentSchema.extend({
  file: z.string(),
  providerData: z.record(z.string(), z.any()).optional(),
});

const ToolUseContentSchema = BaseContentSchema.extend({
  id: z.string().optional(),
  name: z.string().optional(),
  input: z.record(z.string(), z.any()).optional(),
});

const ToolResultContentSchema = BaseContentSchema.extend({
  tool_use_id: z.any(),
  content: z.string(),
}).transform((data) => {
  if ("tool_use_id" in data) return data;
  return {
    type: (data as any).type,
    tool_use_id: (data as any).toolCallId,
    content: (data as any).result || (data as any).content,
  };
});

const OutputTextContentSchema = BaseContentSchema.extend({
  text: z.string(),
  annotations: z.array(z.union([z.record(z.string(), z.any()), z.string()])).optional(),
  cache_control: CacheControlSchema.optional(),
});

// Combined message content schema with transform to handle string content
const MessageContentSchema = z
  .union([
    z.string(),
    z.array(
      z.union([
        TextContentSchema,
        ImageContentSchema,
        FileContentSchema,
        ToolUseContentSchema,
        ToolResultContentSchema,
        OutputTextContentSchema,
        InputImageSchema,
        z.object({
          type: z.string(),
          text: z.string(),
        }),
        // Catch-all for other content types
        z.record(z.string(), z.any()),
      ])
    ),
  ])
  .transform((content) => {
    if (Array.isArray(content)) {
      // Try to extract text from array of objects with text field
      const textParts = content
        .map((item) => {
          if (typeof item === "object") {
            // Handle both camelCase and snake_case keys
            if ("text" in item) return item.text;
            if ("content" in item && typeof item.content === "string")
              return item.content;
            if ("result" in item && typeof item.result === "string")
              return item.result;
          }
          return null;
        })
        .filter((text) => text !== null);

      if (textParts.length > 0) {
        return textParts.join("\n");
      }
      // Just return the original array if transformation isn't possible
      return content;
    }
    return content;
  });

// Tool related schemas
const ToolCallFunctionSchema = z
  .object({
    name: z.string().optional(),
    arguments: z.union([z.string(), z.record(z.string(), z.any())]).optional(),
  })
  .catchall(z.any()); // Allow additional properties

const ToolCallSchema = z
  .object({
    type: z.string().default("function"), // Only require type, default to "function"
    id: z.string().optional(),
    function: ToolCallFunctionSchema.optional(),
  })
  .catchall(z.any()) // Allow any additional properties
  .transform((data) => {
    // Create a shallow copy to avoid modifying the original
    const result: Record<string, any> = { ...data };

    // Handle ID mapping for consistency (only if needed)
    if (
      !result.id &&
      ((data as any).toolCallId || (data as any).tool_call_id)
    ) {
      result.id = (data as any).toolCallId || (data as any).tool_call_id;
    }

    // If we have args/toolName but no function object, create basic structure
    if (
      ((data as any).toolName || (data as any).name || (data as any).args) &&
      !result.function
    ) {
      result.function = {};

      if ((data as any).toolName || (data as any).name) {
        result.function.name = (data as any).toolName || (data as any).name;
      }

      if ((data as any).args) {
        result.function.arguments =
          typeof (data as any).args === "string"
            ? (data as any).args
            : JSON.stringify((data as any).args);
      }
    }

    return result;
  });

const ToolChoiceSchema = z
  .object({
    type: z.string(),
    function: z
      .object({
        name: z.string(),
      })
      .optional(),
  })
  .optional();

// Function tool schema for BasicLLMParams
// Accept both OpenAI-style nested `{ type: "function", function: { ... } }`
// and flattened `{ type: "function", name, description, parameters }` shapes.
const FunctionToolSchema = z
  .union([
    z.object({
      type: z.literal("function"),
      function: z.object({
        name: z.string(),
        description: z.string().optional(),
        parameters: z.record(z.string(), z.any()).optional(),
      }),
    }),
    z.object({
      type: z.literal("function"),
      name: z.string(),
      description: z.string().optional(),
      parameters: z.record(z.string(), z.any()).optional(),
    }),
  ])
  .transform((data) => {
    if ("function" in data && data.function) {
      return data;
    }
    // Normalize flat shape into nested function object
    const { name, description, parameters, ...rest } = data as any;
    return {
      ...rest,
      function: {
        name,
        ...(description ? { description } : {}),
        ...(parameters ? { parameters } : {}),
      },
    };
  });

// Base message schema with flexible role
const MessageSchema = z
  .object({
    role: z.string(),
    content: MessageContentSchema.optional(),
    name: z.string().optional(),
    tool_call_id: z.any().optional(),
    tool_calls: z.array(z.record(z.string(), z.any())).optional(),
    experimental_providerMetadata: z
      .object({
        anthropic: z.object({
          cacheControl: z.object({
            type: z.string(),
          }),
        }),
      })
      .optional(),
  })
  .transform((data) => {
    // Handle camelCase to snake_case conversion
    if ("toolCallId" in data) {
      return {
        ...data,
        tool_call_id: data.toolCallId,
      };
    }
    return data;
  });

// Metadata schema
const MetadataSchema = z.record(z.string(), z.any()).optional();

// Usage schema
const UsageSchema = z.object({
  prompt_tokens: z.number().optional(),
  completion_tokens: z.number().optional(),
  total_tokens: z.number().optional(),
  cache_creation_input_tokens: z.number().optional(),
  cache_creation_prompt_tokens: z.number().optional(),
  cache_read_input_tokens: z.number().optional(),
  completion_tokens_details: z.record(z.string(), z.any()).optional(),
  prompt_tokens_details: z.record(z.string(), z.any()).optional(),
});

// Supporting schemas for Respan params
const OverrideConfigSchema = z.object({
  messages_override_mode: z
    .enum(["override", "append"])
    .optional()
    .default("override"),
});

const EvaluatorToRunSchema = z.record(z.string(), z.any()); // Placeholder for evaluator schema
const EvalInputsSchema = z.record(z.string(), z.any()); // Placeholder for eval inputs schema

const EvaluationParamsSchema = z.object({
  evaluators: z.array(EvaluatorToRunSchema).optional().default([]),
  evaluation_identifier: StringOrNumberSchema.default(""),
  last_n_messages: z.number().optional().default(1),
  eval_inputs: EvalInputsSchema.optional().default({}),
  sample_percentage: z.number().optional(),
});

const LoadBalanceModelSchema = z.object({
  model: z.string(),
  credentials: z.record(z.string(), z.any()).optional(),
  weight: z
    .number()
    .refine((val) => val > 0, "Weight has to be greater than 0"),
});

const LoadBalanceGroupSchema = z.object({
  group_id: z.string(),
  models: z.array(LoadBalanceModelSchema).optional(),
});

const PostHogIntegrationSchema = z.object({
  posthog_api_key: z.string(),
  posthog_base_url: z.string(),
});

const CustomerSchema = z.object({
  customer_identifier: StringOrNumberSchema.optional(),
  name: z.string().optional(),
  email: z.string().optional(),
  period_start: DateTimeSchema.optional(),
  period_end: DateTimeSchema.optional(),
  budget_duration: z.enum(["daily", "weekly", "monthly", "yearly"]).optional(),
  period_budget: z.number().optional(),
  markup_percentage: z.number().optional(),
  total_budget: z.number().optional(),
  metadata: z.record(z.string(), z.any()).optional(),
  rate_limit: z.number().optional(),
});

const CacheOptionsSchema = z.object({
  cache_by_customer: z.boolean().optional(),
  omit_log: z.boolean().optional(),
});

const RetryParamsSchema = z.object({
  num_retries: z
    .number()
    .optional()
    .default(3)
    .refine((val) => val > 0, "num_retries has to be greater than 0"),
  retry_after: z
    .number()
    .optional()
    .default(0.2)
    .refine((val) => val > 0, "retry_after has to be greater than 0"),
  retry_enabled: z.boolean().optional().default(true),
});

const RespanAPIControlParamsSchema = z.object({
  block: z.boolean().optional(),
});

const PromptParamSchema = z.object({
  prompt_id: z.string().optional(),
  is_custom_prompt: z.boolean().optional().default(false),
  version: z.number().optional(),
  variables: z.record(z.string(), z.any()).optional(),
  echo: z.boolean().optional().default(true),
  // v2 fields (set schema_version=2 to use v2 prompt processing)
  schema_version: z.number().optional(),
  patch: z.record(z.string(), z.any()).optional(),
  // v1 fields (preserved for back-compat, ignored when schema_version=2)
  override: z.boolean().optional().default(false),
  override_params: z.record(z.string(), z.any()).optional(), // BasicLLMParams placeholder
  override_config: OverrideConfigSchema.optional(),
});

const LinkupParamsSchema = z.record(z.string(), z.any()); // Placeholder
const Mem0ParamsSchema = z.record(z.string(), z.any()); // Placeholder
const ProviderCredentialTypeSchema = z.record(z.string(), z.any()); // Placeholder

// Basic LLM Parameters Schema
const BasicLLMParamsSchema = z.object({
  echo: z.boolean().optional(),
  frequency_penalty: z.number().optional(),
  logprobs: z.boolean().optional(),
  logit_bias: z.record(z.string(), z.number()).optional(),
  messages: z.array(MessageSchema).optional(),
  model: z.string().optional(),
  max_tokens: z.number().optional(),
  max_completion_tokens: z.number().optional(),
  n: z.number().optional(),
  parallel_tool_calls: z.boolean().optional(),
  presence_penalty: z.number().optional(),
  stop: z.union([z.array(z.string()), z.string()]).optional(),
  stream: z.boolean().optional(),
  stream_options: z.record(z.string(), z.any()).optional(),
  temperature: z.number().optional(),
  timeout: z.number().optional(),
  tools: z.array(FunctionToolSchema).optional(),
  response_format: z.record(z.string(), z.any()).optional(),
  reasoning_effort: z.string().optional(),
  tool_choice: z
    .union([z.enum(["auto", "none", "required"]), ToolChoiceSchema])
    .optional(),
  top_logprobs: z.number().optional(),
  top_p: z.number().optional(),
});

// Basic Embedding Parameters Schema (placeholder)
const BasicEmbeddingParamsSchema = z.object({
  input: z.union([z.string(), z.array(z.string())]).optional(),
  model: z.string().optional(),
  encoding_format: z.string().optional(),
  dimensions: z.number().optional(),
  user: z.string().optional(),
});

// Respan Parameters Schema with regions
const RespanParamsSchema = z.object({
  //#region time
  start_time: DateTimeSchema.optional(),
  timestamp: DateTimeSchema.optional(),
  //#endregion time

  //#region unique identifiers
  custom_identifier: StringOrNumberSchema.optional(),
  response_id: z.string().optional(),
  //#endregion unique identifiers

  //#region status
  //#region error handling
  error_message: z.string().optional(),
  warnings: z.string().optional(),
  //#endregion error handling
  status_code: z.number().optional(),
  //#endregion status

  //#region log identifier/grouping
  group_identifier: StringOrNumberSchema.optional(),
  evaluation_identifier: StringOrNumberSchema.optional(),
  //#endregion log identifier/grouping

  //#region log input/output
  input: z.string().optional(),
  output: z.string().optional(),
  prompt_messages: z.array(MessageSchema).optional(),
  completion_message: MessageSchema.optional(),
  completion_messages: z.array(MessageSchema).optional(),
  completion_tokens: z.number().optional(),
  full_request: z.union([z.record(z.string(), z.any()), z.array(z.any())]).optional(),
  full_response: z.union([z.record(z.string(), z.any()), z.array(z.any())]).optional(),
  //#region special response types
  tool_calls: z.array(z.record(z.string(), z.any())).optional(),
  reasoning: z.array(z.record(z.string(), z.any())).optional(),
  //#endregion special response types
  //#endregion log input/output

  //#region cache params
  cache_enabled: z.boolean().optional(),
  cache_options: CacheOptionsSchema.optional(),
  cache_ttl: z.number().optional(),
  //#endregion cache params

  //#region usage
  //#region cost related
  cost: z.number().optional(),
  prompt_unit_price: z.number().optional(),
  completion_unit_price: z.number().optional(),
  //#endregion cost related

  //#region token usage
  prompt_tokens: z.number().optional(),
  prompt_cache_hit_tokens: z.number().optional(),
  prompt_cache_creation_tokens: z.number().optional(),
  usage: UsageSchema.optional(),
  //#endregion token usage
  //#endregion usage

  //#region user analytics
  customer_email: z.string().optional(),
  customer_name: z.string().optional(),
  customer_identifier: StringOrNumberSchema.optional(),
  customer_params: CustomerSchema.optional(),
  //#endregion user analytics

  //#region respan llm response control
  field_name: z.string().optional().default("data: "),
  delimiter: z.string().optional().default("\n\n"),
  disable_log: z.boolean().optional().default(false),
  request_breakdown: z.boolean().optional().default(false),
  //#endregion respan llm response control

  //#region respan logging control
  respan_api_controls: RespanAPIControlParamsSchema.optional(),
  log_method: z.enum(LOG_METHOD_VALUES).optional(),
  log_type: z.enum(LOG_TYPE_VALUES).optional(),
  //#endregion respan logging control

  //#region respan proxy options
  disable_fallback: z.boolean().optional().default(false),
  exclude_models: z.array(z.string()).optional(),
  exclude_providers: z.array(z.string()).optional(),
  fallback_models: z.array(z.string()).optional(),
  load_balance_group: LoadBalanceGroupSchema.optional(),
  load_balance_models: z.array(LoadBalanceModelSchema).optional(),
  retry_params: RetryParamsSchema.optional(),
  respan_params: z.record(z.string(), z.any()).optional(),
  //#region deprecated
  model_name_map: z.record(z.string(), z.string()).optional(),
  //#endregion deprecated
  //#endregion respan proxy options

  //#region embedding
  embedding: z.array(z.number()).optional(),
  //#endregion embedding

  //#region model information
  provider_id: z.string().optional(),
  //#endregion model information

  //#region audio
  audio_input_file: z.string().optional(),
  audio_output_file: z.string().optional(),
  //#endregion audio

  //#region evaluation
  note: z.string().optional(),
  category: z.string().optional(),
  eval_params: EvaluationParamsSchema.optional(),
  for_eval: z.boolean().optional(),
  positive_feedback: z.boolean().optional(),
  //#endregion evaluation

  //#region technical integrations
  linkup_params: LinkupParamsSchema.optional(),
  mem0_params: Mem0ParamsSchema.optional(),
  posthog_integration: PostHogIntegrationSchema.optional(),
  //#endregion technical integrations

  //#region custom properties
  metadata: z.record(z.string(), z.any()).optional(), // Map(String, String) — values coerced to strings
  properties: z.record(z.string(), z.any()).optional(), // Native JSON — preserves types (numbers, booleans, nested)
  //#endregion custom properties

  //#region prompt
  prompt: z.union([PromptParamSchema, z.string()]).optional(),
  variables: z.record(z.string(), z.any()).optional(),
  //#endregion prompt

  //#region llm response timing metrics
  generation_time: z.number().optional(),
  latency: z.number().optional(),
  ttft: z.number().optional(),
  time_to_first_token: z.number().optional(),
  routing_time: z.number().optional(),
  tokens_per_second: z.number().optional(),
  //#endregion llm response timing metrics

  //#region tracing
  trace_unique_id: z.string().optional(),
  trace_name: z.string().optional(),
  trace_group_identifier: z.string().optional(),
  span_unique_id: z.string().optional(),
  span_name: z.string().optional(),
  span_parent_id: z.string().optional(),
  span_path: z.string().optional(),
  span_handoffs: z.array(z.string()).optional(),
  span_tools: z.array(z.string()).optional(),
  span_workflow_name: z.string().optional(),

  //#region thread
  thread_identifier: StringOrNumberSchema.optional(),
  //#endregion thread

  //#endregion tracing
});

// Combined RespanPayloadSchema that merges RespanParams, BasicLLMParams, and BasicEmbeddingParams
export const RespanPayloadSchema = RespanParamsSchema.merge(
  BasicLLMParamsSchema
)
  .merge(BasicEmbeddingParamsSchema)
  .catchall(z.any());

export type RespanPayload = z.input<typeof RespanPayloadSchema>;

// Export individual schemas for use elsewhere
export {
  RespanParamsSchema,
  BasicLLMParamsSchema,
  BasicEmbeddingParamsSchema,
  MessageSchema,
  ToolCallSchema,
  ToolChoiceSchema,
  FunctionToolSchema,
  UsageSchema,
  MetadataSchema,
  PostHogIntegrationSchema,
  CustomerSchema,
  CacheOptionsSchema,
  RetryParamsSchema,
  LoadBalanceGroupSchema,
  LoadBalanceModelSchema,
  EvaluationParamsSchema,
  PromptParamSchema,
  OverrideConfigSchema,
};
