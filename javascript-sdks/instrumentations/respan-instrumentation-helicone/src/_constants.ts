/** Helicone-owned payload and header names stay local to this adapter. */
export const HeliconeFields = {
  EVENT_TYPE: "_type",
  TOOL_NAME: "toolName",
  OPERATION: "operation",
  NAME: "name",
  INPUT: "input",
  ARGUMENTS: "arguments",
  TOP_K: "topK",
  DATABASE_NAME: "databaseName",
  META: "meta",
} as const;

export const HeliconeEventTypes = {
  TOOL: "tool",
  VECTOR_DB: "vector_db",
  DATA: "data",
} as const;

export const HeliconeHeaders = {
  USER_ID: "helicone-user-id",
  SESSION_ID: "helicone-session-id",
  PROPERTY_PREFIX: "helicone-property-",
} as const;

export const MAX_SERIALIZED_BYTES = 1_000_000;
export const MAX_ERROR_MESSAGE_CHARS = 8_192;
export const MAX_INDEXED_PROMPT_MESSAGES = 128;

/**
 * Required by Respan's published span contract. The installed Traceloop
 * semantic-conventions package does not yet export a constant for this key.
 */
export const TraceloopCompatibilityFields = {
  LLM_USAGE_CACHE_READ_INPUT_TOKENS: "llm.usage.cache_read_input_tokens",
} as const;
