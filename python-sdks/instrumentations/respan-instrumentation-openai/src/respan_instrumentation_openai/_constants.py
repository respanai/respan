"""Constants for the native OpenAI SDK instrumentation.

This package is deliberately independent of Traceloop's
``opentelemetry-instrumentation-openai`` / ``opentelemetry-semantic-conventions-ai``.
Every attribute-name string the backend ingests is defined *here* (the SDK owns
its own convention constants), while the few genuinely shared ones come from
``respan_sdk.constants`` (the SDK core, not Traceloop).
"""

from __future__ import annotations

# --- system + span names ----------------------------------------------------

OPENAI_SYSTEM = "openai"

CHAT_SPAN_NAME = "openai.chat"
EMBEDDING_SPAN_NAME = "openai.embeddings"
COMPLETION_SPAN_NAME = "openai.completion"
RESPONSE_SPAN_NAME = "openai.response"

# --- SDK module / class / method targets to monkey-patch --------------------

CHAT_MODULE = "openai.resources.chat.completions"
EMBEDDINGS_MODULE = "openai.resources.embeddings"
COMPLETIONS_MODULE = "openai.resources.completions"
RESPONSES_MODULE = "openai.resources.responses.responses"

SYNC_CHAT_CLASS = "Completions"
ASYNC_CHAT_CLASS = "AsyncCompletions"
SYNC_EMBEDDINGS_CLASS = "Embeddings"
ASYNC_EMBEDDINGS_CLASS = "AsyncEmbeddings"
SYNC_COMPLETIONS_CLASS = "Completions"
ASYNC_COMPLETIONS_CLASS = "AsyncCompletions"
SYNC_RESPONSES_CLASS = "Responses"
ASYNC_RESPONSES_CLASS = "AsyncResponses"

CREATE_METHOD = "create"

# --- request-type values (what the backend keys on via llm.request.type) ----

REQUEST_TYPE_CHAT = "chat"
REQUEST_TYPE_COMPLETION = "completion"
REQUEST_TYPE_EMBEDDING = "embedding"

# --- attribute-name strings (the documented ingest contract) ----------------
# These mirror the wire format the Respan backend parses. We define them here so
# the package carries no Traceloop dependency.

TRACELOOP_SPAN_KIND = "traceloop.span.kind"
TRACELOOP_ENTITY_NAME = "traceloop.entity.name"
TRACELOOP_ENTITY_PATH = "traceloop.entity.path"
TRACELOOP_ENTITY_INPUT = "traceloop.entity.input"
TRACELOOP_ENTITY_OUTPUT = "traceloop.entity.output"

LLM_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
LLM_PROMPTS = "gen_ai.prompt"
LLM_COMPLETIONS = "gen_ai.completion"
LLM_REQUEST_FUNCTIONS = "llm.request.functions"
LLM_RESPONSE_MODEL = "gen_ai.response.model"
LLM_RESPONSE_ID = "gen_ai.response.id"

SPAN_KIND_LLM = "llm"

# --- response / message field keys ------------------------------------------

ROLE_KEY = "role"
CONTENT_KEY = "content"
TOOL_CALLS_KEY = "tool_calls"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
