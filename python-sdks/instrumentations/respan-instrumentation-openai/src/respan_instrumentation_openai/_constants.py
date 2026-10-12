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
PARSE_METHOD = "parse"

# Every request's raw httpx response passes through this method on the base
# clients; it is where the Respan gateway's log-id response header is read.
BASE_CLIENT_MODULE = "openai._base_client"
SYNC_BASE_CLIENT_CLASS = "SyncAPIClient"
ASYNC_BASE_CLIENT_CLASS = "AsyncAPIClient"
PROCESS_RESPONSE_METHOD = "_process_response"

# --- request-type values (what the backend keys on via llm.request.type) ----

REQUEST_TYPE_CHAT = "chat"
REQUEST_TYPE_COMPLETION = "completion"
REQUEST_TYPE_EMBEDDING = "embedding"

# ``opentelemetry-semantic-conventions-ai`` 0.5.x does not expose this key.
GEN_AI_RESPONSE_ID = "gen_ai.response.id"

# --- Respan gateway link ------------------------------------------------------
# Same values as ``respan_sdk.constants`` RESPAN_LOG_ID_RESPONSE_HEADER /
# RESPAN_GATEWAY_LOG_ID; restated here so this package keeps working on the
# released respan-sdk floor.

# The Respan gateway returns the unique_id of the log row it wrote here.
RESPAN_LOG_ID_RESPONSE_HEADER = "X-Respan-Log-Id"
# A span whose call that row records carries the id and no usage/content.
RESPAN_GATEWAY_LOG_ID = "respan.gateway.log_id"

# --- response / message field keys ------------------------------------------

ROLE_KEY = "role"
CONTENT_KEY = "content"
TOOL_CALLS_KEY = "tool_calls"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
