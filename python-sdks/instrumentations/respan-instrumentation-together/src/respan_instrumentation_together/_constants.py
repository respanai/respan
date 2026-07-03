"""Together AI instrumentation-local constants."""

TOGETHER_INSTRUMENTATION_NAME = "together"
TOGETHER_SYSTEM_NAME = "together"

TOGETHER_CHAT_SPAN_NAME = "together.chat"
TOGETHER_COMPLETION_SPAN_NAME = "together.completion"
TOGETHER_EMBEDDING_SPAN_NAME = "together.embedding"
TOGETHER_RERANK_SPAN_NAME = "together.rerank"
TOGETHER_IMAGE_SPAN_NAME = "together.image"

TOGETHER_CHAT_COMPLETIONS_MODULE = "together.resources.chat.completions"
TOGETHER_TEXT_COMPLETIONS_MODULE = "together.resources.completions"
TOGETHER_EMBEDDINGS_MODULE = "together.resources.embeddings"
TOGETHER_IMAGES_MODULE = "together.resources.images"
TOGETHER_RERANK_MODULE = "together.resources.rerank"

COMPLETIONS_RESOURCE_CLASS_NAME = "CompletionsResource"
ASYNC_COMPLETIONS_RESOURCE_CLASS_NAME = "AsyncCompletionsResource"
EMBEDDINGS_RESOURCE_CLASS_NAME = "EmbeddingsResource"
ASYNC_EMBEDDINGS_RESOURCE_CLASS_NAME = "AsyncEmbeddingsResource"
IMAGES_RESOURCE_CLASS_NAME = "ImagesResource"
ASYNC_IMAGES_RESOURCE_CLASS_NAME = "AsyncImagesResource"
RERANK_RESOURCE_CLASS_NAME = "RerankResource"
ASYNC_RERANK_RESOURCE_CLASS_NAME = "AsyncRerankResource"

CREATE_METHOD_NAME = "create"
GENERATE_METHOD_NAME = "generate"

ASSISTANT_ROLE = "assistant"
FUNCTION_KEY = "function"
FUNCTION_TOOL_TYPE = "function"
SYSTEM_ROLE = "system"
TOOL_ROLE = "tool"
USER_ROLE = "user"

ARGUMENTS_KEY = "arguments"
B64_JSON_KEY = "b64_json"
CHOICES_KEY = "choices"
CONTENT_KEY = "content"
DATA_KEY = "data"
DELTA_KEY = "delta"
DESCRIPTION_KEY = "description"
DOCUMENT_KEY = "document"
DOCUMENTS_KEY = "documents"
EMBEDDING_KEY = "embedding"
FINISH_REASON_KEY = "finish_reason"
FUNCTION_CALL_KEY = "function_call"
ID_KEY = "id"
IMAGE_URL_KEY = "image_url"
INDEX_KEY = "index"
INPUT_KEY = "input"
MESSAGE_KEY = "message"
MESSAGES_KEY = "messages"
MODEL_KEY = "model"
NAME_KEY = "name"
PARAMETERS_KEY = "parameters"
PROMPT_KEY = "prompt"
QUERY_KEY = "query"
RELEVANCE_SCORE_KEY = "relevance_score"
RESULTS_KEY = "results"
ROLE_KEY = "role"
STREAM_KEY = "stream"
TEXT_KEY = "text"
TOOLS_KEY = "tools"
TOOL_CALL_ID_KEY = "tool_call_id"
TOOL_CALLS_KEY = "tool_calls"
TYPE_KEY = "type"
USAGE_KEY = "usage"
URL_KEY = "url"

BANNED_ALIAS_ATTRS = frozenset(
    {
        "respan.span.tools",
        "respan.span.tool_calls",
        "respan.span.handoffs",
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "span_tools",
        "has_tool_calls",
        "parallel_tool_calls",
    }
)
