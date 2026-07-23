"""Replicate instrumentation constants."""

REPLICATE_INSTRUMENTATION_NAME = "replicate"
REPLICATE_SYSTEM_NAME = "replicate"

REPLICATE_RUN_SPAN_NAME = "replicate.run"
REPLICATE_STREAM_SPAN_NAME = "replicate.stream"
REPLICATE_PREDICTION_CREATE_SPAN_NAME = "replicate.predictions.create"
REPLICATE_PREDICTION_WAIT_SPAN_NAME = "replicate.prediction.wait"

ASYNC_PREFIX = "async_"

DEPLOYMENT_KEY = "deployment"
ERROR_KEY = "error"
ID_KEY = "id"
INPUT_KEY = "input"
LOGS_KEY = "logs"
METRICS_KEY = "metrics"
MODEL_KEY = "model"
OUTPUT_KEY = "output"
PREDICTION_KEY = "prediction"
PROMPT_KEY = "prompt"
REF_KEY = "ref"
RESPAN_PARAMS_KEY = "respan_params"
RESPAN_PARAMS_MODEL_KEY = "model"
STATUS_KEY = "status"
STREAM_KEY = "stream"
VERSION_KEY = "version"
PREDICTION_RESPAN_MODEL_ATTR = "_respan_reported_model"

ASSISTANT_ROLE = "assistant"
USER_ROLE = "user"

MAX_STREAM_CHUNKS = 200
MAX_TEXT_LENGTH = 16_000

OFF_CONTRACT_ALIASES = {
    "completion_tokens",
    "has_tool_calls",
    "model",
    "parallel_tool_calls",
    "prompt_tokens",
    "span_tools",
    "tool_calls",
    "tools",
    "total_request_tokens",
}
