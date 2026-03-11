from enum import Enum
from typing import Dict, Literal


class LogMethodChoices(Enum):
    INFERENCE = "inference"  # Log from a generation api call postprocessing
    LOGGING_API = "logging_api"  # Log from a direct logging API call
    BATCH = "batch"  # Log from a batch create api call
    PYTHON_TRACING = "python_tracing"  # Log from a python tracing call
    TS_TRACING = "ts_tracing"  # Log from a typescript tracing call
    TRACING_INTEGRATION = "tracing_integration"  # Log from a tracing integration call


LOG_TYPE_TEXT = "text"
LOG_TYPE_CHAT = "chat"
LOG_TYPE_COMPLETION = "completion"
LOG_TYPE_RESPONSE = "response"
LOG_TYPE_EMBEDDING = "embedding"
LOG_TYPE_TRANSCRIPTION = "transcription"
LOG_TYPE_SPEECH = "speech"
LOG_TYPE_WORKFLOW = "workflow"
LOG_TYPE_TASK = "task"
LOG_TYPE_TOOL = "tool"
LOG_TYPE_AGENT = "agent"
LOG_TYPE_HANDOFF = "handoff"
LOG_TYPE_GUARDRAIL = "guardrail"
LOG_TYPE_FUNCTION = "function"
LOG_TYPE_CUSTOM = "custom"
LOG_TYPE_GENERATION = "generation"
LOG_TYPE_UNKNOWN = "unknown"
LOG_TYPE_SCORE = "score"
LOG_TYPE_BATCH = "batch"
LOG_TYPE_SPAN = "span"


class LogTypeChoices(Enum):
    TEXT = LOG_TYPE_TEXT
    CHAT = LOG_TYPE_CHAT
    COMPLETION = LOG_TYPE_COMPLETION
    RESPONSE = LOG_TYPE_RESPONSE  # OpenAI Response API
    EMBEDDING = LOG_TYPE_EMBEDDING
    TRANSCRIPTION = LOG_TYPE_TRANSCRIPTION
    SPEECH = LOG_TYPE_SPEECH
    WORKFLOW = LOG_TYPE_WORKFLOW
    TASK = LOG_TYPE_TASK
    TOOL = LOG_TYPE_TOOL  # Same as task
    AGENT = LOG_TYPE_AGENT  # Same as workflow
    HANDOFF = LOG_TYPE_HANDOFF  # OpenAI Agent
    GUARDRAIL = LOG_TYPE_GUARDRAIL  # OpenAI Agent
    FUNCTION = LOG_TYPE_FUNCTION  # OpenAI Agent
    CUSTOM = LOG_TYPE_CUSTOM  # OpenAI Agent
    GENERATION = LOG_TYPE_GENERATION  # OpenAI Agent
    UNKNOWN = LOG_TYPE_UNKNOWN
    SCORE = LOG_TYPE_SCORE
    BATCH = LOG_TYPE_BATCH
    SPAN = LOG_TYPE_SPAN  # Generic OTel span (no specific kind)

LogType = Literal[
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
    "score",
    "batch",
    "span",
]

LOG_TYPE_MAP: Dict[str, str] = {
    "workflow": LOG_TYPE_WORKFLOW,
    "trace": LOG_TYPE_WORKFLOW,
    "agent": LOG_TYPE_AGENT,
    "task": LOG_TYPE_TASK,
    "step": LOG_TYPE_TASK,
    "tool": LOG_TYPE_TOOL,
    "function": LOG_TYPE_TOOL,
    "llm": LOG_TYPE_GENERATION,
    "generation": LOG_TYPE_GENERATION,
    "model": LOG_TYPE_GENERATION,
    "chat": LOG_TYPE_CHAT,
    "prompt": "prompt",
}
