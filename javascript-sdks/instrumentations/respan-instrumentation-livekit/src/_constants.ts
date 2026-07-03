import { RespanLogType } from "@respan/respan-sdk";

export const LIVEKIT_INSTRUMENTATION_NAME = "livekit";
export const LIVEKIT_INSTRUMENTATION_PACKAGE = "@respan/instrumentation-livekit";
export const LIVEKIT_LOG_METHOD_TS_TRACING = "ts_tracing";

export const LIVEKIT_SPAN_NAMES = {
  AGENT_SESSION: "agent_session",
  AGENT_TURN: "agent_turn",
  USER_TURN: "user_turn",
  LLM_NODE: "llm_node",
  LLM_REQUEST: "llm_request",
  LLM_REQUEST_RUN: "llm_request_run",
  FUNCTION_TOOL: "function_tool",
  TTS_NODE: "tts_node",
  TTS_REQUEST: "tts_request",
  TTS_REQUEST_RUN: "tts_request_run",
  AGENT_SPEAKING: "agent_speaking",
  START_AGENT_ACTIVITY: "start_agent_activity",
  RESUME_AGENT_ACTIVITY: "resume_agent_activity",
  PAUSE_AGENT_ACTIVITY: "pause_agent_activity",
  DRAIN_AGENT_ACTIVITY: "drain_agent_activity",
  ON_ENTER: "on_enter",
  ON_EXIT: "on_exit",
} as const;

export const LIVEKIT_LOG_TYPE_BY_SPAN_NAME: Record<string, RespanLogType> = {
  [LIVEKIT_SPAN_NAMES.AGENT_SESSION]: RespanLogType.WORKFLOW,
  [LIVEKIT_SPAN_NAMES.AGENT_TURN]: RespanLogType.AGENT,
  [LIVEKIT_SPAN_NAMES.USER_TURN]: RespanLogType.TASK,
  [LIVEKIT_SPAN_NAMES.LLM_NODE]: RespanLogType.CHAT,
  [LIVEKIT_SPAN_NAMES.LLM_REQUEST]: RespanLogType.CHAT,
  [LIVEKIT_SPAN_NAMES.LLM_REQUEST_RUN]: RespanLogType.TASK,
  [LIVEKIT_SPAN_NAMES.FUNCTION_TOOL]: RespanLogType.TOOL,
  [LIVEKIT_SPAN_NAMES.TTS_NODE]: RespanLogType.TEXT,
  [LIVEKIT_SPAN_NAMES.TTS_REQUEST]: RespanLogType.TEXT,
  [LIVEKIT_SPAN_NAMES.TTS_REQUEST_RUN]: RespanLogType.TASK,
  [LIVEKIT_SPAN_NAMES.AGENT_SPEAKING]: RespanLogType.TASK,
};

export const LIVEKIT_INTERNAL_ACTIVITY_SPAN_NAMES = new Set<string>([
  LIVEKIT_SPAN_NAMES.START_AGENT_ACTIVITY,
  LIVEKIT_SPAN_NAMES.RESUME_AGENT_ACTIVITY,
  LIVEKIT_SPAN_NAMES.PAUSE_AGENT_ACTIVITY,
  LIVEKIT_SPAN_NAMES.DRAIN_AGENT_ACTIVITY,
  LIVEKIT_SPAN_NAMES.ON_ENTER,
  LIVEKIT_SPAN_NAMES.ON_EXIT,
]);

export const LIVEKIT_SPAN_NAME_ATTRIBUTE = "livekit.span.name";

export const LK_SPEECH_ID = "lk.speech_id";
export const LK_AGENT_LABEL = "lk.agent_label";
export const LK_START_TIME = "lk.start_time";
export const LK_END_TIME = "lk.end_time";
export const LK_RETRY_COUNT = "lk.retry_count";
export const LK_PROVIDER_REQUEST_IDS = "lk.provider_request_ids";
export const LK_PARTICIPANT_ID = "lk.participant_id";
export const LK_PARTICIPANT_IDENTITY = "lk.participant_identity";
export const LK_PARTICIPANT_KIND = "lk.participant_kind";
export const LK_JOB_ID = "lk.job_id";
export const LK_AGENT_NAME = "lk.agent_name";
export const LK_ROOM_NAME = "lk.room_name";
export const LK_SESSION_OPTIONS = "lk.session_options";
export const LK_AGENT_TURN_ID = "lk.generation_id";
export const LK_AGENT_PARENT_TURN_ID = "lk.parent_generation_id";
export const LK_USER_INPUT = "lk.user_input";
export const LK_INSTRUCTIONS = "lk.instructions";
export const LK_SPEECH_INTERRUPTED = "lk.interrupted";
export const LK_CHAT_CTX = "lk.chat_ctx";
export const LK_FUNCTION_TOOLS = "lk.function_tools";
export const LK_PROVIDER_TOOLS = "lk.provider_tools";
export const LK_TOOL_SETS = "lk.tool_sets";
export const LK_RESPONSE_TEXT = "lk.response.text";
export const LK_RESPONSE_FUNCTION_CALLS = "lk.response.function_calls";
export const LK_RESPONSE_TTFT = "lk.response.ttft";
export const LK_FUNCTION_TOOL_ID = "lk.function_tool.id";
export const LK_FUNCTION_TOOL_NAME = "lk.function_tool.name";
export const LK_FUNCTION_TOOL_ARGS = "lk.function_tool.arguments";
export const LK_FUNCTION_TOOL_IS_ERROR = "lk.function_tool.is_error";
export const LK_FUNCTION_TOOL_OUTPUT = "lk.function_tool.output";
export const LK_TTS_INPUT_TEXT = "lk.input_text";
export const LK_TTS_STREAMING = "lk.tts.streaming";
export const LK_TTS_LABEL = "lk.tts.label";
export const LK_RESPONSE_TTFB = "lk.response.ttfb";
export const LK_EOU_PROBABILITY = "lk.eou.probability";
export const LK_EOU_UNLIKELY_THRESHOLD = "lk.eou.unlikely_threshold";
export const LK_EOU_DELAY = "lk.eou.endpointing_delay";
export const LK_EOU_LANGUAGE = "lk.eou.language";
export const LK_EOU_SOURCE = "lk.eou.source";
export const LK_EOU_FROM_CACHE = "lk.eou.from_cache";
export const LK_EOU_DETECTION_DELAY = "lk.eou.detection_delay";
export const LK_USER_TRANSCRIPT = "lk.user_transcript";
export const LK_TRANSCRIPT_CONFIDENCE = "lk.transcript_confidence";
export const LK_TRANSCRIPTION_DELAY = "lk.transcription_delay";
export const LK_END_OF_TURN_DELAY = "lk.end_of_turn_delay";
export const LK_AMD_CATEGORY = "lk.amd.category";
export const LK_AMD_REASON = "lk.amd.reason";
export const LK_AMD_IS_MACHINE = "lk.amd.is_machine";
export const LK_AMD_INTERRUPT_ON_MACHINE = "lk.amd.interrupt_on_machine";
export const LK_AMD_SPEECH_DURATION = "lk.amd.speech_duration";
export const LK_AMD_DELAY = "lk.amd.delay";
export const LK_AMD_TRANSCRIPT = "lk.amd.transcript";
export const LK_IS_INTERRUPTION = "lk.is_interruption";
export const LK_INTERRUPTION_PROBABILITY = "lk.interruption.probability";
export const LK_INTERRUPTION_TOTAL_DURATION = "lk.interruption.total_duration";
export const LK_INTERRUPTION_PREDICTION_DURATION = "lk.interruption.prediction_duration";
export const LK_INTERRUPTION_DETECTION_DELAY = "lk.interruption.detection_delay";
export const LK_LLM_METRICS = "lk.llm_metrics";
export const LK_TTS_METRICS = "lk.tts_metrics";
export const LK_REALTIME_MODEL_METRICS = "lk.realtime_model_metrics";
export const LK_E2E_LATENCY = "lk.e2e_latency";

export const LIVEKIT_RAW_ATTRIBUTE_PREFIX = "lk.";

export const OFF_CONTRACT_ALIAS_KEYS = [
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
] as const;
