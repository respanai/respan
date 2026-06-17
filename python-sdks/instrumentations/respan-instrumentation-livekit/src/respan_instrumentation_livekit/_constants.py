"""LiveKit instrumentation constants.

LiveKit-owned ``lk.*`` keys stay package-local. Respan-owned keys are imported
from ``respan-sdk`` where they are used.
"""

LIVEKIT_INSTRUMENTATION_NAME = "livekit"
LIVEKIT_SCOPE_NAME = "livekit-agents"
LIVEKIT_LLM_REQUEST_SPAN_NAME = "llm_request"
LIVEKIT_CHAT_SPAN_NAME = "livekit.llm.chat"

ASSISTANT_ROLE = "assistant"
FUNCTION_KEY = "function"
FUNCTION_TOOL_TYPE = "function"
ID_KEY = "id"
NAME_KEY = "name"
ROLE_KEY = "role"
TOOL_ROLE = "tool"
TYPE_KEY = "type"
USER_ROLE = "user"

ATTR_LLM_METRICS = "lk.llm_metrics"
ATTR_PROVIDER_REQUEST_IDS = "lk.provider_request_ids"

EVENT_GEN_AI_SYSTEM_MESSAGE = "gen_ai.system.message"
EVENT_GEN_AI_USER_MESSAGE = "gen_ai.user.message"
EVENT_GEN_AI_ASSISTANT_MESSAGE = "gen_ai.assistant.message"
EVENT_GEN_AI_TOOL_MESSAGE = "gen_ai.tool.message"
EVENT_GEN_AI_CHOICE = "gen_ai.choice"

LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR = "lk.respan.function_tools"

