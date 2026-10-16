"""Names owned by the Stagehand instrumentation."""

import re

INSTRUMENTATION_NAME = "stagehand"
TRACER_NAME = "respan.instrumentation.stagehand"

# Patched owners, resolved lazily so importing this package never imports Stagehand.
STAGEHAND_MODULE = "stagehand.stagehand"
PAGE_MODULE = "stagehand.page"
LOCATOR_MODULE = "stagehand.locator"
CONTEXT_MODULE = "stagehand.browser_context"
MODELS_MODULE = "stagehand._generated.models"

SPAN_SESSION = "stagehand.session"
SPAN_LLM = "stagehand.llm"
# act / observe / extract spans are "stagehand.<operation>".
OPERATION_PREFIX = "stagehand."
OPERATIONS = ("act", "observe", "extract")
ACT = "act"
EXTRACT = "extract"

# Browser actions traced as tool spans: "page.goto", "locator.fill", ... The
# names are Stagehand's own RPC method names.
PAGE_OWNER = "page"
LOCATOR_OWNER = "locator"
CONTEXT_OWNER = "context"
PAGE_ACTIONS = (
    "goto",
    "reload",
    "go_back",
    "go_forward",
    "click",
    "hover",
    "scroll",
    "drag_and_drop",
    "type",
    "key_press",
    "evaluate",
    "screenshot",
    "pdf",
    "wait_for_selector",
    "set_viewport_size",
    "close",
)
LOCATOR_ACTIONS = (
    "click",
    "hover",
    "fill",
    "type",
    "select_option",
    "set_input_files",
    "scroll_to",
    "send_click_event",
)
CONTEXT_ACTIONS = ("new_page", "set_active_page")
NAVIGATION_ACTIONS = frozenset({"goto", "reload", "go_back", "go_forward"})
# (owner, method) -> the argument holding typed text.
TYPED_TEXT = {
    (LOCATOR_OWNER, "fill"): "value",
    (LOCATOR_OWNER, "type"): "text",
    (PAGE_OWNER, "type"): "text",
}

# Display-name detail: Respan shows the session as "agent.stagehand".
SESSION_NAME_DETAIL = "stagehand"
LLM_NAME_DETAIL = "llm"
CLIENT_MODEL = "client"

# Span attributes specific to Stagehand. Everything else uses the
# OpenLLMetry / gen_ai names the Respan backend already parses.
ATTR_SESSION_ID = "stagehand.session.id"
ATTR_PAGE_URL = "stagehand.page.url"
ATTR_ACTION_ID = "stagehand.action.id"
ATTR_CACHE_STATUS = "stagehand.cache.status"
ATTR_USAGE_SOURCE = "stagehand.usage.source"
# The chat span's usage came from the operation result, not a client generate call.
USAGE_SOURCE_OPERATION = "operation"
# Stagehand's per-operation usage fields, summed on the session span.
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "inference_time_ms",
)

# Set on a client model's generate callable by with_model_name().
CLIENT_MODEL_NAME_ATTR = "__respan_model_name__"

# Screenshot attached to each act span: JPEG, CSS pixels, short timeout.
SCREENSHOT_OPTIONS = {"type": "jpeg", "quality": 70, "scale": "css", "timeout": 3000}
SCREENSHOT_DEADLINE_SECONDS = 4.0
# Page screenshots larger than this are summarised instead of exported.
MAX_EXPORTED_IMAGE_BASE64 = 2_000_000

IMAGE_PLACEHOLDER = "[image omitted: {mime}, {size} base64 chars]"
REDACTED = "[redacted]"
# Variable values shorter than this would mask unrelated text.
MIN_SECRET_LENGTH = 3

# Text typed into a field whose selector looks like a credential is never exported.
SENSITIVE_SELECTOR = re.compile(
    r"pass(word|code|phrase)?|secret|token|otp|cvv|cvc|card.?num|ssn|api.?key", re.I
)
# A field described by the page (type, name, id, autocomplete, aria-label) that holds a credential.
SENSITIVE_FIELD = re.compile(
    r"type=password|pass(word|code|phrase)?|secret|token|otp|one-time-code|cvv|cvc"
    r"|cc-number|cc-csc|card.?num|ssn|api.?key|\bpin\b",
    re.I,
)
XPATH_PREFIX = "xpath="
