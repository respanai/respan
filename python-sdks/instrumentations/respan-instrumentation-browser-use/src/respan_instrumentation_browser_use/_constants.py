"""Names owned by the browser-use instrumentation."""

INSTRUMENTATION_NAME = "browser-use"
TRACER_NAME = "respan.instrumentation.browser_use"

# Patched owners, resolved lazily so importing this package never imports browser-use.
AGENT_MODULE = "browser_use.agent.service"
TOOLS_MODULE = "browser_use.tools.service"
TOKENS_MODULE = "browser_use.tokens.service"

SPAN_AGENT = "browser_use.agent"
SPAN_STEP = "browser_use.step"
SPAN_LLM = "browser_use.llm"

# Span attributes specific to browser agents. Everything else uses the
# OpenLLMetry / gen_ai names the Respan backend already parses.
ATTR_AGENT_ID = "browser_use.agent.id"
ATTR_SESSION_ID = "browser_use.session.id"
ATTR_STEP_NUMBER = "browser_use.step.number"
ATTR_PAGE_URL = "browser_use.page.url"
ATTR_PAGE_TITLE = "browser_use.page.title"
ATTR_ACTION_NAME = "browser_use.action.name"

# Placeholder for inline media dropped from captured content.
IMAGE_PLACEHOLDER = "[image omitted: {media_type}, {size} base64 chars]"
DATA_URL_PREFIX = "data:"

# Marker set on patched callables so a second activation never double-wraps.
PATCH_MARKER = "__respan_browser_use_wrapped__"
