"""Names owned by the browser-use instrumentation."""

INSTRUMENTATION_NAME = "browser-use"
TRACER_NAME = "respan.instrumentation.browser_use"

# Patched owners, resolved lazily so importing this package never imports browser-use.
AGENT_MODULE = "browser_use.agent.service"
TOOLS_MODULE = "browser_use.tools.service"
TOKENS_MODULE = "browser_use.tokens.service"

SPAN_AGENT = "browser_use.agent"
SPAN_RERUN = "browser_use.rerun"
SPAN_STEP = "browser_use.step"
SPAN_LLM = "browser_use.llm"
SPAN_ACTION = "browser_use.action"  # only when an ActionModel has no set field

# Display-name detail: Respan shows the run as "agent.browser_use".
AGENT_NAME_DETAIL = "browser_use"
RERUN_NAME_DETAIL = "browser_use_rerun"

# Span attributes specific to browser agents. Everything else uses the
# OpenLLMetry / gen_ai names the Respan backend already parses.
ATTR_AGENT_ID = "browser_use.agent.id"
ATTR_SESSION_ID = "browser_use.session.id"
ATTR_STEP_NUMBER = "browser_use.step.number"
ATTR_PAGE_URL = "browser_use.page.url"
ATTR_PAGE_TITLE = "browser_use.page.title"
ATTR_ACTION_NAME = "browser_use.action.name"

# browser-use chat clients whose ChatInvokeUsage.prompt_tokens omits cache writes.
# Matched by module, not provider name: ChatLiteLLM can also report "anthropic"
# but LiteLLM's prompt_tokens already includes cache writes.
ANTHROPIC_USAGE_MODULES = frozenset(
    {"browser_use.llm.anthropic.chat", "browser_use.llm.aws.chat_anthropic"}
)

# Placeholder for inline media dropped from captured content.
IMAGE_PLACEHOLDER = "[image omitted: {media_type}, {size} base64 chars]"
