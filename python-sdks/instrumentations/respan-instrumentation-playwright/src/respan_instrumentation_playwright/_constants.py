"""Names owned by the Playwright instrumentation."""

INSTRUMENTATION_NAME = "playwright"
TRACER_NAME = "respan.instrumentation.playwright"

# Patched owners, resolved lazily so importing this package never imports Playwright.
ASYNC_API_MODULE = "playwright.async_api._generated"
SYNC_API_MODULE = "playwright.sync_api._generated"
ASYNC_BASE_MODULE = "playwright._impl._async_base"
SYNC_BASE_MODULE = "playwright._impl._sync_base"
PAGE_IMPL_MODULE = "playwright._impl._page"
WRAP_HANDLER = "_wrap_handler"

# Span attributes specific to browser automation. Everything else uses the
# OpenLLMetry names the Respan backend already parses.
ATTR_API_NAME = "playwright.api.name"
ATTR_PAGE_URL = "playwright.page.url"
ATTR_PAGE_TITLE = "playwright.page.title"

# screenshot_on values: a page screenshot after the call, for a flip-book of the run.
SCREENSHOT_NEVER = "never"
SCREENSHOT_ON_NAVIGATION = "navigation"
SCREENSHOT_ON_ACTION = "action"
SCREENSHOT_MODES = frozenset({SCREENSHOT_NEVER, SCREENSHOT_ON_NAVIGATION, SCREENSHOT_ON_ACTION})

# Calls that load a document: the span also gets the page title.
NAVIGATION_METHODS = frozenset(
    {"goto", "reload", "go_back", "go_forward", "set_content", "wait_for_url"}
)
# Calls that change the page (plus navigations), for screenshot_on="action".
ACTION_METHODS = NAVIGATION_METHODS | frozenset(
    {
        "check",
        "clear",
        "click",
        "dblclick",
        "dispatch_event",
        "drag_and_drop",
        "drag_to",
        "fill",
        "focus",
        "hover",
        "insert_text",
        "move",
        "press",
        "press_sequentially",
        "scroll_into_view_if_needed",
        "select_option",
        "select_text",
        "set_checked",
        "set_input_files",
        "tap",
        "type",
        "uncheck",
        "wheel",
    }
)
# Calls whose argument is text typed into the page: masked when it lands in a
# password field (or we cannot tell where it landed).
TEXT_ENTRY_ARGS = {"fill": "value", "type": "text", "press_sequentially": "text", "insert_text": "text"}
SCREENSHOT_METHOD = "screenshot"

MASKED_TEXT = "<masked>"
OUTPUT_TEXT_LIMIT = 4000
TRUNCATED_SUFFIX = "... [truncated, {size} chars]"
# Text typed into a password field is masked everywhere afterwards as this name.
LEARNED_SECRET_NAME = "password"
MAX_LEARNED_SECRETS = 32
