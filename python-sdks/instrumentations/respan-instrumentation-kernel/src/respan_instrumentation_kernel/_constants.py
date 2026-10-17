"""Names owned by the Kernel instrumentation."""

INSTRUMENTATION_NAME = "kernel"
TRACER_NAME = "respan.instrumentation.kernel"

# Patched owners, resolved lazily so importing this package never imports Kernel.
# (candidate modules, sync class, async class, API group, public methods); the
# first module that imports wins (Kernel moved some between releases).
BROWSERS_MODULE = ("kernel.resources.browsers.browsers",)
COMPUTER_MODULE = ("kernel.resources.browsers.computer",)
PROCESS_MODULE = ("kernel.resources.browsers.process",)
# A package (with an `executors` sub-resource) in newer releases; a module before.
PLAYWRIGHT_MODULE = (
    "kernel.resources.browsers.playwright.playwright",
    "kernel.resources.browsers.playwright",
)
REPLAYS_MODULE = ("kernel.resources.browsers.replays",)
APP_MODULE = "kernel.app_framework"
TYPES_MODULE = "kernel._types"

GROUP_BROWSERS = "browsers"
GROUP_COMPUTER = "computer"
GROUP_PROCESS = "process"
GROUP_PLAYWRIGHT = "playwright"
GROUP_REPLAYS = "replays"

METHOD_CREATE = "create"
METHOD_CAPTURE_SCREENSHOT = "capture_screenshot"
METHOD_TYPE_TEXT = "type_text"
METHOD_BATCH = "batch"
METHOD_EXECUTE = "execute"
METHOD_START = "start"

RESOURCES = (
    (
        BROWSERS_MODULE,
        "BrowsersResource",
        "AsyncBrowsersResource",
        GROUP_BROWSERS,
        (
            METHOD_CREATE,
            "retrieve",
            "update",
            "list",
            "delete_by_id",
            "load_extensions",
            "curl",
            "repl",
        ),
    ),
    (
        COMPUTER_MODULE,
        "ComputerResource",
        "AsyncComputerResource",
        GROUP_COMPUTER,
        (
            METHOD_BATCH,
            METHOD_CAPTURE_SCREENSHOT,
            "click_mouse",
            "drag_mouse",
            "get_mouse_position",
            "move_mouse",
            "press_key",
            "read_clipboard",
            "scroll",
            "set_cursor_visibility",
            METHOD_TYPE_TEXT,
            "write_clipboard",
        ),
    ),
    (
        PROCESS_MODULE,
        "ProcessResource",
        "AsyncProcessResource",
        GROUP_PROCESS,
        ("exec", "spawn", "status", "stdin", "stdout_stream", "kill", "resize"),
    ),
    (
        PLAYWRIGHT_MODULE,
        "PlaywrightResource",
        "AsyncPlaywrightResource",
        GROUP_PLAYWRIGHT,
        (METHOD_EXECUTE,),
    ),
    (
        REPLAYS_MODULE,
        "ReplaysResource",
        "AsyncReplaysResource",
        GROUP_REPLAYS,
        (METHOD_START, "stop", "list", "download"),
    ),
)

SPAN_PREFIX = "kernel."
SPAN_ACTION_PREFIX = "kernel.action."
# Display-name detail: Respan shows tool spans as "tool.kernel_<group>_<method>".
TOOL_DETAIL_PREFIX = "kernel_"

# Span attributes specific to Kernel. Everything else uses the OpenLLMetry
# names the Respan backend already parses.
ATTR_API_NAME = "kernel.api.name"
ATTR_SESSION_ID = "kernel.session.id"
ATTR_INVOCATION_ID = "kernel.invocation.id"
ATTR_ACTION_NAME = "kernel.action.name"
ATTR_APP_NAME = "kernel.app.name"
ATTR_LIVE_VIEW_URL = "kernel.browser.live_view_url"
ATTR_REPLAY_ID = "kernel.replay.id"
ATTR_REPLAY_VIEW_URL = "kernel.replay.view_url"
ATTR_PROCESS_ID = "kernel.process.id"
ATTR_EXIT_CODE = "kernel.process.exit_code"

# The id argument every per-browser call takes (positional or keyword).
SESSION_ARG = "id_or_name"
# Arguments that are transport plumbing, not the call's input.
TRANSPORT_ARGS = frozenset({"extra_headers", "extra_query", "extra_body", "timeout"})
# Stainless "argument not given" sentinels.
UNSET_TYPES = ("Omit", "NotGiven")

# Process calls: environment values are secrets; base64 payloads are decoded.
ENV_ARG = "env"
STDIN_ARG = "data_b64"
B64_OUTPUT_FIELDS = {"stdout_b64": "stdout", "stderr_b64": "stderr", "data_b64": "data"}
OUTPUT_TEXT_LIMIT = 4000
TRUNCATED_SUFFIX = "... [truncated, {size} chars]"

# Query parameters that carry a bearer token in the URLs Kernel returns
# (cdp_ws_url, webdriver_ws_url, live view): masked wherever they appear.
TOKEN_QUERY_PARAMS = ("jwt", "token")
MASKED_TEXT = "<masked>"
# Values learned from a call (env values, typed passwords) are masked everywhere
# in it only when at least this long: "1" or "true" would mangle unrelated text.
MIN_LEARNED_SECRET_LENGTH = 8
TYPED_TEXT_SECRET_NAME = "typed_text"
API_KEY_SECRET_NAME = "kernel_api_key"

# playwright.execute reports a failure in its response instead of raising.
PLAYWRIGHT_EXECUTION_ERROR = "PlaywrightExecutionError"

# probe_password_fields: is the element that receives typed text a password
# field? Runs inside the browser through playwright.execute. An iframe counts:
# the focused element inside it cannot be seen from the top document.
PASSWORD_PROBE_CODE = (
    "return await page.evaluate(() => { const e = document.activeElement;"
    " return !!e && (e.tagName === 'IFRAME' || e.tagName === 'FRAME'"
    " || (e.getAttribute('type') || '').toLowerCase() === 'password'); });"
)
PASSWORD_PROBE_TIMEOUT_SEC = 10
