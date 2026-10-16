"""Names owned by the Skyvern instrumentation."""

INSTRUMENTATION_NAME = "skyvern"
TRACER_NAME = "respan.instrumentation.skyvern"
# Skyvern's own OpenTelemetry tracer. Its spans carry no Respan span kind, so
# Respan drops them; ours must never be parented to one.
NATIVE_TRACER_NAME = "skyvern"

# Patched owners, resolved lazily so importing this package never imports Skyvern.
CLIENT_MODULE = "skyvern.library.skyvern"  # Skyvern (run_task / run_workflow / login)
API_CLIENT_MODULE = "skyvern.client.client"  # AsyncSkyvern (run_sdk_action / download_files)
# page.agent.run_task / login / ... : polls until the run finishes, so it is the run root.
PAGE_AGENT_MODULE = "skyvern.library.skyvern_browser_page_agent"
# Engine modules exist only with `skyvern[local]` (in-process engine).
AGENT_MODULE = "skyvern.forge.agent"
ACTIONS_MODULE = "skyvern.webeye.actions.handler"
BLOCK_MODULE = "skyvern.forge.sdk.workflow.models.block"
CONTEXT_MODULE = "skyvern.forge.sdk.core.skyvern_context"
FORGE_APP_MODULE = "skyvern.forge"
LITELLM_MODULE = "litellm"
# LLMCaller: the v3 engine's, CUA and OpenRouter model calls.
LLM_HANDLER_MODULE = "skyvern.forge.sdk.api.llm.api_handler_factory"

SPAN_RUN_TASK = "skyvern.run_task"
SPAN_RUN_WORKFLOW = "skyvern.run_workflow"
SPAN_LOGIN = "skyvern.login"
SPAN_DOWNLOAD_FILES = "skyvern.download_files"
SPAN_SDK_ACTION = "skyvern.sdk_action"
SPAN_STEP = "skyvern.step"
SPAN_BLOCK = "skyvern.block"
SPAN_ACTION = "skyvern.action"  # only when an action has no type
SPAN_LLM = "skyvern.llm"

# Display-name detail: Respan shows agent/tool spans as "<kind>.<detail>".
RUN_NAME_DETAILS = {
    SPAN_RUN_TASK: "skyvern_task",
    SPAN_RUN_WORKFLOW: "skyvern_workflow",
    SPAN_LOGIN: "skyvern_login",
    SPAN_DOWNLOAD_FILES: "skyvern_download_files",
}

STEP_NAME_DETAIL = "skyvern_step"

# Span attributes specific to Skyvern. Everything else uses the OpenLLMetry /
# gen_ai names the Respan backend already parses.
ATTR_RUN_ID = "skyvern.run.id"
ATTR_ENGINE = "skyvern.run.engine"
ATTR_TASK_ID = "skyvern.task.id"
ATTR_WORKFLOW_RUN_ID = "skyvern.workflow_run.id"
ATTR_STEP_ID = "skyvern.step.id"
ATTR_STEP_NUMBER = "skyvern.step.number"
ATTR_STEP_RETRY = "skyvern.step.retry"
ATTR_PAGE_URL = "skyvern.page.url"
ATTR_ACTION_TYPE = "skyvern.action.type"
ATTR_BLOCK_TYPE = "skyvern.block.type"
ATTR_BLOCK_LABEL = "skyvern.block.label"
ATTR_PROMPT_NAME = "skyvern.llm.prompt_name"
ATTR_RECORDING_URL = "skyvern.run.recording_url"
ATTR_APP_URL = "skyvern.run.app_url"

# Skyvern sets this on its own LLM request span; read to name our chat span.
NATIVE_PROMPT_NAME_ATTR = "prompt_name"

# Skyvern run statuses that mean the run did not succeed.
FAILED_RUN_STATUSES = frozenset({"failed", "terminated", "timed_out", "canceled"})

# Action fields that are bookkeeping (ids, timestamps) or bulky DOM data.
ACTION_FIELDS_DROPPED = frozenset(
    {
        "organization_id",
        "workflow_run_id",
        "task_id",
        "step_id",
        "step_order",
        "action_order",
        "source_action_id",
        "skyvern_element_hash",
        "skyvern_element_data",
        "screenshot_artifact_id",
        "created_at",
        "modified_at",
        "started_at",
        "finished_at",
        "created_by",
    }
)

IMAGE_PLACEHOLDER = "[image omitted: {size} base64 chars]"
PNG_DATA_URL_PREFIX = "data:image/png;base64,"
