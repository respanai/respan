import copy
import importlib.metadata
import json
import logging
import traceback
from typing import Optional, Set, Callable

from ..instruments import Instruments


logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "opentelemetry_instrumentor"

# Map Instruments enum values to entry point names.
# Only needed when they differ (most match by convention).
_ENUM_TO_ENTRY_POINT: dict[str, str] = {
    "grpc": "grpc_client",
}

# Post-init hooks keyed by entry point name.
# These run after instrument() for specific instrumentors.
_POST_INIT_HOOKS: dict[str, Callable] = {}


def _register_hook(name: str):
    """Decorator to register a post-init hook by entry point name."""
    def decorator(fn):
        _POST_INIT_HOOKS[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def _discover_instrumentors() -> dict[str, object]:
    """Discover all installed OTEL instrumentors via entry points.

    Returns:
        Dict mapping entry point name to the entry point object.
    """
    try:
        eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        return {ep.name: ep for ep in eps}
    except Exception as e:
        logger.warning(f"Failed to discover instrumentors: {e}")
        return {}


def _enum_to_entry_point_name(instrument: Instruments) -> str:
    """Convert an Instruments enum to the entry point name."""
    return _ENUM_TO_ENTRY_POINT.get(instrument.value, instrument.value)


def _instrument_entry_point(ep, ep_name: str) -> bool:
    """Load and instrument a single entry point.

    Returns True if instrumentation succeeded.
    """
    try:
        instrumentor_cls = ep.load()
        instrumentor = instrumentor_cls()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()

        hook = _POST_INIT_HOOKS.get(ep_name)
        if hook is not None:
            hook()

        return True
    except Exception as e:
        logger.error(f"Failed to initialize {ep_name} instrumentation: {e}")
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_instrumentations(
    instruments: Optional[Set[Instruments]] = None,
    block_instruments: Optional[Set[Instruments]] = None,
) -> bool:
    """
    Initialize OpenTelemetry instrumentations via entry point auto-discovery.

    Every installed ``opentelemetry-instrumentation-*`` package registers an
    entry point under the ``opentelemetry_instrumentor`` group.  This function
    discovers all of them at runtime and instruments each one — no manual
    registry needed.

    Args:
        instruments: If provided, only these instruments are enabled.
                     If None (default), **all** discovered instrumentors run.
        block_instruments: Instruments to explicitly skip.

    Returns:
        True if at least one instrumentor was successfully initialized.

    Note:
        THREADING is always auto-included (unless explicitly blocked) because
        it is critical for OTel context propagation across threads.
    """
    block_instruments = block_instruments or set()
    block_names = {_enum_to_entry_point_name(i) for i in block_instruments}

    discovered = _discover_instrumentors()

    if instruments is not None:
        # Explicit set — resolve enum values to entry point names
        allowed_names = {_enum_to_entry_point_name(i) for i in instruments}
        # Always include threading unless blocked
        if Instruments.THREADING not in block_instruments:
            allowed_names.add(_enum_to_entry_point_name(Instruments.THREADING))
    else:
        # Auto-discover: instrument everything found
        allowed_names = set(discovered.keys())

    # Remove blocked
    allowed_names -= block_names

    instrument_count = 0

    for name in allowed_names:
        ep = discovered.get(name)
        if ep is None:
            # Entry point not installed — skip silently
            continue
        try:
            if _instrument_entry_point(ep, name):
                instrument_count += 1
        except Exception as e:
            logger.warning(f"Failed to initialize {name} instrumentation: {e}")

    if instrument_count == 0:
        logger.warning("No instrumentations were successfully initialized")
        return False

    logger.info(f"Successfully initialized {instrument_count} instrumentations")
    return True


# ---------------------------------------------------------------------------
# Post-init hooks — special-case patches that run after instrument()
# ---------------------------------------------------------------------------

@_register_hook("openai")
def _patch_chat_prompt_capture():
    """
    Replace the async chat _handle_request with a sync version.

    Root cause: opentelemetry-instrumentation-openai v0.52+ has _handle_request
    as async def (for optional base64 image upload). In sync contexts, it runs
    through run_async() which either calls asyncio.run() or spawns a thread.
    Both paths can silently lose prompt attributes when:
      - _set_request_attributes (NOT @dont_throw) raises on response_format
        handling, killing the entire _handle_request before _set_prompts runs
      - asyncio.run() / thread path has environment-specific issues (Lambda, etc.)

    The embeddings wrapper is fully sync and works correctly. This patch makes
    the chat path match the embeddings path: fully synchronous with fault
    isolation between each section.

    The only async code in _set_prompts was for Config.upload_base64_image
    (rarely used). For list content (multimodal), we json.dumps as-is — the
    base64 data stays inline, which is the default behavior anyway.
    """
    try:
        from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw
        from opentelemetry.instrumentation.openai.shared import (
            _set_request_attributes,
            _set_client_attributes,
            _set_functions_attributes,
            _set_span_attribute,
            set_tools_attributes,
            model_as_dict,
            propagate_trace_context,
        )
        from opentelemetry.instrumentation.openai.shared.config import Config
        from opentelemetry.instrumentation.openai.utils import (
            should_send_prompts,
            should_emit_events,
            is_openai_v1,
        )
        from opentelemetry.semconv._incubating.attributes import (
            gen_ai_attributes as GenAIAttributes,
        )
        from opentelemetry.semconv_ai import SpanAttributes

        def _set_prompts_sync(span, messages):
            if not span.is_recording() or messages is None:
                return

            for i, msg in enumerate(messages):
                prefix = f"{GenAIAttributes.GEN_AI_PROMPT}.{i}"
                msg = msg if isinstance(msg, dict) else model_as_dict(msg)

                _set_span_attribute(span, f"{prefix}.role", msg.get("role"))
                if msg.get("content"):
                    content = copy.deepcopy(msg.get("content"))
                    if isinstance(content, list):
                        content = json.dumps(content)
                    _set_span_attribute(span, f"{prefix}.content", content)
                if msg.get("tool_call_id"):
                    _set_span_attribute(
                        span, f"{prefix}.tool_call_id", msg.get("tool_call_id")
                    )
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for j, tool_call in enumerate(tool_calls):
                        if is_openai_v1():
                            tool_call = model_as_dict(tool_call)
                        function = tool_call.get("function")
                        _set_span_attribute(
                            span, f"{prefix}.tool_calls.{j}.id", tool_call.get("id")
                        )
                        _set_span_attribute(
                            span, f"{prefix}.tool_calls.{j}.name", function.get("name")
                        )
                        _set_span_attribute(
                            span,
                            f"{prefix}.tool_calls.{j}.arguments",
                            function.get("arguments"),
                        )

        def _handle_request_sync(span, kwargs, instance):
            # Section 1: Request attributes (fault-isolated from prompts)
            try:
                _set_request_attributes(span, kwargs, instance)
            except Exception:
                logging.warning(
                    "respan-tracing: _set_request_attributes failed (response_format may be incompatible). "
                    "Request attributes like model/temperature may be incomplete on this span. "
                    "Error: %s",
                    traceback.format_exc(),
                )

            try:
                _set_client_attributes(span, instance)
            except Exception:
                pass

            # Section 2: Prompt/event capture
            try:
                if should_emit_events():
                    from opentelemetry.instrumentation.openai.shared.event_emitter import emit_event
                    from opentelemetry.instrumentation.openai.shared.event_models import MessageEvent
                    for message in kwargs.get("messages", []):
                        emit_event(
                            MessageEvent(
                                content=message.get("content"),
                                role=message.get("role"),
                                tool_calls=cw._parse_tool_calls(
                                    message.get("tool_calls", None)
                                ),
                            )
                        )
                else:
                    if should_send_prompts():
                        _set_prompts_sync(span, kwargs.get("messages"))
                        if kwargs.get("functions"):
                            _set_functions_attributes(span, kwargs.get("functions"))
                        elif kwargs.get("tools"):
                            set_tools_attributes(span, kwargs.get("tools"))
            except Exception:
                logging.warning(
                    "respan-tracing: chat prompt capture failed. "
                    "Input messages may not appear on the dashboard for this span. "
                    "Error: %s",
                    traceback.format_exc(),
                )

            # Section 3: Trace propagation + reasoning
            try:
                if Config.enable_trace_context_propagation:
                    propagate_trace_context(span, kwargs)
                reasoning_effort = kwargs.get("reasoning_effort")
                _set_span_attribute(
                    span,
                    SpanAttributes.LLM_REQUEST_REASONING_EFFORT,
                    reasoning_effort or (),
                )
            except Exception:
                pass

        async def _noop():
            pass

        def _patched_handle_request(span, kwargs, instance):
            _handle_request_sync(span, kwargs, instance)
            return _noop()

        cw._handle_request = _patched_handle_request
        logger.debug("respan-tracing: patched chat prompt capture to sync path")

    except Exception as e:
        logger.warning(f"respan-tracing: failed to patch chat prompt capture: {e}")
