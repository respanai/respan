"""Claude Agent SDK OTEL instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import StatusCode

from respan_instrumentation_claude_agent_sdk._processor import (  # type: ignore[reportMissingImports]
    ClaudeAgentSDKSpanProcessor,
    _safe_json_loads,
)
from respan_sdk.constants.span_attributes import RESPAN_METADATA

logger = logging.getLogger(__name__)


@dataclass
class _ProcessorRegistration:
    tracer_provider: Any
    processor: ClaudeAgentSDKSpanProcessor
    owners: int = 0


@dataclass
class _UpstreamRegistration:
    instrumentor: Any
    patched_modules: list[tuple[Any, str, Any]]
    owns_instrumentation: bool
    processor_registration: _ProcessorRegistration
    owners: int = 0


# Upstream instrumentation patches SDK methods globally, while span processors
# belong to providers. Keep their lifetimes separate and serialize ownership
# changes so simultaneous initialization cannot install duplicate normalizers.
_ownership_lock = RLock()
_processor_registrations: dict[int, _ProcessorRegistration] = {}
_upstream_registrations: dict[type[Any], _UpstreamRegistration] = {}


def _get_span_attr_value(span: Any, key: str) -> Any:
    attributes = getattr(span, "attributes", None)
    if attributes is None:
        attributes = getattr(span, "_attributes", None)
    if not isinstance(attributes, Mapping):
        return None
    return attributes.get(key)


def _load_upstream_instrumentor_class() -> type[Any]:
    upstream_module = importlib.import_module(
        "opentelemetry.instrumentation.claude_agent_sdk"
    )
    instrumentor_class = getattr(upstream_module, "ClaudeAgentSdkInstrumentor", None)
    if instrumentor_class is None:
        raise AttributeError(
            "opentelemetry.instrumentation.claude_agent_sdk.ClaudeAgentSdkInstrumentor"
        )
    return instrumentor_class


class ClaudeAgentSDKInstrumentor:
    """Respan instrumentor for the Claude Agent SDK."""

    name = "claude-agent-sdk"

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        capture_content: bool = True,
    ) -> None:
        """
        Args:
            agent_name: Optional name applied to the root agent span.
            capture_content: Capture prompts, tool arguments, and tool results as
                span content. Defaults to True to match Respan's other
                instrumentations (which capture content by default); pass False to
                emit structure-only spans when content must not leave the process.
        """
        self._agent_name = agent_name
        self._capture_content = capture_content
        self._otel_instrumentor = None
        self._processor = None
        self._tracer_provider = None
        self._upstream_instrumentor_class = None
        self._is_instrumented = False
        self._patched_modules: list[tuple[Any, str, Any]] = []

    @staticmethod
    def _register_processor(
        tracer_provider: Any,
        processor: ClaudeAgentSDKSpanProcessor,
    ) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )

        if active_span_processor is not None and processors is not None:
            remaining_processors = tuple(
                existing_processor
                for existing_processor in processors
                if existing_processor is not processor
            )
            # Normalize Claude spans before exporters or other processors read them.
            active_span_processor._span_processors = (processor, *remaining_processors)
            return

        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)

    @staticmethod
    def _unregister_processor(
        tracer_provider: Any,
        processor: ClaudeAgentSDKSpanProcessor,
    ) -> bool:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )
        if active_span_processor is None or processors is None:
            return False
        active_span_processor._span_processors = tuple(
            existing_processor
            for existing_processor in processors
            if existing_processor is not processor
        )
        return True

    def _acquire_processor(self, tracer_provider: Any) -> None:
        registration = _processor_registrations.get(id(tracer_provider))
        if registration is None:
            processor = ClaudeAgentSDKSpanProcessor()
            try:
                self._register_processor(tracer_provider, processor)
            except Exception:
                self._unregister_processor(tracer_provider, processor)
                raise
            registration = _ProcessorRegistration(tracer_provider, processor)
            _processor_registrations[id(tracer_provider)] = registration
        registration.owners += 1
        self._tracer_provider = tracer_provider
        self._processor = registration.processor

    @staticmethod
    def _release_processor_registration(registration: _ProcessorRegistration) -> None:
        registration.owners -= 1
        if registration.owners == 0:
            removed = ClaudeAgentSDKInstrumentor._unregister_processor(
                registration.tracer_provider, registration.processor
            )
            if removed:
                del _processor_registrations[id(registration.tracer_provider)]
                registration.processor.shutdown()
            # Providers exposing only add_span_processor cannot detach processors.
            # Keep their registration available for reuse on the next activation.

    def _release_processor(self) -> None:
        if self._tracer_provider is None:
            return
        registration = _processor_registrations[id(self._tracer_provider)]
        self._tracer_provider = None
        self._processor = None
        self._release_processor_registration(registration)

    def _patch_upstream_helpers(self) -> bool:
        if self._patched_modules:
            return True

        try:
            query_module = importlib.import_module("claude_agent_sdk._internal.query")
            Query = getattr(query_module, "Query")
            constants_module = importlib.import_module(
                "opentelemetry.instrumentation.claude_agent_sdk._constants"
            )
            context_module = importlib.import_module(
                "opentelemetry.instrumentation.claude_agent_sdk._context"
            )
            instrumentor_module = importlib.import_module(
                "opentelemetry.instrumentation.claude_agent_sdk._instrumentor"
            )
            spans_module = importlib.import_module(
                "opentelemetry.instrumentation.claude_agent_sdk._spans"
            )
            hooks_module = importlib.import_module(
                "opentelemetry.instrumentation.claude_agent_sdk._hooks"
            )
            from ._tool_lifecycle import (
                _InvocationMessageStream,
                _capture_failure_output,
                _reconcile_sdk_message,
            )

            instrumentor_class = _load_upstream_instrumentor_class()
            original_set_tool_error_attributes = hooks_module.set_tool_error_attributes

            # Read the upstream constant names inside the guarded block so a version
            # that renames or drops one degrades to a logged no-op instead of an
            # uncaught AttributeError out of activate().
            output_messages_attr = constants_module.GEN_AI_OUTPUT_MESSAGES
            error_type_attr = constants_module.ERROR_TYPE
            usage_input_tokens_attr = constants_module.GEN_AI_USAGE_INPUT_TOKENS
            usage_output_tokens_attr = constants_module.GEN_AI_USAGE_OUTPUT_TOKENS
            usage_cache_creation_tokens_attr = (
                constants_module.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS
            )
            usage_cache_read_tokens_attr = (
                constants_module.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
            )
        except (AttributeError, ImportError) as exc:
            logger.warning(
                "Failed to patch Claude Agent SDK helpers — missing dependency: %s",
                exc,
            )
            return False

        serialize_value = getattr(spans_module, "_to_serializable", lambda value: value)

        original_set_response_content = spans_module.set_response_content
        original_set_result_attributes = spans_module.set_result_attributes
        original_wrap_client_query = instrumentor_class._wrap_client_query
        original_instrumented_receive_response = (
            instrumentor_class._instrumented_receive_response
        )
        original_handle_control_request = Query._handle_control_request
        original_instrumented_query = getattr(instrumentor_class, "_instrumented_query", None)

        def patched_set_tool_error_attributes(span: Any, error: str) -> None:
            ctx = context_module.get_invocation_context()
            capture_content = bool(getattr(ctx, "capture_content", False))
            _capture_failure_output(
                span, error, capture_content=capture_content,
            )
            original_set_tool_error_attributes(
                span, error if capture_content else "Tool execution failed",
            )

        async def patched_instrumented_query(
            instrumentor: Any, wrapped: Any, args: tuple[Any, ...], kwargs: dict[str, Any],
        ) -> Any:
            from claude_agent_sdk import ResultMessage

            previous_invocation_ctx = context_module.get_invocation_context()
            source = _InvocationMessageStream(wrapped)
            messages = original_instrumented_query(instrumentor, source, args, kwargs)
            after_result = False
            try:
                async for message in messages:
                    _reconcile_sdk_message(context_module.get_invocation_context(), message)
                    after_result = isinstance(message, ResultMessage)
                    yield message
            finally:
                try:
                    await source.close(messages, after_result=after_result)
                finally:
                    context_module.set_invocation_context(previous_invocation_ctx)

        def patched_set_response_content(span: Any, content: Any) -> None:
            if content is None:
                return

            existing_messages = _safe_json_loads(
                _get_span_attr_value(span=span, key=output_messages_attr)
            )
            if not isinstance(existing_messages, list):
                existing_messages = []

            appended_messages = [
                *existing_messages,
                {
                    "role": "assistant",
                    "content": serialize_value(content),
                },
            ]
            try:
                span.set_attribute(
                    output_messages_attr,
                    json.dumps(serialize_value(appended_messages), default=str),
                )
            except (TypeError, ValueError):
                span.set_attribute(output_messages_attr, str(appended_messages))

        def patched_set_result_attributes(span: Any, result_message: Any) -> None:
            original_set_result_attributes(span, result_message)
            if getattr(result_message, "is_error", False) is True:
                # A caller can stop at this result before the SDK raises its later
                # exception. Classify the observed failure without exposing content.
                span.set_attribute(error_type_attr, "claude_agent_sdk_error")
                span.set_status(StatusCode.ERROR, "Claude Agent SDK invocation failed")

            usage = getattr(result_message, "usage", None)
            if isinstance(usage, dict):
                input_tokens = int(usage.get("input_tokens", 0) or 0)
                output_tokens = int(usage.get("output_tokens", 0) or 0)
                cache_creation_tokens = int(
                    usage.get("cache_creation_input_tokens", 0) or 0
                )
                cache_read_tokens = int(usage.get("cache_read_input_tokens", 0) or 0)

                # Emit the raw Anthropic input/output token counts. The span
                # processor (ClaudeAgentSDKSpanProcessor) derives prompt/completion/
                # total from these — with cache-token normalization and the override
                # attr that rolls up into total_request_tokens — so writing those here
                # would only pre-empt it. output_tokens was previously missing, which
                # starved that roll-up and left total_request_tokens at 0 (A7).
                span.set_attribute(usage_input_tokens_attr, input_tokens)
                span.set_attribute(usage_output_tokens_attr, output_tokens)
                if cache_creation_tokens > 0:
                    span.set_attribute(
                        usage_cache_creation_tokens_attr,
                        cache_creation_tokens,
                    )
                if cache_read_tokens > 0:
                    span.set_attribute(
                        usage_cache_read_tokens_attr,
                        cache_read_tokens,
                    )

            total_cost = getattr(result_message, "total_cost_usd", None)
            if isinstance(total_cost, (int, float)) and not isinstance(total_cost, bool):
                # The backend reads cost from respan.metadata.response_cost (matches the
                # LiteLLM/OpenAI instrumentors), not a bare "cost" attribute (A7).
                span.set_attribute(f"{RESPAN_METADATA}.response_cost", str(total_cost))

        def patched_wrap_client_query(
            instrumentor: Any,
            wrapped: Any,
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            result = original_wrap_client_query(
                instrumentor,
                wrapped,
                instance,
                args,
                kwargs,
            )
            invocation_ctx = getattr(instance, "_otel_invocation_ctx", None)
            query = getattr(instance, "_query", None)
            if invocation_ctx is not None and query is not None:
                query._otel_invocation_ctx = invocation_ctx
            return result

        async def patched_instrumented_receive_response(
            instrumentor: Any,
            wrapped: Any,
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            from claude_agent_sdk import ResultMessage

            previous_invocation_ctx = context_module.get_invocation_context()
            invocation_ctx = getattr(instance, "_otel_invocation_ctx", None)
            query = getattr(instance, "_query", None)
            terminal_result = None
            source = _InvocationMessageStream(wrapped)
            messages = original_instrumented_receive_response(
                instrumentor, source, instance, args, kwargs,
            )
            if invocation_ctx is not None:
                context_module.set_invocation_context(invocation_ctx)
                if query is not None:
                    query._otel_invocation_ctx = invocation_ctx

            try:
                async for message in messages:
                    if terminal_result is not None:
                        yield terminal_result
                        terminal_result = None
                    _reconcile_sdk_message(invocation_ctx, message)
                    if isinstance(message, ResultMessage):
                        terminal_result = message
                    else:
                        yield message
            finally:
                try:
                    await source.close(messages, after_result=False)
                finally:
                    if query is not None:
                        query._otel_invocation_ctx = None
                    context_module.set_invocation_context(previous_invocation_ctx)
            if terminal_result is not None:
                yield terminal_result

        async def patched_handle_control_request(query: Any, request: Any) -> Any:
            previous_invocation_ctx = context_module.get_invocation_context()
            query_invocation_ctx = getattr(query, "_otel_invocation_ctx", None)
            if query_invocation_ctx is not None:
                context_module.set_invocation_context(query_invocation_ctx)
            try:
                return await original_handle_control_request(query, request)
            finally:
                context_module.set_invocation_context(previous_invocation_ctx)

        for module in (spans_module, instrumentor_module):
            self._patched_modules.append(
                (module, "set_response_content", getattr(module, "set_response_content"))
            )
            self._patched_modules.append(
                (module, "set_result_attributes", getattr(module, "set_result_attributes"))
            )
            module.set_response_content = patched_set_response_content
            module.set_result_attributes = patched_set_result_attributes

        self._patched_modules.append(
            (
                instrumentor_class,
                "_wrap_client_query",
                getattr(instrumentor_class, "_wrap_client_query"),
            )
        )
        self._patched_modules.append(
            (
                instrumentor_class,
                "_instrumented_receive_response",
                getattr(instrumentor_class, "_instrumented_receive_response"),
            )
        )
        instrumentor_class._wrap_client_query = patched_wrap_client_query
        instrumentor_class._instrumented_receive_response = (
            patched_instrumented_receive_response
        )

        self._patched_modules.append(
            (Query, "_handle_control_request", getattr(Query, "_handle_control_request"))
        )
        Query._handle_control_request = patched_handle_control_request

        self._patched_modules.append(
            (hooks_module, "set_tool_error_attributes", original_set_tool_error_attributes)
        )
        hooks_module.set_tool_error_attributes = patched_set_tool_error_attributes
        if original_instrumented_query is not None:
            self._patched_modules.append(
                (instrumentor_class, "_instrumented_query", original_instrumented_query)
            )
            instrumentor_class._instrumented_query = patched_instrumented_query

        return True

    @staticmethod
    def _restore_helper_patches(patches: list[tuple[Any, str, Any]]) -> None:
        while patches:
            module, attr_name, original = patches.pop()
            setattr(module, attr_name, original)

    def _restore_upstream_helpers(self) -> None:
        self._restore_helper_patches(self._patched_modules)

    def _patch_standalone_query_seam(self, *, strip_module_query_wrap: bool) -> None:
        """Make ``from claude_agent_sdk import query`` traceable (issue A6).

        Upstream wraps ``claude_agent_sdk.query`` — a *module attribute*. Code that
        does ``from claude_agent_sdk import query`` binds the original function before
        instrumentation runs, so a bare ``query(...)`` is never traced. The standalone
        ``query()`` always delegates to ``InternalClient.process_query`` — a method
        resolved at call time and used *only* by the standalone path — so wrapping that
        seam (with upstream's own span logic) captures the call regardless of how
        ``query`` was imported, without touching the ``ClaudeSDKClient`` path. The
        now-redundant module-level wrap is then dropped so each call yields one span.

        Both patches are recorded in ``self._patched_modules`` so the generic
        ``_restore_upstream_helpers`` loop undoes them. Best-effort: on failure the
        upstream module-level wrap is left intact, so behaviour is no worse than before.
        """
        try:
            import wrapt
            import claude_agent_sdk
            from claude_agent_sdk._internal.client import InternalClient
        except ImportError as exc:
            logger.warning(
                "Claude Agent SDK: could not instrument the internal query seam; "
                "`from claude_agent_sdk import query` may not be traced: %s",
                exc,
            )
            return

        # Idempotency: a second activation (or an already-wrapped seam) must not
        # stack another wrapper, which would emit two spans per standalone query().
        # getattr (not __dict__) so an inherited process_query is still found.
        process_query = getattr(InternalClient, "process_query", None)
        if process_query is None or hasattr(process_query, "__wrapped__"):
            return

        # Record the original before wrapping so the restore loop unwraps the seam.
        self._patched_modules.append((InternalClient, "process_query", process_query))
        try:
            wrapt.wrap_function_wrapper(
                "claude_agent_sdk._internal.client",
                "InternalClient.process_query",
                self._otel_instrumentor._wrap_query,
            )
        except Exception as exc:
            self._patched_modules.pop()
            logger.warning(
                "Claude Agent SDK: could not instrument the internal query seam; "
                "`from claude_agent_sdk import query` may not be traced: %s",
                exc,
            )
            return

        # Drop the now-redundant module-level `query` wrap so module-qualified and
        # from-imported calls behave identically (one span). Only strip a wrap our
        # own instrument() added — never a user's or another vendor's pre-existing
        # wrap — and record the pristine original so the restore loop puts it back.
        if not strip_module_query_wrap:
            return
        module_query = getattr(claude_agent_sdk, "query", None)
        if hasattr(module_query, "__wrapped__"):
            self._patched_modules.append(
                (claude_agent_sdk, "query", module_query.__wrapped__)
            )
            claude_agent_sdk.query = module_query.__wrapped__

    def activate(self) -> None:
        with _ownership_lock:
            if self._is_instrumented:
                return

            try:
                upstream_instrumentor_class = _load_upstream_instrumentor_class()
            except (AttributeError, ImportError) as exc:
                logger.warning(
                    "Failed to activate Claude Agent SDK instrumentation — missing dependency: %s",
                    exc,
                )
                return

            upstream = _upstream_registrations.get(upstream_instrumentor_class)
            instrument_attempted = False
            try:
                tracer_provider = trace.get_tracer_provider()
                self._acquire_processor(tracer_provider)
                if upstream is None:
                    if not self._patch_upstream_helpers():
                        self._restore_upstream_helpers()
                        self._release_processor()
                        return

                    self._otel_instrumentor = upstream_instrumentor_class()
                    already_instrumented = bool(
                        getattr(
                            self._otel_instrumentor,
                            "is_instrumented_by_opentelemetry",
                            False,
                        )
                    )
                    # Preserve pre-existing query wrappers, including wrappers
                    # owned by instrumentation activated outside this plugin.
                    import claude_agent_sdk

                    module_query_prewrapped = hasattr(
                        getattr(claude_agent_sdk, "query", None), "__wrapped__"
                    )
                    if not already_instrumented:
                        instrument_attempted = True
                        self._otel_instrumentor.instrument(
                            tracer_provider=tracer_provider,
                            agent_name=self._agent_name,
                            capture_content=self._capture_content,
                        )

                    self._patch_standalone_query_seam(
                        strip_module_query_wrap=(
                            instrument_attempted and not module_query_prewrapped
                        )
                    )
                    upstream = _UpstreamRegistration(
                        self._otel_instrumentor,
                        self._patched_modules,
                        owns_instrumentation=instrument_attempted,
                        processor_registration=_processor_registrations[
                            id(tracer_provider)
                        ],
                    )
                    # The global upstream tracer keeps using this provider even
                    # if its original owner deactivates before owners on other
                    # providers. Retain normalization until the global wrap ends.
                    upstream.processor_registration.owners += 1
                    _upstream_registrations[upstream_instrumentor_class] = upstream
                else:
                    # Upstream patches are global. Its first active owner's
                    # provider, agent name, and content settings remain in effect.
                    self._otel_instrumentor = upstream.instrumentor
                    self._patched_modules = upstream.patched_modules
            except Exception as exc:
                if instrument_attempted and self._otel_instrumentor is not None:
                    try:
                        self._otel_instrumentor.uninstrument()
                    except Exception:
                        logger.debug(
                            "Failed to roll back Claude Agent SDK instrumentation",
                            exc_info=True,
                        )
                self._restore_upstream_helpers()
                self._otel_instrumentor = None
                self._release_processor()
                logger.warning(
                    "Failed to activate Claude Agent SDK instrumentation: %s", exc
                )
                return

            upstream.owners += 1
            self._upstream_instrumentor_class = upstream_instrumentor_class
            self._is_instrumented = True
            logger.info("Claude Agent SDK instrumentation activated")

    def deactivate(self) -> None:
        with _ownership_lock:
            if not self._is_instrumented:
                return

            upstream_class = self._upstream_instrumentor_class
            upstream = _upstream_registrations[upstream_class]
            upstream.owners -= 1
            try:
                if upstream.owners == 0:
                    del _upstream_registrations[upstream_class]
                    try:
                        if upstream.owns_instrumentation:
                            upstream.instrumentor.uninstrument()
                    finally:
                        try:
                            self._restore_helper_patches(upstream.patched_modules)
                        finally:
                            self._release_processor_registration(
                                upstream.processor_registration
                            )
            finally:
                self._otel_instrumentor = None
                self._patched_modules = []
                self._upstream_instrumentor_class = None
                self._is_instrumented = False
                self._release_processor()

            logger.info("Claude Agent SDK instrumentation deactivated")
