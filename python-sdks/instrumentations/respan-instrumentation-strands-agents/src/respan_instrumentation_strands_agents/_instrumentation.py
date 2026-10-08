"""Strands Agents OTEL instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
import os
from threading import RLock
from typing import Any, ClassVar

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_strands_agents._constants import (
    STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN,
)
from respan_instrumentation_strands_agents._policy import (
    content_enabled,
    filter_attributes,
    suppressed,
)
from respan_instrumentation_strands_agents._processor import (
    StrandsAgentsSpanProcessor,
)
from respan_instrumentation_strands_agents._serialization import payload_text

logger = logging.getLogger(__name__)


class StrandsAgentsInstrumentor:
    """Respan instrumentor for Strands Agents."""

    name = "strands-agents"
    _lock = RLock()
    _activation_count = 0
    _shared_processor: StrandsAgentsSpanProcessor | None = None
    _shared_provider: Any = None
    _shared_include_tool_definitions: bool | None = None
    _shared_previous_semconv_opt_in: str | None = None
    _shared_installed_semconv_opt_in: str | None = None
    _shared_tracer_module: Any = None
    _shared_tracer_instance: Any = None
    _shared_previous_tracer_provider: Any = None
    _shared_previous_tracer: Any = None
    _shared_previous_include_tool_definitions: bool | None = None
    _shared_installed_tracer_provider: Any = None
    _shared_installed_tracer: Any = None
    _shared_installed_include_tool_definitions: bool | None = None
    _response_provider_class: type[Any] | None = None
    _original_response_format: Any = None
    _installed_response_format: Any = None
    _provider_class: type[Any] | None = None
    _original_provider_format: Any = None
    _installed_provider_format: Any = None
    _agent_class: type[Any] | None = None
    _original_agent_start: Any = None
    _installed_agent_start: Any = None
    _tracer_class: type[Any] | None = None
    _original_tracer_methods: ClassVar[dict[str, Any]] = {}
    _installed_tracer_methods: ClassVar[dict[str, Any]] = {}

    def __init__(self, *, include_tool_definitions: bool = True) -> None:
        self._include_tool_definitions = include_tool_definitions
        self._is_instrumented = False

    @staticmethod
    def _register_processor(
        tracer_provider: Any,
        processor: StrandsAgentsSpanProcessor,
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
            active_span_processor._span_processors = (processor, *remaining_processors)
            return

        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)

    @staticmethod
    def _unregister_processor(
        tracer_provider: Any,
        processor: StrandsAgentsSpanProcessor,
    ) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )
        if active_span_processor is None or processors is None:
            return
        active_span_processor._span_processors = tuple(
            existing_processor
            for existing_processor in processors
            if existing_processor is not processor
        )

    @classmethod
    def _enable_semconv_opt_ins(cls) -> None:
        if not cls._shared_include_tool_definitions:
            return

        cls._shared_previous_semconv_opt_in = os.environ.get(
            "OTEL_SEMCONV_STABILITY_OPT_IN"
        )
        opt_in_values = {
            value.strip()
            for value in (cls._shared_previous_semconv_opt_in or "").split(",")
            if value.strip()
        }
        opt_in_values.add(STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN)
        installed = ",".join(sorted(opt_in_values))
        os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = installed
        cls._shared_installed_semconv_opt_in = installed

    @classmethod
    def _restore_semconv_opt_ins(cls) -> None:
        if not cls._shared_include_tool_definitions:
            return
        if (
            os.environ.get("OTEL_SEMCONV_STABILITY_OPT_IN")
            == cls._shared_installed_semconv_opt_in
        ):
            if cls._shared_previous_semconv_opt_in is None:
                os.environ.pop("OTEL_SEMCONV_STABILITY_OPT_IN", None)
            else:
                os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = (
                    cls._shared_previous_semconv_opt_in
                )
        cls._shared_previous_semconv_opt_in = None
        cls._shared_installed_semconv_opt_in = None

    @staticmethod
    def _refresh_strands_tracer(tracer_module: Any, tracer_provider: Any) -> None:
        tracer_instance = getattr(tracer_module, "_tracer_instance", None)
        if tracer_instance is None:
            return

        tracer_instance.tracer_provider = tracer_provider
        service_name = getattr(tracer_instance, "service_name", tracer_module.__name__)
        tracer_instance.tracer = tracer_provider.get_tracer(service_name)
        parse_opt_in = getattr(tracer_instance, "_parse_semconv_opt_in", None)
        if callable(parse_opt_in):
            opt_in_values = parse_opt_in()
            if hasattr(tracer_instance, "_include_tool_definitions"):
                tracer_instance._include_tool_definitions = (
                    STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN in opt_in_values
                )

    @classmethod
    def _patch_tracer_tool_propagation(cls, tracer_module: Any) -> None:
        tracer_class = getattr(tracer_module, "Tracer", None)
        if tracer_class is None or cls._tracer_class is not None:
            return
        originals = {
            name: getattr(tracer_class, name) for name in ("_start_span", "_add_event")
        }

        installed_processor = cls._shared_processor

        def start_span(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                cls._activation_count == 0
                or cls._shared_processor is not installed_processor
            ):
                return originals["_start_span"](instance, *args, **kwargs)
            if suppressed():
                return trace.INVALID_SPAN
            allowed = content_enabled()
            positional = list(args)
            attributes = kwargs.get("attributes")
            if attributes is None and len(positional) > 2:
                attributes = positional[2]
            filtered = filter_attributes(attributes, allowed)
            if len(positional) > 2:
                positional[2] = filtered
            else:
                kwargs["attributes"] = filtered
            span = originals["_start_span"](instance, *positional, **kwargs)
            processor = cls._shared_processor
            if processor is not None and span.is_recording():
                processor.register_native(span, allowed, filtered)
            return span

        def add_event(
            instance: Any,
            span: Any,
            event_name: str,
            event_attributes: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            processor = cls._shared_processor
            if cls._activation_count == 0 or processor is not installed_processor:
                return originals["_add_event"](
                    instance, span, event_name, event_attributes, *args, **kwargs
                )
            if span is None or not span.is_recording():
                return None
            allowed = processor.content_allowed(span) and content_enabled()
            if not allowed:
                return None
            bounded = {
                key: payload_text(value)
                for key, value in (event_attributes or {}).items()
            }
            return originals["_add_event"](
                instance, span, event_name, bounded, *args, **kwargs
            )

        installed = {"_start_span": start_span, "_add_event": add_event}
        cls._tracer_class = tracer_class
        cls._original_tracer_methods = originals
        cls._installed_tracer_methods = installed
        for name, wrapper in installed.items():
            setattr(tracer_class, name, wrapper)

        try:
            agent_class = importlib.import_module("strands.agent.agent").Agent
        except ImportError:
            return
        original_agent_start = agent_class._start_agent_trace_span

        def agent_start(instance: Any, *args: Any, **kwargs: Any) -> Any:
            span = original_agent_start(instance, *args, **kwargs)
            processor = cls._shared_processor
            if (
                cls._activation_count
                and processor is installed_processor
                and span.is_recording()
            ):
                processor.set_model(span, instance.model)
            return span

        cls._agent_class = agent_class
        cls._original_agent_start = original_agent_start
        cls._installed_agent_start = agent_start
        agent_class._start_agent_trace_span = agent_start

        try:
            provider_class = importlib.import_module(
                "strands.models.openai"
            ).OpenAIModel
        except ImportError:
            return
        original_format = provider_class.format_chunk

        def format_chunk(instance: Any, event: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_format(instance, event, *args, **kwargs)
            if (
                cls._shared_processor is installed_processor
                and cls._activation_count
                and not suppressed()
                and event.get("chunk_type") == "metadata"
            ):
                span = trace.get_current_span()
                processor = cls._shared_processor
                if (
                    span.is_recording()
                    and processor is not None
                    and processor.owns(span)
                ):
                    usage = event.get("data")
                    prompt_details = getattr(usage, "prompt_tokens_details", None)
                    completion_details = getattr(
                        usage, "completion_tokens_details", None
                    )
                    processor.record_provider_usage(
                        span, usage, "prompt_tokens", "completion_tokens"
                    )
                    for key, value in (
                        (
                            SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                            getattr(prompt_details, "cached_tokens", None),
                        ),
                        (
                            SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                            getattr(prompt_details, "cache_write_tokens", None),
                        ),
                        (
                            SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
                            getattr(completion_details, "reasoning_tokens", None),
                        ),
                    ):
                        if type(value) is int and value >= 0:
                            span.set_attribute(key, value)
            return result

        cls._provider_class = provider_class
        cls._original_provider_format = original_format
        cls._installed_provider_format = format_chunk
        provider_class.format_chunk = format_chunk

        try:
            responses_class = importlib.import_module(
                "strands.models.openai_responses"
            ).OpenAIResponsesModel
        except ImportError:
            return
        original_response_format = responses_class._format_chunk

        def responses_format(
            instance: Any, event: Any, *args: Any, **kwargs: Any
        ) -> Any:
            result = original_response_format(instance, event, *args, **kwargs)
            if (
                cls._shared_processor is installed_processor
                and cls._activation_count
                and not suppressed()
                and event.get("chunk_type") == "metadata"
            ):
                span = trace.get_current_span()
                if span.is_recording() and installed_processor.owns(span):
                    usage = event.get("data")
                    installed_processor.record_provider_usage(
                        span, usage, "input_tokens", "output_tokens"
                    )
                    details = getattr(usage, "input_tokens_details", None)
                    output_details = getattr(usage, "output_tokens_details", None)
                    for key, value in (
                        (
                            SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                            getattr(details, "cached_tokens", None),
                        ),
                        (
                            SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                            getattr(details, "cache_write_tokens", None),
                        ),
                        (
                            SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
                            getattr(output_details, "reasoning_tokens", None),
                        ),
                    ):
                        if type(value) is int and value >= 0:
                            span.set_attribute(key, value)
            return result

        cls._response_provider_class = responses_class
        cls._original_response_format = original_response_format
        cls._installed_response_format = responses_format
        responses_class._format_chunk = responses_format

    @classmethod
    def _restore_tracer_tool_propagation(cls) -> None:
        if cls._tracer_class is not None:
            for name, original in cls._original_tracer_methods.items():
                if getattr(
                    cls._tracer_class, name, None
                ) is cls._installed_tracer_methods.get(name):
                    setattr(cls._tracer_class, name, original)
        if (
            cls._agent_class is not None
            and cls._agent_class._start_agent_trace_span is cls._installed_agent_start
        ):
            cls._agent_class._start_agent_trace_span = cls._original_agent_start
        if (
            cls._provider_class is not None
            and cls._provider_class.format_chunk is cls._installed_provider_format
        ):
            cls._provider_class.format_chunk = cls._original_provider_format
        if (
            cls._response_provider_class is not None
            and cls._response_provider_class._format_chunk
            is cls._installed_response_format
        ):
            cls._response_provider_class._format_chunk = cls._original_response_format
        cls._response_provider_class = None
        cls._original_response_format = None
        cls._installed_response_format = None
        cls._provider_class = None
        cls._original_provider_format = None
        cls._installed_provider_format = None
        cls._agent_class = None
        cls._original_agent_start = None
        cls._installed_agent_start = None
        cls._tracer_class = None
        cls._original_tracer_methods = {}
        cls._installed_tracer_methods = {}

    @classmethod
    def _remember_refreshed_tracer(cls, tracer_module: Any) -> None:
        cls._shared_tracer_module = tracer_module
        tracer_instance = getattr(tracer_module, "_tracer_instance", None)
        if tracer_instance is None:
            return
        cls._shared_tracer_instance = tracer_instance
        cls._shared_previous_tracer_provider = getattr(
            tracer_instance, "tracer_provider", None
        )
        cls._shared_previous_tracer = getattr(tracer_instance, "tracer", None)
        cls._shared_previous_include_tool_definitions = getattr(
            tracer_instance, "_include_tool_definitions", None
        )

    @classmethod
    def _record_installed_tracer_state(cls) -> None:
        tracer_instance = cls._shared_tracer_instance
        if tracer_instance is None:
            return
        cls._shared_installed_tracer_provider = getattr(
            tracer_instance, "tracer_provider", None
        )
        cls._shared_installed_tracer = getattr(tracer_instance, "tracer", None)
        cls._shared_installed_include_tool_definitions = getattr(
            tracer_instance, "_include_tool_definitions", None
        )

    @classmethod
    def _restore_refreshed_tracer(cls) -> None:
        tracer_instance = cls._shared_tracer_instance
        if tracer_instance is None and cls._shared_tracer_module is not None:
            tracer_instance = getattr(
                cls._shared_tracer_module, "_tracer_instance", None
            )
        if tracer_instance is not None:
            if (
                getattr(tracer_instance, "tracer_provider", None)
                is cls._shared_installed_tracer_provider
            ):
                tracer_instance.tracer_provider = cls._shared_previous_tracer_provider
            if getattr(tracer_instance, "tracer", None) is cls._shared_installed_tracer:
                tracer_instance.tracer = cls._shared_previous_tracer
            if (
                getattr(tracer_instance, "_include_tool_definitions", None)
                == cls._shared_installed_include_tool_definitions
            ):
                tracer_instance._include_tool_definitions = (
                    cls._shared_previous_include_tool_definitions
                )
            elif cls._shared_tracer_instance is None:
                parse_opt_in = getattr(tracer_instance, "_parse_semconv_opt_in", None)
                if callable(parse_opt_in):
                    tracer_instance._include_tool_definitions = (
                        STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN in parse_opt_in()
                    )
        cls._shared_tracer_module = None
        cls._shared_tracer_instance = None
        cls._shared_previous_tracer_provider = None
        cls._shared_previous_tracer = None
        cls._shared_previous_include_tool_definitions = None
        cls._shared_installed_tracer_provider = None
        cls._shared_installed_tracer = None
        cls._shared_installed_include_tool_definitions = None

    def activate(self) -> None:
        cls = type(self)
        with cls._lock:
            if self._is_instrumented:
                return
            if cls._activation_count:
                if (
                    cls._shared_include_tool_definitions
                    != self._include_tool_definitions
                ):
                    raise ValueError(
                        "Strands Agents instrumentation is already active with a "
                        "different include_tool_definitions setting"
                    )
                cls._activation_count += 1
                self._is_instrumented = True
                return

            try:
                tracer_module = importlib.import_module("strands.telemetry.tracer")
            except ImportError as exc:
                logger.warning(
                    "Failed to activate Strands Agents instrumentation - "
                    "missing dependency: %s",
                    exc,
                )
                return

            tracer_provider = trace.get_tracer_provider()
            processor = StrandsAgentsSpanProcessor(
                include_tool_definitions=self._include_tool_definitions
            )
            cls._shared_include_tool_definitions = self._include_tool_definitions
            cls._shared_processor = processor
            cls._shared_provider = tracer_provider
            try:
                cls._remember_refreshed_tracer(tracer_module)
                cls._enable_semconv_opt_ins()
                cls._activation_count = 1
                self._patch_tracer_tool_propagation(tracer_module)
                self._register_processor(tracer_provider, processor)
                try:
                    self._refresh_strands_tracer(tracer_module, tracer_provider)
                finally:
                    cls._record_installed_tracer_state()
            except BaseException:
                self._unregister_processor(tracer_provider, processor)
                cls._restore_semconv_opt_ins()
                cls._activation_count = 0
                cls._restore_tracer_tool_propagation()
                cls._restore_refreshed_tracer()
                cls._shared_processor = None
                cls._shared_provider = None
                cls._shared_include_tool_definitions = None
                raise
            cls._activation_count = 1
            self._is_instrumented = True
            logger.info("Strands Agents instrumentation activated")

    def deactivate(self) -> None:
        cls = type(self)
        with cls._lock:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            cls._activation_count = max(0, cls._activation_count - 1)
            if cls._activation_count:
                return
            if cls._shared_processor is not None and cls._shared_provider is not None:
                self._unregister_processor(
                    cls._shared_provider,
                    cls._shared_processor,
                )
                cls._shared_processor.shutdown()
            cls._restore_semconv_opt_ins()
            cls._restore_tracer_tool_propagation()
            cls._restore_refreshed_tracer()
            cls._shared_processor = None
            cls._shared_provider = None
            cls._shared_include_tool_definitions = None
            logger.info("Strands Agents instrumentation deactivated")
