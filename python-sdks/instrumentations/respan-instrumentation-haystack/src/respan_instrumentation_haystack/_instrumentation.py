"""Haystack instrumentation plugin for Respan."""

import contextvars
import importlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from respan_instrumentation_haystack._constants import (
    HAYSTACK_ASYNC_PIPELINE_CLASS_NAME,
    HAYSTACK_ASYNC_PIPELINE_MODULE,
    HAYSTACK_COMPONENT_DECORATOR_ATTRIBUTE,
    HAYSTACK_COMPONENT_MODULE,
    HAYSTACK_COMPONENT_NAME_PARAMETER,
    HAYSTACK_COMPONENT_REGISTRY_ATTRIBUTE,
    HAYSTACK_INSTRUMENTATION_NAME,
    HAYSTACK_NATIVE_PROCESSING_ATTRIBUTES,
    HAYSTACK_NATIVE_SPAN_NAMES,
    HAYSTACK_PIPELINE_CLASS_NAME,
    HAYSTACK_PIPELINE_MODULE,
    HAYSTACK_PIPELINE_SPAN_NAMES,
    HAYSTACK_RUN_ASYNC_GENERATOR_METHOD_NAME,
    HAYSTACK_RUN_ASYNC_METHOD_NAME,
    HAYSTACK_RUN_COMPONENT_ASYNC_METHOD_NAME,
    HAYSTACK_RUN_COMPONENT_METHOD_NAME,
    HAYSTACK_RUN_METHOD_NAME,
    OPENINFERENCE_HAYSTACK_MODULE,
    OPENINFERENCE_HAYSTACK_INSTRUMENTOR_CLASS_NAME,
    OPENINFERENCE_TRANSLATOR_CLASS_NAME,
    RESPAN_HAYSTACK_COMPONENT_CONTEXT_VAR_NAME,
    RESPAN_HAYSTACK_MAIN_COMPONENT_PATCH_FLAG,
    RESPAN_HAYSTACK_PIPELINE_CONTEXT_VAR_NAME,
)
from respan_sdk.utils.data_processing.id_processing import format_span_id
from respan_instrumentation_openinference import OpenInferenceInstrumentor
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)


@dataclass
class _HaystackPipelineRunContext:
    graph: Any
    completed_span_id_by_component: dict[str, str] = field(default_factory=dict)
    completion_order_by_component: dict[str, int] = field(default_factory=dict)
    completion_counter: int = 0
    pipeline_span_id: str | None = None

    def record_completion(self, component_name: str, span_id: str) -> None:
        self.completion_counter += 1
        self.completed_span_id_by_component[component_name] = span_id
        self.completion_order_by_component[component_name] = self.completion_counter


@dataclass(frozen=True)
class _HaystackComponentRunContext:
    component_name: str
    pipeline_context: _HaystackPipelineRunContext | None


_CURRENT_PIPELINE_RUN_CONTEXT: contextvars.ContextVar[
    _HaystackPipelineRunContext | None
] = contextvars.ContextVar(
    RESPAN_HAYSTACK_PIPELINE_CONTEXT_VAR_NAME,
    default=None,
)
_CURRENT_COMPONENT_RUN_CONTEXT: contextvars.ContextVar[
    _HaystackComponentRunContext | None
] = contextvars.ContextVar(
    RESPAN_HAYSTACK_COMPONENT_CONTEXT_VAR_NAME,
    default=None,
)
_PIPELINE_CONTEXT_PATCH_APPLIED = False


def _load_openinference_haystack_class() -> type:
    haystack_module = importlib.import_module(OPENINFERENCE_HAYSTACK_MODULE)
    return getattr(haystack_module, OPENINFERENCE_HAYSTACK_INSTRUMENTOR_CLASS_NAME)


def _resolve_registered_component_class(
    *, module_name: str, wrapper_path: str
) -> type[Any] | None:
    class_name, separator, _ = wrapper_path.partition(".")
    if separator != "." or not class_name:
        return None

    try:
        component_module = importlib.import_module(HAYSTACK_COMPONENT_MODULE)
    except ImportError:
        return None

    component_decorator = getattr(
        component_module,
        HAYSTACK_COMPONENT_DECORATOR_ATTRIBUTE,
        None,
    )
    component_registry = getattr(
        component_decorator,
        HAYSTACK_COMPONENT_REGISTRY_ATTRIBUTE,
        None,
    )
    if not isinstance(component_registry, dict):
        return None

    for component_class in component_registry.values():
        if (
            getattr(component_class, "__module__", None) == module_name
            and getattr(component_class, "__name__", None) == class_name
        ):
            return component_class
    return None


def _patch_main_component_wrapping() -> None:
    haystack_module = importlib.import_module(OPENINFERENCE_HAYSTACK_MODULE)
    if getattr(haystack_module, RESPAN_HAYSTACK_MAIN_COMPONENT_PATCH_FLAG, False):
        return

    original_wrap_function_wrapper = getattr(
        haystack_module,
        "wrap_function_wrapper",
        None,
    )
    if original_wrap_function_wrapper is None:
        return

    def compatible_wrap_function_wrapper(module: Any, name: str, wrapper: Any) -> Any:
        try:
            return original_wrap_function_wrapper(
                module=module,
                name=name,
                wrapper=wrapper,
            )
        except AttributeError:
            if not isinstance(module, str):
                raise

            component_class = _resolve_registered_component_class(
                module_name=module,
                wrapper_path=name,
            )
            if component_class is None:
                raise

            _, _, method_name = name.partition(".")
            return original_wrap_function_wrapper(
                module=component_class,
                name=method_name,
                wrapper=wrapper,
            )

    haystack_module.wrap_function_wrapper = compatible_wrap_function_wrapper
    setattr(haystack_module, RESPAN_HAYSTACK_MAIN_COMPONENT_PATCH_FLAG, True)


def _patch_pipeline_context_wrapping() -> None:
    global _PIPELINE_CONTEXT_PATCH_APPLIED

    if _PIPELINE_CONTEXT_PATCH_APPLIED:
        return

    try:
        haystack_module = importlib.import_module(OPENINFERENCE_HAYSTACK_MODULE)
        async_pipeline_module = importlib.import_module(HAYSTACK_ASYNC_PIPELINE_MODULE)
        pipeline_module = importlib.import_module(HAYSTACK_PIPELINE_MODULE)
    except ImportError:
        return

    wrap_function_wrapper = getattr(haystack_module, "wrap_function_wrapper", None)
    if wrap_function_wrapper is None:
        return

    async_pipeline_class = getattr(
        async_pipeline_module,
        HAYSTACK_ASYNC_PIPELINE_CLASS_NAME,
    )
    pipeline_class = getattr(pipeline_module, HAYSTACK_PIPELINE_CLASS_NAME)

    wrap_function_wrapper(
        module=pipeline_class,
        name=HAYSTACK_RUN_METHOD_NAME,
        wrapper=_pipeline_run_context_wrapper,
    )
    wrap_function_wrapper(
        module=async_pipeline_class,
        name=HAYSTACK_RUN_METHOD_NAME,
        wrapper=_pipeline_run_context_wrapper,
    )
    wrap_function_wrapper(
        module=async_pipeline_class,
        name=HAYSTACK_RUN_ASYNC_METHOD_NAME,
        wrapper=_async_pipeline_run_context_wrapper,
    )
    wrap_function_wrapper(
        module=async_pipeline_class,
        name=HAYSTACK_RUN_ASYNC_GENERATOR_METHOD_NAME,
        wrapper=_async_pipeline_run_async_generator_context_wrapper,
    )
    wrap_function_wrapper(
        module=pipeline_class,
        name=HAYSTACK_RUN_COMPONENT_METHOD_NAME,
        wrapper=_component_run_context_wrapper,
    )
    wrap_function_wrapper(
        module=async_pipeline_class,
        name=HAYSTACK_RUN_COMPONENT_ASYNC_METHOD_NAME,
        wrapper=_async_component_run_context_wrapper,
    )
    _PIPELINE_CONTEXT_PATCH_APPLIED = True


def _pipeline_run_context_wrapper(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    token = _CURRENT_PIPELINE_RUN_CONTEXT.set(
        _HaystackPipelineRunContext(graph=getattr(instance, "graph", None))
    )
    try:
        return wrapped(*args, **kwargs)
    finally:
        _CURRENT_PIPELINE_RUN_CONTEXT.reset(token)


async def _async_pipeline_run_context_wrapper(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    token = _CURRENT_PIPELINE_RUN_CONTEXT.set(
        _HaystackPipelineRunContext(graph=getattr(instance, "graph", None))
    )
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _CURRENT_PIPELINE_RUN_CONTEXT.reset(token)


def _async_pipeline_run_async_generator_context_wrapper(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    async def run_with_context():
        token = _CURRENT_PIPELINE_RUN_CONTEXT.set(
            _HaystackPipelineRunContext(graph=getattr(instance, "graph", None))
        )
        try:
            async for output in wrapped(*args, **kwargs):
                yield output
        finally:
            _CURRENT_PIPELINE_RUN_CONTEXT.reset(token)

    return run_with_context()


def _component_run_context_wrapper(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    component_name = _get_component_name(args=args, kwargs=kwargs)
    token = _CURRENT_COMPONENT_RUN_CONTEXT.set(
        _HaystackComponentRunContext(
            component_name=component_name,
            pipeline_context=_CURRENT_PIPELINE_RUN_CONTEXT.get(),
        )
    )
    try:
        return wrapped(*args, **kwargs)
    finally:
        _CURRENT_COMPONENT_RUN_CONTEXT.reset(token)


async def _async_component_run_context_wrapper(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    component_name = _get_component_name(args=args, kwargs=kwargs)
    token = _CURRENT_COMPONENT_RUN_CONTEXT.set(
        _HaystackComponentRunContext(
            component_name=component_name,
            pipeline_context=_CURRENT_PIPELINE_RUN_CONTEXT.get(),
        )
    )
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _CURRENT_COMPONENT_RUN_CONTEXT.reset(token)


def _get_component_name(*, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    if args:
        return str(args[0])
    return str(kwargs.get(HAYSTACK_COMPONENT_NAME_PARAMETER, ""))


def _get_span_id(span: Any) -> str | None:
    get_span_context = getattr(span, "get_span_context", None)
    if get_span_context is None:
        return None

    span_context = get_span_context()
    span_id = getattr(span_context, "span_id", None)
    if not span_id:
        return None
    return format_span_id(span_id)


def _get_parent_span_id(span: Any) -> str | None:
    parent = getattr(span, "parent", None)
    span_id = getattr(parent, "span_id", None)
    if not span_id:
        return None
    return format_span_id(span_id)


def _is_haystack_native_span(span: Any) -> bool:
    return getattr(span, "name", None) in HAYSTACK_NATIVE_SPAN_NAMES


def _suppress_haystack_native_span_export(span: Any) -> None:
    attributes = getattr(span, "_attributes", None)
    if attributes is None:
        return

    # ReadableSpan stores ended-span attributes in an immutable
    # BoundedAttributes instance on newer OpenTelemetry releases. Replace the
    # private snapshot instead of mutating it so native Haystack spans remain
    # unprocessable without raising during processor shutdown.
    span._attributes = {
        name: value
        for name, value in attributes.items()
        if name not in HAYSTACK_NATIVE_PROCESSING_ATTRIBUTES
    }


def _parse_json_attr(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _json_attr(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _message_content(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        text_parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if text is not None:
                    text_parts.append(str(text))
            elif item is not None:
                text_parts.append(str(item))
        return "\n".join(text_parts)

    if value is None:
        return ""
    return str(value)


def _normalize_haystack_message(value: Any, *, fallback_role: str) -> dict[str, Any] | None:
    if isinstance(value, str):
        return {"role": fallback_role, "content": value}

    if not isinstance(value, dict):
        return None

    role = value.get("role") or fallback_role
    content = _message_content(value.get("content"))
    message: dict[str, Any] = {"role": str(role), "content": content}

    name = value.get("name")
    if name is not None:
        message["name"] = name

    return message


def _haystack_input_messages(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _parse_json_attr(attrs.get("haystack.component.input"))
    if not isinstance(payload, dict):
        return []

    messages = payload.get("messages")
    if isinstance(messages, list):
        normalized = [
            message
            for item in messages
            if (
                message := _normalize_haystack_message(
                    item,
                    fallback_role="user",
                )
            )
            is not None
        ]
        if normalized:
            return normalized

    for key in ("query", "question", "prompt"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return [{"role": "user", "content": value}]

    return []


def _haystack_completion_message(attrs: dict[str, Any]) -> dict[str, Any] | None:
    payload = _parse_json_attr(attrs.get("haystack.component.output"))
    if not isinstance(payload, dict):
        return None

    replies = payload.get("replies")
    if isinstance(replies, list):
        for item in reversed(replies):
            message = _normalize_haystack_message(item, fallback_role="assistant")
            if message is not None and message.get("content"):
                return message
        for item in reversed(replies):
            message = _normalize_haystack_message(item, fallback_role="assistant")
            if message is not None:
                return message

    answers = payload.get("answers")
    if isinstance(answers, list):
        for item in reversed(answers):
            if isinstance(item, dict):
                data = item.get("data") or item.get("answer")
                if data is not None:
                    return {"role": "assistant", "content": str(data)}
            elif isinstance(item, str):
                return {"role": "assistant", "content": item}

    return None


def _set_indexed_messages(
    attrs: dict[str, Any],
    *,
    prefix: str,
    messages: list[dict[str, Any]],
) -> None:
    for index, message in enumerate(messages):
        role = message.get("role")
        if role is not None:
            attrs[f"{prefix}.{index}.role"] = str(role)
        content = message.get("content")
        if content is not None:
            attrs[f"{prefix}.{index}.content"] = str(content)


def _enrich_haystack_io_attrs(attrs: dict[str, Any]) -> None:
    input_messages = _haystack_input_messages(attrs)
    if input_messages:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_attr(input_messages)
        _set_indexed_messages(
            attrs,
            prefix=SpanAttributes.LLM_PROMPTS,
            messages=input_messages,
        )

    completion_message = _haystack_completion_message(attrs)
    if completion_message is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_attr(completion_message)
        _set_indexed_messages(
            attrs,
            prefix=SpanAttributes.LLM_COMPLETIONS,
            messages=[completion_message],
        )


class _HaystackParentSpanProcessor(SpanProcessor):
    """Suppress native Haystack spans while preserving parent remapping.

    Haystack creates native pipeline/component spans around OpenInference spans.
    Those native spans are not useful Respan log rows, but their IDs are needed
    so exported child spans do not point at missing parents.
    """

    def __init__(self) -> None:
        self._parent_by_span_id: dict[str, str | None] = {}
        self._context_by_span_id: dict[str, Any] = {}
        self._component_context_by_span_id: dict[str, _HaystackComponentRunContext] = {}
        self._native_span_ids: set[str] = set()

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        span_id = _get_span_id(span)
        if span_id is None:
            return

        self._parent_by_span_id[span_id] = _get_parent_span_id(span)
        self._context_by_span_id[span_id] = span.get_span_context()
        if _is_haystack_native_span(span):
            self._native_span_ids.add(span_id)

        pipeline_context = _CURRENT_PIPELINE_RUN_CONTEXT.get()
        if (
            pipeline_context is not None
            and pipeline_context.pipeline_span_id is None
            and getattr(span, "name", None) in HAYSTACK_PIPELINE_SPAN_NAMES
        ):
            pipeline_context.pipeline_span_id = span_id

        component_context = _CURRENT_COMPONENT_RUN_CONTEXT.get()
        if component_context is not None and component_context.component_name:
            self._component_context_by_span_id[span_id] = component_context

    def on_end(self, span: ReadableSpan) -> None:
        span_id = _get_span_id(span)
        if _is_haystack_native_span(span):
            _suppress_haystack_native_span_export(span)
            return

        attributes = getattr(span, "_attributes", None)
        if attributes is not None:
            _enrich_haystack_io_attrs(attributes)

        parent_id = _get_parent_span_id(span)
        component_context = (
            self._component_context_by_span_id.get(span_id)
            if span_id is not None
            else None
        )
        if self._is_pipeline_component_span(
            parent_id=parent_id,
            component_context=component_context,
        ):
            exported_parent_id = self._graph_parent_span_id(component_context)
            if exported_parent_id is not None:
                exported_parent_context = self._context_by_span_id.get(
                    exported_parent_id
                )
                if exported_parent_context is None:
                    return
                span._parent = exported_parent_context
            elif parent_id in self._native_span_ids:
                exported_parent_id = self._nearest_exported_parent_id(parent_id)
                if exported_parent_id is None:
                    return
                exported_parent_context = self._context_by_span_id.get(
                    exported_parent_id
                )
                if exported_parent_context is None:
                    return
                span._parent = exported_parent_context

            if (
                span_id is not None
                and component_context is not None
                and component_context.pipeline_context is not None
            ):
                component_context.pipeline_context.record_completion(
                    component_context.component_name,
                    span_id,
                )
            return

        if parent_id is None or parent_id not in self._native_span_ids:
            return

        exported_parent_id = self._nearest_exported_parent_id(parent_id)
        if exported_parent_id is None:
            return

        exported_parent_context = self._context_by_span_id.get(exported_parent_id)
        if exported_parent_context is None:
            return

        span._parent = exported_parent_context

    def shutdown(self) -> None:
        self._parent_by_span_id.clear()
        self._context_by_span_id.clear()
        self._component_context_by_span_id.clear()
        self._native_span_ids.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def _nearest_exported_parent_id(self, span_id: str) -> str | None:
        seen: set[str] = set()
        current_id: str | None = span_id

        while current_id is not None:
            if current_id in seen:
                return None
            seen.add(current_id)

            parent_id = self._parent_by_span_id.get(current_id)
            if parent_id is None:
                return None
            if parent_id not in self._native_span_ids:
                return parent_id
            current_id = parent_id

        return None

    def _graph_parent_span_id(
        self,
        component_context: _HaystackComponentRunContext | None,
    ) -> str | None:
        if component_context is None or component_context.pipeline_context is None:
            return None

        pipeline_context = component_context.pipeline_context
        graph = pipeline_context.graph
        if graph is None:
            return None

        try:
            predecessors = tuple(graph.predecessors(component_context.component_name))
        except Exception:
            return None

        candidates: list[tuple[int, str]] = []
        for predecessor in predecessors:
            completed_span_id = pipeline_context.completed_span_id_by_component.get(
                predecessor
            )
            if completed_span_id is None:
                continue
            candidates.append(
                (
                    pipeline_context.completion_order_by_component.get(
                        predecessor,
                        0,
                    ),
                    completed_span_id,
                )
            )

        if not candidates:
            return None
        _, span_id = max(candidates)
        return span_id

    def _is_pipeline_component_span(
        self,
        *,
        parent_id: str | None,
        component_context: _HaystackComponentRunContext | None,
    ) -> bool:
        if (
            parent_id is None
            or component_context is None
            or component_context.pipeline_context is None
        ):
            return False

        if parent_id in self._native_span_ids:
            return True

        return parent_id == component_context.pipeline_context.pipeline_span_id


def _register_haystack_parent_processor(
    processor: _HaystackParentSpanProcessor,
) -> None:
    tracer_provider = trace.get_tracer_provider()
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    span_processors = (
        getattr(active_span_processor, "_span_processors", None)
        if active_span_processor is not None
        else None
    )

    if span_processors is None:
        add_span_processor = getattr(tracer_provider, "add_span_processor", None)
        if add_span_processor is not None:
            add_span_processor(processor)
        return

    remaining_processors = [
        span_processor
        for span_processor in span_processors
        if not isinstance(span_processor, _HaystackParentSpanProcessor)
    ]
    insert_index = 0
    for index, span_processor in enumerate(remaining_processors):
        if span_processor.__class__.__name__ == OPENINFERENCE_TRANSLATOR_CLASS_NAME:
            insert_index = index + 1
            break

    active_span_processor._span_processors = tuple(
        [
            *remaining_processors[:insert_index],
            processor,
            *remaining_processors[insert_index:],
        ]
    )


def _remove_haystack_parent_processor(
    processor: _HaystackParentSpanProcessor,
) -> None:
    tracer_provider = trace.get_tracer_provider()
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    span_processors = (
        getattr(active_span_processor, "_span_processors", None)
        if active_span_processor is not None
        else None
    )
    if span_processors is None:
        return

    active_span_processor._span_processors = tuple(
        span_processor
        for span_processor in span_processors
        if span_processor is not processor
    )


class HaystackInstrumentor:
    """Respan instrumentor for Haystack.

    Activates the OpenInference Haystack instrumentor and registers Respan's
    OpenInference translator so Haystack spans reach the Respan OTLP pipeline
    with canonical ``traceloop.*``, ``gen_ai.*``, and ``respan.*`` fields.

    Usage::

        from respan import Respan
        from respan_instrumentation_haystack import HaystackInstrumentor

        respan = Respan(instrumentations=[HaystackInstrumentor()])
    """

    name = HAYSTACK_INSTRUMENTATION_NAME

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        self._instrumentor_kwargs = instrumentor_kwargs
        self._delegate = None
        self._parent_processor = _HaystackParentSpanProcessor()
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Instrument Haystack via OpenInference and Respan's translator."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Haystack instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            haystack_instrumentor_class = _load_openinference_haystack_class()
            _patch_main_component_wrapping()
        except ImportError as exc:
            logger.warning(
                "Failed to activate Haystack instrumentation - missing dependency: %s",
                exc,
            )
            return

        try:
            self._delegate = OpenInferenceInstrumentor(
                instrumentor_class=haystack_instrumentor_class,
                **self._instrumentor_kwargs,
            )
            self._delegate.activate()
            _patch_pipeline_context_wrapping()
            _register_haystack_parent_processor(self._parent_processor)
            self._is_instrumented = True
            logger.info("Haystack instrumentation activated")
        except Exception:
            _remove_haystack_parent_processor(self._parent_processor)
            if self._delegate is not None:
                try:
                    self._delegate.deactivate()
                except Exception:
                    logger.exception("Failed to clean up Haystack instrumentation")
            self._parent_processor.shutdown()
            self._delegate = None
            self._is_instrumented = False
            logger.exception("Failed to activate Haystack instrumentation")

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        if self._is_instrumented and self._delegate is not None:
            try:
                _remove_haystack_parent_processor(self._parent_processor)
                self._delegate.deactivate()
            except Exception:
                logger.exception("Failed to deactivate Haystack instrumentation")
        self._parent_processor.shutdown()
        self._delegate = None
        self._is_instrumented = False
        logger.info("Haystack instrumentation deactivated")
