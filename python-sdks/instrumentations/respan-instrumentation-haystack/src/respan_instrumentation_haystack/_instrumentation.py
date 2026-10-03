"""Haystack instrumentation plugin for Respan."""

import importlib
import inspect
import logging
import threading
from typing import Any, ClassVar

from opentelemetry import trace
from respan_instrumentation_openinference import OpenInferenceInstrumentor
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_haystack._compat import (
    install_compatibility,
    restore_compatibility,
)
from respan_instrumentation_haystack._constants import (
    HAYSTACK_ASYNC_PIPELINE_CLASS_NAME,
    HAYSTACK_ASYNC_PIPELINE_MODULE,
    HAYSTACK_COMPONENT_DECORATOR_ATTRIBUTE,
    HAYSTACK_COMPONENT_MODULE,
    HAYSTACK_COMPONENT_REGISTRY_ATTRIBUTE,
    HAYSTACK_INSTRUMENTATION_NAME,
    HAYSTACK_PIPELINE_CLASS_NAME,
    HAYSTACK_PIPELINE_MODULE,
    HAYSTACK_RUN_ASYNC_GENERATOR_METHOD_NAME,
    HAYSTACK_RUN_ASYNC_METHOD_NAME,
    HAYSTACK_RUN_COMPONENT_ASYNC_METHOD_NAME,
    HAYSTACK_RUN_COMPONENT_METHOD_NAME,
    HAYSTACK_RUN_METHOD_NAME,
    OPENINFERENCE_HAYSTACK_INSTRUMENTOR_CLASS_NAME,
    OPENINFERENCE_HAYSTACK_MODULE,
    OPENINFERENCE_HAYSTACK_WRAPPERS_MODULE,
    OPENINFERENCE_TRANSLATOR_CLASS_NAME,
    RESPAN_HAYSTACK_MAIN_COMPONENT_PATCH_FLAG,
)
from respan_instrumentation_haystack._context import (
    _async_component_run_context_wrapper,
    _async_pipeline_run_async_generator_context_wrapper,
    _async_pipeline_run_context_wrapper,
    _component_run_context_wrapper,
    _pipeline_run_context_wrapper,
)
from respan_instrumentation_haystack._processor import _HaystackParentSpanProcessor

logger = logging.getLogger(__name__)


_PIPELINE_CONTEXT_PATCH_APPLIED = False
_PIPELINE_CONTEXT_PATCHES = []
_MAIN_COMPONENT_PATCH = None


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
    global _MAIN_COMPONENT_PATCH
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
            return original_wrap_function_wrapper(module, name, wrapper)
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
            return original_wrap_function_wrapper(component_class, method_name, wrapper)

    _MAIN_COMPONENT_PATCH = (
        haystack_module,
        original_wrap_function_wrapper,
        compatible_wrap_function_wrapper,
    )
    haystack_module.wrap_function_wrapper = compatible_wrap_function_wrapper
    setattr(haystack_module, RESPAN_HAYSTACK_MAIN_COMPONENT_PATCH_FLAG, True)


def _patch_late_component_registration(delegate: Any) -> tuple[Any, Any, Any] | None:
    """Wrap components imported after OpenInference activation.

    OpenInference wraps the component registry only once during activation.
    Haystack applications commonly initialize tracing before importing their
    components, so later direct ``component.run()`` calls otherwise disappear.
    """
    openinference_instrumentor = getattr(delegate, "_instrumentor", None)
    tracer = getattr(openinference_instrumentor, "_tracer", None)
    sync_originals = getattr(
        openinference_instrumentor,
        "_original_component_run_methods",
        None,
    )
    async_originals = getattr(
        openinference_instrumentor,
        "_original_component_run_async_methods",
        None,
    )
    if (
        tracer is None
        or not isinstance(sync_originals, dict)
        or not isinstance(async_originals, dict)
    ):
        return None

    component_module = importlib.import_module(HAYSTACK_COMPONENT_MODULE)
    haystack_module = importlib.import_module(OPENINFERENCE_HAYSTACK_MODULE)
    wrappers_module = importlib.import_module(OPENINFERENCE_HAYSTACK_WRAPPERS_MODULE)
    component_decorator = getattr(
        component_module,
        HAYSTACK_COMPONENT_DECORATOR_ATTRIBUTE,
    )
    original_component = component_decorator._component
    wrap_function_wrapper = haystack_module.wrap_function_wrapper

    def wrap_registered_component(component_class: type[Any]) -> None:
        run_method = getattr(component_class, "run", None)
        if callable(run_method) and component_class not in sync_originals:
            sync_originals[component_class] = run_method
            wrap_function_wrapper(
                component_class,
                "run",
                wrappers_module._ComponentRunWrapper(tracer=tracer),
            )

        run_async_method = getattr(component_class, "run_async", None)
        if callable(run_async_method) and component_class not in async_originals:
            async_originals[component_class] = run_async_method
            wrap_function_wrapper(
                component_class,
                "run_async",
                wrappers_module._AsyncComponentRunWrapper(tracer=tracer),
            )

    def component_with_late_wrapping(component_class: type[Any]) -> type[Any]:
        registered_component = original_component(component_class)
        wrap_registered_component(registered_component)
        return registered_component

    component_decorator._component = component_with_late_wrapping
    return component_decorator, original_component, component_with_late_wrapping


def _restore_late_component_registration(
    patch: tuple[Any, Any, Any] | None,
) -> None:
    if patch is None:
        return
    component_decorator, original_component, patched_component = patch
    if getattr(component_decorator, "_component", None) is patched_component:
        component_decorator._component = original_component


def _patch_pipeline_context_wrapping() -> None:
    global _PIPELINE_CONTEXT_PATCH_APPLIED

    if _PIPELINE_CONTEXT_PATCH_APPLIED:
        return

    try:
        haystack_module = importlib.import_module(OPENINFERENCE_HAYSTACK_MODULE)
        pipeline_module = importlib.import_module(HAYSTACK_PIPELINE_MODULE)
    except ImportError:
        return

    wrap_function_wrapper = getattr(haystack_module, "wrap_function_wrapper", None)
    if wrap_function_wrapper is None:
        return

    pipeline_class = getattr(pipeline_module, HAYSTACK_PIPELINE_CLASS_NAME)
    try:
        async_pipeline_module = importlib.import_module(HAYSTACK_ASYNC_PIPELINE_MODULE)
        async_pipeline_class = getattr(
            async_pipeline_module, HAYSTACK_ASYNC_PIPELINE_CLASS_NAME
        )
    except ImportError:
        async_pipeline_class = pipeline_class
    upstream_wrap = wrap_function_wrapper

    def wrap_function_wrapper(owner, name, wrapper):
        original = inspect.getattr_static(owner, name)
        upstream_wrap(owner, name, wrapper)
        _PIPELINE_CONTEXT_PATCHES.append(
            (owner, name, original, inspect.getattr_static(owner, name))
        )

    wrap_function_wrapper(
        pipeline_class,
        HAYSTACK_RUN_METHOD_NAME,
        _pipeline_run_context_wrapper,
    )
    if async_pipeline_class is not pipeline_class:
        wrap_function_wrapper(
            async_pipeline_class,
            HAYSTACK_RUN_METHOD_NAME,
            _pipeline_run_context_wrapper,
        )
    wrap_function_wrapper(
        async_pipeline_class,
        HAYSTACK_RUN_ASYNC_METHOD_NAME,
        _async_pipeline_run_context_wrapper,
    )
    wrap_function_wrapper(
        async_pipeline_class,
        HAYSTACK_RUN_ASYNC_GENERATOR_METHOD_NAME,
        _async_pipeline_run_async_generator_context_wrapper,
    )
    wrap_function_wrapper(
        pipeline_class,
        HAYSTACK_RUN_COMPONENT_METHOD_NAME,
        _component_run_context_wrapper,
    )
    wrap_function_wrapper(
        async_pipeline_class,
        HAYSTACK_RUN_COMPONENT_ASYNC_METHOD_NAME,
        _async_component_run_context_wrapper,
    )
    _PIPELINE_CONTEXT_PATCH_APPLIED = True


def _restore_context_patches() -> None:
    global _PIPELINE_CONTEXT_PATCH_APPLIED, _MAIN_COMPONENT_PATCH
    for owner, name, original, replacement in reversed(_PIPELINE_CONTEXT_PATCHES):
        if inspect.getattr_static(owner, name) is replacement:
            setattr(owner, name, original)
    _PIPELINE_CONTEXT_PATCHES.clear()
    _PIPELINE_CONTEXT_PATCH_APPLIED = False
    if _MAIN_COMPONENT_PATCH is not None:
        owner, original, replacement = _MAIN_COMPONENT_PATCH
        if owner.wrap_function_wrapper is replacement:
            owner.wrap_function_wrapper = original
        setattr(owner, RESPAN_HAYSTACK_MAIN_COMPONENT_PATCH_FLAG, False)
        _MAIN_COMPONENT_PATCH = None


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

    active_span_processor._span_processors = (
        *remaining_processors[:insert_index],
        processor,
        *remaining_processors[insert_index:],
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
    _lock: ClassVar[threading.RLock] = threading.RLock()
    _owner: ClassVar[Any] = None
    _owner_count: ClassVar[int] = 0

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        self._instrumentor_kwargs = instrumentor_kwargs
        self._delegate = None
        self._late_component_patch = None
        self._compatibility_patches = []
        self._parent_processor = _HaystackParentSpanProcessor()
        self._is_instrumented = False

    @property
    def is_instrumented(self) -> bool:
        """Whether the upstream instrumentor and Respan processor are active."""
        return self._is_instrumented

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Instrument Haystack via OpenInference and Respan's translator."""
        with self._lock:
            if self._is_instrumented:
                return
            cls = HaystackInstrumentor
            if cls._owner is not None:
                if self._instrumentor_kwargs != cls._owner._instrumentor_kwargs:
                    logger.warning(
                        "Haystack instrumentation is already active; the first settings remain in effect"
                    )
                self._is_instrumented = True
                cls._owner_count += 1
                return
            self._activate()
            if self._is_instrumented:
                cls._owner = self
                cls._owner_count = 1

    def _activate(self) -> None:
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
            self._compatibility_patches = install_compatibility(self._delegate)
            self._late_component_patch = _patch_late_component_registration(
                self._delegate
            )
            _patch_pipeline_context_wrapping()
            _register_haystack_parent_processor(self._parent_processor)
            self._is_instrumented = True
            logger.info("Haystack instrumentation activated")
        except Exception:
            restore_compatibility(self._compatibility_patches)
            self._compatibility_patches = []
            _restore_context_patches()
            _restore_late_component_registration(self._late_component_patch)
            self._late_component_patch = None
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
        """Restore shared patches after the last owner deactivates."""
        with self._lock:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            cls = HaystackInstrumentor
            cls._owner_count -= 1
            if cls._owner_count:
                return
            owner = cls._owner
            cls._owner = None
            owner._deactivate()

    def _deactivate(self) -> None:
        if self._delegate is not None:
            try:
                _remove_haystack_parent_processor(self._parent_processor)
                _restore_context_patches()
                restore_compatibility(self._compatibility_patches)
                self._compatibility_patches = []
                _restore_late_component_registration(self._late_component_patch)
                self._late_component_patch = None
                self._delegate.deactivate()
            except Exception:
                logger.exception("Failed to deactivate Haystack instrumentation")
        self._parent_processor.shutdown()
        self._delegate = None
        self._late_component_patch = None
        self._is_instrumented = False
        logger.info("Haystack instrumentation deactivated")
