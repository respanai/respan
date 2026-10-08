"""Share native LlamaIndex handlers without replacing application iterators."""

from __future__ import annotations

import importlib
import logging
from threading import RLock
from typing import Any, ClassVar

from opentelemetry import trace
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_llama_index._constants import (
    LLAMA_INDEX_INSTRUMENTATION_NAME,
    LLAMA_INDEX_ROOT_MODULE,
)
from respan_instrumentation_llama_index._handlers import (
    RespanLlamaIndexEventHandler,
    RespanLlamaIndexSpanHandler,
)

logger = logging.getLogger(__name__)


class LlamaIndexInstrumentor:
    """One owned dispatcher registration shared by compatible instances."""

    name = LLAMA_INDEX_INSTRUMENTATION_NAME
    _lock: ClassVar[RLock] = RLock()
    _owners: ClassVar[int] = 0
    _shared_span: ClassVar[Any] = None
    _shared_event: ClassVar[Any] = None
    _shared_root: ClassVar[Any] = None
    _shared_provider: ClassVar[Any] = None
    _patches: ClassVar[list[tuple[Any, str, Any, Any]]] = []

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._span_handler = RespanLlamaIndexSpanHandler(
            capture_content=capture_content
        )
        self._event_handler = RespanLlamaIndexEventHandler(
            capture_content=capture_content, span_handler=self._span_handler
        )
        self._span_handler._event_handler = self._event_handler
        self._root_dispatcher = None
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        return tracer is None or bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        cls = type(self)
        with cls._lock:
            if self._is_instrumented:
                return
            if not self._is_respan_tracing_enabled():
                logger.info(
                    "LlamaIndex instrumentation skipped because Respan tracing is disabled"
                )
                return
            provider = trace.get_tracer_provider()
            if cls._owners:
                if (
                    cls._shared_provider is not provider
                    or cls._shared_span.capture_content != self._capture_content
                ):
                    raise ValueError(
                        "LlamaIndex instrumentation is already active with different settings or provider"
                    )
                self._span_handler, self._event_handler = (
                    cls._shared_span,
                    cls._shared_event,
                )
                self._root_dispatcher = cls._shared_root
                cls._owners += 1
                self._is_instrumented = True
                return
            try:
                root = importlib.import_module(LLAMA_INDEX_ROOT_MODULE).root_dispatcher
            except ImportError as exc:
                logger.warning(
                    "Failed to activate LlamaIndex instrumentation — missing dependency: %s",
                    exc,
                )
                return
            self._root_dispatcher = root
            try:
                self._register_handlers()
                (
                    cls._shared_span,
                    cls._shared_event,
                    cls._shared_root,
                    cls._shared_provider,
                ) = self._span_handler, self._event_handler, root, provider
                cls._patch_embeddings_usage()
            except BaseException:
                self._remove_handlers()
                cls._restore_embeddings_usage()
                cls._shared_span = cls._shared_event = cls._shared_root = (
                    cls._shared_provider
                ) = None
                raise
            cls._owners = 1
            self._is_instrumented = True

    def _register_handlers(self) -> None:
        if not any(
            handler is self._span_handler
            for handler in self._root_dispatcher.span_handlers
        ):
            self._root_dispatcher.add_span_handler(self._span_handler)
        if not any(
            handler is self._event_handler
            for handler in self._root_dispatcher.event_handlers
        ):
            self._root_dispatcher.add_event_handler(self._event_handler)

    def _remove_handlers(self) -> None:
        if self._root_dispatcher is not None:
            self._root_dispatcher.span_handlers = [
                h
                for h in self._root_dispatcher.span_handlers
                if h is not self._span_handler
            ]
            self._root_dispatcher.event_handlers = [
                h
                for h in self._root_dispatcher.event_handlers
                if h is not self._event_handler
            ]
        self._event_handler.close()
        self._span_handler.close()
        self._root_dispatcher = None

    @classmethod
    def _patch_embeddings_usage(cls) -> None:
        # LlamaIndex embedding events omit provider usage. Observe the released
        # OpenAI response only inside an active native embedding event.
        try:
            from openai.resources.embeddings import AsyncEmbeddings, Embeddings
        except ImportError:
            return
        installed_handler = cls._shared_event

        def observe(result: Any) -> None:
            if cls._owners and cls._shared_event is installed_handler:
                try:
                    installed_handler.record_embedding_usage(result)
                except Exception:  # noqa: BLE001 - telemetry cannot alter native provider results
                    logger.debug("Could not capture native embedding usage")

        def wrap(original: Any, asynchronous: bool) -> Any:
            if asynchronous:

                async def call(instance: Any, *args: Any, **kwargs: Any) -> Any:
                    result = await original(instance, *args, **kwargs)
                    observe(result)
                    return result
            else:

                def call(instance: Any, *args: Any, **kwargs: Any) -> Any:
                    result = original(instance, *args, **kwargs)
                    observe(result)
                    return result

            return call

        for resource, asynchronous in ((Embeddings, False), (AsyncEmbeddings, True)):
            original = resource.create
            wrapper = wrap(original, asynchronous)
            cls._patches.append((resource, "create", original, wrapper))
            resource.create = wrapper

    @classmethod
    def _restore_embeddings_usage(cls) -> None:
        for owner, name, original, wrapper in reversed(cls._patches):
            if getattr(owner, name) is wrapper:
                setattr(owner, name, original)
        cls._patches = []

    def deactivate(self) -> None:
        cls = type(self)
        with cls._lock:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            cls._owners -= 1
            if cls._owners:
                return
            cls._restore_embeddings_usage()
            self._remove_handlers()
            cls._shared_span = cls._shared_event = cls._shared_root = (
                cls._shared_provider
            ) = None
