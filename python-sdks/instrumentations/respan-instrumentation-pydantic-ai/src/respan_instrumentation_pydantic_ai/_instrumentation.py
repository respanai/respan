"""Respan plugin for direct PydanticAI span instrumentation."""

from __future__ import annotations

import logging
from threading import RLock
from typing import Any, ClassVar

from opentelemetry import trace

from respan_instrumentation_pydantic_ai._embeddings import (
    install_embedding_capture,
    load_embedder_class,
)
from respan_instrumentation_pydantic_ai._processor import PydanticAISpanProcessor

logger = logging.getLogger(__name__)

_UNSET = object()


class PydanticAIInstrumentor:
    """Enable native Pydantic AI telemetry and normalize its ended spans."""

    name = "pydantic-ai"
    _lifecycle_lock: ClassVar[RLock] = RLock()
    _shared_processor: ClassVar[PydanticAISpanProcessor | None] = None
    _shared_provider: ClassVar[Any | None] = None
    _processor_refcount: ClassVar[int] = 0
    _global_refcount: ClassVar[int] = 0
    _global_config: ClassVar[tuple[bool, bool, int] | None] = None
    _global_agent_class: ClassVar[Any | None] = None
    _global_previous: ClassVar[Any] = _UNSET
    _global_embedder_class: ClassVar[Any | None] = None
    _global_embedder_previous: ClassVar[Any] = _UNSET
    _embedding_capture_restore: ClassVar[Any] = None
    _specific_agents: ClassVar[dict[int, dict[str, Any]]] = {}

    def __init__(
        self,
        agent: Any | None = None,
        *,
        embedder: Any | None = None,
        include_content: bool = True,
        include_binary_content: bool = True,
        version: int = 5,
    ) -> None:
        if agent is not None and embedder is not None:
            raise ValueError("Pass either agent or embedder, not both")
        self._agent = agent if agent is not None else embedder
        self._include_content = include_content
        self._include_binary_content = include_binary_content
        self._version = version
        self._is_instrumented = False
        self._activation_mode: tuple[str, int | None] | None = None

    @property
    def _config(self) -> tuple[bool, bool, int]:
        return (
            self._include_content,
            self._include_binary_content,
            self._version,
        )

    @staticmethod
    def _register_processor(
        tracer_provider: Any,
        processor: PydanticAISpanProcessor,
    ) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )
        if processors is not None:
            if any(existing is processor for existing in processors):
                return
            active_span_processor._span_processors = (processor, *processors)
            return
        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)

    @staticmethod
    def _unregister_processor(
        tracer_provider: Any,
        processor: PydanticAISpanProcessor,
    ) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )
        if processors is None:
            return
        active_span_processor._span_processors = tuple(
            existing for existing in processors if existing is not processor
        )

    @classmethod
    def _acquire_processor(cls, tracer_provider: Any) -> None:
        if cls._processor_refcount:
            if cls._shared_provider is not tracer_provider:
                raise RuntimeError(
                    "all active PydanticAIInstrumentor instances must use the same "
                    "tracer provider"
                )
            cls._processor_refcount += 1
            return

        processor = PydanticAISpanProcessor()
        restore_capture = install_embedding_capture()
        try:
            cls._register_processor(tracer_provider, processor)
        except Exception:
            restore_capture()
            raise
        cls._embedding_capture_restore = restore_capture
        cls._shared_processor = processor
        cls._shared_provider = tracer_provider
        cls._processor_refcount = 1

    @classmethod
    def _release_processor(cls) -> None:
        if not cls._processor_refcount:
            return
        cls._processor_refcount -= 1
        if cls._processor_refcount:
            return
        if cls._shared_processor is not None and cls._shared_provider is not None:
            cls._unregister_processor(cls._shared_provider, cls._shared_processor)
        cls._shared_processor = None
        cls._shared_provider = None
        if cls._embedding_capture_restore is not None:
            cls._embedding_capture_restore()
            cls._embedding_capture_restore = None

    def activate(self) -> None:
        if self._is_instrumented:
            return
        try:
            from pydantic_ai.agent import Agent
            from pydantic_ai.models.instrumented import InstrumentationSettings
        except ImportError as exc:
            logger.warning(
                "Failed to activate PydanticAI instrumentation — missing dependency: %s",
                exc,
            )
            return

        cls = type(self)
        tracer_provider = trace.get_tracer_provider()
        settings = InstrumentationSettings(
            tracer_provider=tracer_provider,
            include_content=self._include_content,
            include_binary_content=self._include_binary_content,
            version=self._version,
        )
        with cls._lifecycle_lock:
            if self._is_instrumented:
                return
            cls._acquire_processor(tracer_provider)
            try:
                if self._agent is None:
                    if cls._global_refcount:
                        if cls._global_config != self._config:
                            raise ValueError(
                                "all active global PydanticAIInstrumentor instances "
                                "must use the same content and version settings"
                            )
                        cls._global_refcount += 1
                    else:
                        previous = getattr(Agent, "_instrument_default", _UNSET)
                        Agent.instrument_all(instrument=settings)
                        embedder_class = load_embedder_class()
                        try:
                            if embedder_class is not None:
                                cls._global_embedder_previous = (
                                    embedder_class._instrument_default
                                )
                                embedder_class.instrument_all(instrument=settings)
                                cls._global_embedder_class = embedder_class
                        except Exception:
                            Agent.instrument_all(
                                instrument=False if previous is _UNSET else previous
                            )
                            cls._global_embedder_previous = _UNSET
                            raise
                        cls._global_previous = previous
                        cls._global_agent_class = Agent
                        cls._global_config = self._config
                        cls._global_refcount = 1
                    self._activation_mode = ("global", None)
                else:
                    key = id(self._agent)
                    active = cls._specific_agents.get(key)
                    if active is not None:
                        if (
                            active["agent"] is not self._agent
                            or active["config"] != self._config
                        ):
                            raise ValueError(
                                "all active instrumentors for one Pydantic AI agent "
                                "must use the same settings"
                            )
                        active["count"] += 1
                    else:
                        previous = getattr(self._agent, "instrument", _UNSET)
                        self._agent.instrument = settings
                        cls._specific_agents[key] = {
                            "agent": self._agent,
                            "config": self._config,
                            "count": 1,
                            "previous": previous,
                        }
                    self._activation_mode = ("agent", key)
            except Exception:
                cls._release_processor()
                raise
            self._is_instrumented = True
        logger.info("PydanticAI instrumentation activated")

    def deactivate(self) -> None:
        cls = type(self)
        with cls._lifecycle_lock:
            if not self._is_instrumented:
                return
            mode = self._activation_mode
            try:
                if mode == ("global", None):
                    cls._global_refcount = max(cls._global_refcount - 1, 0)
                    if not cls._global_refcount and cls._global_agent_class is not None:
                        previous = (
                            False
                            if cls._global_previous is _UNSET
                            else cls._global_previous
                        )
                        cls._global_agent_class.instrument_all(instrument=previous)
                        if cls._global_embedder_class is not None:
                            previous_embedding = (
                                False
                                if cls._global_embedder_previous is _UNSET
                                else cls._global_embedder_previous
                            )
                            cls._global_embedder_class.instrument_all(
                                instrument=previous_embedding
                            )
                            cls._global_embedder_class = None
                            cls._global_embedder_previous = _UNSET
                        cls._global_agent_class = None
                        cls._global_previous = _UNSET
                        cls._global_config = None
                elif mode is not None and mode[0] == "agent" and mode[1] is not None:
                    active = cls._specific_agents.get(mode[1])
                    if active is not None:
                        active["count"] = max(active["count"] - 1, 0)
                        if not active["count"]:
                            previous = active["previous"]
                            active["agent"].instrument = (
                                None if previous is _UNSET else previous
                            )
                            cls._specific_agents.pop(mode[1], None)
            finally:
                cls._release_processor()
                self._is_instrumented = False
                self._activation_mode = None
        logger.info("PydanticAI instrumentation deactivated")
