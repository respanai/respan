"""Semantic Kernel instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from opentelemetry import trace

from respan_instrumentation_semantic_kernel._constants import (
    SEMANTIC_KERNEL_ENABLE_OTEL_DIAGNOSTICS_ENV,
    SEMANTIC_KERNEL_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE_ENV,
    SEMANTIC_KERNEL_INSTRUMENTATION_NAME,
    SEMANTIC_KERNEL_MODEL_DIAGNOSTICS_LOGGER,
    SEMANTIC_KERNEL_MODEL_DIAGNOSTICS_MODULES,
    SEMANTIC_KERNEL_ROOT_MODULE,
)
from respan_instrumentation_semantic_kernel._processor import (
    SemanticKernelLogRecordHandler,
    SemanticKernelSpanProcessor,
    insert_span_processor_before_export,
    remove_span_processor,
)
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_UNSET = object()


class SemanticKernelInstrumentor:
    """Respan instrumentor for Microsoft Semantic Kernel Python."""

    name = SEMANTIC_KERNEL_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._processor: SemanticKernelSpanProcessor | None = None
        self._log_handler: SemanticKernelLogRecordHandler | None = None
        self._is_instrumented = False
        self._previous_env: dict[str, str | object] = {}
        self._previous_settings: list[tuple[Any, str, Any]] = []
        self._diagnostics_logger: logging.Logger | None = None
        self._previous_logger_level: int | None = None

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Activate Semantic Kernel diagnostics and Respan span normalization."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Semantic Kernel instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            importlib.import_module(SEMANTIC_KERNEL_ROOT_MODULE)
        except ImportError as exc:
            logger.warning(
                "Failed to activate Semantic Kernel instrumentation - missing dependency: %s",
                exc,
            )
            return

        tracer_provider = trace.get_tracer_provider()
        try:
            self._enable_semantic_kernel_diagnostics()
            self._processor = SemanticKernelSpanProcessor()
            insert_span_processor_before_export(tracer_provider, self._processor)
            self._install_log_handler()
            self._is_instrumented = True
            logger.info("Semantic Kernel instrumentation activated")
        except Exception:
            if self._processor is not None:
                remove_span_processor(tracer_provider, self._processor)
            self._processor = None
            self._remove_log_handler()
            self._restore_semantic_kernel_diagnostics()
            self._is_instrumented = False
            logger.exception("Failed to activate Semantic Kernel instrumentation")

    def deactivate(self) -> None:
        """Deactivate the instrumentation and restore local diagnostics settings."""
        tracer_provider = trace.get_tracer_provider()
        if self._processor is not None:
            remove_span_processor(tracer_provider, self._processor)
        self._processor = None
        self._remove_log_handler()
        self._restore_semantic_kernel_diagnostics()
        self._is_instrumented = False
        logger.info("Semantic Kernel instrumentation deactivated")

    def _enable_semantic_kernel_diagnostics(self) -> None:
        self._set_env(SEMANTIC_KERNEL_ENABLE_OTEL_DIAGNOSTICS_ENV, "true")
        self._set_env(
            SEMANTIC_KERNEL_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE_ENV,
            "true" if self._capture_content else "false",
        )

        for module_name in SEMANTIC_KERNEL_MODEL_DIAGNOSTICS_MODULES:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            settings = getattr(module, "MODEL_DIAGNOSTICS_SETTINGS", None)
            if settings is None:
                continue
            self._set_setting(settings, "enable_otel_diagnostics", True)
            self._set_setting(
                settings,
                "enable_otel_diagnostics_sensitive",
                self._capture_content,
            )

    def _restore_semantic_kernel_diagnostics(self) -> None:
        for settings, field_name, previous_value in reversed(self._previous_settings):
            setattr(settings, field_name, previous_value)
        self._previous_settings = []

        for env_name, previous_value in self._previous_env.items():
            if previous_value is _UNSET:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = str(previous_value)
        self._previous_env = {}

    def _set_env(self, name: str, value: str) -> None:
        if name not in self._previous_env:
            self._previous_env[name] = os.environ.get(name, _UNSET)
        os.environ[name] = value

    def _set_setting(self, settings: Any, field_name: str, value: bool) -> None:
        self._previous_settings.append(
            (settings, field_name, getattr(settings, field_name))
        )
        setattr(settings, field_name, value)

    def _install_log_handler(self) -> None:
        self._diagnostics_logger = logging.getLogger(
            SEMANTIC_KERNEL_MODEL_DIAGNOSTICS_LOGGER
        )
        self._previous_logger_level = self._diagnostics_logger.level
        self._diagnostics_logger.setLevel(logging.INFO)
        self._log_handler = SemanticKernelLogRecordHandler()
        self._log_handler.setLevel(logging.INFO)
        self._diagnostics_logger.addHandler(self._log_handler)

    def _remove_log_handler(self) -> None:
        if self._diagnostics_logger is not None and self._log_handler is not None:
            self._diagnostics_logger.removeHandler(self._log_handler)
        if (
            self._diagnostics_logger is not None
            and self._previous_logger_level is not None
        ):
            self._diagnostics_logger.setLevel(self._previous_logger_level)
        self._diagnostics_logger = None
        self._previous_logger_level = None
        self._log_handler = None
