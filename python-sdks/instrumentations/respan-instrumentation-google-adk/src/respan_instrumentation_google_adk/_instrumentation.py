"""Google ADK instrumentation plugin for Respan."""

import importlib
import logging
import threading
from typing import Any, ClassVar

from opentelemetry import trace
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_google_adk._compat import patch_legacy_agent_iterator
from respan_instrumentation_google_adk._ownership import PatchTransaction
from respan_instrumentation_google_adk._processor import (
    GoogleADKSpanProcessor,
    insert_span_processor_before_export,
    remove_span_processor,
)

logger = logging.getLogger(__name__)

GOOGLE_ADK_INSTRUMENTATION_NAME = "google-adk"
OPENINFERENCE_GOOGLE_ADK_MODULE = "openinference.instrumentation.google_adk"


def _load_openinference_google_adk_class() -> type:
    google_adk_module = importlib.import_module(OPENINFERENCE_GOOGLE_ADK_MODULE)
    return google_adk_module.GoogleADKInstrumentor


class GoogleADKInstrumentor:
    """Respan instrumentor for Google ADK.

    Activates the OpenInference Google ADK instrumentor and registers a
    Google-ADK-specific span processor so ADK spans reach the Respan OTEL
    pipeline with canonical ``traceloop.*`` and ``gen_ai.*`` fields.
    """

    name = GOOGLE_ADK_INSTRUMENTATION_NAME
    _lock: ClassVar[threading.RLock] = threading.RLock()
    _owner: ClassVar[Any] = None
    _owner_count: ClassVar[int] = 0

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        self._instrumentor_kwargs = dict(instrumentor_kwargs)
        self._instrumentor = None
        self._processor = None
        self._undo_legacy_iterator = None
        self._is_instrumented = False
        self._provider = None
        self._undo_workflows = None
        self._patch_transaction = None

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        with self._lock:
            if self._is_instrumented:
                return
            cls = GoogleADKInstrumentor
            if cls._owner is not None:
                if self._instrumentor_kwargs != cls._owner._instrumentor_kwargs:
                    raise ValueError(
                        "Google ADK is already active with a different configuration; deactivate its owners before changing content/privacy settings"
                    )
                self._instrumentor = cls._owner._instrumentor
                self._processor = cls._owner._processor
                self._provider = cls._owner._provider
                self._is_instrumented = True
                cls._owner_count += 1
                return
            self._activate()
            if self._is_instrumented:
                cls._owner = self
                cls._owner_count = 1

    def _activate(self) -> None:
        """Instrument Google ADK via OpenInference and Respan's ADK processor."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Google ADK instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            google_adk_instrumentor_class = _load_openinference_google_adk_class()
        except ImportError as exc:
            logger.warning(
                "Failed to activate Google ADK instrumentation - missing dependency: %s",
                exc,
            )
            return

        tracer_provider = (
            self._instrumentor_kwargs.get("tracer_provider")
            or trace.get_tracer_provider()
        )
        self._provider = tracer_provider
        try:
            upstream = google_adk_instrumentor_class()
            if getattr(upstream, "is_instrumented_by_opentelemetry", False):
                logger.warning(
                    "Google ADK instrumentation is already active under another "
                    "owner; deactivate that owner before activating this adapter"
                )
                return
            self._patch_transaction = PatchTransaction()
            self._processor = GoogleADKSpanProcessor(
                config=self._instrumentor_kwargs.get("config")
            )
            insert_span_processor_before_export(tracer_provider, self._processor)
            self._instrumentor = upstream
            self._instrumentor.instrument(
                tracer_provider=tracer_provider,
                **{
                    key: value
                    for key, value in self._instrumentor_kwargs.items()
                    if key != "tracer_provider"
                },
            )
            # OTel instrumentors may log dependency conflicts and return without
            # raising. Do not report an active adapter or retain its processor.
            if not getattr(
                self._instrumentor, "is_instrumented_by_opentelemetry", True
            ):
                raise RuntimeError(
                    "OpenInference Google ADK instrumentation did not activate"
                )
            self._undo_legacy_iterator = patch_legacy_agent_iterator()
            from ._workflow import patch_workflow_nodes

            self._undo_workflows = patch_workflow_nodes(
                getattr(upstream, "_tracer", None)
            )
            self._patch_transaction.guard()
            self._is_instrumented = True
            logger.info("Google ADK instrumentation activated")
        except Exception:
            if self._undo_workflows is not None:
                self._undo_workflows()
                self._undo_workflows = None
            if self._undo_legacy_iterator is not None:
                self._undo_legacy_iterator()
                self._undo_legacy_iterator = None
            if self._instrumentor is not None:
                try:
                    if self._patch_transaction is not None:
                        self._patch_transaction.restore(
                            self._instrumentor.uninstrument, partial=True
                        )
                        self._patch_transaction = None
                    else:
                        self._instrumentor.uninstrument()
                except Exception:
                    logger.exception("Failed to clean up Google ADK instrumentation")
            if self._processor is not None:
                remove_span_processor(tracer_provider, self._processor)
            self._instrumentor = None
            self._processor = None
            self._is_instrumented = False
            logger.exception("Failed to activate Google ADK instrumentation")

    def deactivate(self) -> None:
        with self._lock:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            cls = GoogleADKInstrumentor
            cls._owner_count -= 1
            if cls._owner_count:
                return
            owner, cls._owner = cls._owner, None
            owner._deactivate()

    def _deactivate(self) -> None:
        """Deactivate the instrumentation."""
        tracer_provider = self._provider
        if self._undo_workflows is not None:
            self._undo_workflows()
            self._undo_workflows = None
        if self._undo_legacy_iterator is not None:
            self._undo_legacy_iterator()
            self._undo_legacy_iterator = None
        if self._instrumentor is not None:
            try:
                if self._patch_transaction is not None:
                    self._patch_transaction.restore(self._instrumentor.uninstrument)
                    self._patch_transaction = None
                else:
                    self._instrumentor.uninstrument()
            except Exception:
                logger.exception("Failed to deactivate Google ADK instrumentation")
        if self._processor is not None:
            remove_span_processor(tracer_provider, self._processor)
        if self._processor is not None:
            self._processor.shutdown()
        self._instrumentor = None
        self._processor = None
        self._provider = None
        self._is_instrumented = False
        logger.info("Google ADK instrumentation deactivated")
