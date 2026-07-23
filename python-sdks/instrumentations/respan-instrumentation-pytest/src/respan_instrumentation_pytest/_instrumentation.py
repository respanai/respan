"""Pytest hook implementation that emits canonical Respan spans."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind
from respan_instrumentation_pytest._constants import (
    ENV_EXAMPLE_RUN_ID,
    MAX_ATTRIBUTE_CHARS,
    MAX_COLLECTION_ITEMS,
    MAX_SERIALIZATION_DEPTH,
    PYTEST_INSTRUMENTATION_NAME,
    WORKFLOW_LOG_TYPE,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SPAN_CUSTOM_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing import RespanTelemetry
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)


def _to_jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > MAX_SERIALIZATION_DEPTH:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        items = list(value.items())
        payload = {
            str(key): _to_jsonable(item, depth=depth + 1)
            for key, item in items[:MAX_COLLECTION_ITEMS]
        }
        if len(items) > MAX_COLLECTION_ITEMS:
            payload["_respan_truncated_items"] = len(items) - MAX_COLLECTION_ITEMS
        return payload
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        payload = [
            _to_jsonable(item, depth=depth + 1) for item in items[:MAX_COLLECTION_ITEMS]
        ]
        if len(items) > MAX_COLLECTION_ITEMS:
            payload.append(
                {"_respan_truncated_items": len(items) - MAX_COLLECTION_ITEMS}
            )
        return payload
    return repr(value)


def _json_dumps(value: Any, *, max_chars: int = MAX_ATTRIBUTE_CHARS) -> str:
    serialized = json.dumps(
        _to_jsonable(value),
        default=str,
        ensure_ascii=False,
        sort_keys=True,
    )
    if len(serialized) <= max_chars:
        return serialized
    return json.dumps(
        {
            "truncated": True,
            "original_characters": len(serialized),
            "preview": serialized[:max_chars],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _safe_name(value: str) -> str:
    return "_".join(value.strip().split())[:160] or "pytest_session"


@dataclass
class _TestState:
    span: Any
    context_manager: Any
    reports: dict[str, Any] = field(default_factory=dict)
    durations: dict[str, float] = field(default_factory=dict)
    error: BaseException | None = None
    error_when: str | None = None


class PytestInstrumentor:
    """Actual Pytest plugin and Respan instrumentation lifecycle entrypoint."""

    name = PYTEST_INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        capture_content: bool = True,
        workflow_name: str | None = None,
        tracer: Any = None,
        max_attribute_chars: int = MAX_ATTRIBUTE_CHARS,
    ) -> None:
        self._capture_content = capture_content
        self._workflow_name = workflow_name
        self._provided_tracer = tracer
        self._tracer: Any = None
        self._telemetry: RespanTelemetry | None = None
        self._max_attribute_chars = max(512, int(max_attribute_chars))
        self._is_instrumented = False
        self._session_span: Any = None
        self._session_context_manager: Any = None
        self._session: Any = None
        self._test_states: dict[str, _TestState] = {}
        self._outcome_counts: Counter[str] = Counter()
        self._run_id = os.getenv(ENV_EXAMPLE_RUN_ID) or uuid4().hex[:12]

    def _resolve_tracer(self) -> Any:
        if self._provided_tracer is not None:
            return self._provided_tracer
        if getattr(RespanTracer, "_instance", None) is None:
            self._telemetry = RespanTelemetry(
                app_name=self._workflow_name or "pytest",
                api_key=os.getenv("RESPAN_API_KEY"),
                base_url=os.getenv("RESPAN_BASE_URL"),
                is_auto_instrument=False,
                is_batching_enabled=False,
            )
        return trace.get_tracer(__name__)

    def activate(self) -> None:
        """Prepare tracing. Pytest hook registration is handled by `plugin.py`."""
        if self._is_instrumented:
            return
        self._tracer = self._resolve_tracer()
        self._is_instrumented = True
        logger.info("Pytest instrumentation activated")

    def deactivate(self) -> None:
        """Close any live spans and flush telemetry without altering test results."""
        for nodeid in tuple(self._test_states):
            state = self._test_states.pop(nodeid)
            try:
                self._finish_test_span(nodeid, state)
            finally:
                state.context_manager.__exit__(None, None, None)
        if self._session_context_manager is not None:
            try:
                self._finish_session_span(exitstatus=None)
            finally:
                self._session_context_manager.__exit__(None, None, None)
        self._session_context_manager = None
        self._session_span = None
        self._session = None
        if self._telemetry is not None:
            try:
                self._telemetry.flush()
            except Exception:
                logger.debug("Failed to flush Pytest telemetry", exc_info=True)
        self._is_instrumented = False
        logger.info("Pytest instrumentation deactivated")

    def _resolved_workflow_name(self, config: Any) -> str:
        if self._workflow_name:
            return _safe_name(self._workflow_name)
        rootpath = getattr(config, "rootpath", Path.cwd())
        root_name = Path(rootpath).name or "tests"
        return _safe_name(f"pytest_{root_name}_workflow")

    def _metadata(self, *, worker_id: str | None = None) -> str:
        metadata = {
            "integration": "pytest",
            "workflow_name": self._workflow_name,
            "example_run_id": self._run_id,
        }
        if worker_id:
            metadata["worker_id"] = worker_id
        return _json_dumps(metadata, max_chars=self._max_attribute_chars)

    def _set_common_attributes(
        self,
        span: Any,
        *,
        log_type: str,
        entity_name: str,
        entity_path: str,
        input_payload: dict[str, Any],
        worker_id: str | None = None,
    ) -> None:
        span.set_attribute(RESPAN_LOG_TYPE, log_type)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_path)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            _json_dumps(input_payload, max_chars=self._max_attribute_chars),
        )
        if self._workflow_name:
            span.set_attribute(RESPAN_TRACE_GROUP_ID, self._workflow_name)
        span.set_attribute(RESPAN_METADATA, self._metadata(worker_id=worker_id))

    def pytest_sessionstart(self, session: Any) -> None:
        if not self._is_instrumented or self._session_span is not None:
            return
        try:
            self._session = session
            self._workflow_name = self._resolved_workflow_name(session.config)
            worker_id = os.getenv("PYTEST_XDIST_WORKER")
            self._session_context_manager = self._tracer.start_as_current_span(
                "pytest.session", kind=SpanKind.INTERNAL
            )
            self._session_span = self._session_context_manager.__enter__()
            self._set_common_attributes(
                self._session_span,
                log_type=WORKFLOW_LOG_TYPE,
                entity_name=self._workflow_name,
                entity_path="",
                input_payload={
                    "rootpath": str(getattr(session.config, "rootpath", "")),
                    "arguments": (
                        list(getattr(session.config, "args", ()) or ())
                        if self._capture_content
                        else []
                    ),
                    "content_captured": self._capture_content,
                },
                worker_id=worker_id,
            )
            self._session_span.set_attribute(
                RESPAN_SPAN_CUSTOM_ID,
                f"{self._workflow_name}-{self._run_id}"
                + (f"-{worker_id}" if worker_id else ""),
            )
        except Exception:
            logger.exception("Failed to start Pytest session span")
            self._session_context_manager = None
            self._session_span = None

    def pytest_collection_finish(self, session: Any) -> None:
        if self._session_span is None:
            return
        items = list(getattr(session, "items", ()) or ())
        payload: dict[str, Any] = {
            "rootpath": str(getattr(session.config, "rootpath", "")),
            "collected": len(items),
            "content_captured": self._capture_content,
        }
        if self._capture_content:
            payload["tests"] = [getattr(item, "nodeid", str(item)) for item in items]
        self._session_span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            _json_dumps(payload, max_chars=self._max_attribute_chars),
        )

    def _test_nodeid(self, item: Any) -> str:
        nodeid = str(item.nodeid)
        return nodeid if self._capture_content else nodeid.split("[", 1)[0]

    def _test_input(self, item: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "nodeid": self._test_nodeid(item),
            "content_captured": self._capture_content,
        }
        if not self._capture_content:
            return payload
        callspec = getattr(item, "callspec", None)
        params = getattr(callspec, "params", None)
        if params:
            payload["parameters"] = params
        fixture_names = list(getattr(item, "fixturenames", ()) or ())
        if fixture_names:
            payload["fixtures"] = fixture_names
        iter_markers = getattr(item, "iter_markers", None)
        if callable(iter_markers):
            markers = sorted({marker.name for marker in iter_markers()})
            if markers:
                payload["markers"] = markers
        return payload
