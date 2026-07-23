"""Runtime Pytest hooks built on the public instrumentation lifecycle."""

from __future__ import annotations

import logging
import os
from typing import Any

import pytest
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode

from respan_instrumentation_pytest._constants import TASK_LOG_TYPE
from respan_instrumentation_pytest._instrumentation import (
    PytestInstrumentor,
    _TestState,
    _json_dumps,
)

logger = logging.getLogger(__name__)


class PytestRuntimePlugin(PytestInstrumentor):
    """Pytest hook implementation registered by the `pytest11` entry point."""

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item: Any, nextitem: Any):
        if not self._is_instrumented or self._session_span is None:
            yield
            return

        context_manager = None
        state = None
        try:
            context_manager = self._tracer.start_as_current_span(
                "pytest.test", kind=SpanKind.INTERNAL
            )
            span = context_manager.__enter__()
            state = _TestState(span=span, context_manager=context_manager)
            self._test_states[item.nodeid] = state
            display_nodeid = self._test_nodeid(item)
            self._set_common_attributes(
                span,
                log_type=TASK_LOG_TYPE,
                entity_name=(
                    getattr(item, "name", item.nodeid)
                    if self._capture_content
                    else display_nodeid.rsplit("::", 1)[-1]
                ),
                entity_path=display_nodeid,
                input_payload=self._test_input(item),
                worker_id=os.getenv("PYTEST_XDIST_WORKER"),
            )
        except Exception:
            logger.exception("Failed to start Pytest test span for %s", item.nodeid)
            if context_manager is not None:
                context_manager.__exit__(None, None, None)
            yield
            return

        try:
            yield
        finally:
            try:
                self._finish_test_span(item.nodeid, state)
            except Exception:
                logger.exception(
                    "Failed to finish Pytest test span for %s", item.nodeid
                )
            finally:
                self._test_states.pop(item.nodeid, None)
                context_manager.__exit__(None, None, None)

    @pytest.hookimpl(hookwrapper=True, trylast=True)
    def pytest_runtest_makereport(self, item: Any, call: Any):
        outcome = yield
        state = self._test_states.get(item.nodeid)
        if state is None:
            return
        try:
            report = outcome.get_result()
            state.reports[report.when] = report
            state.durations[report.when] = float(getattr(report, "duration", 0.0))
            if report.failed:
                excinfo = getattr(call, "excinfo", None)
                exception = getattr(excinfo, "value", None)
                if isinstance(exception, BaseException):
                    state.error = exception
                    state.error_when = report.when
                    if isinstance(exception, Exception) and self._capture_content:
                        state.span.record_exception(exception)
                    message = (
                        str(exception)
                        if self._capture_content
                        else type(exception).__name__
                    )
                else:
                    state.error_when = report.when
                    message = "Pytest phase failed"
                state.span.set_status(Status(StatusCode.ERROR, message))
                state.span.set_attribute("status_code", 500)
                state.span.set_attribute("error.message", message)
        except Exception:
            logger.exception("Failed to process Pytest report for %s", item.nodeid)

    @staticmethod
    def _test_outcome(state: _TestState) -> str:
        if any(getattr(report, "failed", False) for report in state.reports.values()):
            return "failed"
        if any(getattr(report, "skipped", False) for report in state.reports.values()):
            if any(hasattr(report, "wasxfail") for report in state.reports.values()):
                return "xfailed"
            return "skipped"
        return "passed"

    def _finish_test_span(self, nodeid: str, state: _TestState) -> None:
        outcome = self._test_outcome(state)
        self._outcome_counts[outcome] += 1
        phases = {
            phase: {
                "outcome": getattr(report, "outcome", "unknown"),
                "duration_seconds": state.durations.get(phase, 0.0),
            }
            for phase, report in state.reports.items()
        }
        output: dict[str, Any] = {
            "outcome": outcome,
            "phases": phases,
            "duration_seconds": sum(state.durations.values()),
            "content_captured": self._capture_content,
        }
        if state.error is not None:
            error = {
                "type": type(state.error).__name__,
                "phase": state.error_when,
            }
            if self._capture_content:
                error["message"] = str(state.error)
            output["error"] = error
        if outcome != "failed":
            state.span.set_attribute("status_code", 200)
        state.span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            _json_dumps(output, max_chars=self._max_attribute_chars),
        )

    def _finish_session_span(self, exitstatus: Any) -> None:
        if self._session_span is None:
            return
        try:
            numeric_status = int(exitstatus) if exitstatus is not None else None
        except (TypeError, ValueError):
            numeric_status = None
        failed = numeric_status is not None and numeric_status not in {0, 5}
        output = {
            "exit_status": numeric_status,
            "outcomes": dict(self._outcome_counts),
            "total": sum(self._outcome_counts.values()),
        }
        self._session_span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            _json_dumps(output, max_chars=self._max_attribute_chars),
        )
        if failed:
            message = f"Pytest exited with status {numeric_status}"
            self._session_span.set_status(Status(StatusCode.ERROR, message))
            self._session_span.set_attribute("status_code", 500)
            self._session_span.set_attribute("error.message", message)
        else:
            self._session_span.set_attribute("status_code", 200)

    def pytest_sessionfinish(self, session: Any, exitstatus: Any) -> None:
        if self._session_context_manager is None:
            return
        try:
            self._finish_session_span(exitstatus)
        finally:
            self._session_context_manager.__exit__(None, None, None)
            self._session_context_manager = None
            self._session_span = None
            if self._telemetry is not None:
                try:
                    self._telemetry.flush()
                except Exception:
                    logger.debug("Failed to flush Pytest telemetry", exc_info=True)
