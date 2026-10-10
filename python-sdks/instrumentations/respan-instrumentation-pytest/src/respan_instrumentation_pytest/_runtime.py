"""Pass-through native pytest hooks; telemetry faults never alter outcomes."""

import logging

import pytest
from opentelemetry.semconv_ai import SpanAttributes

from ._instrumentation import PytestInstrumentor
from ._serialization import safe_text

logger = logging.getLogger(__name__)


class PytestRuntimePlugin(PytestInstrumentor):
    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        state = None
        try:
            if self._session_state:
                nodeid = self._test_nodeid(item)
                state = self._start(
                    "pytest.test", "task", nodeid.rsplit("::", 1)[-1], nodeid
                )
                if state:
                    self._test_states[item.nodeid] = state
                    self._payload(
                        state,
                        SpanAttributes.TRACELOOP_ENTITY_INPUT,
                        lambda: self._test_input(item),
                    )
        except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
            logger.debug("Pytest telemetry test startup failed")
        try:
            outcome = yield
            if state and outcome.excinfo:
                state.interrupted = True
                state.error_type = type.__getattribute__(
                    type(outcome.excinfo[1]), "__name__"
                )
        finally:
            if state:
                try:
                    self._finish_test_span(self._test_nodeid(item), state)
                except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
                    state.capture = False
                    logger.debug("Pytest telemetry test finalization failed")
                finally:
                    self._end(state)
                    self._test_states.pop(item.nodeid, None)

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        state = self._test_states.get(item.nodeid)
        if state is None:
            return
        try:
            report = outcome.get_result()
            state.reports[report.when] = {
                "outcome": report.outcome,
                "duration_seconds": report.duration,
                "xfail": hasattr(report, "wasxfail"),
            }
            self._check(state)
            if report.failed:
                state.error_when = report.when
                exc = call.excinfo.value if call.excinfo else None
                if exc is not None:
                    state.error_type = type.__getattribute__(type(exc), "__name__")
                    if state.capture:
                        args = BaseException.args.__get__(exc)
                        state.error_message = safe_text(
                            next((v for v in args if type(v) is str), state.error_type)
                        )
                elif state.capture and type(report.longrepr) is str:
                    state.error_message = safe_text(report.longrepr)
        except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
            state.capture = False
            logger.debug("Pytest telemetry report observation failed")
