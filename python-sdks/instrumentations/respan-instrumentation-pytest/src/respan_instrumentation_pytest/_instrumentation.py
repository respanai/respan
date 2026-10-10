"""Opt-in pytest ownership, native contexts and sourced report summaries."""

import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from uuid import uuid4

from opentelemetry import context, trace
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_MESSAGE,
    ERROR_TYPE,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing import RespanTelemetry
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

from ._policy import Policy, permitted, suppressed
from ._serialization import json_dumps as _json_dumps
from ._serialization import safe_text

logger = logging.getLogger(__name__)


@dataclass
class _TestState:
    span: object
    token: object
    private_token: object = None
    capture: bool = True
    reports: dict = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    error_when: str | None = None
    interrupted: bool = False
    done: bool = False


class PytestInstrumentor:
    name = "pytest"

    def __init__(
        self,
        *,
        capture_content=True,
        workflow_name=None,
        tracer=None,
        tracer_provider=None,
        max_attribute_chars=None,
    ):
        self._capture_content = capture_content
        self._workflow_name = workflow_name
        self._provided_tracer = tracer
        self._provider = tracer_provider
        self._max_attribute_chars = max_attribute_chars
        self._telemetry = None
        self._is_instrumented = False
        self._tracer = None
        self._policy = None
        self._test_states = {}
        self._session_state = None
        self._session_span = None
        self._outcome_counts = Counter()
        self._run_id = os.getenv("RESPAN_EXAMPLE_RUN_ID") or uuid4().hex
        self._metadata = {}

    def activate(self):
        if self._is_instrumented:
            return
        try:
            if self._provided_tracer is not None:
                self._tracer = self._provided_tracer
            else:
                provider = self._provider or trace.get_tracer_provider()
                if not hasattr(provider, "add_span_processor"):
                    self._telemetry = RespanTelemetry(
                        app_name=self._workflow_name or "pytest",
                        api_key=os.getenv("RESPAN_API_KEY"),
                        base_url=os.getenv("RESPAN_BASE_URL"),
                        is_auto_instrument=False,
                        is_batching_enabled=False,
                    )
                    provider = trace.get_tracer_provider()
                self._tracer = provider.get_tracer("respan.instrumentation.pytest")
            processor = getattr(self._tracer, "span_processor", None)
            if processor is not None and hasattr(processor, "add_span_processor"):
                self._policy = Policy(processor)
            self._is_instrumented = True
        except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
            if self._policy:
                self._policy.close()
            self._policy = None
            logger.debug("Pytest telemetry activation failed")

    def deactivate(self):
        self._is_instrumented = False
        for state in reversed(list(self._test_states.values())):
            self._end(state, empty=True)
        self._test_states.clear()
        if self._session_state:
            self._end(self._session_state, empty=True)
        self._session_state = None
        self._session_span = None
        if self._policy:
            self._policy.close()
        self._policy = None
        if self._telemetry:
            try:
                self._telemetry.flush()
            except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
                logger.debug("Pytest telemetry flush failed")
        self._telemetry = None

    def _check(self, state):
        if self._policy:
            return self._policy.check(state)
        state.capture = False
        Policy.clear(state.span)
        return False

    def _start(self, name, kind, entity, path):
        if not self._is_instrumented or suppressed():
            return None
        private = None
        span = None
        token = None
        try:
            allowed = bool(
                self._capture_content
                and self._policy
                and permitted()
                and self._policy.enroll(trace.get_current_span())
            )
            if not allowed:
                private = context.attach(
                    context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
                )
            span = self._tracer.start_span(name)
            token = context.attach(trace.set_span_in_context(span))
            state = _TestState(span, token, private, allowed)
            if not span.is_recording():
                state.capture = False
            if span.is_recording():
                span.set_attribute(RESPAN_LOG_TYPE, kind)
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity)
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, path)
                span.set_attribute(
                    SpanAttributes.TRACELOOP_WORKFLOW_NAME, self._workflow_name
                )
                span.set_attribute(RESPAN_TRACE_GROUP_ID, self._workflow_name)
                self._set_metadata(state)
            self._check(state)
            return state
        except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
            if span:
                Policy.clear(span)
                try:
                    span.end()
                except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
                    logger.debug("Pytest partial span cleanup failed")
            if token:
                context.detach(token)
            if private:
                context.detach(private)
            logger.debug("Pytest telemetry span startup failed")
            return None

    def _set_metadata(self, state, extra=None):
        if not state.span.is_recording():
            return
        data = {**self._metadata, **(extra or {})}
        state.span.set_attribute(RESPAN_METADATA, _json_dumps(data))
        for k, v in data.items():
            if type(v) in (str, int, float, bool):
                state.span.set_attribute(RESPAN_METADATA + "." + k, v)

    def _payload(self, state, field, builder):
        if not self._check(state):
            return
        try:
            state.span.set_attribute(
                field, _json_dumps(builder(), max_bytes=self._max_attribute_chars)
            )
        except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
            state.capture = False
            Policy.clear(state.span)

    def _end(self, state, *, empty=False):
        if state.done:
            return
        state.done = True
        try:
            if empty:
                state.capture = False
            self._check(state)
        except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
            Policy.clear(state.span)
        finally:
            try:
                context.detach(state.token)
            except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
                logger.debug("Pytest telemetry context detach failed")
            if state.private_token:
                try:
                    context.detach(state.private_token)
                except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
                    logger.debug("Pytest telemetry privacy detach failed")
            try:
                state.span.end()
            except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
                logger.debug("Pytest telemetry span end failed")
            state.error_message = None
            state.reports.clear()

    def _test_nodeid(self, item):
        return safe_text(item.nodeid.split("[", 1)[0])

    def _test_input(self, item):
        result = {"nodeid": safe_text(item.nodeid)}
        callspec = getattr(item, "callspec", None)
        if callspec is not None:
            result["parameters"] = callspec.params
        result["fixtures"] = list(item.fixturenames)
        result["markers"] = sorted({m.name for m in item.iter_markers()})
        return result

    def pytest_sessionstart(self, session):
        if self._session_state:
            return
        try:
            self._workflow_name = "_".join(
                safe_text(
                    self._workflow_name
                    or "pytest_" + session.config.rootpath.name + "_workflow"
                ).split()
            )[:160]
            self._metadata = {
                "integration": "pytest",
                "workflow_name": self._workflow_name,
                "run_id": self._run_id,
                "example_run_id": self._run_id,
            }
            worker = os.getenv("PYTEST_XDIST_WORKER")
            if worker:
                self._metadata["worker_id"] = worker
            self._outcome_counts.clear()
            state = self._start("pytest.session", "workflow", self._workflow_name, "")
            self._session_state = state
            self._session_span = state.span if state else None
            if state:
                self._payload(
                    state,
                    SpanAttributes.TRACELOOP_ENTITY_INPUT,
                    lambda: {
                        "rootpath": session.config.rootpath.name,
                        "arguments": list(session.config.args),
                    },
                )
        except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
            logger.debug("Pytest telemetry session startup failed")

    def pytest_collection_finish(self, session):
        state = self._session_state
        if state:
            self._payload(
                state,
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                lambda: {
                    "rootpath": session.config.rootpath.name,
                    "collected": len(session.items),
                    "tests": [safe_text(i.nodeid) for i in session.items],
                },
            )

    def _finish_test_span(self, nodeid, state):
        if any(r["outcome"] == "failed" for r in state.reports.values()):
            outcome = "failed"
        elif state.interrupted:
            outcome = "interrupted"
        elif any(r["outcome"] == "skipped" for r in state.reports.values()):
            outcome = (
                "xfailed"
                if any(r["xfail"] for r in state.reports.values())
                else "skipped"
            )
        else:
            outcome = (
                "xpassed"
                if any(r["xfail"] for r in state.reports.values())
                else "passed"
            )
        self._outcome_counts[outcome] += 1
        self._set_metadata(
            state,
            {
                "pytest.nodeid": nodeid,
                "pytest.outcome": outcome,
                "pytest.phases": _json_dumps(state.reports),
            },
        )
        if outcome in ("failed", "interrupted"):
            state.span.set_status(Status(StatusCode.ERROR))
            if state.error_type:
                state.span.set_attribute(ERROR_TYPE, state.error_type)
            if state.error_message and self._check(state):
                state.span.set_attribute(ERROR_MESSAGE, state.error_message)
        self._payload(
            state,
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            lambda: {
                "outcome": outcome,
                "phases": state.reports,
                "duration_seconds": sum(
                    r["duration_seconds"] for r in state.reports.values()
                ),
            },
        )

    def pytest_sessionfinish(self, session, exitstatus):
        state = self._session_state
        if not state:
            return
        try:
            code = int(exitstatus)
            self._set_metadata(
                state,
                {
                    "pytest.exit_status": code,
                    "pytest.outcomes": _json_dumps(dict(self._outcome_counts)),
                },
            )
            if code not in (0, 5):
                state.span.set_status(Status(StatusCode.ERROR))
                state.span.set_attribute(ERROR_TYPE, "pytest.ExitCode")
            self._payload(
                state,
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                lambda: {
                    "exit_status": code,
                    "outcomes": dict(self._outcome_counts),
                    "total": sum(self._outcome_counts.values()),
                },
            )
        except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
            state.capture = False
            Policy.clear(state.span)
        finally:
            self._end(state)
            self._session_state = None
            self._session_span = None
        if self._telemetry:
            try:
                self._telemetry.flush()
            except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
                logger.debug("Pytest telemetry flush failed")
