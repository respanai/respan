"""Observe native SageMaker client calls and consumption without replacing objects."""

from __future__ import annotations

import functools
import inspect
import logging
import threading
import weakref
from contextlib import contextmanager

from botocore.client import BaseClient
from botocore.eventstream import EventStream
from botocore.exceptions import ClientError, EventStreamError
from botocore.response import StreamingBody
from opentelemetry import context, trace
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import SpanAttributes as AI
from opentelemetry.trace import SpanKind, Status, StatusCode
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

from ._constants import SUPPORTED_OPERATIONS
from ._otel_emitter import build_sagemaker_attrs
from ._policy import Policy, key, permitted, suppressed
from ._serialization import safe_text, to_jsonable
from ._translator import StreamData, decode, request_body

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_OWNERS = set()
_PATCH = None
_MANAGER = None


def _safe(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:  # noqa: BLE001 - observer faults must preserve native results/errors.
            self.allowed = False
            try:
                self.scrub()
            finally:
                self.finish()
            logger.debug("SageMaker telemetry observation failed")
            return None

    return call


def _http_code(response):
    if type(response) is dict and type(response.get("ResponseMetadata")) is dict:
        code = response["ResponseMetadata"].get("HTTPStatusCode")
        return code if type(code) is int else None
    return None


class _Manager:
    def __init__(self, capture, provider):
        self.capture = capture
        self.provider = provider
        self.policies = []
        self.states = weakref.WeakSet()
        self.enabled = True
        self.observe()

    def observe(self):
        provider = self.provider or trace.get_tracer_provider()
        for existing, policy in self.policies:
            if existing is provider:
                return provider, policy
        policy = (
            Policy(provider, self.scrub)
            if hasattr(provider, "add_span_processor")
            else None
        )
        if policy:
            self.policies.append((provider, policy))
        return provider, policy

    def scrub(self):
        for state in list(self.states):
            if not state.done and not state.check():
                state.scrub()

    def close(self):
        self.enabled = False
        for state in list(self.states):
            state.allowed = False
            state.finish()
        for _, policy in self.policies:
            policy.close()
        self.policies.clear()


class _Call:
    def __init__(self, manager, operation, params):
        self.manager = manager
        self.operation = operation
        self.ctx = context.get_current()
        self.span = None
        self.done = False
        self.allowed = False
        self.undo = []
        self.chunks = bytearray()
        self.stream = StreamData()
        self.params = {}
        self.body = None
        self.fields = {}
        self.streaming = False
        self.native_reads = 0
        self.http = None
        provider, self.policy = manager.observe()
        self.allowed = bool(
            manager.capture
            and permitted(self.ctx)
            and self.policy
            and self.policy.enroll(trace.get_current_span(self.ctx))
        )
        private = (
            context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
            if not self.allowed
            else None
        )
        try:
            self.span = provider.get_tracer(__name__).start_span(
                "sagemaker." + operation, kind=SpanKind.CLIENT
            )
            self.span.set_attribute(RESPAN_LOG_TYPE, "task")
            self.span.set_attribute(AI.TRACELOOP_ENTITY_NAME, "sagemaker." + operation)
            self.span.set_attribute(AI.TRACELOOP_ENTITY_PATH, "")
            self.allowed = self.allowed and self.span.is_recording()
            self.structural = {
                k: v
                for k, v in (getattr(self.span, "attributes", None) or {}).items()
                if k == RESPAN_METADATA or k.startswith(RESPAN_METADATA + ".")
            }
            manager.states.add(self)
            if self.check():
                self.params = to_jsonable(params)
                self.body = request_body(params)
        except Exception:
            if self.span:
                self.scrub()
                try:
                    self.span.end()
                except Exception:  # noqa: BLE001 - telemetry faults must preserve native SDK behavior.
                    logger.debug("SageMaker partial span cleanup failed")
            raise
        finally:
            if private:
                context.detach(private)

    def check(self):
        self.allowed = bool(
            self.allowed
            and not self.done
            and self.manager.enabled
            and permitted(self.ctx)
            and self.policy
            and self.policy.enabled
            and self.policy.bound(key(self.span))
        )
        if not self.allowed:
            if self.policy and not self.policy.scrubbing:
                self.policy.deny(key(self.span))
            self.scrub()
        return self.allowed and self.span.is_recording()

    def scrub(self):
        self.params = {}
        self.body = None
        self.fields = {}
        self.chunks.clear()
        self.stream.clear()
        attrs = getattr(self.span, "_attributes", None)
        if attrs is not None:
            for name in list(attrs):
                if name in (
                    AI.TRACELOOP_ENTITY_INPUT,
                    AI.TRACELOOP_ENTITY_OUTPUT,
                    AI.LLM_REQUEST_FUNCTIONS,
                    ERROR_MESSAGE,
                    RESPAN_METADATA + ".sagemaker",
                ) or name.startswith((AI.LLM_PROMPTS + ".", AI.LLM_COMPLETIONS + ".")):
                    attrs.pop(name, None)
        if hasattr(self.span, "_events"):
            self.span._events = BoundedList(0)
        if (
            getattr(getattr(self.span, "status", None), "status_code", None)
            is StatusCode.ERROR
        ):
            self.span._status = Status(StatusCode.ERROR)

    def tap(self, obj, name, factory):
        original = getattr(obj, name)
        stored = obj.__dict__.get(name)
        present = name in obj.__dict__
        owned = factory(original)
        setattr(obj, name, owned)
        self.undo.append((weakref.ref(obj), name, stored, present, owned))

    @_safe
    def response(self, response):
        self.http = _http_code(response)
        if self.http is not None:
            self.span.set_attribute(HTTP_RESPONSE_STATUS_CODE, self.http)
        if not self.check():
            self.finish()
            return
        if type(response) is not dict:
            self.finish()
            return
        self.fields = to_jsonable(
            {k: v for k, v in response.items() if k not in ("Body", "ResponseMetadata")}
        )
        native_meta = response.get("ResponseMetadata")
        if type(native_meta) is dict:
            self.fields["ResponseMetadata"] = {
                k: native_meta[k]
                for k in ("RequestId", "HTTPStatusCode", "RetryAttempts")
                if k in native_meta
            }
        body = response.get("Body")
        if isinstance(body, EventStream):
            self.streaming = True
            self.tap_events(body)
        elif isinstance(body, StreamingBody):
            self.tap_body(body)
        else:
            self.finish(
                payload=self.fields if self.operation == "InvokeEndpointAsync" else None
            )

    def tap_body(self, body):
        def reader(original, raw=False):
            @functools.wraps(original)
            def read(*args, **kwargs):
                if raw and self.native_reads:
                    return original(*args, **kwargs)
                try:
                    if not raw:
                        self.native_reads += 1
                    try:
                        result = original(*args, **kwargs)
                    finally:
                        if not raw:
                            self.native_reads -= 1
                except BaseException as error:
                    self.finish(error=error)
                    raise
                self.read_result(result, args, kwargs)
                return result

            return read

        self.tap(body, "read", reader)
        if hasattr(body, "readinto"):

            def into(original):
                def readinto(buffer):
                    try:
                        self.native_reads += 1
                        try:
                            count = original(buffer)
                        finally:
                            self.native_reads -= 1
                    except BaseException as error:
                        self.finish(error=error)
                        raise
                    self.into_result(buffer, count)
                    return count

                return readinto

            self.tap(body, "readinto", into)
        self.tap_close(body)
        from urllib3.response import HTTPResponse

        raw = body._raw_stream
        if isinstance(raw, HTTPResponse):
            self.tap(raw, "read", lambda original: reader(original, True))
            self.tap_close(raw)

    @_safe
    def read_result(self, result, args, kwargs):
        if self.check() and type(result) is bytes:
            self.chunks.extend(result)
        amt = args[0] if args else kwargs.get("amt")
        if amt is None or (amt != 0 and result == b""):
            self.finish(payload=decode(bytes(self.chunks)))

    @_safe
    def into_result(self, buffer, count):
        if (
            self.check()
            and type(count) is int
            and count > 0
            and type(buffer) in (bytearray, memoryview)
        ):
            self.chunks.extend(buffer[:count])
        if count == 0 and len(buffer) > 0:
            self.finish(payload=decode(bytes(self.chunks)))

    def tap_events(self, body):
        def parser(original):
            def parse(event):
                try:
                    result = original(event)
                except BaseException as error:
                    self.finish(error=error)
                    raise
                self.event_result(result)
                return result

            return parse

        self.tap(body, "_parse_event", parser)
        original = body._event_generator

        def frames():
            try:
                yield from original
            except Exception as error:
                self.finish(error=error)
                raise
            finally:
                self.finish_observed()

        body._event_generator = frames()
        self.tap_close(body)

    @_safe
    def event_result(self, event):
        if self.check():
            self.stream.add(event)

    def finish_observed(self):
        try:
            payload = (
                (
                    self.stream.payload()
                    if self.streaming
                    else (decode(bytes(self.chunks)) if self.chunks else None)
                )
                if not self.done and self.check()
                else None
            )
            self.finish(payload=payload)
        except Exception:  # noqa: BLE001 - native generator cleanup must not expose observer faults.
            self.allowed = False
            self.finish()

    def tap_close(self, body):
        def closer(original):
            def close(*args, **kwargs):
                try:
                    return original(*args, **kwargs)
                except BaseException as error:
                    self.finish(error=error)
                    raise
                finally:
                    self.finish_observed()

            return close

        self.tap(body, "close", closer)

    def finish(self, payload=None, error=None):
        if self.done:
            return
        try:
            allowed = self.check()
            if self.span.is_recording():
                if allowed:
                    attrs = build_sagemaker_attrs(
                        operation_name=self.operation,
                        params=self.params,
                        body=self.body,
                        payload=payload,
                        response_fields=self.fields,
                        streaming=self.streaming,
                    )
                    if error is not None and payload is None:
                        payload = (
                            self.stream.payload()
                            if self.streaming and self.stream.data
                            else (decode(bytes(self.chunks)) if self.chunks else None)
                        )
                        attrs = build_sagemaker_attrs(
                            operation_name=self.operation,
                            params=self.params,
                            body=self.body,
                            payload=payload,
                            response_fields=self.fields,
                            streaming=self.streaming,
                        )
                    self.span.set_attributes(attrs)
                    self.span.set_attributes(self.structural)
                if self.http is not None:
                    self.span.set_attribute(HTTP_RESPONSE_STATUS_CODE, self.http)
                if error is not None:
                    self.span.set_status(Status(StatusCode.ERROR))
                    self.span.set_attribute(
                        ERROR_TYPE, type.__getattribute__(type(error), "__name__")
                    )
                    if isinstance(error, (ClientError, EventStreamError)):
                        code = _http_code(error.response)
                        if self.http is None and code is not None:
                            self.span.set_attribute(HTTP_RESPONSE_STATUS_CODE, code)
                        source = error.response.get("Error", {})
                        message = (
                            source.get("Message") if type(source) is dict else None
                        )
                    else:
                        message = next(
                            (
                                v
                                for v in BaseException.args.__get__(error)
                                if type(v) is str
                            ),
                            None,
                        )
                    if allowed and type(message) is str:
                        self.span.set_attribute(ERROR_MESSAGE, safe_text(message))
                self.check()
        except Exception:  # noqa: BLE001 - telemetry errors never replace native outcomes.
            self.allowed = False
            self.scrub()
            logger.debug("SageMaker telemetry finalization failed")
        finally:
            self.done = True
            try:
                self.span.end()
            except Exception:  # noqa: BLE001 - telemetry faults must preserve native SDK behavior.
                logger.debug("SageMaker telemetry span end failed")
            if self.policy:
                self.policy.on_end(self.span)
            for ref, name, stored, present, owned in self.undo:
                obj = ref()
                if obj is not None and getattr(obj, name, None) is owned:
                    try:
                        if present:
                            setattr(obj, name, stored)
                        else:
                            delattr(obj, name)
                    except Exception:  # noqa: BLE001 - telemetry faults must preserve native SDK behavior.
                        logger.debug("SageMaker telemetry tap cleanup failed")
            self.undo.clear()
            self.params = {}
            self.body = None
            self.chunks.clear()
            self.stream.clear()
            self.fields = {}
            self.manager.states.discard(self)


@contextmanager
def _scope(state):
    token = None
    private = None
    try:
        try:
            if state:
                if not state.check():
                    private = context.attach(
                        context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
                    )
                token = context.attach(trace.set_span_in_context(state.span))
        except Exception:  # noqa: BLE001 - telemetry context faults cannot prevent native calls.
            if state:
                state.allowed = False
                state.scrub()
            logger.debug("SageMaker telemetry context startup failed")
        yield
    finally:
        try:
            if state and not state.done:
                state.check()
        except Exception:  # noqa: BLE001 - telemetry context faults cannot mask native outcomes.
            if state:
                state.allowed = False
                state.scrub()
            logger.debug("SageMaker telemetry policy check failed")
        if token:
            try:
                context.detach(token)
            except Exception:  # noqa: BLE001 - telemetry faults must preserve native SDK behavior.
                logger.debug("SageMaker telemetry context cleanup failed")
        if private:
            try:
                context.detach(private)
            except Exception:  # noqa: BLE001 - telemetry faults must preserve native SDK behavior.
                logger.debug("SageMaker telemetry privacy cleanup failed")


def _wrap(original):
    @functools.wraps(original)
    def call(client, operation_name, api_params=None):
        if (
            _MANAGER is None
            or operation_name not in SUPPORTED_OPERATIONS
            or client.meta.service_model.service_name != "sagemaker-runtime"
            or suppressed()
        ):
            return original(client, operation_name, api_params)
        state = None
        try:
            state = _Call(_MANAGER, operation_name, api_params)
        except Exception:  # noqa: BLE001 - telemetry faults must preserve native SDK behavior.
            logger.debug("SageMaker telemetry startup failed")
        with _scope(state):
            try:
                result = original(client, operation_name, api_params)
            except BaseException as error:
                if state:
                    state.finish(error=error)
                raise
            if state:
                state.response(result)
            return result

    return call


class SageMakerInstrumentor:
    name = "sagemaker"

    def __init__(self, *, capture_content=True, tracer_provider=None):
        self.capture_content = capture_content
        self.provider = tracer_provider
        self.active = False

    def activate(self):
        global _PATCH, _MANAGER
        with _LOCK:
            if self.active:
                return
            if _MANAGER is not None and (
                _MANAGER.capture != self.capture_content
                or _MANAGER.provider is not self.provider
            ):
                raise RuntimeError(
                    "SageMaker instrumentation already active with different configuration"
                )
            if not _OWNERS:
                manager = None
                try:
                    manager = _Manager(self.capture_content, self.provider)
                    original = inspect.getattr_static(BaseClient, "_make_api_call")
                    owned = _wrap(original)
                    BaseClient._make_api_call = owned
                    _PATCH = (original, owned)
                    _MANAGER = manager
                except Exception:
                    if manager:
                        manager.close()
                    raise
            _OWNERS.add(self)
            self.active = True

    def deactivate(self):
        global _PATCH, _MANAGER
        with _LOCK:
            if not self.active:
                return
            self.active = False
            _OWNERS.discard(self)
            if _OWNERS:
                return
            if _MANAGER:
                _MANAGER.close()
            if (
                _PATCH
                and inspect.getattr_static(BaseClient, "_make_api_call") is _PATCH[1]
            ):
                BaseClient._make_api_call = _PATCH[0]
            _PATCH = None
            _MANAGER = None
