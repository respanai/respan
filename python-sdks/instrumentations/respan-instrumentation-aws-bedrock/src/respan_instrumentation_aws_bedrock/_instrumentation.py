"""Observe native botocore Bedrock calls without consuming or replacing bodies."""

from __future__ import annotations

import builtins
import functools
import importlib
import json
import logging
import threading
import weakref

from opentelemetry import context, trace
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from respan_sdk.constants.otlp_constants import ERROR_MESSAGE_ATTR
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_aws_bedrock._constants import (
    AWS_BEDROCK_INSTRUMENTATION_NAME,
    BEDROCK_RUNTIME_SERVICE_NAME,
    SUPPORTED_OPERATIONS,
)
from respan_instrumentation_aws_bedrock._otel_emitter import build_bedrock_attrs
from respan_instrumentation_aws_bedrock._privacy import (
    PolicyObserver,
    content_allowed,
    suppressed,
    text,
    value,
)

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_OWNERS = set()
_PATCH = None
_CONFIG = None
_OBSERVERS = []
_PENDING = weakref.WeakSet()


def _load_base_client_class():
    return importlib.import_module("botocore.client").BaseClient


def _provider(config):
    return config[1] if config[1] is not None else RespanTracer().tracer_provider


def _observer(provider):
    with _LOCK:
        for existing, observer in _OBSERVERS:
            if existing is provider:
                return observer
        observer = PolicyObserver()
        provider.add_span_processor(observer)
        _OBSERVERS.append((provider, observer))
        return observer


def _http_code(response):
    if type(response) is dict and type(response.get("ResponseMetadata")) is dict:
        code = response["ResponseMetadata"].get("HTTPStatusCode")
        return code if type(code) is int else None
    return None


class _Call:
    def __init__(self, operation, params, config, client):
        self.operation = operation
        from botocore import exceptions

        self.error_types = tuple(
            cls
            for module in (exceptions, builtins)
            for cls in vars(module).values()
            if type(cls) is type and issubclass(cls, BaseException)
        ) + tuple(client.exceptions._code_to_exception.values())
        self.ctx = context.get_current()
        self.carrier = trace.get_current_span(self.ctx)
        provider = _provider(config)
        self.observer = _observer(provider)
        self.allowed = (
            config[0]
            and content_allowed(self.ctx)
            and self.observer.allowed(self.carrier)
        )
        self.span = trace.get_tracer(__name__, tracer_provider=provider).start_span(
            operation, context=self.ctx, kind=SpanKind.CLIENT
        )
        self.recording = self.span.is_recording()
        self.allowed = self.allowed and self.recording
        try:
            self.params = value(params) if self.allowed else _model_only(params)
        except Exception:  # noqa: BLE001 - failed telemetry serde cannot leak a started span.
            self.allowed = False
            self.params = _model_only(params)
        self.events = []
        self.chunks = []
        self.done = False
        self.undo = []
        self.http_code = None
        self.native_reads = 0
        _PENDING.add(self)

    def policy(self):
        self.allowed = (
            self.allowed
            and content_allowed(self.ctx)
            and content_allowed()
            and self.observer.allowed(self.carrier)
            and self.observer.allowed(trace.get_current_span())
        )
        if not self.allowed:
            self.observer.deny(self.span)
            self.params = _model_only(self.params)
            self.events.clear()
            self.chunks.clear()
        return self.allowed

    def tap(self, owner, name, replacement):
        original = getattr(owner, name)
        # The tap is stored on the native instance. No global retains its body.
        setattr(owner, name, replacement(original))
        self.undo.append((weakref.ref(owner), name, original, getattr(owner, name)))

    def finish(self, payload=None, error=None):
        if self.done:
            return
        self.done = True
        ambient = context.get_current()
        try:
            allowed = self.policy()
            if self.recording:
                attrs = build_bedrock_attrs(
                    operation_name=self.operation,
                    api_params=self.params,
                    response_payload=value(payload)
                    if allowed and payload is not None
                    else None,
                    stream_events=self.events if allowed and self.events else None,
                    capture_content=allowed,
                )
                if self.http_code is not None:
                    attrs[HTTP_RESPONSE_STATUS_CODE] = self.http_code
                if error is not None:
                    error_class = type(error)
                    attrs[ERROR_TYPE] = type.__getattribute__(error_class, "__name__")
                    native_error = any(
                        error_class is installed for installed in self.error_types
                    )
                    native_args = BaseException.args.__get__(error)
                    message = (
                        text(native_args[0])
                        if allowed
                        and native_error
                        and type(native_args) is tuple
                        and len(native_args) == 1
                        and type(native_args[0]) is str
                        else None
                    )
                    if message:
                        attrs[ERROR_MESSAGE_ATTR] = message
                    self.span.set_status(Status(StatusCode.ERROR, message))
                self.span.set_attributes(attrs)
                # A processor/attribute callback can tighten policy while attrs
                # are written. Veto immediately before ending and exporting.
                if not self.policy():
                    for key in tuple(self.span.attributes or {}):
                        if key.startswith(
                            (
                                f"{SpanAttributes.LLM_PROMPTS}.",
                                f"{SpanAttributes.LLM_COMPLETIONS}.",
                            )
                        ) or key in (
                            SpanAttributes.TRACELOOP_ENTITY_INPUT,
                            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                            SpanAttributes.LLM_REQUEST_FUNCTIONS,
                            ERROR_MESSAGE_ATTR,
                        ):
                            self.span._attributes.pop(key, None)
                    if self.span.status.status_code is StatusCode.ERROR:
                        self.span.set_status(Status(StatusCode.ERROR))
        except Exception:  # noqa: BLE001 - telemetry must not change native outcomes.
            logger.debug("Bedrock telemetry mapping failed")
        finally:
            try:
                self.span.end()
            except Exception:  # noqa: BLE001 - telemetry must not change native outcomes.
                logger.debug("Bedrock telemetry end failed")
            _restore_context(ambient)
            for ref, name, original, owned in self.undo:
                owner = ref()
                if owner is not None and getattr(owner, name, None) is owned:
                    try:
                        setattr(owner, name, original)
                    except Exception:  # noqa: BLE001 - telemetry must not change native outcomes.
                        logger.debug("Bedrock restoration failed")
            self.undo.clear()
            self.params = None
            self.ctx = None
            self.carrier = None
            self.events.clear()
            self.chunks.clear()
            _PENDING.discard(self)

    def body(self, body):
        def read_tap(original):
            @functools.wraps(original)
            def read(*args, **kwargs):
                try:
                    self.native_reads += 1
                    try:
                        result = original(*args, **kwargs)
                    finally:
                        self.native_reads -= 1
                except BaseException as error:
                    self.finish(error=error)
                    raise
                try:
                    if self.policy() and type(result) is bytes:
                        self.chunks.append(result)
                    amt = args[0] if args else kwargs.get("amt")
                    if amt is None or (amt != 0 and not result):
                        self.finish(payload=_decode(b"".join(self.chunks)))
                except Exception:  # noqa: BLE001 - telemetry must not change native outcomes.
                    self.finish()
                return result

            return read

        self.tap(body, "read", read_tap)
        if hasattr(body, "readinto"):

            def readinto_tap(original):
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
                    try:
                        if (
                            self.policy()
                            and count
                            and type(buffer) in (bytearray, memoryview)
                        ):
                            self.chunks.append(bytes(buffer[:count]))
                        if count == 0 and len(buffer):
                            self.finish(payload=_decode(b"".join(self.chunks)))
                    except Exception:  # noqa: BLE001 - telemetry must not change native outcomes.
                        self.finish()
                    return count

                return readinto

            self.tap(body, "readinto", readinto_tap)
        self.close_tap(body)
        # StreamingBody.__enter__ returns its original raw HTTPResponse. Observe
        # that native path too while avoiding double capture during body.read.
        from urllib3.response import HTTPResponse

        raw = body._raw_stream
        if type(raw) is HTTPResponse:

            def raw_read_tap(original):
                def read(*args, **kwargs):
                    try:
                        result = original(*args, **kwargs)
                    except BaseException as error:
                        self.finish(error=error)
                        raise
                    if not self.native_reads:
                        try:
                            if self.policy() and type(result) is bytes:
                                self.chunks.append(result)
                            amt = args[0] if args else kwargs.get("amt")
                            if amt is None or (amt != 0 and not result):
                                self.finish(payload=_decode(b"".join(self.chunks)))
                        except Exception:  # noqa: BLE001 - telemetry must not change native outcomes.
                            self.finish()
                    return result

                return read

            self.tap(raw, "read", raw_read_tap)
            self.close_tap(raw)

    def event_stream(self, stream):
        def parse_tap(original):
            def parse(event):
                try:
                    result = original(event)
                except BaseException as error:
                    self.finish(error=error)
                    raise
                try:
                    if self.policy() and type(result) is dict:
                        self.events.append(value(result))
                except Exception:  # noqa: BLE001 - telemetry must not change native outcomes.
                    self.finish()
                return result

            return parse

        self.tap(stream, "_parse_event", parse_tap)
        native_generator = stream._event_generator

        def messages():
            try:
                yield from native_generator
            except BaseException as error:
                self.finish(error=error)
                raise
            finally:
                self.finish()

        stream._event_generator = messages()
        self.close_tap(stream)

    def close_tap(self, body):
        def close_tap(original):
            def close(*args, **kwargs):
                try:
                    return original(*args, **kwargs)
                except BaseException as error:
                    self.finish(error=error)
                    raise
                finally:
                    self.finish()

            return close

        self.tap(body, "close", close_tap)


def _decode(data):
    try:
        return json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return None


def _model_only(params):
    if type(params) is dict and type(params.get("modelId")) is str:
        return {"modelId": text(params["modelId"])}
    return {}


def _wrap(original):
    @functools.wraps(original)
    def call(client, operation_name, api_params):
        if (
            _CONFIG is None
            or operation_name not in SUPPORTED_OPERATIONS
            or client.meta.service_model.service_name != BEDROCK_RUNTIME_SERVICE_NAME
            or suppressed()
        ):
            return original(client, operation_name, api_params)
        state = None
        token = None
        ambient = context.get_current()
        try:
            state = _Call(operation_name, api_params, _CONFIG, client)
            token = context.attach(trace.set_span_in_context(state.span))
        except Exception:  # noqa: BLE001 - telemetry must not change native outcomes.
            if state is not None:
                state.finish()
            state = None
            _restore_context(ambient)
        try:
            response = original(client, operation_name, api_params)
        except BaseException as error:
            if state is not None:
                error_fields = BaseException.__dict__["__dict__"].__get__(error)
                state.http_code = _http_code(
                    error_fields.get("response") if type(error_fields) is dict else None
                )
                state.finish(error=error)
            raise
        else:
            if state is not None:
                try:
                    from botocore.eventstream import EventStream
                    from botocore.response import StreamingBody

                    state.http_code = _http_code(response)
                    body = (
                        response.get("stream", response.get("body"))
                        if type(response) is dict
                        else None
                    )
                    if type(body) is EventStream:
                        state.event_stream(body)
                    elif type(body) is StreamingBody:
                        state.body(body)
                    else:
                        state.finish(
                            payload=response if operation_name == "Converse" else None
                        )
                except Exception:  # noqa: BLE001 - telemetry must not change native outcomes.
                    state.finish()
            return response
        finally:
            if token is not None:
                try:
                    if state is not None and not state.done:
                        state.policy()
                except Exception:  # noqa: BLE001 - telemetry must not change native outcomes.
                    logger.debug("Bedrock policy observation failed")
                try:
                    _detach(token, ambient)
                except Exception:  # noqa: BLE001 - preserve the native result under all runtime faults.
                    logger.debug("Bedrock context restoration failed")

    return call


def _restore_context(ambient):
    try:
        if context.get_current() is ambient:
            return
        runtime = context._RUNTIME_CONTEXT
        current = getattr(runtime, "_current_context", None)
        if current is not None:
            current.set(ambient)
        else:
            runtime.attach(ambient)
    except Exception:  # noqa: BLE001 - never replace a native outcome with telemetry cleanup faults.
        logger.debug("Bedrock context restoration failed")


def _detach(token, ambient):
    try:
        context.detach(token)
    except Exception:  # noqa: BLE001 - telemetry context faults cannot leak into native calls.
        logger.debug("Bedrock context detach failed")
    if context.get_current() is ambient:
        return
    try:
        context._RUNTIME_CONTEXT.detach(token)
    except Exception:  # noqa: BLE001 - restore the exact prior native ContextVar value.
        logger.debug("Bedrock runtime detach failed")
    _restore_context(ambient)


class AWSBedrockInstrumentor:
    """Instrument four synchronous boto3 Bedrock inference operations."""

    name = AWS_BEDROCK_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content=True, tracer_provider=None):
        self.config = (capture_content, tracer_provider)
        self._is_instrumented = False

    def activate(self):
        global _PATCH, _CONFIG
        with _LOCK:
            if self in _OWNERS:
                return
            if _OWNERS and self.config != _CONFIG:
                raise ValueError(
                    "Bedrock is active with a different content/provider configuration"
                )
            if not _OWNERS:
                try:
                    owner = _load_base_client_class()
                    original = owner._make_api_call
                    _observer(_provider(self.config))
                    replacement = _wrap(original)
                    owner._make_api_call = replacement
                    _PATCH = (owner, original, replacement)
                    _CONFIG = self.config
                except ImportError:
                    return
                except Exception:  # noqa: BLE001 - telemetry must not change native outcomes.
                    _cleanup()
                    logger.debug("Bedrock activation failed")
                    return
            _OWNERS.add(self)
            self._is_instrumented = True

    def deactivate(self):
        with _LOCK:
            _OWNERS.discard(self)
            self._is_instrumented = False
            if not _OWNERS:
                _cleanup()


def _cleanup():
    global _PATCH, _CONFIG
    for state in tuple(_PENDING):
        state.finish()
    if _PATCH:
        owner, original, replacement = _PATCH
        if owner._make_api_call is replacement:
            owner._make_api_call = original
    _PATCH = None
    _CONFIG = None
    for provider, observer in _OBSERVERS:
        processor = getattr(provider, "_active_span_processor", None)
        processors = getattr(processor, "_span_processors", ())
        if observer in processors:
            processor._span_processors = tuple(
                item for item in processors if item is not observer
            )
    _OBSERVERS.clear()
