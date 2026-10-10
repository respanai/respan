"""Native sampled spans around released Replicate operations and consumption."""

from __future__ import annotations

import contextvars
import functools
import logging
import threading
import types
import weakref
from contextlib import contextmanager

from opentelemetry import context, trace
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import SpanAttributes as AI
from opentelemetry.trace import SpanKind, Status, StatusCode
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

from ._policy import Policy, key, permitted, suppressed
from ._serialization import native_value, register_native_types, safe_text
from ._translator import attributes, usage

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_MANAGER = None
_OWNERS = set()
_PATCHES = []
_CURRENT = contextvars.ContextVar("replicate_native_call", default=None)
_MISSING = object()


class _Manager:
    def __init__(self, provider, capture):
        self.provider = provider
        self.capture = capture
        self.policy = None
        self.bound_provider = None
        self.states = weakref.WeakSet()
        self.enabled = True

    def refresh(self):
        provider = self.provider or trace.get_tracer_provider()
        if provider is not self.bound_provider and hasattr(
            provider, "add_span_processor"
        ):
            if self.policy:
                self.policy.close()
            self.policy = Policy(provider, self.scrub)
            self.bound_provider = provider
        return provider

    def scrub(self):
        for state in list(self.states):
            if not state.done and not state.check():
                state.scrub()

    def close(self):
        self.enabled = False
        for state in list(self.states):
            state.safe(state.finish)
        if self.policy:
            self.policy.close()


class _Call:
    def __init__(self, manager, name, args, kwargs, instance):
        self.manager = manager
        self.done = False
        self.span = None
        self.policy = None
        self.allowed = False
        self.request = None
        self.frames = []
        self.prediction = None
        self.result = None
        self.has_result = False
        self.model = None
        self.http_status = None
        self.http_statuses = []
        self.usage_attrs = {}
        self.error_type = None
        self.error_message = None
        self.kind = "task"
        self.name = name
        self.operation = name.rsplit(".", 1)[-1]
        try:
            provider = manager.refresh()
            self.policy = manager.policy
            parent = trace.get_current_span()
            self.parent = key(parent)
            initial = (
                manager.capture
                and permitted()
                and bool(self.policy and self.policy.enroll(parent))
            )
            self.span = provider.get_tracer("replicate").start_span(
                name, kind=SpanKind.CLIENT, attributes={RESPAN_LOG_TYPE: "task"}
            )
            self.allowed = (
                initial
                and self.span.is_recording()
                and bool(self.policy and self.policy.bound(key(self.span)))
            )
            self.structural = (
                {
                    k: v
                    for k, v in (self.span.attributes or {}).items()
                    if k.startswith(RESPAN_METADATA)
                }
                if self.span.is_recording()
                else {}
            )
            manager.states.add(self)
            if self.span.is_recording():
                from replicate.prediction import Prediction

                raw = (
                    object.__getattribute__(instance, "__dict__")
                    if type(instance) is Prediction
                    else {}
                )
                ref = (
                    kwargs.get("model")
                    or kwargs.get("version")
                    or kwargs.get("deployment")
                    or (args[0] if args else None)
                    or raw.get("model")
                    or raw.get("version")
                )
                if type(ref) is str:
                    self.model = safe_text(ref)
                body = (
                    kwargs.get("input")
                    if "input" in kwargs
                    else args[1]
                    if len(args) > 1
                    else None
                )
                if (
                    self.operation in ("run", "async_run", "stream", "async_stream")
                    and type(body) is dict
                    and all(type(k) is str for k in body)
                ):
                    self.kind = (
                        "chat"
                        if "messages" in body
                        else "text"
                        if "prompt" in body
                        else "task"
                    )
                params = kwargs.get("respan_params")
                if type(params) is dict and type(params.get("model")) is str:
                    self.model = safe_text(params["model"])
                self.span.set_attribute(RESPAN_LOG_TYPE, self.kind)
                if self.allowed:
                    self.request = {
                        "args": native_value(args),
                        "kwargs": native_value(kwargs),
                    }
                if type(instance) is Prediction:
                    self.observe_prediction(instance)
        except Exception:
            self.scrub()
            if self.span:
                try:
                    self.span.end()
                except Exception:  # noqa: BLE001 - cleanup cannot replace native SDK outcomes.
                    logger.debug("Replicate telemetry cleanup failed")
            raise

    def check(self):
        if self.done:
            return self.allowed
        self.allowed = (
            self.allowed
            and permitted()
            and bool(self.policy and self.policy.bound(key(self.span)))
        )
        if not self.allowed:
            self.scrub()
        return self.allowed

    def scrub(self):
        self.request = None
        self.frames.clear()
        self.prediction = None
        self.result = None
        self.has_result = False
        self.error_message = None
        if self.span and self.span.is_recording():
            for attr in list(self.span._attributes):
                if attr in (
                    AI.TRACELOOP_ENTITY_INPUT,
                    AI.TRACELOOP_ENTITY_OUTPUT,
                    AI.LLM_REQUEST_FUNCTIONS,
                    ERROR_MESSAGE,
                ) or attr.startswith(
                    (
                        AI.LLM_PROMPTS + ".",
                        AI.LLM_COMPLETIONS + ".",
                        RESPAN_METADATA + ".replicate",
                    )
                ):
                    self.span._attributes.pop(attr, None)
            self.span._events = BoundedList(maxlen=self.span._events._dq.maxlen)
            if self.span.status.status_code == StatusCode.ERROR:
                self.span._status = Status(StatusCode.ERROR)

    def discard(self):
        # Guaranteed bodyless fallback, including when a custom observer raises.
        self.allowed = False
        self.request = None
        self.frames.clear()
        self.prediction = None
        self.result = None
        self.has_result = False
        self.error_message = None
        if self.done:
            return
        self.done = True
        try:
            if self.span and self.span.is_recording():
                from opentelemetry.attributes import BoundedAttributes

                self.span._attributes = BoundedAttributes(
                    attributes={
                        **self.structural,
                        RESPAN_LOG_TYPE: self.kind,
                        AI.TRACELOOP_ENTITY_NAME: self.name,
                    },
                    immutable=False,
                )
                if self.error_type:
                    self.span.set_attribute(ERROR_TYPE, self.error_type)
                self.span._events = BoundedList(maxlen=self.span._events._dq.maxlen)
                if self.span.status.status_code == StatusCode.ERROR:
                    self.span._status = Status(StatusCode.ERROR)
        except Exception:  # noqa: BLE001 - discard cleanup cannot replace SDK outcomes.
            logger.debug("Replicate telemetry discard cleanup failed")
        try:
            if self.span:
                self.span.end()
        except Exception:  # noqa: BLE001 - native consumer outcomes have priority.
            logger.debug("Replicate telemetry discard end failed")
        finally:
            if self.policy:
                self.policy.on_end(self.span)

    def safe(self, fn, *args, **kwargs):
        if self.done:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception:  # noqa: BLE001 - isolate all observer faults.
            logger.debug("Replicate telemetry observation failed")
            self.discard()
            return None

    def observe_prediction(self, prediction):
        if not self.span.is_recording():
            return
        from replicate.prediction import Prediction

        if type(prediction) is not Prediction:
            return
        self.usage_attrs.update(usage(prediction))
        data = object.__getattribute__(prediction, "__dict__")
        if self.model is None and type(data.get("model")) is str:
            self.model = safe_text(data["model"])
        if self.check():
            self.prediction = prediction

    def observe_result(self, result):
        if not self.span.is_recording():
            return
        self.observe_prediction(result)
        if self.check():
            self.result = native_value(result)
            self.has_result = result is not None

    def observe_frame(self, value):
        if self.check():
            self.frames.append(native_value(value))

    def error(self, error):
        if not self.span.is_recording():
            return
        self.error_type = type.__getattribute__(type(error), "__name__")
        self.span.set_attribute(ERROR_TYPE, self.error_type)
        self.span.set_status(Status(StatusCode.ERROR))
        from replicate.exceptions import ModelError, ReplicateError

        if type(error) is ModelError:
            self.observe_prediction(
                object.__getattribute__(error, "__dict__").get("prediction")
            )
        data = BaseException.__dict__["__dict__"].__get__(error)
        status = data.get("status") if type(error) is ReplicateError else None
        if type(status) is int:
            self.http_status = status
        if self.check():
            message = data.get("detail") if type(error) is ReplicateError else None
            if type(message) is not str:
                message = next(
                    (v for v in BaseException.args.__get__(error) if type(v) is str),
                    None,
                )
            if message is not None:
                message = safe_text(message)
                self.error_message = message
                self.span.set_attribute(ERROR_MESSAGE, message)
                self.span.set_status(Status(StatusCode.ERROR, message))

    def finish(self):
        if self.done:
            return
        try:
            allowed = self.check()
            if self.span.is_recording():
                if allowed:
                    mapped = attributes(
                        self.name,
                        self.request or {},
                        self.result,
                        self.prediction,
                        self.frames,
                        has_result=self.has_result,
                        model=self.model,
                        operation=self.operation,
                    )
                    self.span.set_attributes(mapped)
                    self.kind = mapped[RESPAN_LOG_TYPE]
                # Usage is structural even when content is disabled; only source numeric fields.
                if self.kind in ("chat", "text", "embedding") and self.usage_attrs:
                    self.span.set_attributes(self.usage_attrs)
                if self.model is not None and self.kind in (
                    "chat",
                    "text",
                    "embedding",
                ):
                    self.span.set_attribute(AI.LLM_REQUEST_MODEL, self.model)
                if self.http_status is not None:
                    self.span.set_attribute(HTTP_RESPONSE_STATUS_CODE, self.http_status)
                if self.error_type:
                    self.span.set_attribute(ERROR_TYPE, self.error_type)
                if allowed and self.error_message:
                    self.span.set_attribute(ERROR_MESSAGE, self.error_message)
                self.span.set_attributes(self.structural)
                self.check()
        except Exception:  # noqa: BLE001 - discard incomplete telemetry while preserving the native result.
            self.allowed = False
            self.scrub()
            logger.debug("Replicate telemetry mapping failed")
        finally:
            self.done = True
            try:
                self.span.end()
            except Exception:  # noqa: BLE001 - telemetry cleanup must preserve native outcomes.
                logger.debug("Replicate telemetry span end failed")
            finally:
                if self.policy:
                    self.policy.on_end(self.span)
                self.request = None
                self.frames.clear()
                self.prediction = None
                self.result = None


@contextmanager
def _scope(state):
    current = span_token = private_token = None
    try:
        if state and not state.done:
            state.safe(state.check)
            current = _CURRENT.set(state)
            try:
                if not state.allowed:
                    private_token = context.attach(
                        context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
                    )
                span_token = context.attach(trace.set_span_in_context(state.span))
            except Exception:  # noqa: BLE001 - telemetry cleanup must preserve native outcomes.
                state.allowed = False
                state.scrub()
                logger.debug("Replicate telemetry context attach failed")
        yield
    finally:
        if state and not state.done:
            state.safe(state.check)
        for token in (span_token, private_token):
            if token is not None:
                try:
                    context.detach(token)
                except Exception:  # noqa: BLE001 - telemetry cleanup must preserve native outcomes.
                    if state and state.policy:
                        try:
                            state.policy.original_detach(token)
                        except Exception:  # noqa: BLE001, S110 - original detach fallback cannot replace the native outcome.
                            pass
                    logger.debug("Replicate telemetry context cleanup failed")
        if current is not None:
            _CURRENT.reset(current)


class _Iterator:
    def __init__(self, source, state):
        self.source = source
        self.state = state

    def __del__(self):
        try:
            self.state.discard()
        except BaseException:  # noqa: BLE001 - telemetry cleanup must preserve native outcomes.
            logger.debug("Replicate abandoned iterator cleanup failed")

    def __iter__(self):
        return self

    def __getattr__(self, name):
        return getattr(self.source, name)

    def _pull(self, fn, *args):
        try:
            with _scope(self.state):
                value = fn(*args)
        except StopIteration:
            self.state.safe(self.state.finish)
            raise
        except BaseException as error:
            self.state.safe(self.state.error, error)
            self.state.safe(self.state.finish)
            raise
        self.state.safe(self.state.observe_frame, value)
        return value

    def __next__(self):
        return self._pull(next, self.source)

    def send(self, value):
        return self._pull(self.source.send, value)

    def throw(self, *args):
        return self._pull(self.source.throw, *args)

    def close(self):
        try:
            with _scope(self.state):
                return self.source.close()
        except BaseException as error:
            self.state.safe(self.state.error, error)
            raise
        finally:
            self.state.safe(self.state.finish)


class _AsyncIterator:
    def __init__(self, source, state):
        self.source = source
        self.state = state

    def __del__(self):
        try:
            self.state.discard()
        except BaseException:  # noqa: BLE001 - telemetry cleanup must preserve native outcomes.
            logger.debug("Replicate abandoned async iterator cleanup failed")

    def __aiter__(self):
        return self

    def __getattr__(self, name):
        return getattr(self.source, name)

    async def _pull(self, fn, *args):
        try:
            with _scope(self.state):
                value = await fn(*args)
        except StopAsyncIteration:
            self.state.safe(self.state.finish)
            raise
        except BaseException as error:
            self.state.safe(self.state.error, error)
            self.state.safe(self.state.finish)
            raise
        self.state.safe(self.state.observe_frame, value)
        return value

    async def __anext__(self):
        return await self._pull(anext, self.source)

    async def asend(self, value):
        return await self._pull(self.source.asend, value)

    async def athrow(self, *args):
        return await self._pull(self.source.athrow, *args)

    async def aclose(self):
        try:
            with _scope(self.state):
                return await self.source.aclose()
        except BaseException as error:
            self.state.safe(self.state.error, error)
            raise
        finally:
            self.state.safe(self.state.finish)


def _finish_result(state, result):
    if not state:
        return result
    if type(result) is types.GeneratorType:
        return _Iterator(result, state)
    if type(result) is types.AsyncGeneratorType:
        return _AsyncIterator(result, state)
    state.safe(state.observe_result, result)
    state.safe(state.finish)
    return result


def _new(manager, name, args, kwargs, instance):
    try:
        return _Call(manager, name, args, kwargs, instance)
    except Exception:  # noqa: BLE001 - telemetry cleanup must preserve native outcomes.
        logger.debug("Replicate telemetry startup failed")
        return None


def _wrap(original, name, *, method=True, coroutine=False):
    if coroutine:

        @functools.wraps(original)
        async def wrapper(*args, **kwargs):
            manager = _MANAGER
            active = _CURRENT.get()
            call_kwargs = dict(kwargs)
            if manager and manager.enabled:
                call_kwargs.pop("respan_params", None)
            if not manager or not manager.enabled or suppressed():
                return await original(*args, **call_kwargs)
            if active:
                result = await original(*args, **call_kwargs)
                active.safe(active.observe_prediction, result)
                return result
            values = args[1:] if method else args
            state = _new(
                manager, name, values, kwargs, args[0] if method and args else None
            )
            try:
                with _scope(state):
                    result = await original(*args, **call_kwargs)
            except BaseException as error:
                if state:
                    state.safe(state.error, error)
                    state.safe(state.finish)
                raise
            return _finish_result(state, result)
    else:

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            manager = _MANAGER
            active = _CURRENT.get()
            call_kwargs = dict(kwargs)
            if manager and manager.enabled:
                call_kwargs.pop("respan_params", None)
            if not manager or not manager.enabled or suppressed():
                return original(*args, **call_kwargs)
            if active:
                result = original(*args, **call_kwargs)
                active.safe(active.observe_prediction, result)
                return result
            values = args[1:] if method else args
            state = _new(
                manager, name, values, kwargs, args[0] if method and args else None
            )
            try:
                with _scope(state):
                    result = original(*args, **call_kwargs)
            except BaseException as error:
                if state:
                    state.safe(state.error, error)
                    state.safe(state.finish)
                raise
            return _finish_result(state, result)

    return wrapper


def _http(original, coroutine=False):
    if coroutine:

        @functools.wraps(original)
        async def wrapped(*args, **kwargs):
            result = await original(*args, **kwargs)
            state = _CURRENT.get()
            if state:
                state.safe(_http_result, state, result)
            return result
    else:

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            state = _CURRENT.get()
            if state:
                state.safe(_http_result, state, result)
            return result

    return wrapped


def _http_result(state, response):
    if not state.span.is_recording():
        return
    import httpx

    if type(response) is httpx.Response:
        state.http_status = response.status_code
        state.http_statuses.append(response.status_code)


def _patch(owner, name, replacement):
    original = getattr(owner, name)
    present = name in vars(owner)
    stored = vars(owner).get(name)
    setattr(owner, name, replacement)
    _PATCHES.append((owner, name, original, replacement, present, stored))


def _restore():
    for owner, name, original, replacement, present, stored in reversed(_PATCHES):
        if getattr(owner, name, None) is replacement:
            if present:
                setattr(owner, name, stored)
            else:
                delattr(owner, name)
    _PATCHES.clear()


class ReplicateInstrumentor:
    name = "replicate"

    def __init__(self, *, tracer_provider=None, capture_content=True):
        self.provider = tracer_provider
        self.capture = capture_content
        self._is_instrumented = False

    def activate(self):
        global _MANAGER
        with _LOCK:
            if self._is_instrumented:
                return
            if _MANAGER and (
                _MANAGER.provider is not self.provider
                or _MANAGER.capture != self.capture
            ):
                raise ValueError(
                    "Incompatible shared Replicate instrumentation configuration"
                )
            if not _MANAGER:
                try:
                    import replicate
                    from replicate.deployment import DeploymentsPredictions
                    from replicate.model import ModelsPredictions
                    from replicate.prediction import Prediction, Predictions

                    register_native_types()
                    manager = _Manager(self.provider, self.capture)
                    manager.refresh()
                    for owner, namespace, names in [
                        (
                            replicate.Client,
                            "",
                            ("run", "async_run", "stream", "async_stream"),
                        ),
                        (
                            Predictions,
                            "predictions",
                            (
                                "create",
                                "async_create",
                                "get",
                                "async_get",
                                "list",
                                "async_list",
                                "cancel",
                                "async_cancel",
                            ),
                        ),
                        (
                            ModelsPredictions,
                            "models.predictions",
                            ("create", "async_create"),
                        ),
                        (
                            DeploymentsPredictions,
                            "deployments.predictions",
                            ("create", "async_create"),
                        ),
                        (
                            Prediction,
                            "prediction",
                            (
                                "wait",
                                "async_wait",
                                "reload",
                                "async_reload",
                                "cancel",
                                "async_cancel",
                                "stream",
                                "async_stream",
                            ),
                        ),
                    ]:
                        for name in names:
                            original = getattr(owner, name, None)
                            if original is not None:
                                _patch(
                                    owner,
                                    name,
                                    _wrap(
                                        original,
                                        "replicate."
                                        + (namespace + "." if namespace else "")
                                        + name,
                                        coroutine=name.startswith("async_")
                                        and name not in ("async_stream",)
                                        or name == "async_stream"
                                        and owner is replicate.Client,
                                    ),
                                )
                    for name in ("run", "async_run", "stream", "async_stream"):
                        _patch(
                            replicate,
                            name,
                            _wrap(
                                getattr(replicate, name),
                                "replicate." + name,
                                method=False,
                                coroutine=name.startswith("async_"),
                            ),
                        )
                    from replicate.stream import EventSource

                    original_init = EventSource.__init__

                    @functools.wraps(original_init)
                    def event_init(instance, client, response, **kwargs):
                        original_init(instance, client, response, **kwargs)
                        state = _CURRENT.get()
                        if state:
                            state.safe(_http_result, state, response)

                    _patch(EventSource, "__init__", event_init)
                    for name in ("_request", "_async_request"):
                        _patch(
                            replicate.Client,
                            name,
                            _http(
                                getattr(replicate.Client, name),
                                name.startswith("_async"),
                            ),
                        )
                    _MANAGER = manager
                except BaseException:
                    _restore()
                    if "manager" in locals():
                        manager.close()
                    raise
            _OWNERS.add(self)
            self._is_instrumented = True

    def deactivate(self):
        global _MANAGER
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _OWNERS.discard(self)
            if not _OWNERS:
                manager = _MANAGER
                _MANAGER = None
                if manager:
                    manager.close()
                _restore()
