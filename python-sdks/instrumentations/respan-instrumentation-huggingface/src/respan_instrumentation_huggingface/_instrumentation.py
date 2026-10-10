"""Keep the upstream Transformers delegate; repair native payload observation."""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
import threading
import types
import weakref
from contextvars import ContextVar
from functools import wraps
from typing import Any

from opentelemetry import context, trace
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import SpanAttributes as AI
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA
from respan_tracing.core.tracer import RespanTracer

from ._constants import TRANSFORMERS_SCOPE_NAME, TRANSFORMERS_TEXT_GENERATION_SPAN_NAME
from ._policy import CapturePolicy, PrivacyObserver, explicit_capture
from ._serialization import (
    json_string,
    json_value,
    native_dict,
    native_storage,
    redact_text,
)

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_CURRENT: ContextVar[Any] = ContextVar("respan_transformers_native_scope", default=None)
_OWNERS: set[Any] = set()
_MANAGER = None


class HuggingFaceSpanContractProcessor(SpanProcessor):
    """Canonicalize only native upstream Transformers text-generation spans."""

    def on_end(self, span):
        if (
            span.name != TRANSFORMERS_TEXT_GENERATION_SPAN_NAME
            or getattr(getattr(span, "instrumentation_scope", None), "name", None)
            != TRANSFORMERS_SCOPE_NAME
        ):
            return
        try:
            attrs = dict(span.attributes or {})
            attrs[AI.LLM_SYSTEM] = "huggingface"
            attrs.setdefault(AI.LLM_REQUEST_TYPE, "completion")
            attrs.setdefault(RESPAN_LOG_TYPE, "text")
            # Standalone processor mode maps upstream indexed data. The owned
            # delegate observer already supplies complete native payloads.
            for prefix, canonical in (
                (AI.LLM_PROMPTS, AI.TRACELOOP_ENTITY_INPUT),
                (AI.LLM_COMPLETIONS, AI.TRACELOOP_ENTITY_OUTPUT),
            ):
                if canonical not in attrs:
                    contents = [
                        value
                        for key, value in sorted(attrs.items())
                        if key.startswith(prefix + ".") and key.endswith(".content")
                    ]
                    if contents:
                        attrs[canonical] = json_string(contents)
            span._attributes = attrs
        except Exception:  # noqa: BLE001
            logger.debug("Transformers contract mapping failed")


def _register(provider, processor):
    active = getattr(provider, "_active_span_processor", None)
    if active is None:
        provider.add_span_processor(processor)
    else:
        with active._lock:
            active._span_processors = (
                processor,
                *(p for p in active._span_processors if p is not processor),
            )


def _remove(provider, processor):
    active = getattr(provider, "_active_span_processor", None)
    if active is not None:
        with active._lock:
            active._span_processors = tuple(
                p for p in active._span_processors if p is not processor
            )


def _same_options(left, right):
    if left.keys() != right.keys():
        return False
    return all(
        left[key] is right[key]
        or (
            type(left[key]) is type(right[key])
            and type(left[key]) in (str, bool, int, float, type(None))
            and left[key] == right[key]
        )
        for key in left
    )


class _State:
    def __init__(self, manager, tracer, name, options):
        self.manager = manager
        self.done = False
        self.finishing = False
        self.lazy = False
        self.exhausted = False
        self.error = False
        self.span = None
        self.policy = None
        self.extra_policy = None
        self.input = None
        self.output = None
        self.kwargs = {}
        self.observed = []
        self.outputs = []
        self.keys = set()
        self.model = None
        self.chat = False
        self.defaults = {}
        self.marker = None
        self.was_recording = False
        supplied = options.get("context", manager.context)
        self.policy = CapturePolicy(manager.observer, manager.capture, supplied)
        if manager.context is not None and supplied is not manager.context:
            self.extra_policy = CapturePolicy(
                manager.observer, manager.capture, manager.context
            )
        start_options = dict(options)
        start_options.pop("end_on_exit", None)
        start_options.setdefault("context", self.policy.context)
        self.span = tracer.start_span(name, **start_options)
        self.was_recording = self.span.is_recording()
        self.marker = (getattr(self.span, "attributes", None) or {}).get(
            RESPAN_METADATA
        )
        self.manager.observer.states.add(self)

    def safe(self, method, *args):
        if self.done:
            return None
        try:
            return getattr(self, method)(*args)
        except Exception:  # noqa: BLE001
            self.abort()
            return None

    def drop_body(self):
        self.input = None
        self.output = None
        self.kwargs = {}
        self.observed = []
        self.outputs = []
        self.defaults = {}

    def remove_body_attributes(self):
        attrs = getattr(self.span, "_attributes", None)
        if attrs is not None:
            for key in self.keys:
                attrs.pop(key, None)
        if (
            getattr(getattr(self.span, "status", None), "status_code", None)
            == StatusCode.ERROR
        ):
            self.span._status = Status(StatusCode.ERROR)

    def clean(self):
        try:
            self.scrub()
        except Exception:  # noqa: BLE001
            self.drop_body()
            try:
                self.remove_body_attributes()
            except Exception:  # noqa: BLE001
                logger.debug("Transformers observer attribute discard failed")

    def release(self):
        self.drop_body()
        self.keys.clear()
        for policy in (self.policy, self.extra_policy):
            if policy is not None:
                policy.allowed = False
                policy.context = None
                policy.supplied = None
                policy.parent = trace.INVALID_SPAN
                policy.supplied_parent = None

    def abort(self):
        if self.done:
            return
        if self.policy is not None:
            self.policy.allowed = False
        self.clean()
        self.finishing = True
        try:
            if self.span is not None:
                self.span.end()
        except Exception:  # noqa: BLE001
            logger.debug("Transformers observer span discard failed")
        finally:
            self.done = True
            self.finishing = False
            self.manager.observer.states.discard(self)
            self.release()

    def check(self):
        allowed = self.policy.check() and (
            self.span.is_recording() or (self.finishing and self.was_recording)
        )
        if self.extra_policy is not None:
            allowed = allowed and self.extra_policy.check()
        if not allowed:
            self.scrub()
        return allowed

    def scrub(self):
        self.drop_body()
        self.remove_body_attributes()

    def capture(self, key, value):
        if self.check():
            self.keys.add(key)
            try:
                self.span.set_attribute(key, value)
            finally:
                self.keys.add(key)
            self.check()

    def start(self, instance, args, kwargs):
        # Read only native owned storage, never arbitrary conversion/property hooks.
        storage = native_storage(instance, self.manager.pipeline_base) or {}
        model = storage.get("model")
        for module_name, base_name in (
            ("transformers.modeling_utils", "PreTrainedModel"),
            ("transformers.modeling_tf_utils", "TFPreTrainedModel"),
        ):
            native_module = sys.modules.get(module_name)
            module_data = (
                native_storage(native_module, types.ModuleType)
                if native_module is not None
                else None
            )
            base = module_data.get(base_name) if module_data else None
            if base is not None:
                model_data = native_storage(model, base)
                if model_data is not None:
                    config = model_data.get("config")
                    config_data = native_storage(config, self.manager.config_base)
                    if config_data is not None:
                        self.model = config_data.get("_name_or_path")
                    break
        if type(self.model) is str and self.model:
            self.model = redact_text(self.model)
            self.span.set_attribute(AI.LLM_REQUEST_MODEL, self.model)
        if not self.check():
            return
        prompt = args[0] if args else kwargs.get("text_inputs")
        if type(prompt) in (str, list, tuple, dict) or native_dict(prompt) is not None:
            self.input = json_value(prompt)
        self.chat = (
            type(self.input) is list
            and bool(self.input)
            and all(type(x) is dict and "role" in x for x in self.input)
        )
        self.kwargs = json_value(kwargs)
        forward = storage.get("_forward_params")
        self.defaults = json_value(forward) if type(forward) is dict else {}

    def preprocess(self, prompt):
        if self.check():
            if type(prompt).__module__.startswith("transformers.") and type(
                prompt
            ).__name__ in ("Chat", "_Chat"):
                data = native_dict(prompt)
                prompt = data.get("messages") if data is not None else None
            self.observed.append(json_value(prompt))

    def result(self, response):
        if (
            self.manager.iterator_type is not None
            and type(response) is self.manager.iterator_type
        ):
            self.lazy = True
            self.manager.iterators[response] = self
            weakref.finalize(response, self.safe, "finish")
        else:
            if (
                self.span.is_recording()
                and type(response) in (list, tuple, dict, str)
                and len(response)
            ):
                self.span.set_status(Status(StatusCode.OK))
            if self.check():
                self.output = json_value(response)

    def consume(self, result):
        if self.check():
            self.outputs.append(json_value(result))

    def failure(self, error):
        self.error = True
        self.span._status = Status(StatusCode.ERROR)
        self.span.set_attribute(
            ERROR_TYPE, type.__getattribute__(type(error), "__name__")
        )
        if self.check():
            for argument in BaseException.args.__get__(error):
                if type(argument) is str:
                    self.capture(ERROR_MESSAGE, redact_text(argument))
                    break

    def finish(self):
        if self.done or self.finishing:
            return
        self.finishing = True
        try:
            if (
                self.lazy
                and self.exhausted
                and not self.error
                and self.span.is_recording()
            ):
                self.span.set_status(Status(StatusCode.OK))
            if self.check():
                input_value = (
                    self.input
                    if self.input is not None
                    else self.observed
                    if self.observed
                    else None
                )
                output_value = (
                    (self.outputs if self.outputs or self.exhausted else None)
                    if self.lazy
                    else self.output
                )
                # Convenience projections precede complete canonical bodies.
                prompts = input_value if type(input_value) is list else [input_value]
                for index, prompt in enumerate(prompts):
                    if prompt is not None:
                        role = prompt.get("role") if type(prompt) is dict else "user"
                        if type(role) is str:
                            self.capture(f"{AI.LLM_PROMPTS}.{index}.role", role)
                        self.capture(
                            f"{AI.LLM_PROMPTS}.{index}.content",
                            prompt["content"]
                            if type(prompt) is dict
                            and type(prompt.get("content")) is str
                            else prompt
                            if type(prompt) is str
                            else json_string(prompt.get("content"))
                            if type(prompt) is dict and "role" in prompt
                            else json_string(prompt),
                        )
                projected = []

                def collect(value):
                    if type(value) is list:
                        for item in value:
                            collect(item)
                    elif type(value) is dict and "generated_text" in value:
                        text = value["generated_text"]
                        if type(text) is list and text and type(text[-1]) is dict:
                            text = text[-1].get("content")
                        if text is not None:
                            projected.append(text)

                collect(output_value)
                for index, text in enumerate(projected):
                    self.capture(
                        f"{AI.LLM_COMPLETIONS}.{index}.content",
                        text if type(text) is str else json_string(text),
                    )
                options = {}
                for source_options in (self.defaults or {}, self.kwargs):
                    config_options = source_options.get("generation_config")
                    if type(config_options) is dict:
                        options.update(config_options)
                    options.update(source_options)
                for source, key in [
                    ("max_new_tokens", AI.LLM_REQUEST_MAX_TOKENS),
                    ("temperature", AI.LLM_REQUEST_TEMPERATURE),
                    ("top_p", AI.LLM_REQUEST_TOP_P),
                    ("repetition_penalty", AI.LLM_REQUEST_REPETITION_PENALTY),
                ]:
                    if type(options.get(source)) in (int, float):
                        self.span.set_attribute(key, options[source])
                if input_value is not None:
                    self.capture(AI.TRACELOOP_ENTITY_INPUT, json_string(input_value))
                if output_value is not None:
                    self.capture(AI.TRACELOOP_ENTITY_OUTPUT, json_string(output_value))
                self.capture(
                    RESPAN_METADATA + ".huggingface.request",
                    json_string(
                        {"pipeline_forward": self.defaults, "call": self.kwargs}
                    ),
                )
                tokenizer_kwargs = options.get("tokenizer_encode_kwargs") or {}
                if type(tokenizer_kwargs) is dict and "tools" in tokenizer_kwargs:
                    self.capture(
                        AI.LLM_REQUEST_FUNCTIONS, json_string(tokenizer_kwargs["tools"])
                    )
                if "tools" in options:
                    self.capture(
                        AI.LLM_REQUEST_FUNCTIONS, json_string(options["tools"])
                    )
            self.span.set_attribute(AI.LLM_SYSTEM, "huggingface")
            self.span.set_attribute(
                AI.LLM_REQUEST_TYPE, "chat" if self.chat else "completion"
            )
            self.span.set_attribute(RESPAN_LOG_TYPE, "chat" if self.chat else "text")
            if type(self.model) is str and self.model:
                self.span.set_attribute(AI.LLM_REQUEST_MODEL, self.model)
            if type(self.marker) is str:
                self.span.set_attribute(RESPAN_METADATA, self.marker)
            self.check()
        except Exception:  # noqa: BLE001
            self.policy.allowed = False
            self.clean()
        finally:
            try:
                self.span.end()
            except Exception:  # noqa: BLE001
                self.policy.allowed = False
                self.clean()
                logger.debug("Transformers telemetry end failed")
            finally:
                self.done = True
                self.finishing = False
                self.manager.observer.states.discard(self)
                self.release()


class _Scope:
    def __init__(self, manager, tracer, name, options):
        self.options = options
        self.manager = manager
        self.tracer = tracer
        self.name = name
        self.state = None
        self.cm = None
        self.token = None

    def __enter__(self):
        try:
            self.manager.observe()
            tracer = self.manager.native_tracer(*self.tracer)
            self.state = _State.__new__(_State)
            self.state.__init__(self.manager, tracer, self.name, self.options)
            self.cm = trace.use_span(
                self.state.span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            )
            self.cm.__enter__()
        except Exception:  # noqa: BLE001
            if self.state:
                self.state.abort()
            self.state = None
        self.token = _CURRENT.set(self.state)
        return self.state.span if self.state else trace.INVALID_SPAN

    def __exit__(self, kind, error, tb):
        if self.state:
            if error:
                self.state.safe("failure", error)
            self.state.safe("check")
            if error or not self.state.lazy:
                self.state.safe("finish")
        try:
            if self.cm:
                self.cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            logger.debug("Transformers telemetry detach failed")
        if self.token is not None:
            _CURRENT.reset(self.token)
        return False


class _Tracer:
    def __init__(self, manager, args, kwargs):
        self.manager = manager
        self.args = (args, kwargs)

    def start_as_current_span(self, name, **kwargs):
        return _Scope(self.manager, self.args, name, kwargs)


class _Provider:
    def __init__(self, manager):
        self.manager = manager

    def get_tracer(self, *args, **kwargs):
        return _Tracer(self.manager, args, kwargs)


class _ObservedReturn:
    """Internal delegate sentinel; the caller always gets the original result."""

    def __init__(self, result):
        self.result = result

    def __bool__(self):
        return False


class _Manager:
    def __init__(self, capture, provider, ctx):
        self.capture = capture
        self.provider = provider
        self.context = ctx
        self.enabled = True
        self.observer = PrivacyObserver()
        self.contract = HuggingFaceSpanContractProcessor()
        self.providers = []
        self.iterators = weakref.WeakKeyDictionary()
        self.patches = []
        self.delegate = None

    def observe(self):
        provider = self.provider or trace.get_tracer_provider()
        if not any(provider is owned for owned in self.providers) and hasattr(
            provider, "add_span_processor"
        ):
            self.providers.append(provider)
            _register(provider, self.contract)
            _register(provider, self.observer)
        return provider

    def native_tracer(self, args, kwargs):
        return self.observe().get_tracer(*args, **kwargs)

    def patch(self, obj, name, replacement):
        namespace = (
            type.__getattribute__(obj, "__dict__")
            if isinstance(obj, type)
            else native_storage(obj, type(obj))
        )
        present = namespace is not None and name in namespace
        stored = namespace.get(name) if present else None
        original = (
            inspect.getattr_static(obj, name)
            if isinstance(obj, type)
            else getattr(obj, name)
        )
        self.patches.append((obj, name, original, replacement, present, stored))
        setattr(obj, name, replacement)

    def close(self):
        self.enabled = False
        for state in list(self.observer.states):
            state.policy.allowed = False
            state.safe("finish")
        for obj, name, _original, replacement, present, stored in reversed(
            self.patches
        ):
            if inspect.getattr_static(obj, name, None) is replacement:
                if present:
                    setattr(obj, name, stored)
                else:
                    delattr(obj, name)
        for provider in self.providers:
            _remove(provider, self.observer)
            _remove(provider, self.contract)
        self.providers.clear()
        self.iterators.clear()
        if hasattr(self, "config_owned"):
            for field, before, owned in zip(
                ("exception_logger", "use_legacy_attributes"),
                self.config_before,
                self.config_owned,
            ):
                if getattr(self.config, field) is owned:
                    setattr(self.config, field, before)


class HuggingFaceInstrumentor:
    name = "huggingface"

    def __init__(
        self,
        *,
        exception_logger=None,
        use_legacy_attributes=True,
        capture_content=True,
        tracer_provider=None,
        context=None,
        **instrumentor_kwargs,
    ):
        self.capture_content = capture_content
        self.provider = tracer_provider
        self.context = context
        self.constructor = {
            "exception_logger": exception_logger,
            "use_legacy_attributes": use_legacy_attributes,
        }
        self.kwargs = instrumentor_kwargs
        self._is_instrumented = False

    def activate(self):
        global _MANAGER
        with _LOCK:
            if self._is_instrumented:
                return
            tracer = getattr(RespanTracer, "_instance", None)
            if tracer is not None and not getattr(tracer, "is_enabled", True):
                return
            if _OWNERS:
                if (
                    _MANAGER.capture != self.capture_content
                    or self.provider is not _MANAGER.provider
                    or self.context is not _MANAGER.context
                    or not _same_options(self.constructor, _MANAGER.constructor)
                    or not _same_options(self.kwargs, _MANAGER.options)
                ):
                    raise RuntimeError(
                        "Hugging Face instrumentation configuration conflict"
                    )
                _OWNERS.add(self)
                self._is_instrumented = True
                return
            try:
                upstream = importlib.import_module(
                    "opentelemetry.instrumentation.transformers"
                )
                cls = upstream.TransformersInstrumentor
                prior = cls.__new__(cls)
                if prior.is_instrumented_by_opentelemetry:
                    logger.warning(
                        "Foreign Transformers instrumentation is already active"
                    )
                    return
                manager = _Manager(self.capture_content, self.provider, self.context)
                manager.constructor = dict(self.constructor)
                manager.options = dict(self.kwargs)
                config = importlib.import_module(
                    "opentelemetry.instrumentation.transformers.config"
                ).Config
                manager.config = config
                manager.config_before = (
                    config.exception_logger,
                    config.use_legacy_attributes,
                )
                manager.observe()
                from transformers import PretrainedConfig, TextGenerationPipeline
                from transformers.pipelines.base import Pipeline

                manager.pipeline_base = Pipeline
                manager.config_base = PretrainedConfig

                try:
                    from transformers.pipelines.pt_utils import PipelineIterator
                except ImportError:
                    PipelineIterator = None
                manager.iterator_type = PipelineIterator

                wrapper_module = importlib.import_module(
                    "opentelemetry.instrumentation.transformers.text_generation_pipeline_wrapper"
                )
                factory = upstream.text_generation_pipeline_wrapper

                def wrapped_factory(tracer, logger, to_wrap):
                    base = factory(tracer, logger, to_wrap)

                    def wrapper(wrapped, instance, args, kwargs):
                        if not manager.enabled:
                            return wrapped(*args, **kwargs)
                        holder = []

                        def observed(*native_args, **native_kwargs):
                            result = wrapped(*native_args, **native_kwargs)
                            state = _CURRENT.get()
                            if state:
                                state.safe("result", result)
                            holder.append(result)
                            return _ObservedReturn(result)

                        result = base(observed, instance, args, kwargs)
                        return holder[0] if holder else result

                    return wrapper

                def handle_input(span, logger, instance, args, kwargs):
                    state = _CURRENT.get()
                    if state:
                        state.safe("start", instance, args, kwargs)

                manager.patch(
                    upstream, "text_generation_pipeline_wrapper", wrapped_factory
                )
                manager.patch(
                    upstream,
                    "WRAPPED_METHODS",
                    [
                        {**item, "wrapper": wrapped_factory}
                        if item.get("wrapper") is factory
                        else item
                        for item in upstream.WRAPPED_METHODS
                    ],
                )
                manager.patch(wrapper_module, "_handle_input", handle_input)
                original_preprocess = TextGenerationPipeline.preprocess

                @wraps(original_preprocess)
                def preprocess(instance, prompt, *args, **kwargs):
                    state = _CURRENT.get()
                    if state:
                        state.safe("preprocess", prompt)
                    return original_preprocess(instance, prompt, *args, **kwargs)

                manager.patch(TextGenerationPipeline, "preprocess", preprocess)
                if PipelineIterator is not None:
                    original_next = PipelineIterator.__next__

                    @wraps(original_next)
                    def next_item(iterator):
                        state = manager.iterators.get(iterator)
                        if state is None:
                            return original_next(iterator)
                        token = _CURRENT.set(state)
                        cm = None
                        try:
                            cm = trace.use_span(
                                state.span,
                                end_on_exit=False,
                                record_exception=False,
                                set_status_on_exception=False,
                            )
                            cm.__enter__()
                        except Exception:  # noqa: BLE001
                            state.safe("scrub")
                            cm = None
                        try:
                            state.safe("check")
                            result = original_next(iterator)
                            state.safe("consume", result)
                            return result
                        except StopIteration:
                            state.exhausted = True
                            state.safe("finish")
                            manager.iterators.pop(iterator, None)
                            raise
                        except BaseException as error:
                            state.safe("failure", error)
                            state.safe("finish")
                            manager.iterators.pop(iterator, None)
                            raise
                        finally:
                            try:
                                if cm:
                                    cm.__exit__(None, None, None)
                            except Exception:  # noqa: BLE001
                                logger.debug("Transformers consumption detach failed")
                            _CURRENT.reset(token)

                    manager.patch(PipelineIterator, "__next__", next_item)
                runtime = context._RUNTIME_CONTEXT
                original_detach = runtime.detach

                def detach(*args, **kwargs):
                    try:
                        if manager.enabled and not explicit_capture():
                            manager.observer.notice()
                    except Exception:  # noqa: BLE001
                        logger.debug("Transformers context observer failed")
                    return original_detach(*args, **kwargs)

                manager.patch(runtime, "detach", detach)
                original = inspect.getattr_static(TextGenerationPipeline, "__call__")
                manager.original_call = (TextGenerationPipeline, original)
                manager.delegate = cls(**self.constructor)
                manager.config_owned = (
                    config.exception_logger,
                    config.use_legacy_attributes,
                )
                manager.delegate.instrument(
                    tracer_provider=_Provider(manager), **self.kwargs
                )
                manager.native_call = (
                    TextGenerationPipeline,
                    original,
                    inspect.getattr_static(TextGenerationPipeline, "__call__"),
                )
                _MANAGER = manager
                _OWNERS.add(self)
                self._is_instrumented = True
            except Exception:  # noqa: BLE001
                if "manager" in locals():
                    if hasattr(manager, "original_call"):
                        pipeline, original = manager.original_call
                        current = inspect.getattr_static(pipeline, "__call__")
                        if getattr(current, "__wrapped__", None) is original:
                            pipeline.__call__ = original
                    if manager.delegate is not None:
                        manager.delegate._is_instrumented_by_opentelemetry = False
                    manager.close()
                logger.debug("Hugging Face activation failed")

    def deactivate(self):
        global _MANAGER
        with _LOCK:
            if self not in _OWNERS:
                return
            _OWNERS.remove(self)
            self._is_instrumented = False
            if _OWNERS:
                return
            manager = _MANAGER
            upstream = importlib.import_module(
                "opentelemetry.instrumentation.transformers"
            )
            cls, _original, owned = manager.native_call
            native_unwrap = upstream.unwrap

            def unwrap(module, name):
                if inspect.getattr_static(cls, "__call__") is owned:
                    native_unwrap(module, name)

            upstream.unwrap = unwrap
            try:
                manager.delegate.uninstrument()
            finally:
                if upstream.unwrap is unwrap:
                    upstream.unwrap = native_unwrap
                manager.close()
                _MANAGER = None
