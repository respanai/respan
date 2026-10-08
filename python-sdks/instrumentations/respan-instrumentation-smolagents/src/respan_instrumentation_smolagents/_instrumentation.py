"""Native smolagents spans with detached iterator context and shared ownership."""

from __future__ import annotations

import inspect
import json
import logging
import os
import threading
from collections.abc import Generator, Iterator, Mapping
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

from opentelemetry import context, trace
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv.attributes.exception_attributes import (
    EXCEPTION_MESSAGE,
    EXCEPTION_TYPE,
)
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_smolagents._serialization import json_string, redact_text

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_RUNTIME: _Runtime | None = None
_CONTENT_BOUND = "smolagents.capture_content_bound"
_CURRENT_MODEL: ContextVar[Any] = ContextVar("smolagents.current_model", default=None)
_CURRENT_TOOL: ContextVar[Any] = ContextVar("smolagents.current_tool", default=None)
_CURRENT_STEP: ContextVar[Any] = ContextVar("smolagents.current_step", default=None)


def _content_allowed(options: dict | None = None) -> bool:
    options = options or {}
    config = options.get("config")
    if options.get("trace_content") is False or any(
        getattr(config, field, False)
        for field in (
            "hide_inputs",
            "hide_outputs",
            "hide_input_messages",
            "hide_output_messages",
            "hide_input_text",
            "hide_output_text",
            "hide_input_images",
            "hide_llm_tools",
            "hide_llm_invocation_parameters",
        )
    ):
        return False
    return (
        context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
        and context.get_value(_CONTENT_BOUND) is not False
        and os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower()
        not in {"0", "false", "no", "off"}
    )


def _get(value: Any, key: str, default: Any = None) -> Any:
    try:
        return (
            value.get(key, default)
            if isinstance(value, Mapping)
            else getattr(value, key, default)
        )
    except Exception:  # noqa: BLE001 - telemetry getters must fail open
        return default


def _arguments(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> dict:
    try:
        bound = inspect.signature(wrapped).bind(instance, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.update(arguments.pop("kwargs", {}))
        return arguments
    except (TypeError, ValueError):
        return dict(kwargs)


def _role(value: Any) -> str:
    value = getattr(value, "value", value)
    return {"tool-call": "assistant", "tool-response": "tool"}.get(
        value, value or "assistant"
    )


def _tool_calls(calls: Any) -> list[dict]:
    return [
        {
            "id": _get(call, "id"),
            "type": _get(call, "type", "function"),
            "function": {
                "name": _get(_get(call, "function"), "name"),
                "arguments": (
                    _get(_get(call, "function"), "arguments")
                    if isinstance(_get(_get(call, "function"), "arguments"), str)
                    else json_string(_get(_get(call, "function"), "arguments"))
                ),
            },
        }
        for call in calls or ()
    ]


def _messages(attributes: dict, prefix: str, messages: Any) -> None:
    for index, message in enumerate(messages or ()):
        root = f"{prefix}.{index}"
        attributes[f"{root}.role"] = _role(_get(message, "role"))
        content = _get(message, "content")
        if content is not None:
            attributes[f"{root}.content"] = (
                redact_text(content)
                if isinstance(content, str)
                else json_string(content)
            )
        calls = _tool_calls(_get(message, "tool_calls"))
        if calls:
            attributes[f"{root}.tool_calls"] = json_string(calls)
        call_id = _get(message, "tool_call_id")
        if call_id:
            attributes[f"{root}.tool_call_id"] = call_id


def _usage(attributes: dict, usage: Any) -> None:
    if usage is None:
        return
    values = {name: _get(usage, name) for name in ("input_tokens", "output_tokens")}
    valid = {
        name: isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for name, value in values.items()
    }
    for name, keys in (
        (
            "input_tokens",
            (
                gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS,
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            ),
        ),
        (
            "output_tokens",
            (
                gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            ),
        ),
    ):
        if valid[name]:
            for key in keys:
                attributes[key] = values[name]
    # SDK TokenUsage.total_tokens is computed from both counts, not an independent
    # provider total. A positive computed total cannot validate an invalid count.
    if all(valid.values()):
        total = _get(usage, "total_tokens")
        if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
            attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total


def _model_metadata(attributes: dict, message: Any) -> None:
    _usage(attributes, _get(message, "token_usage"))
    raw_usage = _get(_get(message, "raw"), "usage")
    if raw_usage is not None:
        for details, field_name, keys in (
            (
                "prompt_tokens_details",
                "cached_tokens",
                (
                    SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                    SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
                ),
            ),
            (
                "completion_tokens_details",
                "reasoning_tokens",
                (
                    SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
                    SpanAttributes.LLM_USAGE_REASONING_TOKENS,
                ),
            ),
        ):
            value = _get(_get(raw_usage, details), field_name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                for key in keys:
                    attributes[key] = value


def _safe(operation: Any) -> None:
    try:
        operation()
    except Exception:
        logger.debug("smolagents telemetry failed open", exc_info=True)


@dataclass
class _Step:
    value: Any
    used_ids: set[str] = field(default_factory=set)
    lock: Any = field(default_factory=threading.Lock)

    def call_id(self, name: str, arguments: Any) -> str | None:
        message = getattr(self.value, "model_output_message", None)
        with self.lock:
            candidates = []
            for call in getattr(message, "tool_calls", None) or ():
                function = _get(call, "function")
                expected = _get(function, "arguments")
                if isinstance(expected, str):
                    try:
                        expected = json.loads(expected)
                    except ValueError:
                        pass
                call_id = _get(call, "id")
                if _get(function, "name") == name and expected == arguments and call_id:
                    candidates.append(call_id)
            # The SDK passes only name/arguments to Tool.__call__, not the
            # source call ID. Scheduler order cannot identify identical calls.
            if len(candidates) == 1 and candidates[0] not in self.used_ids:
                self.used_ids.add(candidates[0])
                return candidates[0]

        return None


class _Call:
    def __init__(
        self,
        runtime: _Runtime,
        kind: str,
        instance: Any,
        arguments: dict,
        streaming: bool = False,
        operation: str = "",
    ):
        self.kind, self.instance, self.arguments = kind, instance, arguments
        self.options = runtime.options
        self.operation = operation
        self.parent_context = context.get_current()
        name = getattr(instance, "name", None) or type(instance).__name__
        if kind == LOG_TYPE_CHAT:
            name = "smolagents.model"
        elif kind == LOG_TYPE_TASK:
            name = (
                "smolagents.plan"
                if operation == "_generate_planning_step"
                else "smolagents.step"
            )
        self.span = runtime.tracer.start_span(
            name,
            attributes={
                RESPAN_LOG_TYPE: kind,
                SpanAttributes.TRACELOOP_ENTITY_NAME: name,
                SpanAttributes.TRACELOOP_ENTITY_PATH: "",
            },
            context=self.parent_context,
        )
        self.capture = self.span.is_recording() and _content_allowed(self.options)
        self.veto = False
        self.closed = False
        self.content: dict = {}
        self.output: Any = None
        self.stream_calls: dict[int, dict] = {}
        self.stream_text = ""
        self.stream_observed_content = False
        self.stream_usage: dict[str, int] = {}
        self.stream_invalid_total = False
        self.step = (
            _Step(arguments.get("memory_step")) if kind == LOG_TYPE_TASK else None
        )
        self.streaming = streaming
        _safe(self._setup)

    def _setup(self) -> None:
        if self.kind == LOG_TYPE_CHAT:
            self.span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, "chat")
            model = getattr(self.instance, "model_id", None)
            if model:
                self.span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
            provider = getattr(self.instance, "provider", None)
            if not provider and isinstance(model, str) and "/" in model:
                prefix = model.partition("/")[0]
                if prefix in {
                    "openai",
                    "anthropic",
                    "azure",
                    "gemini",
                    "huggingface",
                    "bedrock",
                    "mistral",
                    "cohere",
                }:
                    provider = {"azure": "openai", "gemini": "google"}.get(
                        prefix, prefix
                    )
            if not provider:
                provider = {
                    "OpenAIServerModel": "openai",
                    "AzureOpenAIServerModel": "openai",
                    "InferenceClientModel": "huggingface",
                }.get(type(self.instance).__name__)
            if provider:
                self.span.set_attribute(SpanAttributes.LLM_SYSTEM, provider)
            self.span.set_attribute(SpanAttributes.GEN_AI_IS_STREAMING, self.streaming)
            parameters = dict(self.arguments)
            parameters.update(getattr(self.instance, "kwargs", {}) or {})
            for field_name, key in (
                ("temperature", SpanAttributes.LLM_REQUEST_TEMPERATURE),
                ("top_p", SpanAttributes.LLM_REQUEST_TOP_P),
                ("max_tokens", SpanAttributes.LLM_REQUEST_MAX_TOKENS),
                ("reasoning_effort", SpanAttributes.GEN_AI_REQUEST_REASONING_EFFORT),
            ):
                value = parameters.get(field_name)
                if getattr(
                    self.options.get("config"), "hide_llm_invocation_parameters", False
                ):
                    continue
                if isinstance(value, str | int | float) and not isinstance(value, bool):
                    self.span.set_attribute(
                        key, redact_text(value) if isinstance(value, str) else value
                    )
            if self.capture and parameters.get("response_format") is not None:
                self.content[SpanAttributes.LLM_REQUEST_STRUCTURED_OUTPUT_SCHEMA] = (
                    json_string(parameters["response_format"])
                )
        elif self.kind == LOG_TYPE_AGENT:
            for key in ("reset", "max_steps", "return_full_result", "stream"):
                value = self.arguments.get(key)
                if isinstance(value, bool | int):
                    self.span.set_attribute(f"smolagents.{key}", value)
        if not self.capture:
            return
        if self.kind == LOG_TYPE_CHAT:
            messages = self.arguments.get("messages", ())
            _messages(self.content, SpanAttributes.LLM_PROMPTS, messages)
            tools = self.arguments.get("tools_to_call_from")
            if tools:
                from smolagents.models import get_tool_json_schema

                self.content[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json_string(
                    [get_tool_json_schema(tool) for tool in tools]
                )
        elif self.kind == LOG_TYPE_TOOL:
            args = dict(self.arguments.get("kwargs", {}))
            args.update(
                {
                    key: value
                    for key, value in self.arguments.items()
                    if key not in {"self", "args", "kwargs", "sanitize_inputs_outputs"}
                }
            )
            positional = self.arguments.get("args", ())
            if positional:
                inputs = getattr(self.instance, "inputs", {})
                args.update(zip(inputs, positional, strict=False))
            self.content[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string(
                {"name": self.instance.name, "arguments": args}
            )
            step = _CURRENT_STEP.get()
            if step and (call_id := step.call_id(self.instance.name, args)):
                self.span.set_attribute(gen_ai_attributes.GEN_AI_TOOL_CALL_ID, call_id)
        elif self.kind == LOG_TYPE_AGENT:
            self.content[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string(
                {key: value for key, value in self.arguments.items() if key != "self"}
            )
        elif self.kind == LOG_TYPE_TASK:
            self.content[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string(
                {
                    "step_number": getattr(
                        self.step.value, "step_number", self.arguments.get("step")
                    )
                }
            )

    def enter(self) -> tuple[Any, list]:
        if not _content_allowed(self.options):
            self.veto = True
        current = trace.set_span_in_context(self.span, self.parent_context)
        current = context.set_value(
            _CONTENT_BOUND, self.capture and not self.veto, current
        )
        token = context.attach(current)
        variables = []
        if self.kind == LOG_TYPE_CHAT:
            variables.append((_CURRENT_MODEL, _CURRENT_MODEL.set(self.instance)))
        elif self.kind == LOG_TYPE_TOOL:
            variables.append((_CURRENT_TOOL, _CURRENT_TOOL.set(self.instance)))
        elif self.step:
            variables.append((_CURRENT_STEP, _CURRENT_STEP.set(self.step)))
        return token, variables

    def exit(self, tokens: tuple) -> None:
        if not _content_allowed(self.options):
            self.veto = True
        token, variables = tokens
        for variable, value in reversed(variables):
            variable.reset(value)
        context.detach(token)

    def observe(self, value: Any) -> None:
        if not self.span.is_recording():
            return
        if self.kind == LOG_TYPE_CHAT:
            usage = _get(value, "token_usage")
            if usage is not None:
                values = {
                    key: _get(usage, key) for key in ("input_tokens", "output_tokens")
                }
                complete = all(
                    isinstance(count, int)
                    and not isinstance(count, bool)
                    and count >= 0
                    for count in values.values()
                )
                self.stream_invalid_total = self.stream_invalid_total or not complete
                if complete:
                    values["total_tokens"] = _get(usage, "total_tokens")
                for key, count in values.items():
                    if (
                        isinstance(count, int)
                        and not isinstance(count, bool)
                        and count >= 0
                    ):
                        self.stream_usage[key] = self.stream_usage.get(key, 0) + count
        if not self.capture or self.veto or not _content_allowed(self.options):
            return
        if self.kind == LOG_TYPE_CHAT:
            content = _get(value, "content")
            if isinstance(content, str):
                self.stream_text += content
                self.stream_observed_content = True
            for delta in _get(value, "tool_calls") or ():
                index = _get(delta, "index")
                if index is None:
                    continue
                entry = self.stream_calls.setdefault(
                    index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                for key in ("id", "type"):
                    if _get(delta, key):
                        entry[key] = _get(delta, key)
                function = _get(delta, "function")
                for key in ("name", "arguments"):
                    if _get(function, key):
                        entry["function"][key] += _get(function, key)
        elif self.kind == LOG_TYPE_TASK and type(value).__name__ == "PlanningStep":
            self.output = getattr(value, "plan", None)
        elif self.kind == LOG_TYPE_AGENT and (
            type(value).__name__ == "FinalAnswerStep"
            or getattr(value, "is_final_answer", False)
        ):
            self.output = getattr(value, "output", None)

    def finish(self, result: Any = None, error: BaseException | None = None) -> None:
        if self.closed:
            return
        self.closed = True
        if not _content_allowed(self.options):
            self.veto = True

        def finalize():
            if error is not None and not isinstance(error, GeneratorExit):
                self.span.set_status(trace.StatusCode.ERROR)
                # Exception messages can contain prompts, credentials, or tool inputs.
                # Only emit the class when capture is vetoed.
                if self.capture and not self.veto:
                    args = BaseException.args.__get__(error)
                    message = (
                        redact_text(args[0])
                        if len(args) == 1 and isinstance(args[0], str)
                        else json_string(args)
                    )
                    self.span.add_event(
                        "exception",
                        {
                            EXCEPTION_TYPE: type(error).__name__,
                            EXCEPTION_MESSAGE: message or "",
                        },
                    )
                else:
                    self.span.add_event(
                        "exception", {EXCEPTION_TYPE: type(error).__name__}
                    )
            if self.kind == LOG_TYPE_CHAT and self.span.is_recording():
                metadata = {}
                if self.streaming:
                    _usage(metadata, self.stream_usage)
                    if self.stream_invalid_total:
                        metadata.pop(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, None)
                elif result is not None:
                    _model_metadata(metadata, result)
                self.span.set_attributes(metadata)
            if self.capture and not self.veto:
                if self.kind == LOG_TYPE_CHAT:
                    if self.streaming and (
                        self.stream_observed_content or self.stream_calls
                    ):
                        message = {
                            "role": "assistant",
                            "content": self.stream_text,
                            "tool_calls": [
                                self.stream_calls[i] for i in sorted(self.stream_calls)
                            ],
                            "token_usage": self.stream_usage or None,
                        }
                        _messages(
                            self.content, SpanAttributes.LLM_COMPLETIONS, [message]
                        )
                    elif not self.streaming and result is not None:
                        _messages(
                            self.content, SpanAttributes.LLM_COMPLETIONS, [result]
                        )
                elif self.kind == LOG_TYPE_AGENT:
                    output = (
                        self.output
                        if self.streaming
                        else getattr(result, "output", result)
                    )
                    if output is not None:
                        self.content[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = (
                            json_string(output)
                        )
                elif self.kind == LOG_TYPE_TASK:
                    output = (
                        self.output
                        if self.operation == "_generate_planning_step"
                        else getattr(self.step.value, "observations", None)
                    )
                    if output is not None:
                        self.content[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = (
                            json_string(output)
                        )
                elif error is None and result is not None:
                    self.content[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_string(
                        result
                    )
                self.span.set_attributes(
                    {
                        key: value
                        for key, value in self.content.items()
                        if value is not None
                    }
                )

        _safe(finalize)
        _safe(self.span.end)
        self.content.clear()
        self.output = None
        self.arguments.clear()
        self.stream_calls.clear()
        self.stream_text = ""


class _Stream(Generator):
    """Delegate the native generator protocol and attach context per advance."""

    def __init__(self, native: Iterator, call: _Call):
        self.native, self.call = native, call

    def __iter__(self):
        return self

    def __next__(self):
        return self._advance(next, self.native)

    def send(self, value):
        return self._advance(self.native.send, value)

    def throw(self, *args):
        return self._advance(self.native.throw, *args)

    def close(self):
        try:
            return self._advance(self.native.close)
        finally:
            self.call.finish()

    def _advance(self, operation, *args):
        tokens = self.call.enter()
        try:
            value = operation(*args)
            _safe(lambda: self.call.observe(value))
            return value
        except StopIteration as exc:
            self.call.finish(result=exc.value)
            raise
        except BaseException as exc:
            self.call.finish(error=exc)
            raise
        finally:
            self.call.exit(tokens)


class _Runtime:
    def __init__(self, provider: Any, options: dict):
        self.provider, self.options = provider, options
        self.tracer = provider.get_tracer("respan.instrumentation.smolagents")
        self.active = True
        self.count = 1
        self.patches: list[tuple[type, str, Any, Any]] = []

    def patch(self, cls: type, name: str, kind: str) -> None:
        original = cls.__dict__.get(name)
        if original is None or any(
            owner is cls and key == name for owner, key, _, _ in self.patches
        ):
            return

        @wraps(original)
        def wrapper(instance, *args, **kwargs):
            if (
                not self.active
                or context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY)
                or (kind == LOG_TYPE_CHAT and _CURRENT_MODEL.get() is instance)
                or (kind == LOG_TYPE_TOOL and _CURRENT_TOOL.get() is instance)
            ):
                return original(instance, *args, **kwargs)
            arguments = _arguments(original, instance, args, kwargs)
            streaming = name == "generate_stream" or (
                kind == LOG_TYPE_AGENT and bool(arguments.get("stream"))
            )
            try:
                call = _Call(self, kind, instance, arguments, streaming, name)
            except Exception:
                logger.debug("smolagents span creation failed open", exc_info=True)
                return original(instance, *args, **kwargs)
            tokens = call.enter()
            try:
                result = original(instance, *args, **kwargs)
                if isinstance(result, Iterator):
                    call.streaming = True
                    return _Stream(result, call)
                call.finish(result=result)
                return result
            except BaseException as exc:
                call.finish(error=exc)
                raise
            finally:
                call.exit(tokens)

        setattr(cls, name, wrapper)
        self.patches.append((cls, name, original, wrapper))

    def patch_models(self, cls: type) -> None:
        for name in ("generate", "generate_stream"):
            self.patch(cls, name, LOG_TYPE_CHAT)
        for child in cls.__subclasses__():
            self.patch_models(child)

    def patch_tools(self, cls: type) -> None:
        self.patch(cls, "__call__", LOG_TYPE_TOOL)
        for child in cls.__subclasses__():
            self.patch_tools(child)

    def subclass_hook(self, base: type, model: bool = True) -> None:
        original = base.__dict__.get("__init_subclass__")

        def hook(cls, **kwargs):
            if original is None:
                super(base, cls).__init_subclass__(**kwargs)
            else:
                original.__get__(None, cls)(**kwargs)
            if self.active:
                with _LOCK:
                    (self.patch_models if model else self.patch_tools)(cls)

        replacement = classmethod(hook)
        base.__init_subclass__ = replacement
        self.patches.append((base, "__init_subclass__", original, replacement))

    def patch_executor(self) -> None:
        from smolagents import local_python_executor

        original = local_python_executor.ThreadPoolExecutor
        runtime = self

        class ContextExecutor(original):
            def submit(self, fn, *args, **kwargs):
                if runtime.active:
                    saved = copy_context()
                    return super().submit(saved.run, fn, *args, **kwargs)
                return super().submit(fn, *args, **kwargs)

        local_python_executor.ThreadPoolExecutor = ContextExecutor
        self.patches.append(
            (local_python_executor, "ThreadPoolExecutor", original, ContextExecutor)
        )

    def restore(self) -> None:
        self.active = False
        for cls, name, original, wrapper in reversed(self.patches):
            if cls.__dict__.get(name) is wrapper:
                if original is None:
                    delattr(cls, name)
                else:
                    setattr(cls, name, original)
        self.patches.clear()


class SmolagentsInstrumentor:
    """Trace released smolagents agents, steps, models, and local tools.

    Instances share one registration on a provider. Model subclasses created
    during activation are covered, and teardown restores only owned hooks.
    """

    name = "smolagents"

    def __init__(self, **instrumentor_kwargs: Any):
        self._options = dict(instrumentor_kwargs)
        self._runtime = None

    def activate(self) -> None:
        global _RUNTIME
        with _LOCK:
            if self._runtime is not None:
                return
            tracer = getattr(RespanTracer, "_instance", None)
            if tracer is not None and not getattr(tracer, "is_enabled", True):
                return
            provider = (
                self._options.get("tracer_provider") or trace.get_tracer_provider()
            )
            if _RUNTIME is not None:
                if (
                    _RUNTIME.provider is not provider
                    or _RUNTIME.options != self._options
                ):
                    raise ValueError(
                        "smolagents instrumentation is already active with different settings"
                    )
                _RUNTIME.count += 1
                self._runtime = _RUNTIME
                return
            from smolagents import (
                CodeAgent,
                Model,
                MultiStepAgent,
                Tool,
                ToolCallingAgent,
            )

            runtime = _Runtime(provider, self._options)
            try:
                runtime.patch(MultiStepAgent, "run", LOG_TYPE_AGENT)
                runtime.patch(MultiStepAgent, "__call__", LOG_TYPE_TOOL)
                runtime.patch(MultiStepAgent, "_generate_planning_step", LOG_TYPE_TASK)
                for cls in (CodeAgent, ToolCallingAgent):
                    runtime.patch(cls, "_step_stream", LOG_TYPE_TASK)
                runtime.patch_tools(Tool)
                runtime.subclass_hook(Tool, model=False)
                runtime.patch_models(Model)
                runtime.subclass_hook(Model)
                runtime.patch_executor()
            except BaseException:
                runtime.restore()
                raise
            _RUNTIME = self._runtime = runtime

    def deactivate(self) -> None:
        global _RUNTIME
        with _LOCK:
            if self._runtime is None:
                return
            runtime, self._runtime = self._runtime, None
            runtime.count -= 1
            if runtime.count:
                return
            runtime.restore()
            if _RUNTIME is runtime:
                _RUNTIME = None
