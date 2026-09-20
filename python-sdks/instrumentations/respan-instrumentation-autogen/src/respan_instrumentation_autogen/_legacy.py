"""Adapt OpenInference's AG2 wrappers to the earlier ``autogen`` APIs."""

from __future__ import annotations

from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from inspect import getattr_static
from typing import Any

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.ag2 import AG2Instrumentor
from openinference.instrumentation.ag2._wrappers import (
    _ChatWrapper,
    _ReplyWrapper,
    _ToolWrapper,
)
from opentelemetry import trace
from opentelemetry.instrumentation.dependencies import get_dependency_conflicts
from wrapt import wrap_function_wrapper


class _ChatResult:
    def __init__(self, result: Any, agent: Any, recipient: Any) -> None:
        self.result = result
        self.chat_history = agent.chat_messages.get(recipient, [])


class _LegacyChatWrapper(_ChatWrapper):
    def _span(self, instance: Any, method: str, bound: Any) -> Any:
        # pyautogen 0.2.2 accepts message inside **context and returns None.
        bound = {**bound.get("context", {}), **bound}
        return super()._span(instance, method, bound)

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        @wraps(wrapped)
        def with_result(*call_args: Any, **call_kwargs: Any) -> Any:
            result = wrapped(*call_args, **call_kwargs)
            recipient = call_kwargs.get("recipient", call_args[0] if call_args else None)
            return _ChatResult(result, instance, recipient) if result is None else result

        result = super().__call__(with_result, instance, args, kwargs)
        return result.result if isinstance(result, _ChatResult) else result

    async def async_call(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        @wraps(wrapped)
        async def with_result(*call_args: Any, **call_kwargs: Any) -> Any:
            result = await wrapped(*call_args, **call_kwargs)
            recipient = call_kwargs.get("recipient", call_args[0] if call_args else None)
            return _ChatResult(result, instance, recipient) if result is None else result

        result = await super().async_call(with_result, instance, args, kwargs)
        return result.result if isinstance(result, _ChatResult) else result


class _LegacyReplyWrapper(_ReplyWrapper):
    def _span(self, instance: Any, method: str, bound: Any) -> Any:
        if bound.get("messages") is None:
            bound = {**bound, "messages": instance.chat_messages.get(bound.get("sender"), [])}
        return super()._span(instance, method, bound)


class _LegacyToolWrapper(_ToolWrapper):
    @staticmethod
    def _normalized_call(args: Any, kwargs: Any) -> Any:
        # The traced and suppressed paths must retain the SDK's validation,
        # including its rejection of a bare string instead of a function dict.
        return args, kwargs


def _gate(wrapper: Any, generation: dict[str, bool]) -> Any:
    def call(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        if not generation["enabled"]:
            return wrapped(*args, **kwargs)
        return wrapper(wrapped, instance, args, kwargs)

    return call


class LegacyAutoGenInstrumentor(AG2Instrumentor):
    """Use the upstream lifecycle and wrappers with verified legacy contracts.

    AG2's default dependency check excludes pyautogen 0.2. These versions share
    the six ConversableAgent methods but differ in chat arguments and results.
    """

    # BaseInstrumentor is a per-class singleton; never inherit an already
    # constructed AG2Instrumentor instance from the parent class.
    _instance = None

    def _check_dependency_conflicts(self) -> Any:
        requirements = ("pyautogen>=0.2.2,<0.3", "autogen>=0.7,<0.8")
        conflicts = [get_dependency_conflicts([requirement]) for requirement in requirements]
        return None if any(conflict is None for conflict in conflicts) else conflicts[0]

    def _instrument(self, **kwargs: Any) -> None:
        import autogen
        from autogen import ConversableAgent

        if AG2Instrumentor().is_instrumented_by_opentelemetry:
            raise RuntimeError("Deactivate the existing AG2 instrumentor before activating legacy AutoGen")

        installed = []
        for distribution in ("pyautogen", "autogen", "ag2"):
            try:
                installed.append((distribution, version(distribution)))
            except PackageNotFoundError:
                pass
        # autogen 0.7 is a metapackage for pyautogen of the same version.
        if not installed or any(item[1] != autogen.__version__ for item in installed):
            raise RuntimeError(
                "Legacy AutoGen requires matching autogen distribution versions; "
                "install either pyautogen 0.2.x or autogen 0.7.x in a clean environment"
            )

        config = kwargs.get("config") or TraceConfig()
        if not isinstance(config, TraceConfig):
            raise TypeError("config must be an instance of TraceConfig")
        self._tracer = OITracer(
            trace.get_tracer(
                "openinference.instrumentation.ag2",
                tracer_provider=kwargs.get("tracer_provider"),
            ),
            config=config,
        )
        chat, reply, tool = (
            _LegacyChatWrapper(self._tracer),
            _LegacyReplyWrapper(self._tracer),
            _LegacyToolWrapper(self._tracer),
        )
        wrappers = {
            "initiate_chat": chat,
            "a_initiate_chat": chat.async_call,
            "generate_reply": reply,
            "a_generate_reply": reply.async_call,
            "execute_function": tool,
            "a_execute_function": tool.async_call,
        }
        self._original_methods = {name: getattr_static(ConversableAgent, name) for name in wrappers}
        self._installed_methods = {}
        self._generation = {"enabled": False}
        try:
            for name, wrapper in wrappers.items():
                wrap_function_wrapper(ConversableAgent, name, _gate(wrapper, self._generation))
                self._installed_methods[name] = getattr_static(ConversableAgent, name)
            self._generation["enabled"] = True
        except BaseException:
            self._uninstrument()
            raise

    def _uninstrument(self, **kwargs: Any) -> None:
        from autogen import ConversableAgent

        # Another library may wrap our method after activation. Retain that
        # wrapper and disable our captured generation so it becomes inert,
        # including after a later activation creates a new generation.
        if generation := getattr(self, "_generation", None):
            generation["enabled"] = False
        for name, original in getattr(self, "_original_methods", {}).items():
            if getattr_static(ConversableAgent, name) is self._installed_methods.get(name):
                setattr(ConversableAgent, name, original)
        self._original_methods = {}
        self._installed_methods = {}
