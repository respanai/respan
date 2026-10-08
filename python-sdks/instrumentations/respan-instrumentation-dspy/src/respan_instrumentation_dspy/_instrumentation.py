"""Shared native callback registration and owned DSPy data-boundary hooks."""

from __future__ import annotations

import inspect
import logging
import threading
from functools import wraps
from typing import Any
from uuid import uuid4

from opentelemetry import trace
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_dspy._callback import (
    _TOOL_CALLS,
    DSPyInstrumentationCallback,
)

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_RUNTIME: _Runtime | None = None
_UNSET = object()


class _Runtime:
    def __init__(self, dspy, provider):
        self.dspy, self.provider = dspy, provider
        self.bindings = {}
        self.active = True
        self.patches = []
        self.callback = DSPyInstrumentationCallback(
            tracer_provider=provider, policy=self.policy
        )

    def policy(self, instance):
        return all(
            binding["content"]
            for binding in self.bindings.values()
            if binding["target"] is None or binding["target"] is instance
        )

    def registered(self, instance):
        return (
            0 in self.bindings or id(instance) in self.bindings
        ) and self.callback in (
            list(self.dspy.settings.get("callbacks", []))
            + list(getattr(instance, "callbacks", []))
        )

    def patch(self, owner, name, factory):
        original = getattr(owner, name, None)
        if original is None:
            return
        replacement = factory(original)
        setattr(owner, name, replacement)
        self.patches.append((owner, name, original, replacement))

    def install_hooks(self):
        runtime = self

        def legacy_factory(original):
            @wraps(original)
            def wrapper(instance, response, *args, **kwargs):
                if runtime.active:
                    runtime.callback.capture_result(instance, response)
                return original(instance, response, *args, **kwargs)

            return wrapper

        self.patch(self.dspy.BaseLM, "_process_lm_response", legacy_factory)
        try:
            from dspy.clients import execution
        except ImportError:
            pass
        else:

            def factory(original):
                @wraps(original)
                def wrapper(lm, call, result):
                    if runtime.active:
                        runtime.callback.capture_result(lm, result)
                    return original(lm, call, result)

                return wrapper

            self.patch(execution, "finalize", factory)

        def embedding_factory(original):
            def start(instance, args, kwargs):
                key = uuid4().hex
                try:
                    inputs = dict(
                        inspect.signature(original)
                        .bind(instance, *args, **kwargs)
                        .arguments
                    )
                    inputs.pop("self", None)
                except (ValueError, TypeError):
                    inputs = {"inputs": args[0] if args else kwargs.get("inputs")}
                runtime.callback.on_embedding_start(key, instance, inputs)
                return key

            if inspect.iscoroutinefunction(original):

                @wraps(original)
                async def async_wrapper(instance, *args, **kwargs):
                    if not runtime.active or not runtime.registered(instance):
                        return await original(instance, *args, **kwargs)
                    key = start(instance, args, kwargs)
                    try:
                        result = await original(instance, *args, **kwargs)
                    except BaseException as error:
                        runtime.callback.on_embedding_end(key, None, error)
                        raise
                    runtime.callback.on_embedding_end(key, result)
                    return result

                return async_wrapper

            @wraps(original)
            def wrapper(instance, *args, **kwargs):
                if not runtime.active or not runtime.registered(instance):
                    return original(instance, *args, **kwargs)
                key = start(instance, args, kwargs)
                try:
                    result = original(instance, *args, **kwargs)
                except BaseException as error:
                    runtime.callback.on_embedding_end(key, None, error)
                    raise
                runtime.callback.on_embedding_end(key, result)
                return result

            return wrapper

        self.patch(self.dspy.Embedder, "__call__", embedding_factory)
        self.patch(self.dspy.Embedder, "acall", embedding_factory)

        def dispatch_factory(original):
            if inspect.iscoroutinefunction(original):

                @wraps(original)
                async def async_wrapper(instance, tool_calls, *args, **kwargs):
                    token = (
                        _TOOL_CALLS.set(tuple(tool_calls.tool_calls))
                        if runtime.active
                        else None
                    )
                    try:
                        return await original(instance, tool_calls, *args, **kwargs)
                    finally:
                        if token is not None:
                            _TOOL_CALLS.reset(token)

                return async_wrapper

            @wraps(original)
            def wrapper(instance, tool_calls, *args, **kwargs):
                token = (
                    _TOOL_CALLS.set(tuple(tool_calls.tool_calls))
                    if runtime.active
                    else None
                )
                try:
                    return original(instance, tool_calls, *args, **kwargs)
                finally:
                    if token is not None:
                        _TOOL_CALLS.reset(token)

            return wrapper

        react = getattr(self.dspy, "ReActV2", None)
        if react is not None:
            self.patch(react, "_execute_tool_calls", dispatch_factory)
            self.patch(react, "_aexecute_tool_calls", dispatch_factory)

    def callbacks(self, target):
        return (
            self.dspy.settings.get("callbacks", [])
            if target is None
            else getattr(target, "callbacks", _UNSET)
        )

    def set_callbacks(self, target, value):
        if target is None:
            self.dspy.configure(callbacks=value)
        elif value is _UNSET:
            object.__delattr__(target, "callbacks") if isinstance(
                target, self.dspy.Tool
            ) else delattr(target, "callbacks")
        else:
            if isinstance(target, self.dspy.Tool):
                object.__setattr__(target, "callbacks", value)
            else:
                target.callbacks = value

    def add(self, target, content):
        key = 0 if target is None else id(target)
        if key in self.bindings:
            binding = self.bindings[key]
            if binding["content"] != content:
                raise ValueError(
                    "DSPy target already instrumented with different include_content"
                )
            binding["count"] += 1
            return key
        previous = self.callbacks(target)
        callbacks = [] if previous is _UNSET else list(previous or [])
        if self.callback not in callbacks:
            callbacks.append(self.callback)
        try:
            self.set_callbacks(target, callbacks)
        except BaseException:
            current = self.callbacks(target)
            remaining = [
                cb
                for cb in ([] if current is _UNSET else current)
                if cb is not self.callback
            ]
            try:
                self.set_callbacks(
                    target,
                    _UNSET if previous is _UNSET and not remaining else remaining,
                )
            except Exception:  # noqa: BLE001 - preserve the original registration error
                logger.debug("DSPy callback registration rollback failed")
            raise
        self.bindings[key] = {
            "target": target,
            "content": content,
            "count": 1,
            "had_callbacks": previous is not _UNSET,
        }
        return key

    def release(self, key):
        binding = self.bindings[key]
        binding["count"] -= 1
        if binding["count"]:
            return
        target = binding["target"]
        current = self.callbacks(target)
        remaining = [
            cb
            for cb in ([] if current is _UNSET else current)
            if cb is not self.callback
        ]
        self.set_callbacks(
            target,
            _UNSET if not binding["had_callbacks"] and not remaining else remaining,
        )
        del self.bindings[key]

    def close(self):
        self.active = False
        self.callback.close()
        for owner, name, original, replacement in reversed(self.patches):
            if getattr(owner, name, None) is replacement:
                setattr(owner, name, original)
        self.patches.clear()


class DSPyInstrumentor:
    name = "dspy"

    def __init__(
        self,
        target: Any | None = None,
        *,
        include_content: bool = True,
        tracer_provider: Any = None,
    ):
        self._target, self._include_content, self._provider = (
            target,
            include_content,
            tracer_provider,
        )
        self._runtime = None
        self._callback = None
        self._key = None

    def activate(self):
        global _RUNTIME
        with _LOCK:
            if self._runtime:
                return
            tracer = getattr(RespanTracer, "_instance", None)
            if tracer is not None and not getattr(tracer, "is_enabled", True):
                return
            import dspy

            provider = self._provider or trace.get_tracer_provider()
            if _RUNTIME and _RUNTIME.provider is not provider:
                raise RuntimeError("DSPy callback already active on another provider")
            runtime = _RUNTIME or _Runtime(dspy, provider)
            try:
                key = runtime.add(self._target, self._include_content)
                if _RUNTIME is None:
                    runtime.install_hooks()
            except BaseException:
                if _RUNTIME is None:
                    for key in list(runtime.bindings):
                        runtime.release(key)
                    runtime.close()
                raise
            _RUNTIME = self._runtime = runtime
            self._key = key
            self._callback = runtime.callback

    def deactivate(self):
        global _RUNTIME
        with _LOCK:
            if not self._runtime:
                return
            runtime = self._runtime
            runtime.release(self._key)
            self._runtime = self._callback = None
            if not runtime.bindings:
                runtime.close()
                if _RUNTIME is runtime:
                    _RUNTIME = None


DspyInstrumentor = DSPyInstrumentor
