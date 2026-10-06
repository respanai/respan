"""OpenInference compatibility wrappers that keep Groq stream spans open."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from inspect import signature
from types import SimpleNamespace, TracebackType
from typing import Any, Self

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

_PATCH_FLAG = "_respan_stream_wrappers_patched"


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(exclude_none=True)
        except TypeError:
            return model_dump()
    return str(value)


class _AssembledCompletion:
    def __init__(self, *, model: str | None, usage: Any, choices: list[Any]) -> None:
        self.model = model
        self.usage = usage
        self.choices = choices

    def model_dump_json(self, **_: Any) -> str:
        return json.dumps(
            {
                "model": self.model,
                "choices": [
                    {
                        "index": choice.index,
                        "finish_reason": choice.finish_reason,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                            "tool_calls": _json_value(choice.message.tool_calls),
                        },
                    }
                    for choice in self.choices
                ],
                "usage": _json_value(self.usage),
            },
            separators=(",", ":"),
        )


class _StreamAccumulator:
    """Incrementally assemble Groq chat chunks without retaining every frame."""

    def __init__(self) -> None:
        self.model: str | None = None
        self.usage: Any = None
        self._choices: dict[int, dict[str, Any]] = {}

    def add(self, chunk: Any) -> None:
        model = _field(chunk, "model")
        if model:
            self.model = str(model)

        usage = _field(chunk, "usage")
        if usage is None:
            usage = _field(_field(chunk, "x_groq"), "usage")
        if usage is not None:
            self.usage = usage

        for choice in _field(chunk, "choices", []) or []:
            index = int(_field(choice, "index", 0) or 0)
            state = self._choices.setdefault(
                index,
                {
                    "role": "assistant",
                    "content": [],
                    "finish_reason": None,
                    "tool_calls": {},
                },
            )
            delta = _field(choice, "delta")
            role = _field(delta, "role")
            if role:
                state["role"] = str(role)
            content = _field(delta, "content")
            if content:
                state["content"].append(str(content))
            finish_reason = _field(choice, "finish_reason")
            if finish_reason is not None:
                state["finish_reason"] = str(finish_reason)

            for tool_call in _field(delta, "tool_calls", []) or []:
                tool_index = int(_field(tool_call, "index", 0) or 0)
                tool_state = state["tool_calls"].setdefault(
                    tool_index,
                    {
                        "id": None,
                        "type": "function",
                        "name": None,
                        "arguments": [],
                    },
                )
                tool_call_id = _field(tool_call, "id")
                if tool_call_id:
                    tool_state["id"] = str(tool_call_id)
                tool_type = _field(tool_call, "type")
                if tool_type:
                    tool_state["type"] = str(tool_type)
                function = _field(tool_call, "function")
                function_name = _field(function, "name")
                if function_name:
                    tool_state["name"] = str(function_name)
                arguments = _field(function, "arguments")
                if arguments:
                    tool_state["arguments"].append(str(arguments))

    def completion(self) -> _AssembledCompletion:
        choices: list[Any] = []
        for index in sorted(self._choices):
            state = self._choices[index]
            tool_calls = [
                SimpleNamespace(
                    id=tool_state["id"],
                    type=tool_state["type"],
                    function=SimpleNamespace(
                        name=tool_state["name"],
                        arguments="".join(tool_state["arguments"]),
                    ),
                )
                for _, tool_state in sorted(state["tool_calls"].items())
            ]
            choices.append(
                SimpleNamespace(
                    index=index,
                    finish_reason=state["finish_reason"],
                    message=SimpleNamespace(
                        role=state["role"],
                        content="".join(state["content"]),
                        tool_calls=tool_calls,
                        function_call=None,
                    ),
                )
            )
        if not choices:
            choices.append(
                SimpleNamespace(
                    index=0,
                    finish_reason=None,
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[],
                        function_call=None,
                    ),
                )
            )
        return _AssembledCompletion(
            model=self.model,
            usage=self.usage,
            choices=choices,
        )


class _SyncStreamProxy:
    def __init__(
        self, stream: Any, finish: Callable[[Any, BaseException | None], None]
    ) -> None:
        self._stream = stream
        self._iterator = iter(stream)
        self._finish_callback = finish
        self._accumulator = _StreamAccumulator()
        self._finished = False

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._iterator)
        except StopIteration:
            self._finish(None)
            raise
        except BaseException as exc:
            self._finish(exc)
            raise
        self._accumulator.add(chunk)
        return chunk

    def _finish(self, error: BaseException | None) -> None:
        if self._finished:
            return
        self._finished = True
        self._finish_callback(self._accumulator.completion(), error)

    def close(self) -> Any:
        try:
            close = getattr(self._stream, "close", None)
            return close() if callable(close) else None
        finally:
            self._finish(None)

    def __enter__(self) -> Self:
        enter = getattr(self._stream, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Any:
        try:
            exit_method = getattr(self._stream, "__exit__", None)
            return (
                exit_method(exc_type, exc, traceback)
                if callable(exit_method)
                else False
            )
        finally:
            self._finish(exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _AsyncStreamProxy:
    def __init__(
        self, stream: Any, finish: Callable[[Any, BaseException | None], None]
    ) -> None:
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._finish_callback = finish
        self._accumulator = _StreamAccumulator()
        self._finished = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._finish(None)
            raise
        except BaseException as exc:
            self._finish(exc)
            raise
        self._accumulator.add(chunk)
        return chunk

    def _finish(self, error: BaseException | None) -> None:
        if self._finished:
            return
        self._finished = True
        self._finish_callback(self._accumulator.completion(), error)

    async def aclose(self) -> Any:
        try:
            close = getattr(self._stream, "close", None)
            if not callable(close):
                close = getattr(self._stream, "aclose", None)
            result = close() if callable(close) else None
            if hasattr(result, "__await__"):
                return await result
            return result
        finally:
            self._finish(None)

    async def __aenter__(self) -> Self:
        enter = getattr(self._stream, "__aenter__", None)
        if callable(enter):
            result = enter()
            if hasattr(result, "__await__"):
                await result
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Any:
        try:
            exit_method = getattr(self._stream, "__aexit__", None)
            result = (
                exit_method(exc_type, exc, traceback)
                if callable(exit_method)
                else False
            )
            if hasattr(result, "__await__"):
                return await result
            return result
        finally:
            self._finish(exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _finish_stream(
    *,
    wrappers_module: Any,
    with_span: Any,
    response_extractor: Any,
    request_parameters: Mapping[str, Any],
    completion: Any,
    error: BaseException | None,
) -> None:
    status = trace_api.Status(status_code=trace_api.StatusCode.OK)
    if error is not None:
        with_span.record_exception(error)
        status = trace_api.Status(
            status_code=trace_api.StatusCode.ERROR,
            description=f"{type(error).__name__}: {error}",
        )
    wrappers_module._finish_tracing(
        status=status,
        with_span=with_span,
        attributes=response_extractor.get_attributes(response=completion),
        extra_attributes=response_extractor.get_extra_attributes(
            response=completion,
            request_parameters=request_parameters,
        ),
    )


def _stream_requested(request_parameters: Mapping[str, Any], response: Any) -> bool:
    return bool(request_parameters.get("stream")) or type(response).__name__ in {
        "Stream",
        "AsyncStream",
    }


def patch_openinference_stream_wrappers() -> tuple[Any, type, type] | None:
    """Replace upstream wrappers with stream-aware subclasses before activation."""
    wrappers_module = __import__(
        "openinference.instrumentation.groq._wrappers",
        fromlist=["_CompletionsWrapper", "_AsyncCompletionsWrapper"],
    )
    if getattr(wrappers_module, _PATCH_FLAG, False):
        return None

    original_sync = wrappers_module._CompletionsWrapper
    original_async = wrappers_module._AsyncCompletionsWrapper

    class RespanCompletionsWrapper(original_sync):
        def __call__(
            self,
            wrapped: Any,
            instance: Any,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Any:
            if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
                return wrapped(*args, **kwargs)
            request_parameters = wrappers_module._parse_args(
                signature(wrapped), *args, **kwargs
            )
            with self._start_as_current_span(
                span_name="Completions",
                attributes=self._request_extractor.get_attributes_from_request(
                    request_parameters
                ),
                context_attributes=wrappers_module.get_attributes_from_context(),
                extra_attributes=self._request_extractor.get_extra_attributes_from_request(
                    request_parameters
                ),
            ) as with_span:
                try:
                    response = wrapped(*args, **kwargs)
                except Exception as exc:
                    with_span.record_exception(exc)
                    with_span.finish_tracing(
                        status=trace_api.Status(
                            status_code=trace_api.StatusCode.ERROR,
                            description=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    raise
                if _stream_requested(request_parameters, response):
                    with_span.set_attributes({TLSpanAttributes.LLM_IS_STREAMING: True})
                    return _SyncStreamProxy(
                        response,
                        lambda completion, error: _finish_stream(
                            wrappers_module=wrappers_module,
                            with_span=with_span,
                            response_extractor=self._response_extractor,
                            request_parameters=request_parameters,
                            completion=completion,
                            error=error,
                        ),
                    )
                _finish_stream(
                    wrappers_module=wrappers_module,
                    with_span=with_span,
                    response_extractor=self._response_extractor,
                    request_parameters=request_parameters,
                    completion=response,
                    error=None,
                )
                return response

    class RespanAsyncCompletionsWrapper(original_async):
        async def __call__(
            self,
            wrapped: Any,
            instance: Any,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Any:
            if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
                return await wrapped(*args, **kwargs)
            request_parameters = wrappers_module._parse_args(
                signature(wrapped), *args, **kwargs
            )
            with self._start_as_current_span(
                span_name="AsyncCompletions",
                attributes=self._request_extractor.get_attributes_from_request(
                    request_parameters
                ),
                context_attributes=wrappers_module.get_attributes_from_context(),
                extra_attributes=self._request_extractor.get_extra_attributes_from_request(
                    request_parameters
                ),
            ) as with_span:
                try:
                    response = await wrapped(*args, **kwargs)
                except Exception as exc:
                    with_span.record_exception(exc)
                    with_span.finish_tracing(
                        status=trace_api.Status(
                            status_code=trace_api.StatusCode.ERROR,
                            description=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    raise
                if _stream_requested(request_parameters, response):
                    with_span.set_attributes({TLSpanAttributes.LLM_IS_STREAMING: True})
                    return _AsyncStreamProxy(
                        response,
                        lambda completion, error: _finish_stream(
                            wrappers_module=wrappers_module,
                            with_span=with_span,
                            response_extractor=self._response_extractor,
                            request_parameters=request_parameters,
                            completion=completion,
                            error=error,
                        ),
                    )
                _finish_stream(
                    wrappers_module=wrappers_module,
                    with_span=with_span,
                    response_extractor=self._response_extractor,
                    request_parameters=request_parameters,
                    completion=response,
                    error=None,
                )
                return response

    wrappers_module._CompletionsWrapper = RespanCompletionsWrapper
    wrappers_module._AsyncCompletionsWrapper = RespanAsyncCompletionsWrapper
    setattr(wrappers_module, _PATCH_FLAG, True)
    return wrappers_module, original_sync, original_async


def restore_openinference_stream_wrappers(patch: tuple[Any, type, type] | None) -> None:
    if patch is None:
        return
    wrappers_module, original_sync, original_async = patch
    wrappers_module._CompletionsWrapper = original_sync
    wrappers_module._AsyncCompletionsWrapper = original_async
    setattr(wrappers_module, _PATCH_FLAG, False)
