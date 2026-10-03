"""OpenInference compatibility wrappers that keep Groq stream spans open."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from inspect import iscoroutinefunction, signature
from types import SimpleNamespace, TracebackType
from typing import Any, Self

import opentelemetry.context as context_api
from openinference.instrumentation.groq._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.semconv.trace import SpanAttributes as OISpanAttributes
from opentelemetry import trace as trace_api
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

from ._processor import _is_groq_omit

_PATCH_FLAG = "_respan_stream_wrappers_patched"
logger = logging.getLogger(__name__)
_PATCH_OWNERS: set[object] = set()
_PATCH_STATE: tuple[Any, ...] | None = None


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _json_value(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return _json_value(vars(value))
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
                            "reasoning": getattr(choice.message, "reasoning", None),
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
                    "reasoning": [],
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
            reasoning = _field(delta, "reasoning")
            if reasoning:
                state["reasoning"].append(str(reasoning))
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
                        reasoning="".join(state["reasoning"]),
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
        try:
            self._accumulator.add(chunk)
        except Exception:
            logger.debug("Could not capture Groq chunk", exc_info=True)
        return chunk

    def _finish(self, error: BaseException | None) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            self._finish_callback(self._accumulator.completion(), error)
        except Exception:
            logger.debug("Could not finish Groq stream", exc_info=True)

    def close(self) -> Any:
        try:
            close = getattr(self._stream, "close", None)
            return close() if callable(close) else None
        except BaseException as exc:
            self._finish(exc)
            raise
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
        except BaseException as close_error:
            self._finish(close_error)
            raise
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
        try:
            self._accumulator.add(chunk)
        except Exception:
            logger.debug("Could not capture Groq chunk", exc_info=True)
        return chunk

    def _finish(self, error: BaseException | None) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            self._finish_callback(self._accumulator.completion(), error)
        except Exception:
            logger.debug("Could not finish Groq stream", exc_info=True)

    async def aclose(self) -> Any:
        try:
            close = getattr(self._stream, "close", None)
            if not callable(close):
                close = getattr(self._stream, "aclose", None)
            result = close() if callable(close) else None
            if hasattr(result, "__await__"):
                return await result
            return result
        except BaseException as exc:
            self._finish(exc)
            raise
        finally:
            self._finish(None)

    async def close(self) -> Any:
        """Preserve the native AsyncStream.close() lifecycle."""
        return await self.aclose()

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
        except BaseException as close_error:
            self._finish(close_error)
            raise
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
    config: Any = None,
) -> None:
    status = trace_api.Status(status_code=trace_api.StatusCode.OK)
    if error is not None:
        with_span.record_exception(error)
        status = trace_api.Status(
            status_code=trace_api.StatusCode.ERROR,
            description=f"{type(error).__name__}: {error}",
        )
    try:
        wrappers_module._finish_tracing(
            status=status,
            with_span=with_span,
            attributes=()
            if any(
                getattr(config, key, False)
                for key in (
                    "hide_outputs",
                    "hide_output_messages",
                    "hide_output_text",
                    "hide_choices",
                )
            )
            else response_extractor.get_attributes(response=completion),
            extra_attributes=response_extractor.get_extra_attributes(
                response=completion,
                request_parameters=request_parameters,
            ),
        )
    except Exception:
        logger.debug("Could not extract Groq response attributes", exc_info=True)
        with_span.finish_tracing(status=status)


def _request_parameters(wrapped: Any, args: tuple, kwargs: Mapping) -> dict:
    # Do not serialize SDK transport controls (including authorization headers),
    # or Groq 1.x's Omit defaults into the model request.
    bound = signature(wrapped).bind(*args, **kwargs)
    return {
        key: value
        for key, value in bound.arguments.items()
        if key not in {"extra_headers", "extra_query", "timeout"}
        and not _is_groq_omit(value)
    }


class _RequestExtractor(_RequestAttributesExtractor):
    def __init__(self, config=None):
        self._config = config

    def get_attributes_from_request(self, parameters):
        for key, value in super().get_attributes_from_request(parameters):
            if key in {
                OISpanAttributes.INPUT_VALUE,
                OISpanAttributes.INPUT_MIME_TYPE,
            } and any(
                getattr(self._config, flag, False)
                for flag in (
                    "hide_inputs",
                    "hide_input_messages",
                    "hide_input_text",
                    "hide_input_images",
                    "hide_prompts",
                    "hide_llm_tools",
                )
            ):
                continue
            yield key, value
        yield OISpanAttributes.LLM_SYSTEM, "groq"
        yield OISpanAttributes.LLM_PROVIDER, "groq"


def _capture_response(response: Any, finish: Callable, *, async_stream: bool) -> Any:
    # Raw and streaming HTTP response helpers retain their native object and
    # defer capture until the caller parses or closes it. Never read eagerly.
    if type(response).__module__.startswith("groq") and callable(
        getattr(response, "parse", None)
    ):
        parse, close = response.parse, response.close
        captured: dict[int, Any] = {}

        def capture(value):
            if id(value) not in captured:
                captured[id(value)] = _capture_response(
                    value, finish, async_stream=async_stream
                )
            return captured[id(value)]

        def finish_close():
            for value in captured.values():
                if isinstance(value, _SyncStreamProxy | _AsyncStreamProxy):
                    value._finish(None)
            finish(None, None)

        if iscoroutinefunction(parse):

            async def traced_parse(*args, **kwargs):
                try:
                    return capture(await parse(*args, **kwargs))
                except BaseException as exc:
                    finish(None, exc)
                    raise
        else:

            def traced_parse(*args, **kwargs):
                try:
                    return capture(parse(*args, **kwargs))
                except BaseException as exc:
                    finish(None, exc)
                    raise

        if iscoroutinefunction(close):

            async def traced_close():
                try:
                    return await close()
                except BaseException as exc:
                    finish(None, exc)
                    raise
                finally:
                    finish_close()
        else:

            def traced_close():
                try:
                    return close()
                except BaseException as exc:
                    finish(None, exc)
                    raise
                finally:
                    finish_close()

        response.parse, response.close = traced_parse, traced_close
        return response
    if type(response).__name__ in {"Stream", "AsyncStream"}:
        return (_AsyncStreamProxy if async_stream else _SyncStreamProxy)(
            response, finish
        )
    finish(response, None)
    return response


def patch_openinference_stream_wrappers() -> object:
    """Share the stream-aware upstream wrapper classes across adapter owners."""
    global _PATCH_STATE
    token = object()
    _PATCH_OWNERS.add(token)
    if _PATCH_STATE is not None:
        return token
    wrappers_module = __import__(
        "openinference.instrumentation.groq._wrappers",
        fromlist=["_CompletionsWrapper", "_AsyncCompletionsWrapper"],
    )
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
            materialize = getattr(
                wrappers_module, "_materialize_content_iterables", lambda value: value
            )
            kwargs = dict(materialize(kwargs))
            for key in ("tools", "functions"):
                if isinstance(kwargs.get(key), Iterator):
                    kwargs[key] = list(kwargs[key])
            request_parameters = _request_parameters(wrapped, args, kwargs)
            config = getattr(self._tracer, "_self_config", None)
            self._request_extractor = _RequestExtractor(config)
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
                except BaseException as exc:
                    with_span.record_exception(exc)
                    with_span.finish_tracing(
                        status=trace_api.Status(
                            status_code=trace_api.StatusCode.ERROR,
                            description=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    raise
                if request_parameters.get("stream"):
                    with_span.set_attributes({TLSpanAttributes.LLM_IS_STREAMING: True})
                return _capture_response(
                    response,
                    lambda completion, error: _finish_stream(
                        wrappers_module=wrappers_module,
                        with_span=with_span,
                        response_extractor=self._response_extractor,
                        request_parameters=request_parameters,
                        completion=completion,
                        error=error,
                        config=config,
                    ),
                    async_stream=False,
                )

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
            materialize = getattr(
                wrappers_module, "_materialize_content_iterables", lambda value: value
            )
            kwargs = dict(materialize(kwargs))
            for key in ("tools", "functions"):
                if isinstance(kwargs.get(key), Iterator):
                    kwargs[key] = list(kwargs[key])
            request_parameters = _request_parameters(wrapped, args, kwargs)
            config = getattr(self._tracer, "_self_config", None)
            self._request_extractor = _RequestExtractor(config)
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
                except BaseException as exc:
                    with_span.record_exception(exc)
                    with_span.finish_tracing(
                        status=trace_api.Status(
                            status_code=trace_api.StatusCode.ERROR,
                            description=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    raise
                if request_parameters.get("stream"):
                    with_span.set_attributes({TLSpanAttributes.LLM_IS_STREAMING: True})
                return _capture_response(
                    response,
                    lambda completion, error: _finish_stream(
                        wrappers_module=wrappers_module,
                        with_span=with_span,
                        response_extractor=self._response_extractor,
                        request_parameters=request_parameters,
                        completion=completion,
                        error=error,
                        config=config,
                    ),
                    async_stream=True,
                )

    wrappers_module._CompletionsWrapper = RespanCompletionsWrapper
    wrappers_module._AsyncCompletionsWrapper = RespanAsyncCompletionsWrapper
    setattr(wrappers_module, _PATCH_FLAG, True)
    _PATCH_STATE = (
        wrappers_module,
        original_sync,
        original_async,
        RespanCompletionsWrapper,
        RespanAsyncCompletionsWrapper,
    )
    return token


def restore_openinference_stream_wrappers(token: object | None) -> None:
    global _PATCH_STATE
    _PATCH_OWNERS.discard(token)
    if _PATCH_OWNERS or _PATCH_STATE is None:
        return
    module, original_sync, original_async, owned_sync, owned_async = _PATCH_STATE
    if module._CompletionsWrapper is owned_sync:
        module._CompletionsWrapper = original_sync
    if module._AsyncCompletionsWrapper is owned_async:
        module._AsyncCompletionsWrapper = original_async
    setattr(module, _PATCH_FLAG, False)
    _PATCH_STATE = None
