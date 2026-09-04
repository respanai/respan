"""Lifecycle and public-method patching for the Exa Python SDK."""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
import inspect
import json
import logging
import os
import re
import threading
from collections.abc import AsyncIterable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

from opentelemetry import trace
from opentelemetry.context import get_current
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode, use_span
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.span_attributes import RESPAN_METADATA
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_exa._constants import (
    EXA_INSTRUMENTATION_NAME,
    EXA_INSTRUMENTATION_SCOPE,
    FAMILY_AGENT,
    FAMILY_CHAT,
    FAMILY_TASK,
    FAMILY_TOOL,
    STATUS_CODE_ATTR,
    OperationConfig,
)
from respan_instrumentation_exa._serialization import json_dumps, type_name
from respan_instrumentation_exa._translator import (
    build_start_attributes,
    build_success_attributes,
    stream_result,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ClassSpec:
    module: str
    class_name: str
    methods: Mapping[str, OperationConfig]


@dataclass
class _Patch:
    owner: type
    name: str
    original: Any
    replacement: Any


def _op(
    entity_name: str,
    family: str,
    operation: str,
    *,
    always_streaming: bool = False,
    stream_flag: str | None = None,
    stream_family: str | None = None,
    legacy_research: bool = False,
) -> OperationConfig:
    return OperationConfig(
        entity_name=entity_name,
        family=family,
        operation=operation,
        always_streaming=always_streaming,
        stream_flag=stream_flag,
        stream_family=stream_family,
        legacy_research=legacy_research,
    )


_CORE_METHODS = {
    "search": _op("search", FAMILY_TOOL, "search"),
    "stream_search": _op("search", FAMILY_TOOL, "stream_search", always_streaming=True),
    "search_and_contents": _op(
        "search_and_contents", FAMILY_TOOL, "search_and_contents"
    ),
    "get_contents": _op("get_contents", FAMILY_TOOL, "get_contents"),
    "find_similar": _op("find_similar", FAMILY_TOOL, "find_similar"),
    "find_similar_and_contents": _op(
        "find_similar_and_contents",
        FAMILY_TOOL,
        "find_similar_and_contents",
    ),
    "answer": _op("answer", FAMILY_CHAT, "answer"),
    "stream_answer": _op("answer", FAMILY_CHAT, "stream_answer", always_streaming=True),
}

_AGENT_RUN_METHODS = {
    "create": _op(
        "run",
        FAMILY_AGENT,
        "agent.runs.create",
        stream_flag="stream",
    ),
    "get": _op("run.get", FAMILY_TASK, "agent.runs.get"),
    "list": _op("run.list", FAMILY_TASK, "agent.runs.list"),
    "cancel": _op("run.cancel", FAMILY_TASK, "agent.runs.cancel"),
    "stop": _op("run.stop", FAMILY_TASK, "agent.runs.stop"),
    "delete": _op("run.delete", FAMILY_TASK, "agent.runs.delete"),
    "poll_until_finished": _op("run", FAMILY_AGENT, "agent.runs.poll_until_finished"),
    "create_and_wait": _op("run", FAMILY_AGENT, "agent.runs.create_and_wait"),
}

_RESEARCH_METHODS = {
    "create": _op(
        "research",
        FAMILY_AGENT,
        "research.create",
        legacy_research=True,
    ),
    "get": _op(
        "research.get",
        FAMILY_TASK,
        "research.get",
        stream_flag="stream",
        stream_family=FAMILY_AGENT,
        legacy_research=True,
    ),
    "list": _op(
        "research.list",
        FAMILY_TASK,
        "research.list",
        legacy_research=True,
    ),
    "poll_until_finished": _op(
        "research",
        FAMILY_AGENT,
        "research.poll_until_finished",
        legacy_research=True,
    ),
}

_SPECS = (
    _ClassSpec("exa_py.api", "Exa", _CORE_METHODS),
    _ClassSpec("exa_py.api", "AsyncExa", _CORE_METHODS),
    _ClassSpec("exa_py.agent.client", "AgentRunsClient", _AGENT_RUN_METHODS),
    _ClassSpec("exa_py.agent.client", "AgentBetaRunsClient", _AGENT_RUN_METHODS),
    _ClassSpec(
        "exa_py.agent.client",
        "AgentRunEventsClient",
        {"list": _op("run.events.list", FAMILY_TASK, "agent.runs.events.list")},
    ),
    _ClassSpec(
        "exa_py.agent.client",
        "AgentBetaRunEventsClient",
        {"list": _op("run.events.list", FAMILY_TASK, "agent.runs.events.list")},
    ),
    _ClassSpec("exa_py.agent.async_client", "AsyncAgentRunsClient", _AGENT_RUN_METHODS),
    _ClassSpec(
        "exa_py.agent.async_client", "AsyncAgentBetaRunsClient", _AGENT_RUN_METHODS
    ),
    _ClassSpec(
        "exa_py.agent.async_client",
        "AsyncAgentRunEventsClient",
        {"list": _op("run.events.list", FAMILY_TASK, "agent.runs.events.list")},
    ),
    _ClassSpec(
        "exa_py.agent.async_client",
        "AsyncAgentBetaRunEventsClient",
        {"list": _op("run.events.list", FAMILY_TASK, "agent.runs.events.list")},
    ),
    _ClassSpec("exa_py.research.sync_client", "ResearchClient", _RESEARCH_METHODS),
    _ClassSpec(
        "exa_py.research.async_client", "AsyncResearchClient", _RESEARCH_METHODS
    ),
)

_LOCK = threading.RLock()
_PATCHES: list[_Patch] = []
_REFCOUNT = 0
_ENABLED = False
_CAPTURE_CONTENT: bool | None = None
_ACTIVE_DEPTH: ContextVar[int] = ContextVar("respan_exa_active_depth", default=0)
_STATUS_CODE_PATTERN = re.compile(
    r"\b(?:status(?:\s+code)?|http)\D{0,16}([45]\d{2})\b",
    re.IGNORECASE,
)


def _capture_from_env() -> bool:
    return os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _package_version() -> str:
    try:
        return importlib.metadata.version("respan-instrumentation-exa")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _has_parent() -> bool:
    context = trace.get_current_span().get_span_context()
    return bool(getattr(context, "is_valid", False))


def _call_input(
    original: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        bound = inspect.signature(original).bind_partial(instance, *args, **kwargs)
        return {key: value for key, value in bound.arguments.items() if key != "self"}
    except Exception:  # noqa: BLE001 - vendor signatures are best effort.
        return {"args": list(args), "kwargs": kwargs}


def _is_streaming(config: OperationConfig, call_input: Mapping[str, Any]) -> bool:
    if config.always_streaming:
        return True
    if config.stream_flag:
        return bool(call_input.get(config.stream_flag, False))
    return False


def _set_attributes(span: Any, attributes: Mapping[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def _metadata_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {"value": value}
        return dict(parsed) if isinstance(parsed, Mapping) else {"value": parsed}
    if value is None:
        return {}
    return {"value": value}


def _merge_metadata_attributes(instrumentation: Any, existing: Any) -> str:
    """Merge canonical metadata without requiring a newer tracing runtime."""

    return json.dumps(
        {
            **_metadata_mapping(existing),
            **_metadata_mapping(instrumentation),
        },
        default=str,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _error_status_code(error: BaseException) -> int:
    response = getattr(error, "response", None)
    candidates = (
        getattr(error, "status_code", None),
        getattr(error, "statusCode", None),
        getattr(error, "status", None),
        getattr(error, "code", None),
        getattr(response, "status_code", None),
        getattr(response, "status", None),
    )
    for candidate in candidates:
        try:
            status_code = int(candidate)
        except (TypeError, ValueError):
            continue
        if 400 <= status_code <= 599:
            return status_code
    match = _STATUS_CODE_PATTERN.search(str(error))
    return int(match.group(1)) if match else 500


def _start_span(
    config: OperationConfig,
    call_input: Mapping[str, Any],
    *,
    streaming: bool,
) -> Any:
    tracer = trace.get_tracer(EXA_INSTRUMENTATION_SCOPE, _package_version())
    has_parent = _has_parent()
    return tracer.start_span(
        config.entity_name,
        context=get_current(),
        kind=SpanKind.CLIENT,
        attributes=build_start_attributes(
            config=config,
            call_input=call_input,
            capture_content=bool(_CAPTURE_CONTENT),
            streaming=streaming,
            has_parent=has_parent,
        ),
        record_exception=False,
        set_status_on_exception=False,
    )


def _finish_success(
    span: Any,
    config: OperationConfig,
    call_input: Mapping[str, Any],
    result: Any,
    *,
    streaming: bool,
    stream_completed: bool = True,
) -> None:
    attributes = build_success_attributes(
        config=config,
        call_input=call_input,
        result=result,
        capture_content=bool(_CAPTURE_CONTENT),
        streaming=streaming,
        stream_completed=stream_completed,
    )
    metadata = attributes.get(RESPAN_METADATA)
    span_attributes = getattr(span, "attributes", None)
    existing_metadata = (
        span_attributes.get(RESPAN_METADATA)
        if isinstance(span_attributes, Mapping)
        else None
    )
    if metadata is not None:
        attributes[RESPAN_METADATA] = _merge_metadata_attributes(
            metadata,
            existing_metadata,
        )
    _set_attributes(span, attributes)
    span.set_attribute(STATUS_CODE_ATTR, 200)
    span.set_status(Status(StatusCode.OK))
    span.end()


def _finish_error(span: Any, error: BaseException) -> None:
    message = str(error) or type_name(error)
    span.record_exception(error)
    span.set_attribute(ERROR_MESSAGE_ATTR, message)
    span.set_attribute(STATUS_CODE_ATTR, _error_status_code(error))
    if _CAPTURE_CONTENT:
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json_dumps({"error": type_name(error), "message": message}),
        )
    span.set_status(Status(StatusCode.ERROR, message))
    span.end()


class _SyncStreamProxy:
    def __init__(
        self,
        iterable: Iterable[Any],
        *,
        span: Any,
        config: OperationConfig,
        call_input: Mapping[str, Any],
    ) -> None:
        self._source = iterable
        self._iterator = iter(iterable)
        self._span = span
        self._config = config
        self._call_input = call_input
        self._chunks: list[Any] = []
        self._finished = False

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Any:
        token = _ACTIVE_DEPTH.set(_ACTIVE_DEPTH.get() + 1)
        try:
            with use_span(self._span, end_on_exit=False):
                chunk = next(self._iterator)
        except StopIteration:
            self._finish(completed=True)
            raise
        except BaseException as exc:
            self._fail(exc)
            raise
        finally:
            _ACTIVE_DEPTH.reset(token)
        self._chunks.append(chunk)
        return chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    def close(self) -> None:
        error: BaseException | None = None
        token = _ACTIVE_DEPTH.set(_ACTIVE_DEPTH.get() + 1)
        try:
            close = getattr(self._source, "close", None)
            if callable(close):
                close()
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                if error is None:
                    self._finish(completed=False)
                else:
                    self._fail(error)
            finally:
                _ACTIVE_DEPTH.reset(token)

    def __enter__(self) -> Self:
        enter = getattr(self._source, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Any:
        if exc is not None:
            self._fail(exc)
        else:
            self.close()
        return False

    def _finish(self, *, completed: bool) -> None:
        if self._finished:
            return
        self._finished = True
        _finish_success(
            self._span,
            self._config,
            self._call_input,
            stream_result(self._chunks),
            streaming=True,
            stream_completed=completed,
        )

    def _fail(self, error: BaseException) -> None:
        if self._finished:
            return
        self._finished = True
        _finish_error(self._span, error)


class _AsyncStreamProxy:
    def __init__(
        self,
        iterable: AsyncIterable[Any],
        *,
        span: Any,
        config: OperationConfig,
        call_input: Mapping[str, Any],
    ) -> None:
        self._source = iterable
        self._iterator = iterable.__aiter__()
        self._span = span
        self._config = config
        self._call_input = call_input
        self._chunks: list[Any] = []
        self._finished = False

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Any:
        token = _ACTIVE_DEPTH.set(_ACTIVE_DEPTH.get() + 1)
        try:
            with use_span(self._span, end_on_exit=False):
                chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._finish(completed=True)
            raise
        except BaseException as exc:
            self._fail(exc)
            raise
        finally:
            _ACTIVE_DEPTH.reset(token)
        self._chunks.append(chunk)
        return chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    def close(self) -> None:
        """Close exa-py's async stream, whose public close method is synchronous."""

        if self._finished:
            return
        error: BaseException | None = None
        token = _ACTIVE_DEPTH.set(_ACTIVE_DEPTH.get() + 1)
        try:
            close = getattr(self._source, "close", None)
            if callable(close):
                close()
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                if error is None:
                    self._finish(completed=False)
                else:
                    self._fail(error)
            finally:
                _ACTIVE_DEPTH.reset(token)

    async def aclose(self) -> None:
        if self._finished:
            return
        error: BaseException | None = None
        token = _ACTIVE_DEPTH.set(_ACTIVE_DEPTH.get() + 1)
        try:
            close = getattr(self._source, "aclose", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
            else:
                close = getattr(self._source, "close", None)
                if callable(close):
                    result = close()
                    if inspect.isawaitable(result):
                        await result
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                if error is None:
                    self._finish(completed=False)
                else:
                    self._fail(error)
            finally:
                _ACTIVE_DEPTH.reset(token)

    def _finish(self, *, completed: bool) -> None:
        if self._finished:
            return
        self._finished = True
        _finish_success(
            self._span,
            self._config,
            self._call_input,
            stream_result(self._chunks),
            streaming=True,
            stream_completed=completed,
        )

    def _fail(self, error: BaseException) -> None:
        if self._finished:
            return
        self._finished = True
        _finish_error(self._span, error)


def _sync_wrapper(original: Any, config: OperationConfig) -> Any:
    @functools.wraps(original)
    def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
        if not _ENABLED or _ACTIVE_DEPTH.get() > 0:
            return original(instance, *args, **kwargs)
        call_input = _call_input(original, instance, args, kwargs)
        streaming = _is_streaming(config, call_input)
        span = _start_span(config, call_input, streaming=streaming)
        token = _ACTIVE_DEPTH.set(_ACTIVE_DEPTH.get() + 1)
        try:
            with use_span(span, end_on_exit=False):
                result = original(instance, *args, **kwargs)
        except BaseException as exc:
            _finish_error(span, exc)
            raise
        finally:
            _ACTIVE_DEPTH.reset(token)
        if streaming:
            if not isinstance(result, Iterable):
                error = TypeError(
                    f"{config.operation} returned non-iterable {type_name(result)}"
                )
                _finish_error(span, error)
                return result
            return _SyncStreamProxy(
                result,
                span=span,
                config=config,
                call_input=call_input,
            )
        _finish_success(
            span,
            config,
            call_input,
            result,
            streaming=False,
        )
        return result

    return wrapped


def _async_wrapper(original: Any, config: OperationConfig) -> Any:
    @functools.wraps(original)
    async def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
        if not _ENABLED or _ACTIVE_DEPTH.get() > 0:
            return await original(instance, *args, **kwargs)
        call_input = _call_input(original, instance, args, kwargs)
        streaming = _is_streaming(config, call_input)
        span = _start_span(config, call_input, streaming=streaming)
        token = _ACTIVE_DEPTH.set(_ACTIVE_DEPTH.get() + 1)
        try:
            with use_span(span, end_on_exit=False):
                result = await original(instance, *args, **kwargs)
        except BaseException as exc:
            _finish_error(span, exc)
            raise
        finally:
            _ACTIVE_DEPTH.reset(token)
        if streaming:
            if not isinstance(result, AsyncIterable):
                error = TypeError(
                    f"{config.operation} returned non-async-iterable {type_name(result)}"
                )
                _finish_error(span, error)
                return result
            return _AsyncStreamProxy(
                result,
                span=span,
                config=config,
                call_input=call_input,
            )
        _finish_success(
            span,
            config,
            call_input,
            result,
            streaming=False,
        )
        return result

    return wrapped


class ExaInstrumentor:
    """Respan instrumentor for `exa-py` 2.20 and compatible 2.x releases."""

    name = EXA_INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        capture_content: bool | None = None,
        module_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self._capture_content = (
            _capture_from_env() if capture_content is None else capture_content
        )
        self._module_overrides = dict(module_overrides or {})
        self._is_instrumented = False

    @staticmethod
    def _tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        return tracer is None or bool(getattr(tracer, "is_enabled", True))

    def _module(self, name: str) -> Any:
        return self._module_overrides.get(name) or importlib.import_module(name)

    def activate(self) -> None:
        """Patch supported public SDK methods with shared, reference-counted ownership."""

        global _CAPTURE_CONTENT, _ENABLED, _REFCOUNT
        with _LOCK:
            if self._is_instrumented or not self._tracing_enabled():
                return
            if _REFCOUNT:
                if _CAPTURE_CONTENT != self._capture_content:
                    raise ValueError(
                        "all active ExaInstrumentor instances must use the same "
                        "capture_content setting"
                    )
                _REFCOUNT += 1
                self._is_instrumented = True
                return

            applied: list[_Patch] = []
            try:
                for spec in _SPECS:
                    module = self._module(spec.module)
                    owner = getattr(module, spec.class_name)
                    for method_name, config in spec.methods.items():
                        original = owner.__dict__.get(method_name)
                        if not callable(original):
                            continue
                        replacement = (
                            _async_wrapper(original, config)
                            if inspect.iscoroutinefunction(original)
                            else _sync_wrapper(original, config)
                        )
                        setattr(owner, method_name, replacement)
                        applied.append(
                            _Patch(owner, method_name, original, replacement)
                        )
            except Exception:
                for patch in reversed(applied):
                    if getattr(patch.owner, patch.name, None) is patch.replacement:
                        setattr(patch.owner, patch.name, patch.original)
                raise

            if not applied:
                logger.warning("Exa instrumentation found no supported methods")
                return
            _PATCHES[:] = applied
            _CAPTURE_CONTENT = self._capture_content
            _ENABLED = True
            _REFCOUNT = 1
            self._is_instrumented = True
        logger.info("Exa instrumentation activated")

    def deactivate(self) -> None:
        """Restore methods owned by this instrumentation."""

        global _CAPTURE_CONTENT, _ENABLED, _REFCOUNT
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _REFCOUNT = max(0, _REFCOUNT - 1)
            if _REFCOUNT:
                return
            _ENABLED = False
            for patch in reversed(_PATCHES):
                if getattr(patch.owner, patch.name, None) is patch.replacement:
                    setattr(patch.owner, patch.name, patch.original)
            _PATCHES.clear()
            _CAPTURE_CONTENT = None
        logger.info("Exa instrumentation deactivated")

    def is_active(self) -> bool:
        return self._is_instrumented

    def instrument(self) -> None:
        self.activate()

    def uninstrument(self) -> None:
        self.deactivate()
