"""MCP instrumentation plugin for Respan."""

import asyncio
import importlib
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from threading import RLock
from typing import Any, ClassVar, NamedTuple

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.instrumentation.utils import is_instrumentation_enabled, unwrap
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_instrumentation_openinference import OpenInferenceInstrumentor
from respan_sdk.constants.llm_logging import LOG_TYPE_TASK, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer
from wrapt import wrap_function_wrapper

from respan_instrumentation_mcp._native import (
    owned_operation,
    patch_native_spans,
    restore_native_spans,
)

logger = logging.getLogger(__name__)

MCP_INSTRUMENTATION_NAME = "mcp"
OPENINFERENCE_MCP_MODULE = "openinference.instrumentation.mcp"
MCP_CLIENT_SESSION_MODULE = "mcp.client.session"

_MAX_ATTRIBUTE_CHARS = 16000
_MAX_SERIALIZATION_DEPTH = 6
_MAX_JSON_SCHEMA_DEPTH = 20
_SENSITIVE_KEY_MARKERS = frozenset(
    {
        "apikey",
        "apitoken",
        "authorization",
        "authtoken",
        "bearertoken",
        "cookie",
        "credential",
        "idtoken",
        "password",
        "privatekey",
        "refreshtoken",
        "secret",
        "sessiontoken",
        "accesstoken",
    }
)
_JSON_SCHEMA_KEYS = frozenset(
    {
        "inputSchema",
        "outputSchema",
        "input_schema",
        "output_schema",
        "json_schema",
    }
)


class _OperationConfig(NamedTuple):
    log_type: str
    span_name: str
    entity_name: str


_CLIENT_METHODS: dict[str, _OperationConfig] = {
    "initialize": _OperationConfig(LOG_TYPE_TASK, "mcp.initialize", "mcp.initialize"),
    "list_tools": _OperationConfig(LOG_TYPE_TASK, "mcp.list_tools", "mcp.list_tools"),
    "call_tool": _OperationConfig(LOG_TYPE_TOOL, "mcp.call_tool", "mcp.call_tool"),
    "list_resources": _OperationConfig(
        LOG_TYPE_TASK,
        "mcp.list_resources",
        "mcp.list_resources",
    ),
    "read_resource": _OperationConfig(
        LOG_TYPE_TASK,
        "mcp.read_resource",
        "mcp.read_resource",
    ),
    "list_resource_templates": _OperationConfig(
        LOG_TYPE_TASK,
        "mcp.list_resource_templates",
        "mcp.list_resource_templates",
    ),
    "list_prompts": _OperationConfig(
        LOG_TYPE_TASK,
        "mcp.list_prompts",
        "mcp.list_prompts",
    ),
    "get_prompt": _OperationConfig(LOG_TYPE_TASK, "mcp.get_prompt", "mcp.get_prompt"),
}


def _load_openinference_mcp_class() -> type:
    mcp_module = importlib.import_module(OPENINFERENCE_MCP_MODULE)
    return mcp_module.MCPInstrumentor


def _type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _is_sensitive_key(key: object) -> bool:
    compact = "".join(
        character for character in str(key).lower() if character.isalnum()
    )
    return compact == "token" or any(
        marker in compact for marker in _SENSITIVE_KEY_MARKERS
    )


def _to_jsonable(
    value: Any,
    *,
    depth: int = 0,
    in_json_schema: bool = False,
) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    max_depth = _MAX_JSON_SCHEMA_DEPTH if in_json_schema else _MAX_SERIALIZATION_DEPTH
    if depth > max_depth:
        return {"type": _type_name(value), "truncated": True}
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _to_jsonable(
                item,
                depth=depth + 1,
                in_json_schema=in_json_schema,
            )
            for item in value
        ]
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if _is_sensitive_key(key)
                else _to_jsonable(
                    item,
                    depth=depth + 1,
                    in_json_schema=in_json_schema or str(key) in _JSON_SCHEMA_KEYS,
                )
            )
            for key, item in value.items()
            if not callable(item)
        }

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_jsonable(
                model_dump(mode="json", exclude_none=True, by_alias=True),
                depth=depth + 1,
                in_json_schema=in_json_schema,
            )
        except TypeError:
            try:
                return _to_jsonable(
                    model_dump(),
                    depth=depth + 1,
                    in_json_schema=in_json_schema,
                )
            except Exception:
                logger.debug(
                    "Failed to serialize %s with model_dump",
                    _type_name(value),
                    exc_info=True,
                )
        except Exception:
            logger.debug(
                "Failed to serialize %s with model_dump",
                _type_name(value),
                exc_info=True,
            )

    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        try:
            return _to_jsonable(
                to_dict(),
                depth=depth + 1,
                in_json_schema=in_json_schema,
            )
        except Exception:
            logger.debug(
                "Failed to serialize %s with dict",
                _type_name(value),
                exc_info=True,
            )

    if hasattr(value, "__dict__"):
        public_attributes = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
        if public_attributes:
            return _to_jsonable(
                public_attributes,
                depth=depth + 1,
                in_json_schema=in_json_schema,
            )

    return {"type": _type_name(value)}


def _json_dumps(value: Any) -> str:
    serialized = json.dumps(_to_jsonable(value), default=str, sort_keys=True)
    if len(serialized) <= _MAX_ATTRIBUTE_CHARS:
        return serialized

    def build_wrapper(preview_length: int) -> str:
        return json.dumps(
            {
                "original_length": len(serialized),
                "preview": serialized[:preview_length],
                "truncated": True,
            },
            sort_keys=True,
        )

    lower = 0
    upper = min(len(serialized), _MAX_ATTRIBUTE_CHARS)
    best = build_wrapper(0)
    while lower <= upper:
        middle = (lower + upper) // 2
        candidate = build_wrapper(middle)
        if len(candidate) <= _MAX_ATTRIBUTE_CHARS:
            best = candidate
            lower = middle + 1
        else:
            upper = middle - 1
    return best


def _arg_or_kw(
    args: tuple[Any, ...], kwargs: dict[str, Any], index: int, key: str
) -> Any:
    if key in kwargs:
        return kwargs[key]
    if len(args) > index:
        return args[index]
    return None


def _capture_content() -> bool:
    return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true" or bool(
        context_api.get_value("override_enable_content_tracing")
    )


def _method_input(
    method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    if method_name == "call_tool":
        payload = {
            "name": _arg_or_kw(args, kwargs, 0, "name"),
            "arguments": _arg_or_kw(args, kwargs, 1, "arguments"),
        }
        timeout = _arg_or_kw(args, kwargs, 2, "read_timeout_seconds")
        if timeout is not None:
            payload["read_timeout_seconds"] = timeout
    elif method_name == "read_resource":
        payload = {"uri": _arg_or_kw(args, kwargs, 0, "uri")}
    elif method_name == "get_prompt":
        payload = {
            "name": _arg_or_kw(args, kwargs, 0, "name"),
            "arguments": _arg_or_kw(args, kwargs, 1, "arguments"),
        }
    else:
        payload = {"method": method_name}
        if args:
            payload["args"] = args
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if not callable(value)
        }
        if filtered_kwargs:
            payload["kwargs"] = filtered_kwargs
        return payload
    for key in ("input_responses", "request_state"):
        if kwargs.get(key) is not None:
            payload[key] = kwargs[key]
    return payload


def _entity_name(
    method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str:
    if method_name == "call_tool":
        name = _arg_or_kw(args, kwargs, 0, "name")
        return str(name) if name else _CLIENT_METHODS[method_name].entity_name
    if method_name == "read_resource":
        return _CLIENT_METHODS[method_name].entity_name
    if method_name == "get_prompt":
        name = _arg_or_kw(args, kwargs, 0, "name")
        return str(name) if name else _CLIENT_METHODS[method_name].entity_name
    return _CLIENT_METHODS[method_name].entity_name


def _span_name(method_name: str, entity_name: str) -> str:
    if method_name == "call_tool":
        return f"mcp.tool.{entity_name}"
    if method_name == "read_resource":
        return "mcp.resource.read"
    if method_name == "get_prompt":
        return f"mcp.prompt.{entity_name}"
    return _CLIENT_METHODS[method_name].span_name


class MCPInstrumentor:
    """Respan instrumentor for MCP client operations.

    The upstream OpenInference MCP package propagates trace context through
    MCP transports. This package enables that propagation and adds native
    Respan spans for common ``ClientSession`` calls.
    """

    name = MCP_INSTRUMENTATION_NAME
    _patches_applied = False
    _activation_count = 0
    _shared_delegate: ClassVar[object | None] = None
    _shared_patched_methods: ClassVar[list[str]] = []
    _state_lock: ClassVar[RLock] = RLock()
    _native_patches: ClassVar[list[tuple[Any, Any, Any]]] = []

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        self._instrumentor_kwargs = instrumentor_kwargs
        self._delegate = None
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def _activate_context_propagation(self) -> object | None:
        try:
            openinference_mcp_class = _load_openinference_mcp_class()
        except ImportError as exc:
            logger.warning(
                "Failed to activate MCP context propagation — missing dependency: %s",
                exc,
            )
            return None

        try:
            delegate = OpenInferenceInstrumentor(
                openinference_mcp_class,
                **self._instrumentor_kwargs,
            )
            delegate.activate()
            return delegate
        except Exception:
            logger.exception("Failed to activate MCP context propagation")
            return None

    @classmethod
    def _patch_client_session(cls) -> bool:
        if cls._patches_applied:
            return True

        try:
            session_module = importlib.import_module(MCP_CLIENT_SESSION_MODULE)
        except ImportError as exc:
            logger.warning(
                "Failed to activate MCP instrumentation — missing dependency: %s",
                exc,
            )
            return False

        client_session = getattr(session_module, "ClientSession", None)
        if client_session is None:
            logger.warning("Failed to activate MCP instrumentation — no ClientSession")
            return False

        for method_name in _CLIENT_METHODS:
            if not hasattr(client_session, method_name):
                continue

            async def traced_method(
                wrapped: Callable[..., Any],
                instance: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                _method_name: str = method_name,
            ) -> Any:
                return await cls._trace_async_method(
                    _method_name,
                    wrapped,
                    args,
                    kwargs,
                )

            wrap_function_wrapper(
                MCP_CLIENT_SESSION_MODULE,
                f"ClientSession.{method_name}",
                traced_method,
            )
            cls._shared_patched_methods.append(method_name)

        cls._patches_applied = bool(cls._shared_patched_methods)
        return cls._patches_applied

    @staticmethod
    async def _trace_async_method(
        method_name: str,
        wrapped: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not is_instrumentation_enabled():
            return await wrapped(*args, **kwargs)
        include_content = _capture_content()
        config = _CLIENT_METHODS[method_name]
        entity_name = _entity_name(method_name, args, kwargs)
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span(_span_name(method_name, entity_name)) as span:
            span.set_attribute(RESPAN_LOG_TYPE, config.log_type)
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_name)
            if include_content:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_INPUT,
                    _json_dumps(_method_input(method_name, args, kwargs)),
                )

            try:
                with owned_operation(method_name, entity_name):
                    result = await wrapped(*args, **kwargs)
            except (Exception, asyncio.CancelledError) as exc:
                message = str(exc) or type(exc).__name__
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, message))
                span.set_attribute(ERROR_MESSAGE, message)
                if include_content:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        _json_dumps({"error": type(exc).__name__, "message": message}),
                    )
                raise

            if include_content:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    _json_dumps(result),
                )
            # Tool errors are valid JSON-RPC results, rather than exceptions.
            if method_name == "call_tool" and (
                getattr(result, "is_error", False) or getattr(result, "isError", False)
            ):
                message = "MCP tool returned an error"
                span.set_status(Status(StatusCode.ERROR, message))
                span.set_attribute(ERROR_MESSAGE, message)
            return result

    @classmethod
    def _unpatch_client_session(cls) -> None:
        restore_native_spans(cls._native_patches)
        cls._native_patches.clear()
        for method_name in reversed(cls._shared_patched_methods):
            try:
                unwrap(
                    f"{MCP_CLIENT_SESSION_MODULE}.ClientSession",
                    method_name,
                )
            except Exception:
                logger.debug(
                    "Failed to unwrap MCP ClientSession.%s",
                    method_name,
                    exc_info=True,
                )
        cls._shared_patched_methods.clear()
        cls._patches_applied = False

    def activate(self) -> None:
        """Activate MCP context propagation and native client spans."""
        cls = type(self)
        with cls._state_lock:
            if self._is_instrumented:
                return

            if not self._is_respan_tracing_enabled():
                logger.info(
                    "MCP instrumentation skipped because Respan tracing is disabled"
                )
                return

            if cls._activation_count:
                cls._activation_count += 1
                self._delegate = cls._shared_delegate
                self._is_instrumented = True
                logger.info("MCP instrumentation activation shared")
                return

            delegate = self._activate_context_propagation()
            try:
                patches_applied = cls._patch_client_session()
                cls._native_patches = patch_native_spans()
            except Exception:
                logger.exception("Failed to patch MCP client operations")
                cls._unpatch_client_session()
                if delegate is not None:
                    delegate.deactivate()
                return

            if delegate is None and not patches_applied:
                return

            cls._shared_delegate = delegate
            cls._activation_count = 1
            self._delegate = delegate
            self._is_instrumented = True
        logger.info("MCP instrumentation activated")

    def deactivate(self) -> None:
        """Deactivate MCP context propagation and native client spans."""
        cls = type(self)
        with cls._state_lock:
            if not self._is_instrumented:
                return

            self._delegate = None
            self._is_instrumented = False
            cls._activation_count -= 1
            if cls._activation_count:
                logger.info("MCP instrumentation remains active for another owner")
                return

            cls._unpatch_client_session()

            delegate = cls._shared_delegate
            cls._shared_delegate = None
            if delegate is not None:
                try:
                    delegate.deactivate()
                except Exception:
                    logger.exception("Failed to deactivate MCP context propagation")

        logger.info("MCP instrumentation deactivated")
