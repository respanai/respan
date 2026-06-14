"""Chroma instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_instrumentation_chroma._constants import (
    CHROMA_CLIENT_CLASS_NAME,
    CHROMA_CLIENT_MODULE,
    CHROMA_COLLECTION_CLASS_NAME,
    CHROMA_COLLECTION_MODULE,
    CHROMA_INSTRUMENTATION_NAME,
    CLIENT_METHODS,
    COLLECTION_METHODS,
    DATA_KEYS,
    EMBEDDING_KEYS,
    IMAGE_KEYS,
    MAX_ATTRIBUTE_CHARS,
    MAX_PREVIEW_CHARS,
    MAX_PREVIEW_ITEMS,
    OperationConfig,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _preview_text(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= MAX_PREVIEW_CHARS:
        return text
    return f"{text[:MAX_PREVIEW_CHARS]}..."


def _count_items(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        return 1
    if _is_sequence(value) or isinstance(value, Mapping):
        try:
            return len(value)
        except Exception:
            return None
    return 1


def _is_number_sequence(value: Any) -> bool:
    return _is_sequence(value) and all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        for item in value
    )


def _vector_dimensions(value: Any) -> int | None:
    if _is_number_sequence(value):
        return len(value)
    if _is_sequence(value):
        for item in value:
            dimensions = _vector_dimensions(item)
            if dimensions is not None:
                return dimensions
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            shape_values = [int(part) for part in shape]
        except Exception:
            return None
        if shape_values:
            return shape_values[-1]
    return None


def _embedding_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {"count": 0}

    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            shape_values = [int(part) for part in shape]
            summary: dict[str, Any] = {"shape": shape_values}
            if len(shape_values) == 1:
                summary.update({"count": 1, "dimensions": shape_values[0]})
            elif len(shape_values) >= 2:
                summary.update({"count": shape_values[0], "dimensions": shape_values[-1]})
            return summary
        except Exception:
            return {"type": type(value).__name__}

    if _is_number_sequence(value):
        return {"count": 1, "dimensions": len(value)}

    if _is_sequence(value):
        summary = {"count": len(value)}
        dimensions = _vector_dimensions(value)
        if dimensions is not None:
            summary["dimensions"] = dimensions
        return summary

    return {"count": 1, "type": type(value).__name__}


def _media_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {"count": 0}
    count = _count_items(value)
    summary: dict[str, Any] = {"count": count if count is not None else 1}
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            summary["shape"] = [int(part) for part in shape]
        except Exception:
            summary["type"] = type(value).__name__
    return summary


def _to_jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return repr(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}

    if isinstance(value, tuple):
        return [_to_jsonable(item, depth=depth + 1) for item in value]

    if isinstance(value, list):
        if len(value) > MAX_PREVIEW_ITEMS:
            return {
                "count": len(value),
                "items": [
                    _to_jsonable(item, depth=depth + 1)
                    for item in value[:MAX_PREVIEW_ITEMS]
                ],
                "truncated": True,
            }
        return [_to_jsonable(item, depth=depth + 1) for item in value]

    if isinstance(value, Mapping):
        return {
            str(key): _to_jsonable(item, depth=depth + 1)
            for key, item in value.items()
            if not callable(item)
        }

    if hasattr(value, "tolist"):
        try:
            return _to_jsonable(value.tolist(), depth=depth + 1)
        except Exception:
            pass

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_jsonable(
                model_dump(mode="json", exclude_none=True),
                depth=depth + 1,
            )
        except TypeError:
            return _to_jsonable(model_dump(), depth=depth + 1)
        except Exception:
            return repr(value)

    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        try:
            return _to_jsonable(to_dict(), depth=depth + 1)
        except Exception:
            return repr(value)

    name = getattr(value, "name", None)
    identifier = getattr(value, "id", None)
    if name is not None or identifier is not None:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = str(name)
        if identifier is not None:
            payload["id"] = str(identifier)
        return payload

    return repr(value)


def _flatten_text_items(value: Any) -> list[str]:
    items: list[str] = []

    def visit(item: Any) -> None:
        if len(items) > MAX_PREVIEW_ITEMS:
            return
        if isinstance(item, str):
            items.append(_preview_text(item))
            return
        if _is_sequence(item):
            for nested_item in item:
                visit(nested_item)
            return
        items.append(_preview_text(item))

    visit(value)
    return items


def _count_text_items(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return 1
    if _is_sequence(value):
        return sum(_count_text_items(item) for item in value)
    return 1


def _text_items_summary(value: Any) -> Any:
    if value is None:
        return None
    items = _flatten_text_items(value)
    count = _count_text_items(value)
    return {
        "count": count,
        "items": items[:MAX_PREVIEW_ITEMS],
        "truncated": count > MAX_PREVIEW_ITEMS,
    }


def _summarize_arg(key: str, value: Any) -> Any:
    if key in EMBEDDING_KEYS:
        return _embedding_summary(value)
    if key in IMAGE_KEYS or key in DATA_KEYS:
        return _media_summary(value)
    if key in {"embedding_function", "data_loader"}:
        return None if value is None else type(value).__name__
    if key in {"documents", "query_texts"}:
        return _text_items_summary(value)
    return _to_jsonable(value)


def _json_dumps(value: Any) -> str:
    serialized = json.dumps(_to_jsonable(value), default=str, sort_keys=True)
    if len(serialized) <= MAX_ATTRIBUTE_CHARS:
        return serialized
    return f"{serialized[:MAX_ATTRIBUTE_CHARS]}..."


def _arg_or_kw(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    index: int,
    key: str,
) -> Any:
    if key in kwargs:
        return kwargs[key]
    if len(args) > index:
        return args[index]
    return None


def _collection_name(collection: Any) -> str | None:
    name = getattr(collection, "name", None)
    if name is not None:
        return str(name)
    model = getattr(collection, "_model", None)
    model_name = getattr(model, "name", None)
    if model_name is not None:
        return str(model_name)
    return None


def _collection_identity(collection: Any) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    name = _collection_name(collection)
    if name is not None:
        identity["collection_name"] = name
    for attr_name in ("id", "tenant", "database"):
        value = getattr(collection, attr_name, None)
        if value is not None:
            identity[attr_name] = str(value)
    return identity


def _client_method_input(
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {"operation": f"client.{method_name}"}
    if method_name in {
        "create_collection",
        "get_collection",
        "get_or_create_collection",
        "delete_collection",
    }:
        payload["collection_name"] = _arg_or_kw(args, kwargs, 0, "name")
    if method_name in {"create_collection", "get_or_create_collection"}:
        payload["metadata"] = _summarize_arg(
            "metadata",
            _arg_or_kw(args, kwargs, 1, "metadata"),
        )
        payload["embedding_function"] = _summarize_arg(
            "embedding_function",
            kwargs.get("embedding_function"),
        )
        payload["data_loader"] = _summarize_arg("data_loader", kwargs.get("data_loader"))
    return {key: value for key, value in payload.items() if value is not None}


def _collection_method_input(
    collection: Any,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "operation": f"collection.{method_name}",
        **_collection_identity(collection),
    }

    field_order_by_method = {
        "add": ("ids", "embeddings", "metadatas", "documents", "images", "uris"),
        "upsert": ("ids", "embeddings", "metadatas", "documents", "images", "uris"),
        "update": ("ids", "embeddings", "metadatas", "documents", "images", "uris"),
        "query": (
            "query_embeddings",
            "query_texts",
            "query_images",
            "query_uris",
            "ids",
            "n_results",
            "where",
            "where_document",
            "include",
        ),
        "get": ("ids", "where", "limit", "offset", "where_document", "include"),
        "delete": ("ids", "where", "where_document"),
        "peek": ("limit",),
        "modify": ("name", "metadata", "configuration"),
        "fork": ("new_name",),
        "search": ("searches",),
    }

    for index, field_name in enumerate(field_order_by_method.get(method_name, ())):
        value = _arg_or_kw(args, kwargs, index, field_name)
        if value is not None:
            payload[field_name] = _summarize_arg(field_name, value)
    return payload


def _summarize_chroma_result(result: Any) -> Any:
    if result is None:
        return {"status": "ok"}

    if isinstance(result, Mapping):
        payload: dict[str, Any] = {}
        for key, value in result.items():
            key_str = str(key)
            if key_str in EMBEDDING_KEYS:
                payload[key_str] = _embedding_summary(value)
            elif key_str in IMAGE_KEYS or key_str in DATA_KEYS:
                payload[key_str] = _media_summary(value)
            elif key_str == "documents":
                payload[key_str] = _text_items_summary(value)
            else:
                payload[key_str] = _to_jsonable(value)
        return payload

    if _is_sequence(result):
        return {
            "count": len(result),
            "items": [_to_jsonable(item) for item in result[:MAX_PREVIEW_ITEMS]],
            "truncated": len(result) > MAX_PREVIEW_ITEMS,
        }

    if isinstance(result, (str, int, float, bool)):
        return result

    identity = _collection_identity(result)
    if identity:
        return identity
    return _to_jsonable(result)


class ChromaInstrumentor:
    """Respan instrumentor for Chroma client and collection operations."""

    name = CHROMA_INSTRUMENTATION_NAME
    _patches_applied = False

    def __init__(self) -> None:
        self._is_instrumented = False
        self._patched_methods: list[tuple[str, str]] = []

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def _patch_methods(
        self,
        *,
        module_path: str,
        class_name: str,
        methods: dict[str, OperationConfig],
        is_collection: bool,
    ) -> bool:
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            logger.warning(
                "Failed to activate Chroma instrumentation - missing dependency: %s",
                exc,
            )
            return False

        target_class = getattr(module, class_name, None)
        if target_class is None:
            logger.warning(
                "Failed to activate Chroma instrumentation - no %s.%s",
                module_path,
                class_name,
            )
            return False

        patched = False
        for method_name in methods:
            if not hasattr(target_class, method_name):
                continue

            def traced_method(
                wrapped: Callable[..., Any],
                instance: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                _method_name: str = method_name,
                _is_collection: bool = is_collection,
            ) -> Any:
                return self._trace_method(
                    _method_name,
                    wrapped,
                    instance,
                    args,
                    kwargs,
                    is_collection=_is_collection,
                )

            wrap_function_wrapper(module_path, f"{class_name}.{method_name}", traced_method)
            self._patched_methods.append((module_path, f"{class_name}.{method_name}"))
            patched = True

        return patched

    def _trace_method(
        self,
        method_name: str,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        is_collection: bool,
    ) -> Any:
        method_configs = COLLECTION_METHODS if is_collection else CLIENT_METHODS
        config = method_configs[method_name]
        tracer = trace.get_tracer(__name__)
        entity_name = self._entity_name(config, instance, method_name, is_collection)

        with tracer.start_as_current_span(config.span_name) as span:
            span.set_attribute(RESPAN_LOG_TYPE, config.log_type)
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_name)
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                _json_dumps(
                    _collection_method_input(instance, method_name, args, kwargs)
                    if is_collection
                    else _client_method_input(method_name, args, kwargs)
                ),
            )

            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    _json_dumps(
                        {
                            "error": type(exc).__name__,
                            "message": str(exc),
                        }
                    ),
                )
                raise

            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                _json_dumps(_summarize_chroma_result(result)),
            )
            return result

    def _entity_name(
        self,
        config: OperationConfig,
        instance: Any,
        method_name: str,
        is_collection: bool,
    ) -> str:
        if not is_collection:
            return config.entity_name
        collection_name = _collection_name(instance)
        if collection_name:
            return f"{collection_name}.{method_name}"
        return config.entity_name

    def activate(self) -> None:
        """Activate native Chroma client and collection spans."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info("Chroma instrumentation skipped because Respan tracing is disabled")
            return

        if ChromaInstrumentor._patches_applied:
            self._is_instrumented = True
            return

        client_patched = self._patch_methods(
            module_path=CHROMA_CLIENT_MODULE,
            class_name=CHROMA_CLIENT_CLASS_NAME,
            methods=CLIENT_METHODS,
            is_collection=False,
        )
        collection_patched = self._patch_methods(
            module_path=CHROMA_COLLECTION_MODULE,
            class_name=CHROMA_COLLECTION_CLASS_NAME,
            methods=COLLECTION_METHODS,
            is_collection=True,
        )

        if not client_patched and not collection_patched:
            self._patched_methods.clear()
            return

        ChromaInstrumentor._patches_applied = True
        self._is_instrumented = True
        logger.info("Chroma instrumentation activated")

    def deactivate(self) -> None:
        """Deactivate native Chroma client and collection spans."""
        for module_path, target in reversed(self._patched_methods):
            try:
                unwrap(module_path, target)
            except Exception:
                logger.debug(
                    "Failed to unwrap Chroma %s.%s",
                    module_path,
                    target,
                    exc_info=True,
                )
        self._patched_methods.clear()
        ChromaInstrumentor._patches_applied = False
        self._is_instrumented = False
        logger.info("Chroma instrumentation deactivated")
