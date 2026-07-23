"""Native Pinecone instrumentation for Respan."""

from contextvars import ContextVar
from typing import Any

from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from ._native_instrumentation import (
    NativeClientInstrumentor,
    PatchSpec,
)

_INDEX_OPERATIONS = (
    "cancel_import",
    "create_namespace",
    "delete",
    "delete_namespace",
    "describe_import",
    "describe_index_stats",
    "describe_namespace",
    "fetch",
    "fetch_by_metadata",
    "list",
    "list_imports",
    "list_namespaces",
    "query",
    "query_namespaces",
    "search",
    "search_records",
    "start_import",
    "update",
    "upsert",
    "upsert_from_dataframe",
    "upsert_records",
)


class PineconeInstrumentor(NativeClientInstrumentor):
    """Trace current Pinecone sync, async, gRPC, and resource clients."""

    name = "pinecone"
    vendor = "pinecone"
    _patches_applied = False
    _active_call: ContextVar[bool] = ContextVar(
        "respan_pinecone_active",
        default=False,
    )
    patches = (
        PatchSpec(
            "pinecone.index",
            "Index",
            _INDEX_OPERATIONS,
            label="index",
        ),
        PatchSpec(
            "pinecone.async_client.async_index",
            "AsyncIndex",
            _INDEX_OPERATIONS,
            is_async=True,
            label="index",
        ),
        PatchSpec(
            "pinecone.grpc",
            "GrpcIndex",
            _INDEX_OPERATIONS,
            label="index",
        ),
        PatchSpec(
            "pinecone.client.indexes",
            "Indexes",
            ("configure", "create", "delete", "describe", "exists", "list"),
            label="indexes",
        ),
        PatchSpec(
            "pinecone.client.collections",
            "Collections",
            ("create", "delete", "describe", "list"),
            label="collections",
        ),
        PatchSpec(
            "pinecone.client.backups",
            "Backups",
            ("create", "delete", "describe", "get", "list"),
            label="backups",
        ),
        PatchSpec(
            "pinecone.client.inference",
            "Inference",
            ("embed", "get_model", "list_models", "rerank"),
            label="inference",
        ),
        PatchSpec(
            "pinecone.async_client.indexes",
            "AsyncIndexes",
            ("configure", "create", "delete", "describe", "exists", "list"),
            is_async=True,
            label="indexes",
        ),
        PatchSpec(
            "pinecone.async_client.collections",
            "AsyncCollections",
            ("create", "delete", "describe", "list"),
            is_async=True,
            label="collections",
        ),
        PatchSpec(
            "pinecone.async_client.backups",
            "AsyncBackups",
            ("create", "delete", "describe", "get", "list"),
            is_async=True,
            label="backups",
        ),
        PatchSpec(
            "pinecone.async_client.inference",
            "AsyncInference",
            ("embed", "get_model", "list_models", "rerank"),
            is_async=True,
            label="inference",
        ),
    )

    @classmethod
    def _set_start_attributes(
        cls,
        span: Any,
        operation: str,
        instance: Any,
        wrapped: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        super()._set_start_attributes(
            span,
            operation,
            instance,
            wrapped,
            args,
            kwargs,
        )
        if operation == "inference.embed":
            span.set_attribute(RESPAN_LOG_TYPE, "embedding")
            span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, "embedding")
            model = kwargs.get("model") or (args[0] if args else None)
            if model is not None:
                span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, str(model))
