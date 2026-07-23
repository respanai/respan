"""Native Milvus instrumentation for Respan."""

from contextvars import ContextVar

from ._native_instrumentation import (
    NativeClientInstrumentor,
    PatchSpec,
)

_OPERATIONS = (
    "compact",
    "create_collection",
    "create_database",
    "create_index",
    "create_partition",
    "delete",
    "describe_collection",
    "describe_database",
    "describe_index",
    "drop_collection",
    "drop_database",
    "drop_index",
    "drop_partition",
    "flush",
    "get",
    "get_collection_stats",
    "get_load_state",
    "get_partition_stats",
    "has_collection",
    "has_partition",
    "hybrid_search",
    "insert",
    "list_collections",
    "list_databases",
    "list_indexes",
    "list_partitions",
    "load_collection",
    "load_partitions",
    "optimize",
    "query",
    "query_iterator",
    "release_collection",
    "release_partitions",
    "run_analyzer",
    "search",
    "search_iterator",
    "truncate_collection",
    "upsert",
    "use_database",
)


class MilvusInstrumentor(NativeClientInstrumentor):
    """Trace PyMilvus client operations as canonical Respan task spans."""

    name = "milvus"
    vendor = "milvus"
    _patches_applied = False
    _active_call: ContextVar[bool] = ContextVar(
        "respan_milvus_active",
        default=False,
    )
    patches = (
        PatchSpec(
            "pymilvus",
            "MilvusClient",
            _OPERATIONS,
            label="client",
        ),
        PatchSpec(
            "pymilvus",
            "AsyncMilvusClient",
            _OPERATIONS,
            is_async=True,
            label="client",
        ),
    )
