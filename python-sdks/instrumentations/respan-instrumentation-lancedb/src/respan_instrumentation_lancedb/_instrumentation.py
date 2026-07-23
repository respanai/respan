"""Native LanceDB instrumentation for Respan."""

from contextvars import ContextVar

from ._native_instrumentation import (
    NativeClientInstrumentor,
    PatchSpec,
)


class LanceDBInstrumentor(NativeClientInstrumentor):
    """Trace LanceDB operations as canonical Respan task spans."""

    name = "lancedb"
    vendor = "lancedb"
    _patches_applied = False
    _active_call: ContextVar[bool] = ContextVar(
        "respan_lancedb_active",
        default=False,
    )
    patches = (
        PatchSpec(
            "lancedb.db",
            "LanceDBConnection",
            ("create_table", "open_table", "drop_table", "table_names"),
            label="connection",
        ),
        PatchSpec(
            "lancedb.table",
            "LanceTable",
            (
                "add",
                "delete",
                "update",
                "merge_insert",
                "create_index",
                "optimize",
                "search",
            ),
            label="table",
        ),
        PatchSpec(
            "lancedb.query",
            "LanceQueryBuilder",
            ("to_list", "to_arrow", "to_pandas", "explain_plan"),
            label="query",
        ),
        PatchSpec(
            "lancedb.query",
            "LanceEmptyQueryBuilder",
            ("to_list", "to_arrow", "to_pandas", "explain_plan"),
            label="query",
        ),
        PatchSpec(
            "lancedb.query",
            "LanceFtsQueryBuilder",
            ("to_list", "to_arrow", "to_pandas", "explain_plan"),
            label="query",
        ),
        PatchSpec(
            "lancedb.query",
            "LanceHybridQueryBuilder",
            ("to_list", "to_arrow", "to_pandas", "explain_plan"),
            label="query",
        ),
        PatchSpec(
            "lancedb.query",
            "LanceTakeQueryBuilder",
            ("to_list", "to_arrow", "to_pandas", "explain_plan"),
            label="query",
        ),
        PatchSpec(
            "lancedb.query",
            "LanceVectorQueryBuilder",
            ("to_list", "to_arrow", "to_pandas", "explain_plan"),
            label="query",
        ),
        PatchSpec(
            "lancedb.db",
            "AsyncConnection",
            ("create_table", "open_table", "drop_table", "table_names"),
            is_async=True,
            label="connection",
        ),
        PatchSpec(
            "lancedb.table",
            "AsyncTable",
            (
                "add",
                "delete",
                "update",
                "merge_insert",
                "create_index",
                "optimize",
                "search",
            ),
            is_async=True,
            label="table",
        ),
        PatchSpec(
            "lancedb.query",
            "AsyncQueryBase",
            ("to_list", "to_arrow", "to_pandas", "explain_plan"),
            is_async=True,
            label="query",
        ),
        PatchSpec(
            "lancedb.query",
            "AsyncHybridQuery",
            ("explain_plan",),
            is_async=True,
            label="query",
        ),
    )
