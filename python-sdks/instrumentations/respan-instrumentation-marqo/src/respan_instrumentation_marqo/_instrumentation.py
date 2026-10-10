"""Native Marqo instrumentation for Respan."""

from ._native_instrumentation import (
    NativeClientInstrumentor,
    PatchSpec,
)


class MarqoInstrumentor(NativeClientInstrumentor):
    """Trace Marqo operations as canonical Respan spans."""

    name = "marqo"
    vendor = "marqo"
    patches = (
        PatchSpec(
            "marqo.client",
            "Client",
            (
                "bulk_search",
                "create_index",
                "delete_index",
                "get_indexes",
            ),
            label="client",
        ),
        PatchSpec(
            "marqo.index",
            "Index",
            (
                "add_documents",
                "create",
                "delete",
                "delete_documents",
                "eject_model",
                "embed",
                "get_document",
                "get_documents",
                "get_settings",
                "get_stats",
                "get_status",
                "health",
                "recommend",
                "search",
                "update_documents",
            ),
            label="index",
        ),
    )
