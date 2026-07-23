"""Native Marqo instrumentation for Respan."""

from contextvars import ContextVar
from typing import Any

from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from ._native_instrumentation import (
    NativeClientInstrumentor,
    PatchSpec,
)


class MarqoInstrumentor(NativeClientInstrumentor):
    """Trace Marqo operations as canonical Respan spans."""

    name = "marqo"
    vendor = "marqo"
    _patches_applied = False
    _active_call: ContextVar[bool] = ContextVar(
        "respan_marqo_active",
        default=False,
    )
    patches = (
        PatchSpec(
            "marqo.client",
            "Client",
            (
                "bulk_search",
                "create_index",
                "delete_index",
                "get_index",
                "get_indexes",
                "index",
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
        if operation == "index.embed":
            span.set_attribute(RESPAN_LOG_TYPE, "embedding")
            span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, "embedding")
