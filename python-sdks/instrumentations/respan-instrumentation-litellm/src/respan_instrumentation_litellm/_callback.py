"""LiteLLM CustomLogger callback that emits Respan spans."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from opentelemetry import trace

from respan_instrumentation_litellm._constants import (
    COMPLETE_STREAMING_RESPONSE_KEY,
    EXCEPTION_KEY,
    METADATA_KEY,
    RESPAN_SKIP_CALLBACK_KEY,
    STREAM_KEY,
    TRACEBACK_EXCEPTION_KEY,
)
from respan_instrumentation_litellm._translator import build_litellm_span_data
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
)
from respan_tracing.utils.span_factory import build_readable_span
from respan_tracing.utils.span_factory import inject_span

try:
    from litellm.integrations.custom_logger import CustomLogger
except Exception:  # pragma: no cover - LiteLLM is optional at import time

    class CustomLogger:  # type: ignore[no-redef]
        """Fallback base class used when LiteLLM is not installed."""


logger = logging.getLogger(__name__)


def _metadata_value(kwargs: dict[str, Any], key: str) -> Any:
    metadata = kwargs.get(METADATA_KEY)
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]

    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, dict):
        litellm_metadata = litellm_params.get(METADATA_KEY)
        if isinstance(litellm_metadata, dict):
            return litellm_metadata.get(key)
    return None


def _time_to_ns(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(value.timestamp() * 1_000_000_000)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(float(value) * 1_000_000_000)
    return None


def _current_otel_parent() -> tuple[str | None, str | None]:
    current_span = trace.get_current_span()
    try:
        span_context = current_span.get_span_context()
    except Exception:
        return None, None

    trace_id = getattr(span_context, "trace_id", 0)
    span_id = getattr(span_context, "span_id", 0)
    if not isinstance(trace_id, int) or not isinstance(span_id, int):
        return None, None
    if trace_id == 0 or span_id == 0:
        return None, None
    return format_trace_id(trace_id=trace_id), format_span_id(span_id=span_id)


class RespanLiteLLMCallback(CustomLogger):
    """LiteLLM callback handler that emits canonical Respan spans."""

    def __init__(self, *, include_content: bool = True) -> None:
        super().__init__()
        self._include_content = include_content

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Emit a span for a successful LiteLLM call."""
        self._emit_event(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            error=None,
        )

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Emit a span for a successful async LiteLLM call."""
        self.log_success_event(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
        )

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Emit a span for a failed LiteLLM call."""
        error = kwargs.get(EXCEPTION_KEY) or kwargs.get(TRACEBACK_EXCEPTION_KEY)
        self._emit_event(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            error=error if isinstance(error, Exception) else Exception(str(error)),
        )

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Emit a span for a failed async LiteLLM call."""
        self.log_failure_event(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
        )

    def _emit_event(
        self,
        *,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
        error: Exception | None,
        parent_context: tuple[str | None, str | None] | None = None,
    ) -> None:
        try:
            if _metadata_value(kwargs=kwargs, key=RESPAN_SKIP_CALLBACK_KEY):
                return

            if kwargs.get(STREAM_KEY) and error is None:
                response_obj = (
                    kwargs.get(COMPLETE_STREAMING_RESPONSE_KEY) or response_obj
                )

            span_name, attributes = build_litellm_span_data(
                kwargs=kwargs,
                response_obj=response_obj,
                error=error,
                include_content=self._include_content,
            )
            trace_id, parent_id = parent_context or _current_otel_parent()
            now_ns = time.time_ns()
            start_time_ns = _time_to_ns(start_time) or now_ns
            end_time_ns = _time_to_ns(end_time) or now_ns
            span = build_readable_span(
                name=span_name,
                trace_id=trace_id,
                parent_id=parent_id,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                attributes=attributes,
                status_code=500 if error is not None else 200,
                error_message=str(error) if error is not None else None,
            )
            inject_span(span=span)
        except Exception:
            logger.debug("Failed to emit LiteLLM span", exc_info=True)
