"""AWS SageMaker Runtime instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Iterable, Mapping
from typing import Any

from respan_instrumentation_sagemaker._constants import (
    BODY_KEY,
    SAGEMAKER_INSTRUMENTATION_NAME,
    SAGEMAKER_RUNTIME_SERVICE_NAME,
    STREAMING_OPERATIONS,
    SUPPORTED_OPERATIONS,
)
from respan_instrumentation_sagemaker._otel_emitter import emit_sagemaker_span
from respan_instrumentation_sagemaker._translator import (
    capture_invoke_response_payload,
)
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_original_make_api_call = None


def _load_base_client_class() -> type[Any]:
    module = importlib.import_module("botocore.client")
    base_client = getattr(module, "BaseClient", None)
    if base_client is None:
        raise AttributeError("botocore.client.BaseClient")
    return base_client


def _is_sagemaker_runtime_client(client: Any) -> bool:
    service_model = getattr(getattr(client, "meta", None), "service_model", None)
    return getattr(service_model, "service_name", None) == SAGEMAKER_RUNTIME_SERVICE_NAME


def _status_code_from_response(response: Any) -> int:
    if not isinstance(response, Mapping):
        return 200
    response_metadata = response.get("ResponseMetadata")
    if not isinstance(response_metadata, Mapping):
        return 200
    value = response_metadata.get("HTTPStatusCode")
    return value if isinstance(value, int) else 200


class _InstrumentedEventStream:
    def __init__(
        self,
        *,
        stream: Iterable[Any],
        operation_name: str,
        api_params: Mapping[str, Any] | None,
        start_ns: int,
    ) -> None:
        self._stream = stream
        self._operation_name = operation_name
        self._api_params = api_params
        self._start_ns = start_ns
        self._events: list[Any] = []
        self._emitted = False

    def __iter__(self) -> Iterable[Any]:
        stream_iterable = (
            [self._stream] if isinstance(self._stream, Mapping) else self._stream
        )
        try:
            for event in stream_iterable:
                self._events.append(event)
                yield event
        except Exception as exc:
            self._emit(error_message=str(exc), status_code=500)
            raise
        else:
            self._emit()

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if callable(close):
            close()

    def _emit(
        self, *, error_message: str | None = None, status_code: int = 200
    ) -> None:
        if self._emitted:
            return
        self._emitted = True
        emit_sagemaker_span(
            operation_name=self._operation_name,
            api_params=self._api_params,
            start_ns=self._start_ns,
            stream_events=self._events,
            error_message=error_message,
            status_code=status_code,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _wrap_streaming_response(
    *,
    response: Any,
    operation_name: str,
    api_params: Mapping[str, Any] | None,
    start_ns: int,
) -> Any:
    if not isinstance(response, dict):
        return response

    stream = response.get(BODY_KEY)
    if stream is None:
        emit_sagemaker_span(
            operation_name=operation_name,
            api_params=api_params,
            start_ns=start_ns,
            response_payload=response,
            status_code=_status_code_from_response(response),
        )
        return response

    response[BODY_KEY] = _InstrumentedEventStream(
        stream=stream,
        operation_name=operation_name,
        api_params=api_params,
        start_ns=start_ns,
    )
    return response


def _wrap_make_api_call(original: Any) -> Any:
    def wrapper(
        self: Any, operation_name: str, api_params: Mapping[str, Any] | None = None
    ) -> Any:
        if operation_name not in SUPPORTED_OPERATIONS or not _is_sagemaker_runtime_client(
            self
        ):
            return original(self, operation_name, api_params)

        start_ns = time.time_ns()
        try:
            response = original(self, operation_name, api_params)
        except Exception as exc:
            emit_sagemaker_span(
                operation_name=operation_name,
                api_params=api_params,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise

        if operation_name in STREAMING_OPERATIONS:
            return _wrap_streaming_response(
                response=response,
                operation_name=operation_name,
                api_params=api_params,
                start_ns=start_ns,
            )

        response, response_payload = capture_invoke_response_payload(response)
        if response_payload is None:
            response_payload = response
        emit_sagemaker_span(
            operation_name=operation_name,
            api_params=api_params,
            start_ns=start_ns,
            response_payload=response_payload,
            status_code=_status_code_from_response(response),
        )
        return response

    return wrapper


class SageMakerInstrumentor:
    """Respan instrumentor for the AWS SageMaker Runtime boto3 client."""

    name = SAGEMAKER_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._is_instrumented = False

    def activate(self) -> None:
        """Monkey-patch botocore's SageMaker Runtime call path."""
        global _original_make_api_call

        try:
            base_client = _load_base_client_class()
        except (AttributeError, ImportError) as exc:
            logger.warning(
                "Failed to activate SageMaker instrumentation - missing dependency: %s",
                exc,
            )
            return
        except Exception as exc:
            logger.warning("Failed to activate SageMaker instrumentation: %s", exc)
            return

        if _original_make_api_call is None:
            _original_make_api_call = getattr(base_client, "_make_api_call")
            setattr(
                base_client,
                "_make_api_call",
                _wrap_make_api_call(_original_make_api_call),
            )

        RespanTracer().get_tracer()
        self._is_instrumented = True
        logger.info("SageMaker instrumentation activated")

    def deactivate(self) -> None:
        """Restore botocore's original call path."""
        global _original_make_api_call

        if not self._is_instrumented:
            return

        try:
            base_client = _load_base_client_class()
            if _original_make_api_call is not None:
                setattr(base_client, "_make_api_call", _original_make_api_call)
        except Exception:
            logger.debug("Failed to deactivate SageMaker instrumentation", exc_info=True)
        finally:
            _original_make_api_call = None
            self._is_instrumented = False
            logger.info("SageMaker instrumentation deactivated")
