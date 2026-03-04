"""Instrumentor for Helicone Manual Logger.

This module intercepts `HeliconeManualLogger.send_log` and sends logs to both
Helicone and Respan.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import requests
import wrapt
from respan_sdk.constants.api_constants import DEFAULT_TRACES_INGEST_ENDPOINT
from respan_sdk.constants.llm_logging import LOG_TYPE_GENERATION, LogMethodChoices
from respan_sdk.respan_types.log_types import RespanFullLogParams
from respan_sdk.utils.time import format_timestamp

try:
    from helicone_helpers import HeliconeManualLogger, manual_logger
except ImportError:
    HeliconeManualLogger = None
    manual_logger = None

logger = logging.getLogger(__name__)

HELICONE_LOGGER_MODULE = "helicone_helpers.manual_logger"
HELICONE_SEND_LOG_FUNCTION = "HeliconeManualLogger.send_log"
HELICONE_HEADER_PREFIX = "helicone-"
HELICONE_USER_ID_HEADER = "helicone-user-id"
HELICONE_SESSION_ID_HEADER = "helicone-session-id"

RESPAN_API_KEY_ENV = "RESPAN_API_KEY"
RESPAN_ENDPOINT_ENV = "RESPAN_ENDPOINT"
RESPAN_TRACING_LOG_METHOD = LogMethodChoices.TRACING_INTEGRATION.value
RESPAN_SUCCESS_STATUS_CODES = (200, 201)

AUTHORIZATION_HEADER = "Authorization"
CONTENT_TYPE_HEADER = "Content-Type"
JSON_CONTENT_TYPE = "application/json"
DEFAULT_TIMEOUT_SECONDS = 10

JsonDict = Dict[str, Any]
ResponsePayload = Union[JsonDict, str, None]


class HeliconeInstrumentor:
    """Intercept Helicone logs and forward them to Respan."""

    def __init__(self) -> None:
        self._api_key: Optional[str] = None
        self._endpoint: Optional[str] = None
        self._timeout = DEFAULT_TIMEOUT_SECONDS
        self._is_patched = False
        self._original_send_log: Optional[Any] = None

    def instrument(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Enable instrumentation.

        Args:
            api_key: Respan API key (uses RESPAN_API_KEY env var if not set)
            endpoint: Respan ingest endpoint
            timeout: Network timeout for Respan API requests
        """
        self._api_key = api_key or os.getenv(RESPAN_API_KEY_ENV)
        self._endpoint = endpoint or os.getenv(
            RESPAN_ENDPOINT_ENV,
            DEFAULT_TRACES_INGEST_ENDPOINT,
        )
        self._timeout = timeout if timeout > 0 else DEFAULT_TIMEOUT_SECONDS

        if not self._api_key:
            logger.warning(
                "RESPAN_API_KEY is not set. Helicone logs will not be exported to Respan."
            )
            return

        self._patch()
        logger.info("Helicone instrumentation enabled for Respan")

    def uninstrument(self) -> None:
        """Disable instrumentation by restoring the original send_log."""
        if self._is_patched and manual_logger is not None:
            try:
                if (
                    self._original_send_log is not None
                    and hasattr(manual_logger.HeliconeManualLogger, "send_log")
                ):
                    manual_logger.HeliconeManualLogger.send_log = self._original_send_log
            except Exception as exc:
                logger.warning("Failed to restore Helicone send_log: %s", exc)
        self._is_patched = False
        self._original_send_log = None
        logger.info("Helicone instrumentation disabled")

    def _patch(self) -> None:
        """Patch `HeliconeManualLogger.send_log` to intercept data."""
        if self._is_patched:
            return

        if HeliconeManualLogger is None or manual_logger is None:
            logger.error("helicone-helpers is not installed. Cannot instrument Helicone.")
            return

        self._original_send_log = getattr(
            manual_logger.HeliconeManualLogger, "send_log", None
        )
        if self._original_send_log is None:
            logger.error("HeliconeManualLogger.send_log not found. Cannot instrument.")
            return

        def send_log_wrapper(
            wrapped: Any,
            _instance: Any,
            args: Tuple[Any, ...],
            kwargs: JsonDict,
        ) -> Any:
            """Wrapper for send_log that intercepts log events."""
            provider, request, response, options = self._resolve_send_log_args(
                args=args,
                kwargs=kwargs,
            )

            # Always call original send_log exactly once so Helicone behavior is preserved.
            result = wrapped(*args, **kwargs)

            if self._api_key and isinstance(request, dict):
                self._start_async_export(
                    provider=provider,
                    request=request,
                    response=response,
                    options=options,
                )

            return result

        wrapt.wrap_function_wrapper(
            module=HELICONE_LOGGER_MODULE,
            name=HELICONE_SEND_LOG_FUNCTION,
            wrapper=send_log_wrapper,
        )
        self._is_patched = True

    def _start_async_export(
        self,
        provider: Optional[str],
        request: JsonDict,
        response: ResponsePayload,
        options: Optional[JsonDict],
    ) -> None:
        """Send log export in a daemon thread to avoid blocking Helicone."""
        try:
            threading.Thread(
                target=self._send_to_respan,
                kwargs={
                    "provider": provider,
                    "request": request,
                    "response": response,
                    "options": options,
                },
                daemon=True,
            ).start()
        except Exception as exc:
            logger.error("Error in Helicone wrapper: %s", exc)

    @staticmethod
    def _resolve_send_log_args(
        args: Tuple[Any, ...],
        kwargs: JsonDict,
    ) -> Tuple[Optional[str], Optional[JsonDict], ResponsePayload, Optional[JsonDict]]:
        """Resolve `send_log` args for positional and keyword invocation styles.

        Tightly coupled to HeliconeManualLogger.send_log(provider, request, response, options).
        Tested with helicone-helpers >=1.0.0. Update this resolver if Helicone changes the
        send_log signature.
        """
        provider = kwargs.get("provider", args[0] if len(args) > 0 else None)
        request = kwargs.get("request", args[1] if len(args) > 1 else None)
        response = kwargs.get("response", args[2] if len(args) > 2 else None)
        options = kwargs.get("options", args[3] if len(args) > 3 else None)

        normalized_request = request if isinstance(request, dict) else None
        normalized_options = options if isinstance(options, dict) else None

        return provider, normalized_request, response, normalized_options

    def _send_to_respan(
        self,
        provider: Optional[str],
        request: JsonDict,
        response: ResponsePayload,
        options: Optional[JsonDict],
    ) -> None:
        """Transform Helicone log and send it to Respan API."""
        if not self._endpoint:
            logger.warning("Respan endpoint is not configured. Skipping Helicone export.")
            return

        try:
            options_data = options if isinstance(options, dict) else {}
            payload = self._build_payload(
                provider=provider,
                request=request,
                response=response,
                options=options_data,
            )
            response_obj = requests.post(
                url=self._endpoint,
                json=[payload],
                headers=self._build_headers(),
                timeout=self._timeout,
            )

            if response_obj.status_code not in RESPAN_SUCCESS_STATUS_CODES:
                logger.warning(
                    "Respan export failed: %s - %s",
                    response_obj.status_code,
                    response_obj.text,
                )
        except Exception as exc:
            logger.error("Failed to send Helicone log to Respan: %s", exc, exc_info=True)

    def _build_payload(
        self,
        provider: Optional[str],
        request: JsonDict,
        response: ResponsePayload,
        options: JsonDict,
    ) -> JsonDict:
        """Build and validate the Respan payload from Helicone request data."""
        start_ts = self._extract_timestamp(options=options, key="start_time")
        end_ts = self._extract_timestamp(options=options, key="end_time")

        payload: JsonDict = {
            "log_method": RESPAN_TRACING_LOG_METHOD,
            "log_type": LOG_TYPE_GENERATION,
            "start_time": format_timestamp(ts=start_ts),
            "timestamp": format_timestamp(ts=end_ts),
            "latency": self._calculate_latency(start_ts=start_ts, end_ts=end_ts),
            "provider": provider,
            "provider_id": provider,
            "model": request.get("model"),
            "input": self._extract_input(request=request),
        }

        output_value = self._extract_output(response=response)
        if output_value is not None:
            payload["output"] = output_value

        payload.update(self._extract_usage(response=response))

        metadata = self._extract_helicone_metadata(options=options)
        if metadata:
            payload["metadata"] = metadata
            normalized_metadata = {
                key.lower(): value for key, value in metadata.items()
            }
            if HELICONE_USER_ID_HEADER in normalized_metadata:
                payload["customer_identifier"] = normalized_metadata[HELICONE_USER_ID_HEADER]
            if HELICONE_SESSION_ID_HEADER in normalized_metadata:
                payload["session_identifier"] = normalized_metadata[HELICONE_SESSION_ID_HEADER]

        validated_payload = RespanFullLogParams.model_validate(payload).model_dump(
            mode="json",
            exclude_none=True,
        )
        return validated_payload

    @staticmethod
    def _extract_timestamp(options: JsonDict, key: str) -> Optional[float]:
        """Extract numeric timestamps from Helicone options."""
        timestamp = options.get(key)
        if isinstance(timestamp, bool):
            return None
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
        return None

    @staticmethod
    def _calculate_latency(
        start_ts: Optional[float],
        end_ts: Optional[float],
    ) -> Optional[float]:
        """Return elapsed seconds when both timestamps are available."""
        if start_ts is None or end_ts is None:
            return None
        return max(0.0, end_ts - start_ts)

    @staticmethod
    def _extract_input(request: JsonDict) -> Optional[str]:
        """Extract request input for Respan payload.

        Prefers "messages" or "prompt" when present. Fallback serializes the entire
        request with json.dumps(); this can produce very large payloads (e.g. base64
        images). Output is truncated to 100k chars to avoid oversized ingest.
        """
        if "messages" in request:
            out = json.dumps(request["messages"], default=str)
        elif "prompt" in request:
            prompt_value = request.get("prompt")
            if prompt_value is None:
                return None
            # Ensure prompt is a string for truncation (OpenAI allows string or array)
            out = (
                prompt_value
                if isinstance(prompt_value, str)
                else json.dumps(prompt_value, default=str)
            )
        else:
            out = json.dumps(request, default=str)
        max_len = 100_000
        if len(out) > max_len:
            return out[:max_len] + f"... [truncated, total {len(out)} chars]"
        return out

    @staticmethod
    def _extract_output(response: ResponsePayload) -> Optional[str]:
        """Extract response output for Respan payload."""
        if isinstance(response, dict):
            if "choices" in response:
                return json.dumps(response["choices"], default=str)
            return json.dumps(response, default=str)
        if isinstance(response, str):
            return response
        return None

    @staticmethod
    def _extract_usage(response: ResponsePayload) -> JsonDict:
        """Extract token usage from Helicone response payload."""
        if not isinstance(response, dict):
            return {}

        usage = response.get("usage")
        if not isinstance(usage, dict):
            return {}

        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_request_tokens": usage.get("total_tokens"),
        }

    @staticmethod
    def _extract_helicone_metadata(options: JsonDict) -> Dict[str, Any]:
        """Extract Helicone-prefixed headers into metadata."""
        metadata: Dict[str, Any] = {}
        additional_headers = options.get("additional_headers", {})
        if not isinstance(additional_headers, Mapping):
            return metadata

        for key, value in additional_headers.items():
            if isinstance(key, str) and key.lower().startswith(HELICONE_HEADER_PREFIX):
                metadata[key] = value
        return metadata

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers for the Respan ingest endpoint."""
        return {
            AUTHORIZATION_HEADER: f"Bearer {self._api_key}",
            CONTENT_TYPE_HEADER: JSON_CONTENT_TYPE,
        }
