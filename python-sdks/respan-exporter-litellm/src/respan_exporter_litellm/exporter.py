"""Respan LiteLLM Integration.

RespanLiteLLMCallback - LiteLLM-native callback class for sending traces to Respan.
"""

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://api.respan.ai/api/v1/traces/ingest"


def _format_span_id(raw_id: str) -> str:
    """Format span ID to 16-char hex for Respan compatibility."""
    return hashlib.md5(raw_id.encode()).hexdigest()[:16]


try:
    from litellm.integrations.custom_logger import CustomLogger as LiteLLMCustomLogger
except Exception:  # pragma: no cover - litellm is optional at runtime
    class LiteLLMCustomLogger:  # type: ignore[no-redef]
        """Fallback base class when litellm is unavailable."""


class NamedCallbackRegistry(list):
    """List-like callback registry with name-based access.

    Supports dictionary-style assignment:
        litellm.success_callback["respan"] = callback.log_success_event
    while remaining compatible with list operations expected by LiteLLM.
    """

    def __init__(self, iterable: Optional[List[Any]] = None) -> None:
        super().__init__(iterable or [])
        self._named: Dict[str, Any] = {}

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._named[key]
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            self._named[key] = value
            if value not in self:
                super().append(value)
            return
        super().__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        if isinstance(key, str):
            value = self._named.pop(key)
            if value not in self._named.values():
                try:
                    super().remove(value)
                except ValueError:
                    pass
            return
        removed = self[key]
        super().__delitem__(key)
        self._drop_named_values(removed)

    def pop(self, key: Any = -1) -> Any:
        if isinstance(key, str):
            value = self._named.pop(key)
            if value not in self._named.values():
                try:
                    super().remove(value)
                except ValueError:
                    pass
            return value
        value = super().pop(key)
        self._drop_named_value(value)
        return value

    def remove(self, value: Any) -> None:
        super().remove(value)
        self._drop_named_value(value)

    def clear(self) -> None:
        super().clear()
        self._named.clear()

    def _drop_named_values(self, removed: Any) -> None:
        if isinstance(removed, list):
            for value in removed:
                self._drop_named_value(value)
        else:
            self._drop_named_value(removed)

    def _drop_named_value(self, value: Any) -> None:
        for name, cb in list(self._named.items()):
            if cb is value:
                del self._named[name]


def _ensure_named_callback_registry(callbacks: Any) -> NamedCallbackRegistry:
    """Convert callbacks to NamedCallbackRegistry if needed."""
    if isinstance(callbacks, NamedCallbackRegistry):
        return callbacks
    if callbacks is None:
        callbacks = []
    return NamedCallbackRegistry(list(callbacks))


def _enable_named_callbacks() -> None:
    """Enable name-based access on LiteLLM callback lists."""
    try:
        import litellm
    except Exception:
        return

    litellm.success_callback = _ensure_named_callback_registry(litellm.success_callback)
    litellm.failure_callback = _ensure_named_callback_registry(litellm.failure_callback)
    
    if hasattr(litellm, "_async_success_callback"):
        litellm._async_success_callback = _ensure_named_callback_registry(
            litellm._async_success_callback
        )
    if hasattr(litellm, "_async_failure_callback"):
        litellm._async_failure_callback = _ensure_named_callback_registry(
            litellm._async_failure_callback
        )

class RespanLiteLLMCallback(LiteLLMCustomLogger):
    """LiteLLM callback that sends traces to Respan.
    
    Usage:
        callback = RespanLiteLLMCallback(api_key="...")
        callback.register_litellm_callbacks()
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 10,
    ):
        super().__init__()
        _enable_named_callbacks()
        self.api_key = api_key or os.getenv("RESPAN_API_KEY")
        self.endpoint = endpoint or os.getenv("RESPAN_ENDPOINT", DEFAULT_ENDPOINT)
        self.timeout = timeout
        if not self.api_key:
            logger.warning("Respan API key not provided")

    def register_litellm_callbacks(self, name: str = "respan") -> None:
        """Register success/failure callbacks on LiteLLM by name."""
        try:
            import litellm
        except Exception as exc:
            raise RuntimeError("litellm must be installed to register callbacks") from exc

        _enable_named_callbacks()
        litellm.success_callback[name] = self.log_success_event
        litellm.failure_callback[name] = self.log_failure_event

    # -------------------------------------------------------------------------
    # Callback methods (called by LiteLLM)
    # -------------------------------------------------------------------------
    
    def log_success_event(
        self, kwargs: Dict, response_obj: Any, start_time: datetime, end_time: datetime
    ) -> None:
        """Log successful completion."""
        self._log_event(kwargs, response_obj, start_time, end_time, error=None)
    
    async def async_log_success_event(
        self, kwargs: Dict, response_obj: Any, start_time: datetime, end_time: datetime
    ) -> None:
        """Async log successful completion."""
        threading.Thread(
            target=self._log_event,
            args=(kwargs, response_obj, start_time, end_time, None),
        ).start()
    
    def log_failure_event(
        self, kwargs: Dict, response_obj: Any, start_time: datetime, end_time: datetime
    ) -> None:
        """Log failed completion."""
        error = kwargs.get("exception") or kwargs.get("traceback_exception")
        self._log_event(kwargs, response_obj, start_time, end_time, error=error)
    
    async def async_log_failure_event(
        self, kwargs: Dict, response_obj: Any, start_time: datetime, end_time: datetime
    ) -> None:
        """Async log failed completion."""
        error = kwargs.get("exception") or kwargs.get("traceback_exception")
        threading.Thread(
            target=self._log_event,
            args=(kwargs, response_obj, start_time, end_time, error),
        ).start()

    # -------------------------------------------------------------------------
    # Core logging logic
    # -------------------------------------------------------------------------
    
    def _log_event(
        self,
        kwargs: Dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
        error: Optional[Exception],
    ) -> None:
        """Send event to Respan."""
        if not self.api_key:
            return
        
        try:
            if kwargs.get("stream") and error is None:
                complete_streaming_response = kwargs.get("complete_streaming_response")
                if complete_streaming_response is None:
                    return
                response_obj = complete_streaming_response

            # Extract basic info
            model = kwargs.get("model") or kwargs.get("litellm_params", {}).get("model")
            messages = kwargs.get("messages", [])
            metadata = kwargs.get("litellm_params", {}).get("metadata", {}) or {}
            kw_params = metadata.get("respan_params", {})
            workflow_name = kw_params.get("workflow_name", "litellm")
            
            # Build payload
            payload = {
                "span_name": kw_params.get("span_name", "litellm.completion"),
                "span_workflow_name": workflow_name,
                "log_type": "generation",
                "timestamp": end_time.astimezone(timezone.utc).isoformat(),
                "start_time": start_time.astimezone(timezone.utc).isoformat(),
                "latency": (end_time - start_time).total_seconds(),
                "model": model,
                "input": json.dumps(messages),
                "stream": kwargs.get("stream", False),
            }
            
            # Add trace info
            trace_id = kw_params.get("trace_id")
            parent_id = kw_params.get("parent_span_id")
            
            if trace_id:
                payload["trace_unique_id"] = trace_id
                payload["trace_name"] = kw_params.get("trace_name", workflow_name)
            
            # Add span_unique_id
            self._add_span_id(payload, kw_params, trace_id, parent_id, response_obj, kwargs)
            
            if parent_id:
                payload["span_parent_id"] = parent_id
            
            # Add tools if present
            if kwargs.get("tools"):
                payload["tools"] = kwargs["tools"]
            if kwargs.get("tool_choice"):
                payload["tool_choice"] = kwargs["tool_choice"]
            
            # Add status
            if error:
                payload["status"] = "error"
                payload["error_message"] = str(error)
            else:
                payload["status"] = "success"
            
            # Add custom Respan params
            self._add_respan_params(payload, kw_params)
            
            self._send([payload])
        except Exception as e:
            logger.error(f"Respan logging error: {e}")

    def _add_span_id(
        self,
        payload: Dict,
        kw_params: Dict,
        trace_id: Optional[str],
        parent_id: Optional[str],
        response_obj: Any,
        kwargs: Dict,
    ) -> None:
        """Add span_unique_id to payload."""
        span_id = kw_params.get("span_id")
        if span_id:
            payload["span_unique_id"] = span_id
        elif trace_id and not parent_id:
            # Root span: use trace_id as span_unique_id (Respan convention)
            payload["span_unique_id"] = trace_id
        else:
            # Child span: derive from response.id or litellm_call_id
            raw_id = None
            if response_obj:
                raw_id = getattr(response_obj, "id", None)
                if not raw_id and hasattr(response_obj, "get"):
                    raw_id = response_obj.get("id")
            if not raw_id:
                raw_id = kwargs.get("litellm_call_id")
            if raw_id:
                payload["span_unique_id"] = _format_span_id(str(raw_id))

    def _add_respan_params(self, payload: Dict, kw_params: Dict) -> None:
        """Add Respan-specific params to payload."""
        extra_meta = {}
        
        # Customer identifier
        if "customer_identifier" in kw_params:
            payload["customer_identifier"] = kw_params["customer_identifier"]
        if cp := kw_params.get("customer_params"):
            if isinstance(cp, dict):
                payload["customer_identifier"] = cp.get("customer_identifier")
                extra_meta.update({
                    f"customer_{k}": v for k, v in cp.items() if k != "customer_identifier"
                })
        
        # Session identifier
        if "session_identifier" in kw_params:
            payload["session_identifier"] = kw_params["session_identifier"]

        # Thread identifier
        if "thread_identifier" in kw_params:
            payload["thread_identifier"] = kw_params["thread_identifier"]

        # Custom metadata
        if m := kw_params.get("metadata"):
            if isinstance(m, dict):
                extra_meta.update(m)
        
        # Add remaining params as metadata
        excluded = {
            "customer_identifier", "customer_params", "session_identifier", "thread_identifier", "metadata",
            "workflow_name", "trace_id", "trace_name", "span_id", "parent_span_id", "span_name",
        }
        extra_meta.update({k: v for k, v in kw_params.items() if k not in excluded})
        
        if extra_meta:
            payload["metadata"] = extra_meta
    
    def _send(self, payloads: List[Dict[str, Any]]) -> None:
        """Send payloads to Respan."""
        try:
            response = requests.post(
                self.endpoint,
                json=payloads,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
            if response.status_code != 200:
                logger.warning(f"Respan error: {response.status_code}")
        except Exception as e:
            logger.error(f"Respan request error: {e}")
