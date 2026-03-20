"""
RespanSpanExporterV2 — self-contained batching HTTP exporter for raw log dicts.

Used by plugin instrumentations (e.g. OpenAI Agents SDK) to send spans to
the /v1/traces/ingest endpoint. Unlike the OTEL-based RespanSpanExporter
(which handles ReadableSpan objects via the OTLP pipeline), this exporter
accepts plain Python dicts and manages its own batching and retry logic.
"""

import atexit
import contextvars
import json
import logging
import random
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import requests
from opentelemetry.context import attach, detach, set_value
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context-propagated attributes
# ---------------------------------------------------------------------------

_PROPAGATED_ATTRIBUTES: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "respan_propagated_attributes", default={}
)

# Keys that are merged into every span dict when set via propagate_attributes
_SUPPORTED_ATTRIBUTE_KEYS = frozenset({
    "customer_identifier",
    "customer_email",
    "customer_name",
    "thread_identifier",
    "custom_identifier",
    "group_identifier",
    "environment",
    "metadata",
    "prompt",
})


@contextmanager
def propagate_attributes(**kwargs):
    """Context manager that attaches attributes to all spans exported within its scope.

    Attributes are stored in a ``ContextVar`` and merged into every span dict
    at export time.  This works correctly with ``asyncio`` — each task gets its
    own copy of the context.

    Supported attributes:
        customer_identifier, customer_email, customer_name,
        thread_identifier, custom_identifier, group_identifier,
        environment, metadata (dict — merged, not replaced),
        prompt (dict with prompt_id + variables — triggers server-side
        template resolution).

    Example::

        with propagate_attributes(customer_identifier="user_123"):
            result = await Runner.run(agent, "Hello")

        with propagate_attributes(prompt={"prompt_id": "abc", "variables": {"x": "y"}}):
            result = await Runner.run(agent, "Hello")
    """
    # Merge with any already-active attributes (supports nesting)
    parent = _PROPAGATED_ATTRIBUTES.get()
    merged = {**parent}
    for key, value in kwargs.items():
        if key not in _SUPPORTED_ATTRIBUTE_KEYS:
            logger.warning("Ignoring unsupported attribute: %s", key)
            continue
        if key == "metadata" and isinstance(value, dict):
            # Merge metadata dicts instead of replacing
            merged["metadata"] = {**merged.get("metadata", {}), **value}
        else:
            merged[key] = value

    token = _PROPAGATED_ATTRIBUTES.set(merged)
    try:
        yield
    finally:
        _PROPAGATED_ATTRIBUTES.reset(token)


class RespanSpanExporterV2:
    """Batching HTTP exporter for raw log dicts → /v1/traces/ingest.

    Accepts lists of JSON-serializable dicts via ``export()``, buffers them
    in memory, and flushes to the Respan backend in batches on a background
    daemon thread.

    Args:
        api_key: Bearer token for the Respan API.
        endpoint: Full URL of the ingest endpoint.
        max_batch_size: Maximum number of items per HTTP POST.
        max_queue_size: Maximum items held in memory (oldest dropped on overflow).
        flush_interval: Seconds between automatic flush cycles.
        max_retries: Number of retry attempts for transient failures.
        base_delay: Initial backoff delay in seconds.
        max_delay: Maximum backoff delay in seconds.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://api.respan.ai/api/v1/traces/ingest",
        max_batch_size: int = 128,
        max_queue_size: int = 8192,
        flush_interval: float = 5.0,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        timeout: int = 30,
        default_attributes: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_batch_size = max_batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self._default_attributes = default_attributes or {}

        self._queue: deque = deque(maxlen=max_queue_size)
        self._lock = threading.Lock()
        self._shutdown = False

        # Persistent HTTP session for connection reuse
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
        })
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

        # Background flush thread
        self._flush_event = threading.Event()
        self._worker = threading.Thread(
            target=self._flush_loop, daemon=True, name="RespanExporterV2"
        )
        self._worker.start()

        atexit.register(self.shutdown)

    def export(self, data: List[Dict]) -> None:
        """Enqueue log dicts for batched export.

        Merges default attributes and context-propagated attributes into each
        span dict before queuing.  Context attributes take precedence over
        defaults; explicit span fields are never overwritten.

        Thread-safe. If the queue is full, oldest items are silently dropped
        (deque maxlen behaviour).
        """
        if self._shutdown or not data:
            return

        # Read context-propagated attributes (must happen in caller's thread)
        ctx_attrs = _PROPAGATED_ATTRIBUTES.get()

        # Build merged attributes: defaults < context < span fields
        merged_attrs: Dict[str, Any] = {}
        for key in _SUPPORTED_ATTRIBUTE_KEYS:
            if key in ("metadata", "prompt"):
                continue  # handled separately below
            val = ctx_attrs.get(key) or self._default_attributes.get(key)
            if val is not None:
                merged_attrs[key] = val

        # Merge metadata dicts: defaults < context
        default_meta = self._default_attributes.get("metadata", {})
        ctx_meta = ctx_attrs.get("metadata", {})
        merged_meta = {**default_meta, **ctx_meta}

        # Build prompt logging fields from propagated prompt config
        prompt_cfg = ctx_attrs.get("prompt") or self._default_attributes.get("prompt")

        # Apply to each span dict (don't overwrite fields already on the span)
        enriched = []
        for item in data:
            enriched_item = dict(item)
            for key, val in merged_attrs.items():
                if not enriched_item.get(key):
                    enriched_item[key] = val
            # Prompt logging: send prompt fields + completion_message
            if isinstance(prompt_cfg, dict):
                # Nested prompt object for prompt resolution
                enriched_item.setdefault("prompt", prompt_cfg)
                # Flat fields as fallback (supported by /v1/traces/ingest)
                if "prompt_id" in prompt_cfg:
                    enriched_item.setdefault("prompt_id", prompt_cfg["prompt_id"])
                if "variables" in prompt_cfg:
                    enriched_item.setdefault("variables", prompt_cfg["variables"])
                # completion_message from output
                output = enriched_item.get("output")
                if isinstance(output, dict) and "role" in output:
                    enriched_item.setdefault("completion_message", output)
            if merged_meta:
                existing_meta = enriched_item.get("metadata") or {}
                enriched_item["metadata"] = {**merged_meta, **existing_meta}
            enriched.append(enriched_item)

        with self._lock:
            self._queue.extend(enriched)

    def flush(self) -> None:
        """Force-flush all queued items immediately."""
        self._flush_event.set()
        # Drain synchronously so caller knows items are sent
        self._drain()

    def shutdown(self) -> None:
        """Flush remaining items and stop the background thread."""
        if self._shutdown:
            return
        self._shutdown = True
        self._flush_event.set()
        self._drain()
        self._session.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_loop(self) -> None:
        """Background thread: periodically drains the queue."""
        while not self._shutdown:
            self._flush_event.wait(timeout=self.flush_interval)
            self._flush_event.clear()
            if not self._shutdown:
                self._drain()

    def _drain(self) -> None:
        """Send all queued items in batches."""
        while True:
            batch = self._take_batch()
            if not batch:
                break
            self._send_batch(batch)

    def _take_batch(self) -> List[Dict]:
        """Pop up to max_batch_size items from the queue."""
        with self._lock:
            n = min(len(self._queue), self.max_batch_size)
            if n == 0:
                return []
            return [self._queue.popleft() for _ in range(n)]

    def _send_batch(self, batch: List[Dict]) -> None:
        """POST a batch to the ingest endpoint with retries."""
        if not self.api_key:
            logger.warning("API key is not set, skipping trace export")
            return

        payload = {"data": batch}

        # Suppress OTEL instrumentation to prevent recursive traces
        token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            self._send_with_retry(payload)
        finally:
            detach(token)

    def _send_with_retry(self, payload: Dict) -> None:
        """Exponential backoff with 10% jitter. No retry on 4xx."""
        delay = self.base_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.post(
                    url=self.endpoint,
                    data=json.dumps(payload, default=str),
                    timeout=self.timeout,
                )
                if response.status_code < 300:
                    logger.debug(
                        "Exported %d items to Respan (HTTP %d)",
                        len(payload["data"]),
                        response.status_code,
                    )
                    return
                if 400 <= response.status_code < 500:
                    logger.error(
                        "Respan client error %d: %s",
                        response.status_code,
                        response.text[:500],
                    )
                    return  # Don't retry client errors
                logger.warning(
                    "Server error %d (attempt %d/%d), retrying...",
                    response.status_code,
                    attempt,
                    self.max_retries,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "Request failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

            if attempt < self.max_retries:
                sleep_time = delay + random.uniform(0, 0.1 * delay)
                time.sleep(sleep_time)
                delay = min(delay * 2, self.max_delay)

        logger.error("Max retries reached, giving up on batch of %d items.", len(payload["data"]))
