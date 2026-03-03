"""
Shared export helpers for Respan ingest: payload validation and fire-and-forget HTTP POST.

Used by respan-exporter-dify, respan-exporter-superagent, and other integrations.
Centralizes RetryHandler, RESPAN_DOGFOOD_HEADER, and daemon-thread semantics so no
exporter can forget retry logic or the anti-recursion header.
"""

import logging
import threading
from typing import Any, Dict, List

import requests

from respan_sdk.constants import RESPAN_DOGFOOD_HEADER
from respan_sdk.respan_types import RespanFullLogParams
from respan_sdk.utils.retry_handler import RetryHandler


logger = logging.getLogger(__name__)


def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate payload against RespanFullLogParams and return JSON-serializable dict."""
    validated = RespanFullLogParams(**payload)
    return validated.model_dump(mode="json", exclude_none=True)


def send_payloads(
    *,
    api_key: str,
    endpoint: str,
    timeout: int,
    payloads: List[Dict[str, Any]],
    context: str = "respan ingest",
) -> None:
    """
    POST payloads to Respan ingest in a daemon thread (fire-and-forget).

    Callers build and pass payloads on their own thread (e.g. the Dify client
    calls export_dify_call from the main thread; payload construction runs there).
    Only the HTTP POST is offloaded: this function spawns a daemon thread that
    performs the request. So payload building is synchronous from the caller's
    perspective; only the network send is fire-and-forget.

    Uses RetryHandler for backoff and sends RESPAN_DOGFOOD_HEADER so the server
    can skip emitting traces for the ingest request (anti-recursion).
    """
    def _run() -> None:
        handler = RetryHandler(
            max_retries=3,
            retry_delay=1.0,
            backoff_multiplier=2.0,
            max_delay=30.0,
        )

        def _post() -> None:
            response = requests.post(
                endpoint,
                json=payloads,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    RESPAN_DOGFOOD_HEADER: "1",
                },
                timeout=timeout,
            )
            if response.status_code >= 500:
                raise RuntimeError(
                    f"Respan ingest server error status_code={response.status_code}"
                )
            if response.status_code >= 300:
                logger.warning(
                    "Respan ingest client error status_code=%s",
                    response.status_code,
                )

        try:
            handler.execute(func=_post, context=context)
        except Exception as exc:
            logger.exception("Respan ingest failed after retries: %s", exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
