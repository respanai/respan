import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set

import requests
from respan_sdk.constants import RESPAN_DOGFOOD_HEADER
from respan_sdk.respan_types.log_types import RespanFullLogParams
from respan_sdk.utils.retry_handler import RetryHandler


logger = logging.getLogger(__name__)

_export_threads: Set[threading.Thread] = set()
_export_threads_lock = threading.Lock()


def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate payload against the published Respan log schema."""
    validated = RespanFullLogParams.model_validate(payload)
    return validated.model_dump(mode="json", exclude_none=True)


def send_payloads(
    *,
    api_key: str,
    endpoint: str,
    timeout: int,
    payloads: List[Dict[str, Any]],
    context: str = "respan ingest",
) -> None:
    """POST payloads to Respan ingest in a daemon thread."""

    def _run() -> None:
        handler = RetryHandler(
            max_retries=3,
            retry_delay=1.0,
            backoff_multiplier=2.0,
            max_delay=30.0,
        )

        def _post() -> None:
            response = requests.post(
                url=endpoint,
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

    def _run_and_unregister() -> None:
        try:
            _run()
        finally:
            with _export_threads_lock:
                _export_threads.discard(threading.current_thread())

    thread = threading.Thread(target=_run_and_unregister, daemon=True)
    with _export_threads_lock:
        _export_threads.add(thread)
    thread.start()


def flush_export_threads(timeout: Optional[float] = None) -> None:
    """Wait for in-flight background export threads to finish."""
    deadline = None if timeout is None else time.monotonic() + timeout

    while True:
        with _export_threads_lock:
            active_threads = [thread for thread in _export_threads if thread.is_alive()]
            _export_threads.clear()
            _export_threads.update(active_threads)

        if not active_threads:
            return

        for thread in active_threads:
            remaining = None
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
                if remaining == 0.0:
                    return
            thread.join(timeout=remaining)
