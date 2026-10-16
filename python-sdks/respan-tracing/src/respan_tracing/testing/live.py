"""Check traces after they landed in Respan (live soaks of an instrumentation).

The generic half of an integration's live verifier: read a trace back from the
Respan logs API and report what is structurally wrong with it, so humans only
read content. Framework-specific checks (step counts, judges, ...) stay with
the integration's examples.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from typing import Any, Optional

import requests

ERROR_STATUS = 400


class RespanLogsClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout

    def trace_rows(self, trace_id: str) -> list[dict[str, Any]]:
        response = requests.post(
            f"{self.base_url}/request-logs/list/",
            params={"page_size": 500},
            json={"filters": {"trace_unique_id": {"operator": "", "value": [trace_id]}}},
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("results", payload) if isinstance(payload, dict) else payload

    def detail(self, unique_id: str) -> dict[str, Any]:
        """One log in full; the list API omits e.g. the row's own span id."""
        response = requests.get(
            f"{self.base_url}/request-logs/{unique_id}/",
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_trace(
        self, trace_id: str, *, timeout: float = 180, poll: float = 10
    ) -> list[dict[str, Any]]:
        """Poll until the span count stops changing (ingestion is asynchronous)."""
        deadline, last, rows = time.monotonic() + timeout, -1, []
        while time.monotonic() < deadline:
            rows = self.trace_rows(trace_id)
            if rows and len(rows) == last:
                return rows
            last = len(rows)
            time.sleep(poll)
        return rows


def is_error(row: Mapping[str, Any]) -> bool:
    return (row.get("status_code") or 200) >= ERROR_STATUS


def json_field(row: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = row.get(field)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def trace_problems(
    rows: list[dict[str, Any]],
    *,
    exported: Optional[int] = None,
    customer_identifier: Optional[str] = None,
    metadata: Optional[Mapping[str, str]] = None,
    forbidden: Iterable[str] = (),
) -> list[str]:
    """Generic structural checks for one trace's stored rows.

    - every span the SDK exported is stored (``exported``);
    - exactly one root span;
    - customer id and metadata on every span;
    - one thread id across the trace;
    - token counts and cost on every successful chat span;
    - no forbidden strings (secrets) anywhere.
    """
    problems: list[str] = []
    if not rows:
        return ["trace not found"]
    if exported is not None and len(rows) != exported:
        problems.append(f"SDK exported {exported} spans, Respan has {len(rows)}")
    roots = [r for r in rows if not r.get("span_parent_id")]
    if len(roots) != 1:
        problems.append(f"{len(roots)} root spans")
    if len({r.get("thread_identifier") for r in rows}) > 1:
        problems.append("more than one thread id in the trace")
    forbidden = tuple(forbidden)
    for row in rows:
        name = row.get("span_name")
        if customer_identifier and row.get("customer_identifier") != customer_identifier:
            problems.append(f"{name}: customer_identifier missing")
        for key, value in (metadata or {}).items():
            if (row.get("metadata") or {}).get(key) != value:
                problems.append(f"{name}: metadata.{key} missing")
        if row.get("log_type") == "chat" and not is_error(row):
            for field in ("prompt_tokens", "completion_tokens", "cost"):
                if not (row.get(field) or 0) > 0:
                    problems.append(f"{name}: no {field}")
        if forbidden:
            dumped = json.dumps(row, default=str)
            if any(value in dumped for value in forbidden):
                problems.append(f"{name}: contains a forbidden value")
    return problems
