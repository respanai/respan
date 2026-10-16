"""Read a run's traces back from Respan and check them.

    python verify_live.py runs/<run id>.json [scenario ...]

The generic checks (every exported span stored, one root, customer id and
metadata on every span, one thread id, tokens and cost on every LLM span, no
secrets) come from ``respan_tracing.testing.live``. The Puppeteer-specific ones:

- every Puppeteer call the SDK exported is stored as a ``tool`` span, call for
  call (by ``puppeteer.api.name``);
- the scenario's expected calls, error calls, screenshots and LLM spans.

LLM calls made through the Respan gateway are logged by the gateway itself; a
traced span that only links to that row has no usage of its own, so usage is
checked on the linked row.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env", override=True)

from respan_sdk.constants.span_attributes import RESPAN_GATEWAY_LOG_ID  # noqa: E402
from respan_tracing.testing.live import (  # noqa: E402
    RespanLogsClient,
    is_error,
    trace_problems,
)

API_KEY = os.environ["RESPAN_API_KEY"]
BASE_URL = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api").rstrip("/")
PLATFORM_URL = os.getenv("RESPAN_PLATFORM_URL", "https://platform.respan.ai").rstrip("/")
LOGS = RespanLogsClient(BASE_URL, API_KEY)
API_NAME_ATTR = "puppeteer.api.name"
RECORDING_PATH_ATTR = "puppeteer.recording.path"
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "cost")


def api_name(row: dict[str, Any]) -> str | None:
    return (row.get("metadata") or {}).get(API_NAME_ATTR)


def resolve_gateway_links(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Usage of gateway-served LLM calls lives on the gateway's own row."""
    resolved, linked = [], 0
    for row in rows:
        log_id = (row.get("metadata") or {}).get(RESPAN_GATEWAY_LOG_ID)
        if row.get("log_type") == "chat" and log_id:
            gateway_row = LOGS.detail(log_id)
            row = {**row, **{key: gateway_row.get(key) for key in USAGE_FIELDS}}
            linked += 1
        resolved.append(row)
    return resolved, linked


# The local site's address and the scenarios' typed text: never exported with content off.
CONTENT_MARKERS = ("127.0.0.1", "ada@example.com", "correct-horse", "image_url", "Respan Puppeteer demo")


def calls_of(rows: list[dict[str, Any]]) -> set[str | None]:
    return {api_name(r) for r in rows}


def has_image(row: dict[str, Any]) -> bool:
    return "image_url" in json.dumps(row.get("output") or row.get("completion_message") or "")


def check(scenario: dict[str, Any], run: dict[str, Any]) -> tuple[list[str], list[str], list[str], dict[str, list]]:
    expect = scenario.get("expect") or {}
    exported = scenario.get("traces") or {}
    problems: list[str] = []
    notes: list[str] = []
    if scenario.get("error"):
        problems.append(f"run raised {scenario['error']}")
    if len(exported) != expect.get("traces", 1):
        problems.append(f"{len(exported)} traces, expected {expect.get('traces', 1)}")
    traces = {trace_id: LOGS.wait_for_trace(trace_id) for trace_id in sorted(exported)}
    for trace_id, rows in traces.items():
        rows, linked = resolve_gateway_links(rows)
        if linked:
            notes.append(f"{linked} LLM spans link to gateway rows; usage checked there")
        problems.extend(
            f"trace {trace_id[:8]}: {problem}"
            for problem in trace_problems(
                rows,
                exported=exported[trace_id]["spans"],
                customer_identifier=run["customer"],
                metadata={"scenario": scenario["name"], "run_id": run["run_id"]},
                forbidden=expect.get("forbidden", ()),
            )
        )
        stored = Counter(name for r in rows if (name := api_name(r)))
        missing = Counter(exported[trace_id]["calls"]) - stored
        if missing:
            problems.append(f"trace {trace_id[:8]}: Puppeteer calls missing in Respan: {dict(missing)}")
        for row in rows:
            if api_name(row) and row.get("log_type") != "tool":
                problems.append(f"{api_name(row)}: stored as {row.get('log_type')}, not tool")
    rows = [r for t in traces.values() for r in t]
    settings = run.get("settings") or {}
    if settings.get("captureContent") is False:
        # Content off: no Puppeteer row carries a page URL, typed text, title or image.
        for row in rows:
            if api_name(row) and (leak := next((s for s in CONTENT_MARKERS if s in json.dumps(row)), None)):
                problems.append(f"{api_name(row)}: content off, but the row contains {leak!r}")
    if settings.get("screenshotOn") == "action" and expect.get("callsInclude") and "page.goto" in calls_of(rows):
        if not any(api_name(r) == "page.goto" and has_image(r) for r in rows):
            problems.append("screenshotOn=action, but page.goto carries no screenshot")
    calls = {api_name(r) for r in rows}
    for wanted in expect.get("callsInclude", ()):
        if wanted not in calls:
            problems.append(f"no {wanted} span")
    failed = {api_name(r) or r.get("span_name") for r in rows if is_error(r)}
    for wanted in expect.get("errorCalls", ()):
        if wanted not in failed:
            problems.append(f"{wanted} should be an error span")
    images_on = settings.get("captureContent") is not False and settings.get("captureScreenshots") is not False
    for wanted in expect.get("screenshots", ()) if images_on else ():
        if not any(api_name(r) == wanted and has_image(r) for r in rows):
            problems.append(f"{wanted}: no screenshot in its output")
    if "page.record" in expect.get("callsInclude", ()) and settings.get("captureContent") is not False:
        if not any(api_name(r) == "page.record" and (r.get("metadata") or {}).get(RECORDING_PATH_ATTR) for r in rows):
            problems.append("page.record carries no recording path")
    if expect.get("hasLlm") and not any(r.get("log_type") == "chat" for r in rows):
        problems.append("no LLM span")
    incidental = sorted(str(n) for n in failed - set(expect.get("errorCalls", ())))
    if incidental:
        notes.append(f"other error spans: {incidental}")
    links = []
    for rows_of_trace in traces.values():
        root = next((r for r in rows_of_trace if not r.get("span_parent_id")), None)
        if root:
            links.append(f"{PLATFORM_URL}/platform/requests?unique_id={root['unique_id']}")
    return problems, notes, links, traces


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    run = json.loads(Path(sys.argv[1]).read_text())
    only = set(sys.argv[2:])
    results = []
    for scenario in run["scenarios"]:
        if only and scenario["name"] not in only:
            continue
        problems, notes, links, traces = check(scenario, run)
        spans = sum(len(rows) for rows in traces.values())
        results.append((scenario, problems, notes, links, traces, spans))
        print(f"{'PASS' if not problems else 'FAIL'}  {scenario['name']:<22} {spans:>3} spans  {scenario['title']}")
        for problem in problems:
            print(f"        - {problem}")
        for note in notes:
            print(f"        ~ {note}")
        for trace_id in traces:
            print(f"        trace {trace_id}")
        for link in links:
            print(f"        {link}")
    passed = sum(not r[1] for r in results)
    print(f"\n{passed}/{len(results)} passed (run {run['run_id']}, customer {run['customer']}, settings {run.get('settings')}).")
    report = Path(sys.argv[1]).with_suffix(".md")
    lines = [
        f"# Puppeteer scenarios - run {run['run_id']}",
        "",
        f"Settings: `{json.dumps(run.get('settings'))}`",
        "",
        "| | Scenario | Spans | What to look for | Traces | Problems |",
        "|---|---|---|---|---|---|",
    ]
    for scenario, problems, notes, links, traces, spans in results:
        trace_links = " ".join(f"[{t[:8]}]({link})" for t, link in zip(traces, links))
        lines.append(
            f"| {'PASS' if not problems else 'FAIL'} | `{scenario['name']}` {scenario['title']} | {spans} "
            f"| {scenario['look_for']} | {trace_links} | {'<br>'.join([*problems, *(f'note: {n}' for n in notes)])} |"
        )
    report.write_text("\n".join(lines) + "\n")
    print(f"Report: {report}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
