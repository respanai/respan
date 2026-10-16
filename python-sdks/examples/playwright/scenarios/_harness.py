"""Shared setup for the scenarios: env, Respan, the local demo site, and live checks.

Every scenario runs inside a Respan workflow span (Playwright calls are traced
only under an active span) with `customer_identifier=playwright-scenarios` and
`metadata.scenario=<name>`. After a run, `verify()` reads each trace back from
the Respan logs API and checks it against the scenario's `Expect` - the live
half of the baseline (the offline half is the package's `tests/test_baseline.py`).
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import os
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parents[1]
load_dotenv(HERE / ".env", override=True)

from respan import Respan, workflow  # noqa: E402
from respan_instrumentation_openai import OpenAIInstrumentor  # noqa: E402
from respan_instrumentation_playwright import PlaywrightInstrumentor  # noqa: E402
from respan_sdk.constants.span_attributes import RESPAN_GATEWAY_LOG_ID  # noqa: E402
from respan_tracing.testing.live import (  # noqa: E402
    RespanLogsClient,
    is_error,
    trace_problems,
)

API_KEY = os.getenv("RESPAN_API_KEY")
BASE_URL = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api").rstrip("/")
PLATFORM_URL = os.getenv("RESPAN_PLATFORM_URL", "https://platform.respan.ai").rstrip("/")
MODEL = os.getenv("RESPAN_MODEL", "gpt-4.1-mini")
HEADLESS = os.getenv("HEADED") != "1"
# "chrome" drives the installed Google Chrome; empty = Playwright's own Chromium
# (`playwright install chromium`).
CHANNEL = os.getenv("PLAYWRIGHT_CHANNEL", "chrome") or None
CUSTOMER = "playwright-scenarios"
RUN_ID = os.getenv("SCENARIO_RUN_ID") or time.strftime("%Y%m%d-%H%M%S")
ARTIFACTS = HERE / "artifacts"  # screenshots, downloads, trace.zip (gitignored)
ARTIFACTS.mkdir(exist_ok=True)
if not API_KEY:
    raise RuntimeError("Set RESPAN_API_KEY (copy .env.example to .env).")

SCENARIO_ATTR = "respan.metadata.scenario"
API_NAME_ATTR = "playwright.api.name"
_traces: dict[str, set[str]] = defaultdict(set)
_exported: Counter[str] = Counter()  # spans the SDK exported, per trace id
_exported_calls: dict[str, Counter[str]] = defaultdict(Counter)  # Playwright calls, per trace
_client: Respan | None = None


def _remember_trace(span: Any) -> None:
    if scenario := span.attributes.get(SCENARIO_ATTR):
        trace_id = format(span.context.trace_id, "032x")
        _traces[scenario].add(trace_id)
        _exported[trace_id] += 1
        if api_name := span.attributes.get(API_NAME_ATTR):
            _exported_calls[trace_id][api_name] += 1


def init(
    *, capture_content: bool = True, capture_screenshots: bool = True, screenshot_on: str = "never"
) -> Respan:
    global _client
    if _client is None:
        _client = Respan(
            api_key=API_KEY,
            base_url=BASE_URL,
            app_name="playwright-scenarios",
            customer_identifier=CUSTOMER,
            metadata={"run_id": RUN_ID},
            instrumentations=[
                PlaywrightInstrumentor(
                    capture_content=capture_content,
                    capture_screenshots=capture_screenshots,
                    screenshot_on=screenshot_on,
                ),
                OpenAIInstrumentor(),  # the LLM scenario's model calls
            ],
            span_postprocess_callback=_remember_trace,
        )
    return _client


def launch_options() -> dict[str, Any]:
    return {"headless": HEADLESS, "channel": CHANNEL}


# -- the local demo site ---------------------------------------------------------


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


def _serve_site() -> str:
    handler = functools.partial(_QuietHandler, directory=str(HERE / "site"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


SITE = _serve_site()


# -- scenarios -------------------------------------------------------------------


@dataclass(frozen=True)
class Expect:
    traces: int = 1
    calls_include: tuple[str, ...] = ()  # Playwright API names, e.g. "Page.goto"
    error_calls: tuple[str, ...] = ()  # calls that must be error spans
    screenshots: tuple[str, ...] = ()  # calls whose output carries an image
    forbidden: tuple[str, ...] = ()  # strings no span may contain
    has_llm: bool = False  # chat spans (with tokens and cost) in the trace


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    source: str  # the Playwright for Python docs page it mirrors
    look_for: str  # what to check in Respan
    run: Callable[[], Any]
    expect: Expect = field(default_factory=Expect)
    # The scenario opens its own workflow spans (e.g. one per parallel session).
    own_roots: bool = False


def execute(scenario: Scenario) -> None:
    client = init()
    run = scenario.run if scenario.own_roots else workflow(name=scenario.name)(scenario.run)
    with client.propagate_attributes(metadata={"scenario": scenario.name}):
        if inspect.iscoroutinefunction(scenario.run):
            asyncio.run(run())
        else:
            run()


# -- live verification -----------------------------------------------------------


LOGS = RespanLogsClient(BASE_URL, API_KEY)


def _api_name(row: dict[str, Any]) -> str | None:
    return (row.get("metadata") or {}).get(API_NAME_ATTR)


USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "cost")


def resolve_gateway_links(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """LLM calls made through the Respan gateway are logged by the gateway itself;
    the traced span only links to that row (so the call is not counted twice) and
    has no usage of its own. Check usage on the linked row instead."""
    resolved, linked = [], 0
    for row in rows:
        log_id = (row.get("metadata") or {}).get(RESPAN_GATEWAY_LOG_ID)
        if row.get("log_type") == "chat" and log_id:
            gateway_row = LOGS.detail(log_id)
            row = {**row, **{key: gateway_row.get(key) for key in USAGE_FIELDS}}
            linked += 1
        resolved.append(row)
    return resolved, linked


def _has_image(row: dict[str, Any]) -> bool:
    return "image_url" in json.dumps(row.get("output") or row.get("completion_message") or "")


@dataclass
class Result:
    scenario: Scenario
    traces: dict[str, list[dict[str, Any]]]
    problems: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def links(self) -> list[str]:
        """One link per trace (sorted by trace id) to its root span."""
        roots = [
            next((r for r in self.traces[t] if not r.get("span_parent_id")), None)
            for t in sorted(self.traces)
        ]
        return [
            f"{PLATFORM_URL}/platform/requests?unique_id={r['unique_id']}" for r in roots if r
        ]


def verify(scenario: Scenario) -> Result:
    expect = scenario.expect
    trace_ids = sorted(_traces.get(scenario.name, ()))
    traces = {t: LOGS.wait_for_trace(t) for t in trace_ids}
    rows = [r for t in trace_ids for r in traces[t]]
    problems: list[str] = []
    notes: list[str] = []

    def check(ok: bool, problem: str) -> None:
        if not ok:
            problems.append(problem)

    check(len(trace_ids) == expect.traces, f"{len(trace_ids)} traces, expected {expect.traces}")
    for trace_id, trace_rows in traces.items():
        trace_rows, linked = resolve_gateway_links(trace_rows)
        if linked:
            notes.append(f"{linked} LLM spans link to gateway rows; usage checked there")
        problems.extend(
            f"trace {trace_id[:8]}: {problem}"
            for problem in trace_problems(
                trace_rows,
                exported=_exported[trace_id],
                customer_identifier=CUSTOMER,
                metadata={"scenario": scenario.name, "run_id": RUN_ID},
                forbidden=expect.forbidden,
            )
        )
        # Every Playwright call the SDK traced is stored, call for call.
        stored = Counter(name for r in trace_rows if (name := _api_name(r)))
        if stored != _exported_calls[trace_id]:
            missing = _exported_calls[trace_id] - stored
            check(False, f"trace {trace_id[:8]}: Playwright calls missing in Respan: {dict(missing)}")
        for row in trace_rows:
            if _api_name(row):
                check(row.get("log_type") == "tool", f"{row.get('span_name')}: not a tool span")

    calls = {_api_name(r) for r in rows}
    for wanted in expect.calls_include:
        check(wanted in calls, f"no {wanted} span")
    failed = {_api_name(r) or r.get("span_name") for r in rows if is_error(r)}
    for wanted in expect.error_calls:
        check(wanted in failed, f"{wanted} should be an error span")
    for wanted in expect.screenshots:
        check(
            any(_api_name(r) == wanted and _has_image(r) for r in rows),
            f"{wanted}: no screenshot in its output",
        )
    if expect.has_llm:
        check(any(r.get("log_type") == "chat" for r in rows), "no LLM span")
    incidental = sorted(str(n) for n in failed - set(expect.error_calls))
    if incidental:
        notes.append(f"other error spans: {incidental}")
    return Result(scenario, traces, problems, notes)


def print_result(result: Result) -> None:
    status = "PASS" if not result.problems else "FAIL"
    spans = sum(len(rows) for rows in result.traces.values())
    print(f"{status}  {result.scenario.name:<22} {spans:>3} spans  {result.scenario.title}")
    for problem in result.problems:
        print(f"        - {problem}")
    for note in result.notes:
        print(f"        ~ {note}")
    for trace_id in sorted(result.traces):
        print(f"        trace {trace_id}")
    for link in result.links:
        print(f"        {link}")


def main(scenario: Scenario) -> None:
    """Run one scenario on its own: `python scenarios/<file>.py`."""
    client = init()
    try:
        execute(scenario)
    finally:
        client.flush()
    print_result(verify(scenario))
