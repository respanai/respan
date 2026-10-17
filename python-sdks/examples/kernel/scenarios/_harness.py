"""Shared setup for the scenarios: env, Respan, the Kernel target, and live checks.

Each scenario is a Kernel app action, written the way Kernel's Python templates
write them (`kernel.App(...)`, `@app.action(...)`). The harness invokes the
action the way the Kernel runtime does (`handler(KernelContext, payload)`) with
`customer_identifier=kernel-scenarios` and `metadata.scenario=<name>`. After a run,
`verify()` reads each trace back from the Respan logs API and checks it against
the scenario's `Expect` - the live half of the baseline (the offline half is the
package's `tests/test_baseline.py`).

Two modes:
- live (default): real Kernel cloud browsers. Needs `KERNEL_API_KEY`.
- `KERNEL_FAKE=1` (`run_all.py --fake`): the package's local fake Kernel API
  with a local headless Chrome behind it. No Kernel account; the Kernel side is
  a stand-in, the Respan side is real.
"""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import inspect
import json
import os
import re
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parents[1]
load_dotenv(HERE / ".env", override=True)

FAKE = os.getenv("KERNEL_FAKE") == "1"
FAKE_KERNEL_MODULE = (
    HERE.parents[1]
    / "instrumentations"
    / "respan-instrumentation-kernel"
    / "tests"
    / "fake_kernel.py"
)
FAKE_KERNEL_API_KEY = "sk_kernel_fake_0123456789"


def _start_fake_kernel() -> Any:
    """The package's offline Kernel stand-in; `Kernel()` finds it via KERNEL_BASE_URL."""
    spec = importlib.util.spec_from_file_location("fake_kernel", FAKE_KERNEL_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake = module.FakeKernel()
    os.environ["KERNEL_BASE_URL"] = fake.base_url
    os.environ["KERNEL_API_KEY"] = FAKE_KERNEL_API_KEY
    return fake


FAKE_KERNEL = _start_fake_kernel() if FAKE else None
if not os.getenv("KERNEL_API_KEY"):
    raise RuntimeError(
        "Set KERNEL_API_KEY (https://dashboard.onkernel.com), or run against the local "
        "fake Kernel API with `python run_all.py --fake`."
    )

import kernel  # noqa: E402
from respan import Respan  # noqa: E402
from respan_instrumentation_kernel import KernelInstrumentor  # noqa: E402
from respan_tracing.testing.live import (  # noqa: E402
    RespanLogsClient,
    is_error,
    trace_problems,
)

try:  # scenario 02 only
    from browser_use import ChatOpenAI
    from respan_instrumentation_browser_use import BrowserUseInstrumentor
except ImportError:  # pragma: no cover - browser-use not installed
    ChatOpenAI = BrowserUseInstrumentor = None

API_KEY = os.getenv("RESPAN_API_KEY")
BASE_URL = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api").rstrip("/")
PLATFORM_URL = os.getenv("RESPAN_PLATFORM_URL", "https://platform.respan.ai").rstrip("/")
MODEL = os.getenv("RESPAN_MODEL", "gpt-4.1-mini")
KERNEL_API_KEY = os.environ["KERNEL_API_KEY"]
PROBE_PASSWORDS = os.getenv("KERNEL_PROBE_PASSWORDS") == "1"
# run_all.py --no-content: structure only; content checks become "no content" checks.
CONTENT = os.getenv("RESPAN_TRACE_CONTENT", "true").lower() not in ("0", "false", "no", "off")
HEADLESS = os.getenv("HEADED") != "1"
CUSTOMER = "kernel-scenarios"
RUN_ID = os.getenv("SCENARIO_RUN_ID") or time.strftime("%Y%m%d-%H%M%S")
if not API_KEY:
    raise RuntimeError("Set RESPAN_API_KEY (copy .env.example to .env).")

SCENARIO_ATTR = "respan.metadata.scenario"
API_NAME = "kernel.api.name"
SESSION_ID = "kernel.session.id"
ACTION_NAME = "kernel.action.name"
# A browser token that was not masked (masked ones read `jwt=<masked>`).
UNMASKED_TOKEN = re.compile(r"[?&](?:jwt|token)=(?!<masked>)")
_traces: dict[str, set[str]] = defaultdict(set)
_exported: Counter[str] = Counter()  # spans the SDK exported, per trace id
_exported_calls: dict[str, Counter[str]] = defaultdict(Counter)  # Kernel calls, per trace


def _remember_trace(span: Any) -> None:
    if scenario := span.attributes.get(SCENARIO_ATTR):
        trace_id = format(span.context.trace_id, "032x")
        _traces[scenario].add(trace_id)
        _exported[trace_id] += 1
        if api_name := span.attributes.get(API_NAME):
            _exported_calls[trace_id][api_name] += 1


# Respan starts before the scenario modules are imported: `@app.action`
# handlers are wrapped when they are registered.
CLIENT = Respan(
    api_key=API_KEY,
    base_url=BASE_URL,
    app_name="kernel-scenarios",
    customer_identifier=CUSTOMER,
    metadata={"run_id": RUN_ID},
    instrumentations=[
        KernelInstrumentor(probe_password_fields=PROBE_PASSWORDS),
        *([BrowserUseInstrumentor()] if BrowserUseInstrumentor else []),
    ],
    span_postprocess_callback=_remember_trace,
)


def llm() -> Any:
    """browser-use's ChatOpenAI pointed at the Respan gateway (one key for both)."""
    return ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=BASE_URL)


# -- the page the browser opens ---------------------------------------------------


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


def _serve_site() -> str:
    handler = functools.partial(_QuietHandler, directory=str(HERE / "site"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/index.html"


# A Kernel cloud browser cannot reach this machine: live runs use a public page.
TARGET_URL = _serve_site() if FAKE else os.getenv("KERNEL_TARGET_URL", "https://example.com")


# -- scenarios ----------------------------------------------------------------------


@dataclass(frozen=True)
class Expect:
    calls_include: tuple[str, ...] = ()  # Kernel API names, e.g. "computer.click_mouse"
    error_calls: tuple[str, ...] = ()  # calls that must be error spans
    screenshots: tuple[str, ...] = ()  # calls whose output carries an image
    forbidden: tuple[str, ...] = ()  # strings no span may contain
    has_llm: bool = False  # chat spans (with tokens and cost) in the trace


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    source: str  # the Kernel template / docs page it mirrors
    look_for: str  # what to check in Respan
    app: Any  # the kernel.App
    action: str
    payload: Any = None
    expect: Expect = field(default_factory=Expect)


def execute(scenario: Scenario) -> Any:
    """Invoke the action the way the Kernel runtime does."""
    handler = scenario.app.get_action(scenario.action).handler
    context = kernel.KernelContext(invocation_id=f"local-{uuid.uuid4().hex[:12]}")
    args = (context,) if scenario.payload is None else (context, scenario.payload)
    with CLIENT.propagate_attributes(metadata={"scenario": scenario.name}):
        result = handler(*args)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
    print(f"  -> {json.dumps(result, default=str)[:200]}")
    return result


# -- live verification ------------------------------------------------------------


LOGS = RespanLogsClient(BASE_URL, API_KEY)


def _meta(row: dict[str, Any], key: str) -> Any:
    return (row.get("metadata") or {}).get(key)


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
        roots = [
            next((r for r in self.traces[t] if not r.get("span_parent_id")), None)
            for t in sorted(self.traces)
        ]
        return [f"{PLATFORM_URL}/platform/requests?unique_id={r['unique_id']}" for r in roots if r]


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

    check(len(trace_ids) == 1, f"{len(trace_ids)} traces, expected 1 (one per action run)")
    for trace_id, trace_rows in traces.items():
        problems.extend(
            f"trace {trace_id[:8]}: {problem}"
            for problem in trace_problems(
                trace_rows,
                exported=_exported[trace_id],
                customer_identifier=CUSTOMER,
                metadata={"scenario": scenario.name, "run_id": RUN_ID},
                forbidden=(KERNEL_API_KEY, *expect.forbidden),
            )
        )
        # Every Kernel call the SDK traced is stored, call for call.
        stored = Counter(name for r in trace_rows if (name := _meta(r, API_NAME)))
        if stored != _exported_calls[trace_id]:
            missing = _exported_calls[trace_id] - stored
            check(False, f"trace {trace_id[:8]}: Kernel calls missing in Respan: {dict(missing)}")
        roots = [r for r in trace_rows if not r.get("span_parent_id")]
        # Respan names an agent span "agent.<detail>"; the action name is the detail.
        expected_root = f"agent.{scenario.action}"
        check(
            any(
                r.get("span_name") == expected_root and r.get("log_type") == "agent" for r in roots
            ),
            f"trace {trace_id[:8]}: root is not the agent span {expected_root}",
        )
        sessions = {
            _meta(r, SESSION_ID) for r in trace_rows if _meta(r, API_NAME) == "browsers.create"
        }
        threads = {r.get("thread_identifier") for r in trace_rows}
        check(
            len(sessions) == 1 and threads == sessions,
            f"trace {trace_id[:8]}: thread {threads} is not the Kernel session {sessions}",
        )
        for row in trace_rows:
            if _meta(row, API_NAME):
                check(row.get("log_type") == "tool", f"{row.get('span_name')}: not a tool span")
            if UNMASKED_TOKEN.search(json.dumps(row, default=str)):
                check(False, f"{row.get('span_name')}: an unmasked browser token")
            # Kernel's own spans; other instrumentations decide what "structure" is.
            if not CONTENT and (_meta(row, API_NAME) or _meta(row, ACTION_NAME)):
                check(
                    not row.get("input") and not row.get("output"),
                    f"{row.get('span_name')}: content exported with content off",
                )

    calls = {_meta(r, API_NAME) for r in rows}
    for wanted in expect.calls_include:
        check(wanted in calls, f"no {wanted} span")
    failed = {_meta(r, API_NAME) or r.get("span_name") for r in rows if is_error(r)}
    for wanted in expect.error_calls:
        check(wanted in failed, f"{wanted} should be an error span")
    for wanted in expect.screenshots if CONTENT else ():
        check(
            any(_meta(r, API_NAME) == wanted and _has_image(r) for r in rows),
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
    print(f"{status}  {result.scenario.name:<20} {spans:>3} spans  {result.scenario.title}")
    for problem in result.problems:
        print(f"        - {problem}")
    for note in result.notes:
        print(f"        ~ {note}")
    for trace_id in sorted(result.traces):
        print(f"        trace {trace_id}")
    for link in result.links:
        print(f"        {link}")


def shutdown() -> None:
    CLIENT.flush()
    if FAKE_KERNEL is not None:
        FAKE_KERNEL.close()


def main(scenario: Scenario) -> None:
    """Run one scenario on its own: `python scenarios/<file>.py` (KERNEL_FAKE=1 for the fake)."""
    try:
        execute(scenario)
    finally:
        shutdown()
    print_result(verify(scenario))
