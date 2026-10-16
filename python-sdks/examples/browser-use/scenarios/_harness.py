"""Shared setup for the scenarios: env, Respan, model, browser, and live trace checks.

Every scenario runs under `customer_identifier=browser-use-scenarios` and
`metadata.scenario=<name>`, so its spans are easy to find in Respan. After a run,
`verify()` reads each trace back from the Respan logs API and checks it against the
scenario's `Expect` - the live half of the baseline (the offline half is the
package's `tests/test_baseline.py`).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

from browser_use import Browser, ChatOpenAI  # noqa: E402
from respan import Respan  # noqa: E402
from respan_instrumentation_browser_use import BrowserUseInstrumentor  # noqa: E402
from respan_tracing.testing.live import (  # noqa: E402
    RespanLogsClient,
    is_error,
    json_field,
    trace_problems,
)

API_KEY = os.getenv("RESPAN_API_KEY")
BASE_URL = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api").rstrip("/")
PLATFORM_URL = os.getenv("RESPAN_PLATFORM_URL", "https://platform.respan.ai").rstrip("/")
MODEL = os.getenv("RESPAN_MODEL", "gpt-4.1-mini")
SMALL_MODEL = os.getenv("RESPAN_SMALL_MODEL", "gpt-4.1-nano")
HEADLESS = os.getenv("HEADLESS") == "1"
CUSTOMER = "browser-use-scenarios"
RUN_ID = os.getenv("SCENARIO_RUN_ID") or time.strftime("%Y%m%d-%H%M%S")
if not API_KEY:
    raise RuntimeError("Set RESPAN_API_KEY (copy .env.example to .env).")

SCENARIO_ATTR = "respan.metadata.scenario"
_traces: dict[str, set[str]] = defaultdict(set)
_exported: Counter[str] = Counter()  # spans the SDK exported, per trace id
_client: Respan | None = None


def _remember_trace(span: Any) -> None:
    if scenario := span.attributes.get(SCENARIO_ATTR):
        trace_id = format(span.context.trace_id, "032x")
        _traces[scenario].add(trace_id)
        _exported[trace_id] += 1


def init(*, capture_content: bool = True, capture_screenshots: bool = True) -> Respan:
    global _client
    if _client is None:
        _client = Respan(
            api_key=API_KEY,
            base_url=BASE_URL,
            app_name="browser-use-scenarios",
            customer_identifier=CUSTOMER,
            metadata={"run_id": RUN_ID},
            instrumentations=[
                BrowserUseInstrumentor(
                    capture_content=capture_content,
                    capture_screenshots=capture_screenshots,
                )
            ],
            span_postprocess_callback=_remember_trace,
        )
    return _client


def llm(model: str | None = None) -> ChatOpenAI:
    """browser-use's ChatOpenAI pointed at the Respan gateway (one key for both)."""
    return ChatOpenAI(model=model or MODEL, api_key=API_KEY, base_url=BASE_URL)


def browser(**kwargs: Any) -> Browser:
    return Browser(headless=HEADLESS, **kwargs)


def span_name(model: str) -> str:
    """How Respan names the chat span of `model`."""
    return "llm." + model.replace("/", "_")


@dataclass(frozen=True)
class Expect:
    traces: int = 1
    agent_spans: int | None = None  # default: one per trace
    success: bool | None = True  # the agent's own verdict; None = either
    spans_include: tuple[str, ...] = ()  # e.g. "tool.save_note"
    error_spans: tuple[str, ...] = ()  # spans that must fail
    models_include: tuple[str, ...] = ()  # as requested; the gateway may report a resolved name
    forbidden: tuple[str, ...] = ()  # strings no span may contain
    prompt_images: bool | None = None  # False: no screenshot reached the model


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    source: str  # browser-use's own example this mirrors
    look_for: str  # what to check in Respan
    run: Callable[[], Awaitable[Any]]
    expect: Expect = field(default_factory=Expect)


async def execute(scenario: Scenario) -> None:
    client = init()
    with client.propagate_attributes(metadata={"scenario": scenario.name}):
        await scenario.run()


# -- live verification --------------------------------------------------------


LOGS = RespanLogsClient(BASE_URL, API_KEY)


def fetch_trace(trace_id: str) -> list[dict[str, Any]]:
    return LOGS.trace_rows(trace_id)


def _agent_output(row: dict[str, Any]) -> dict[str, Any]:
    return json_field(row, "output")


def _span_id(row: dict[str, Any]) -> str | None:
    return LOGS.detail(row["unique_id"]).get("span_unique_id")


async def structure_problems(rows: list[dict[str, Any]]) -> list[str]:
    """Spans missing from the tree, judged against browser-use's own record.

    - an agent's task spans (+1 when actions ran before step 1, which
      browser-use records as history step 0) == the `steps` it reports;
    - each step has one tool span per action result and at least one LLM call.
    """
    parents = [r for r in rows if r.get("log_type") in ("agent", "task")]
    ids = await asyncio.gather(*(asyncio.to_thread(_span_id, r) for r in parents))
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        children[row.get("span_parent_id") or ""].append(row)
    problems = []
    for row, span_id in zip(parents, ids, strict=True):
        kids = children.get(span_id or "", [])
        kinds = [k.get("log_type") for k in kids]
        # A step cut short (stop callback, max_steps) has a span but no history entry.
        recorded_steps = sum(
            1 for k in kids if k.get("log_type") == "task" and "results" in _agent_output(k)
        )
        output = _agent_output(row)
        if row.get("span_name") == "agent.browser_use" and output.get("is_done"):
            # browser-use's judge (use_judge=True) grades every finished run; a
            # missing verdict means its LLM call failed somewhere in the trace.
            if "judge_verdict" not in output:
                problems.append("finished run has no judge_verdict (judge call failed)")
        if row.get("span_name") == "agent.browser_use" and "steps" in output:
            recorded = recorded_steps + (1 if "tool" in kinds else 0)
            if recorded != output["steps"]:
                problems.append(f"agent reports {output['steps']} steps, trace has {recorded}")
        # A step whose model reply failed validation records one error result
        # and runs no action, so only steps with actions need tool spans.
        elif row.get("log_type") == "task" and "actions" in output:
            step = (row.get("metadata") or {}).get("browser_use.step.number")
            if kinds.count("tool") != len(output["results"]):
                problems.append(
                    f"step {step}: {len(output['results'])} action results, "
                    f"{kinds.count('tool')} tool spans"
                )
            if "chat" not in kinds:
                problems.append(f"step {step}: no LLM span")
    return problems


def _has_prompt_image(row: dict[str, Any]) -> bool:
    return "image_url" in json.dumps(row.get("prompt_messages") or row.get("input") or "")


@dataclass
class Result:
    scenario: Scenario
    traces: dict[str, list[dict[str, Any]]]
    problems: list[str]
    # Real failures the model ran into on live sites (a blocked search, a bad
    # selector). The spans are right to be errors; they are listed, not failed.
    notes: list[str] = field(default_factory=list)

    @property
    def agent_rows(self) -> list[dict[str, Any]]:
        return [r for rows in self.traces.values() for r in rows if r.get("log_type") == "agent"]

    @property
    def links(self) -> list[str]:
        rows = self.agent_rows or [r for rows in self.traces.values() for r in rows[:1]]
        return [f"{PLATFORM_URL}/platform/requests?unique_id={r['unique_id']}" for r in rows]


async def verify(scenario: Scenario) -> Result:
    expect = scenario.expect
    trace_ids = sorted(_traces.get(scenario.name, ()))
    traces = {t: await asyncio.to_thread(LOGS.wait_for_trace, t) for t in trace_ids}
    rows = [r for t in trace_ids for r in traces[t]]
    problems: list[str] = []

    def check(ok: bool, problem: str) -> None:
        if not ok:
            problems.append(problem)

    check(len(trace_ids) == expect.traces, f"{len(trace_ids)} traces, expected {expect.traces}")
    for trace_id, trace_rows in traces.items():
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
        agents = [r for r in trace_rows if r.get("log_type") == "agent"]
        if agents:
            session = (agents[0].get("metadata") or {}).get("browser_use.session.id")
            threads = {r.get("thread_identifier") for r in trace_rows}
            check(threads == {session}, f"trace {trace_id[:8]}: thread ids {threads} != agent session")

    problems.extend(await structure_problems(rows))
    agents = [r for r in rows if r.get("log_type") == "agent"]
    want_agents = expect.traces if expect.agent_spans is None else expect.agent_spans
    check(len(agents) == want_agents, f"{len(agents)} agent spans, expected {want_agents}")
    for row in rows:
        if expect.prompt_images is False and row.get("log_type") == "chat":
            check(not _has_prompt_image(row), f"{row.get('span_name')}: a screenshot reached the model")

    names = {r.get("span_name") for r in rows}
    for wanted in expect.spans_include:
        check(wanted in names, f"no {wanted} span")
    for model in expect.models_include:
        check(span_name(model) in names, f"no call to {model}")
    failed = {r.get("span_name") for r in rows if is_error(r)}
    for wanted in expect.error_spans:
        check(wanted in failed, f"{wanted} should be an error span")
    if expect.success is not None:
        runs = [r for r in agents if r.get("span_name") == "agent.browser_use"]
        verdicts = {_agent_output(r).get("is_successful") for r in runs}
        check(verdicts == {expect.success}, f"agent success {verdicts}, expected {expect.success}")
    incidental = sorted(failed - set(expect.error_spans) - {"task", "agent.browser_use"})
    notes = [f"model-caused error spans: {incidental}"] if incidental else []
    return Result(scenario, traces, problems, notes)


def print_result(result: Result) -> None:
    status = "PASS" if not result.problems else "FAIL"
    spans = sum(len(rows) for rows in result.traces.values())
    print(f"{status}  {result.scenario.name:<24} {spans:>3} spans  {result.scenario.title}")
    for problem in result.problems:
        print(f"        - {problem}")
    for note in result.notes:
        print(f"        ~ {note}")
    for link in result.links:
        print(f"        {link}")


def main(scenario: Scenario) -> None:
    """Run one scenario on its own: `python scenarios/<file>.py`."""
    client = init()
    try:
        asyncio.run(execute(scenario))
    finally:
        client.flush()
    print_result(asyncio.run(verify(scenario)))
