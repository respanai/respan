"""Shared setup for the scenarios: env, Respan, Skyvern, and live trace checks.

Every scenario runs under `customer_identifier=skyvern-scenarios` and
`metadata.scenario=<name>`, so its spans are easy to find in Respan. After a run,
`verify()` reads each trace back from the Respan logs API and checks it against the
scenario's `Expect` - the live half of the baseline (the offline half is the
package's `tests/test_baseline.py`).

Skyvern runs in local mode (`Skyvern.local()`): the engine runs in this process
with an in-memory database, and its model calls go through the Respan gateway.
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

from respan import Respan  # noqa: E402
from respan_instrumentation_skyvern import SkyvernInstrumentor  # noqa: E402
from respan_tracing.testing.live import (  # noqa: E402
    RespanLogsClient,
    is_error,
    json_field,
    trace_problems,
)
from skyvern import Skyvern  # noqa: E402
from skyvern.schemas.llm import LLMConfig  # noqa: E402

API_KEY = os.getenv("RESPAN_API_KEY")
BASE_URL = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api").rstrip("/")
PLATFORM_URL = os.getenv("RESPAN_PLATFORM_URL", "https://platform.respan.ai").rstrip("/")
MODEL = os.getenv("RESPAN_MODEL", "gpt-4.1-mini")
HEADLESS = os.getenv("HEADLESS") == "1"
CUSTOMER = "skyvern-scenarios"
RUN_ID = os.getenv("SCENARIO_RUN_ID") or time.strftime("%Y%m%d-%H%M%S")
if not API_KEY:
    raise RuntimeError("Set RESPAN_API_KEY (copy .env.example to .env).")

SCENARIO_ATTR = "respan.metadata.scenario"
_traces: dict[str, set[str]] = defaultdict(set)
_exported: Counter[str] = Counter()  # spans the SDK exported, per trace id
_client: Respan | None = None
_skyvern: Skyvern | None = None


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
            app_name="skyvern-scenarios",
            customer_identifier=CUSTOMER,
            metadata={"run_id": RUN_ID},
            instrumentations=[
                SkyvernInstrumentor(
                    capture_content=capture_content,
                    capture_screenshots=capture_screenshots,
                )
            ],
            span_postprocess_callback=_remember_trace,
        )
    return _client


def llm_config(model: str | None = None) -> LLMConfig:
    """Skyvern's model, served by the Respan gateway (OpenAI-compatible, one key)."""
    return LLMConfig(
        model_name=f"openai/{model or MODEL}",
        required_env_vars=[],
        supports_vision=True,
        add_assistant_prefix=False,
        litellm_params={"api_base": BASE_URL, "api_key": API_KEY},
    )


def skyvern() -> Skyvern:
    """One embedded Skyvern per process (Skyvern allows only one)."""
    global _skyvern
    if _skyvern is None:
        _skyvern = Skyvern.local(
            use_in_memory_db=True,
            llm_config=llm_config(),
            settings={"BROWSER_TYPE": "chromium-headless" if HEADLESS else "chromium-headful"},
        )
    return _skyvern


async def close() -> None:
    global _skyvern
    if _skyvern is not None:
        await _skyvern.aclose()
        _skyvern = None


@dataclass(frozen=True)
class Expect:
    traces: int = 1
    statuses: tuple[str, ...] = ("completed",)  # Skyvern's own run status, per run span
    spans_include: tuple[str, ...] = ()  # e.g. "tool.click" or "click"
    error_spans: tuple[str, ...] = ()  # spans that must fail
    prompts_include: tuple[str, ...] = ()  # Skyvern prompt names, e.g. "extract-actions"
    forbidden: tuple[str, ...] = ()  # strings no span may contain
    steps: bool = True  # the run goes through the step engine


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    source: str  # Skyvern's own example this mirrors
    look_for: str  # what to check in Respan
    run: Callable[[], Awaitable[Any]]
    expect: Expect = field(default_factory=Expect)


async def execute(scenario: Scenario) -> None:
    client = init()
    with client.propagate_attributes(metadata={"scenario": scenario.name}):
        await scenario.run()


# -- live verification --------------------------------------------------------


LOGS = RespanLogsClient(BASE_URL, API_KEY)


def _output(row: dict[str, Any]) -> dict[str, Any]:
    return json_field(row, "output")


def _span_id(row: dict[str, Any]) -> str | None:
    return LOGS.detail(row["unique_id"]).get("span_unique_id")


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("metadata") or {}


async def structure_problems(rows: list[dict[str, Any]], expect: Expect) -> list[str]:
    """Spans missing from the tree, judged against Skyvern's own record of the step.

    - each step span has one tool span per action Skyvern recorded, and the
      LLM call that chose them;
    - every LLM span carries the Skyvern prompt name that produced it.
    """
    steps = [r for r in rows if "skyvern.step.number" in _metadata(r)]
    ids = await asyncio.gather(*(asyncio.to_thread(_span_id, r) for r in steps))
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        children[row.get("span_parent_id") or ""].append(row)
    problems = []
    if expect.steps and not steps:
        problems.append("no step spans")
    for row, span_id in zip(steps, ids, strict=True):
        kinds = [k.get("log_type") for k in children.get(span_id or "", [])]
        number = _metadata(row).get("skyvern.step.number")
        output = _output(row)
        if "actions" in output and kinds.count("tool") != len(output["actions"]):
            problems.append(
                f"step {number}: {len(output['actions'])} actions, {kinds.count('tool')} tool spans"
            )
        if "chat" not in kinds:
            problems.append(f"step {number}: no LLM span")
    for row in rows:
        if row.get("log_type") == "chat" and not _metadata(row).get("skyvern.llm.prompt_name"):
            problems.append(f"{row.get('span_name')}: no prompt name")
    return problems


@dataclass
class Result:
    scenario: Scenario
    traces: dict[str, list[dict[str, Any]]]
    problems: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def roots(self) -> list[dict[str, Any]]:
        return [
            r for rows in self.traces.values() for r in rows if not r.get("span_parent_id")
        ]

    @property
    def links(self) -> list[str]:
        return [f"{PLATFORM_URL}/platform/requests?unique_id={r['unique_id']}" for r in self.roots]


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
        root = next((r for r in trace_rows if not r.get("span_parent_id")), None)
        run_id = _metadata(root or {}).get("skyvern.run.id")
        threads = {r.get("thread_identifier") for r in trace_rows}
        check(threads == {run_id}, f"trace {trace_id[:8]}: thread ids {threads} != Skyvern run id")

    problems.extend(await structure_problems(rows, expect))
    statuses = sorted(
        str(_output(r).get("status"))
        for r in rows
        if r.get("log_type") == "agent" and "status" in _output(r)
    )
    check(
        statuses == sorted(expect.statuses),
        f"run statuses {statuses}, expected {sorted(expect.statuses)}",
    )
    names = {r.get("span_name") for r in rows}
    for wanted in expect.spans_include:
        check(wanted in names or f"tool.{wanted}" in names, f"no {wanted} span")
    prompts = {_metadata(r).get("skyvern.llm.prompt_name") for r in rows}
    for wanted in expect.prompts_include:
        check(wanted in prompts, f"no LLM call for prompt {wanted}")
    failed = {r.get("span_name") for r in rows if is_error(r)}
    for wanted in expect.error_spans:
        check(
            wanted in failed or f"tool.{wanted}" in failed, f"{wanted} should be an error span"
        )
    incidental = sorted(
        failed - set(expect.error_spans) - {f"tool.{e}" for e in expect.error_spans}
    )
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

    async def run() -> None:
        try:
            await execute(scenario)
        finally:
            await close()

    try:
        asyncio.run(run())
    finally:
        client.flush()
    print_result(asyncio.run(verify(scenario)))


def dumps(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)
