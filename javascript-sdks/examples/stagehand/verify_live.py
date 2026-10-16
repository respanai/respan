"""Read a run's traces back from Respan and check them.

    python verify_live.py runs/<run id>.json [scenario ...] [--no-metadata-check]

``--no-metadata-check`` skips the customer-metadata check, for SDK setups
whose span metadata does not reach Respan (the other checks still run).

The generic checks (every exported span stored, one root, customer id and
metadata on every span, one thread id, tokens and cost on every LLM span, no
secrets) come from ``respan_tracing.testing.live``. The Stagehand-specific ones:

- one ``agent.stagehand`` session span per trace, and every span's thread id is
  that session's id;
- the session's own operation count matches the task spans in the trace;
- each operation's LLM spans add up to the token usage Stagehand itself
  reports for that operation (``metadata.usage`` on its result);
- the scenario's expected spans, error spans, models and prompt images.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env", override=True)

from respan_tracing.testing.live import (  # noqa: E402
    RespanLogsClient,
    is_error,
    json_field,
    trace_problems,
)

API_KEY = os.environ["RESPAN_API_KEY"]
BASE_URL = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api").rstrip("/")
PLATFORM_URL = os.getenv("RESPAN_PLATFORM_URL", "https://platform.respan.ai").rstrip("/")
LOGS = RespanLogsClient(BASE_URL, API_KEY)
SESSION_SPAN = "agent.stagehand"


def entity(row: dict[str, Any]) -> str | None:
    """The instrumentation's own span name (Respan shows task spans as plain `task`)."""
    return (row.get("metadata") or {}).get("traceloop.entity.name") or row.get("span_name")


def span_id(row: dict[str, Any]) -> str | None:
    return LOGS.detail(row["unique_id"]).get("span_unique_id")


def stagehand_problems(rows: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    sessions = [r for r in rows if r.get("span_name") == SESSION_SPAN]
    if len(sessions) != 1:
        return [f"{len(sessions)} {SESSION_SPAN} spans, expected 1"]
    session = sessions[0]
    session_id = (session.get("metadata") or {}).get("stagehand.session.id")
    threads = {r.get("thread_identifier") for r in rows}
    if not session_id or threads != {session_id}:
        problems.append(f"thread ids {threads} != session id {session_id}")
    operations = [r for r in rows if r.get("log_type") == "task"]
    reported = json_field(session, "output").get("operations")
    if reported is not None and reported != len(operations):
        problems.append(f"session reports {reported} operations, trace has {len(operations)} task spans")
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        children[row.get("span_parent_id") or ""].append(row)
    for op in operations:
        if is_error(op):
            continue
        usage = (json_field(op, "output").get("metadata") or {}).get("usage") or {}
        chats = [c for c in children.get(span_id(op) or "", []) if c.get("log_type") == "chat"]
        want = usage.get("inputTokens", 0)
        got = sum(c.get("prompt_tokens") or 0 for c in chats)
        if want and got != want:
            problems.append(f"{entity(op)}: Stagehand reports {want} input tokens, its LLM spans {got}")
        if want and not chats:
            problems.append(f"{entity(op)}: inference ran but no LLM span")
    return problems


def check(scenario: dict[str, Any], run: dict[str, Any], *, metadata: bool) -> tuple[list[str], list[str], int]:
    expect = scenario.get("expect") or {}
    traces = scenario.get("traces") or {}
    problems: list[str] = []
    if scenario.get("error"):
        problems.append(f"run raised {scenario['error']}")
    if len(traces) != expect.get("traces", 1):
        problems.append(f"{len(traces)} traces, expected {expect.get('traces', 1)}")
    all_rows: list[dict[str, Any]] = []
    links: list[str] = []
    for trace_id, exported in traces.items():
        rows = LOGS.wait_for_trace(trace_id)
        all_rows.extend(rows)
        problems.extend(
            f"trace {trace_id[:8]}: {p}"
            for p in trace_problems(
                rows,
                exported=exported,
                customer_identifier=run["customer"],
                metadata={"scenario": scenario["name"], "run_id": run["run_id"]} if metadata else None,
                forbidden=expect.get("forbidden", ()),
            )
        )
        problems.extend(f"trace {trace_id[:8]}: {p}" for p in stagehand_problems(rows))
        roots = [r for r in rows if not r.get("span_parent_id")]
        links += [f"{PLATFORM_URL}/platform/requests?unique_id={r['unique_id']}" for r in roots]
    names = {entity(r) for r in all_rows}
    for wanted in expect.get("spansInclude", ()):
        if wanted not in names:
            problems.append(f"no {wanted} span")
    failed = {entity(r) for r in all_rows if is_error(r)}
    for wanted in expect.get("errorSpans", ()):
        if wanted not in failed:
            problems.append(f"{wanted} should be an error span")
    for model in expect.get("models", ()):
        # Respan may store the provider-resolved name (e.g. "azure/<model>").
        stored = {r.get("model") or "" for r in all_rows if r.get("log_type") == "chat"}
        if not any(m == model or m.endswith(f"/{model}") for m in stored):
            problems.append(f"no call to {model}")
    if "promptImages" in expect:
        has_image = any(
            "image_url" in json.dumps(r.get("prompt_messages") or r.get("input") or "")
            for r in all_rows
            if r.get("log_type") == "chat"
        )
        if has_image != expect["promptImages"]:
            problems.append(f"prompt images {has_image}, expected {expect['promptImages']}")
    return problems, links, len(all_rows)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    metadata = "--no-metadata-check" not in sys.argv
    run = json.loads(Path(sys.argv[1]).read_text())
    only = set(args)
    results = []
    for scenario in run["scenarios"]:
        if only and scenario["name"] not in only:
            continue
        problems, links, spans = check(scenario, run, metadata=metadata)
        results.append((scenario, problems, links, spans))
        status = "PASS" if not problems else "FAIL"
        print(f"{status}  {scenario['name']:<22} {spans:>3} spans  {scenario['title']}")
        for problem in problems:
            print(f"        - {problem}")
        for link in links:
            print(f"        {link}")
    passed = sum(not p for _, p, _, _ in results)
    print(f"\n{passed}/{len(results)} passed (run {run['run_id']}, customer {run['customer']}).")
    report = Path(sys.argv[1]).with_suffix(".md")
    lines = [f"# Stagehand scenarios - run {run['run_id']}", "", "| | Scenario | Spans | What to look for | Traces | Problems |", "|---|---|---|---|---|---|"]
    for scenario, problems, links, spans in results:
        lines.append(
            f"| {'PASS' if not problems else 'FAIL'} | `{scenario['name']}` {scenario['title']} | {spans} "
            f"| {scenario['look_for']} | {' '.join(f'[{i + 1}]({l})' for i, l in enumerate(links))} | {'<br>'.join(problems)} |"
        )
    report.write_text("\n".join(lines) + "\n")
    print(f"Report: {report}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
