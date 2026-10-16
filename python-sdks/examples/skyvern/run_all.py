"""Run the Skyvern scenarios, then check every trace in Respan.

    python run_all.py                      # every scenario
    python run_all.py quickstart form_filling # some of them (by name)
    python run_all.py --list
    python run_all.py --headless --no-screenshots

Prints PASS/FAIL per scenario with a link to its agent span in Respan, and writes
the same report to runs/<run id>.md.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENARIO_DIR = HERE / "scenarios"


def load_scenarios():
    sys.path.insert(0, str(SCENARIO_DIR))
    scenarios = []
    for path in sorted(SCENARIO_DIR.glob("[0-9][0-9]_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scenarios.append(module.SCENARIO)
    return scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="scenario names (default: all)")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    parser.add_argument("--headless", action="store_true", help="headless Chromium")
    parser.add_argument("--no-content", action="store_true", help="SkyvernInstrumentor(capture_content=False)")
    parser.add_argument("--no-screenshots", action="store_true", help="SkyvernInstrumentor(capture_screenshots=False)")
    return parser.parse_args()


async def run(scenarios, harness) -> list:
    try:
        for scenario in scenarios:
            print(f"\n== {scenario.name}: {scenario.title}")
            try:
                await harness.execute(scenario)
            except Exception as error:  # noqa: BLE001 - keep going; verify() reports it
                print(f"  run raised {type(error).__name__}: {error}")
    finally:
        await harness.close()  # one embedded Skyvern for the whole run
    harness.init().flush()
    print("\nChecking traces in Respan (ingestion takes a moment)...\n")
    return [await harness.verify(scenario) for scenario in scenarios]


def write_report(results, harness) -> Path:
    lines = [
        f"# Skyvern scenarios - run {harness.RUN_ID}",
        "",
        f"All spans: customer `{harness.CUSTOMER}`, metadata `run_id={harness.RUN_ID}`.",
        "",
        "| | Scenario | Spans | What to look for | Links | Problems |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        status = "PASS" if not result.problems else "FAIL"
        spans = sum(len(rows) for rows in result.traces.values())
        links = " ".join(f"[trace {i + 1}]({link})" for i, link in enumerate(result.links))
        problems = "<br>".join([*result.problems, *(f"note: {n}" for n in result.notes)])
        lines.append(
            f"| {status} | `{result.scenario.name}` {result.scenario.title} | {spans} "
            f"| {result.scenario.look_for} | {links} | {problems} |"
        )
    path = HERE / "runs" / f"{harness.RUN_ID}.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    args = parse_args()
    if args.headless:
        os.environ["HEADLESS"] = "1"
    scenarios = load_scenarios()
    import _harness as harness  # after HEADLESS is set; loaded by load_scenarios()

    if args.list:
        for scenario in scenarios:
            print(f"{scenario.name:<22} {scenario.title:<36} {scenario.source}")
        return
    if args.names:
        unknown = set(args.names) - {s.name for s in scenarios}
        if unknown:
            raise SystemExit(f"unknown scenarios: {sorted(unknown)}")
        scenarios = [s for s in scenarios if s.name in args.names]

    harness.init(capture_content=not args.no_content, capture_screenshots=not args.no_screenshots)
    results = asyncio.run(run(scenarios, harness))
    for result in results:
        harness.print_result(result)
    passed = sum(not r.problems for r in results)
    print(f"\n{passed}/{len(results)} passed. Report: {write_report(results, harness)}")
    print(f"All runs: {harness.PLATFORM_URL}/platform/requests?customer_identifier={harness.CUSTOMER}")


if __name__ == "__main__":
    main()
