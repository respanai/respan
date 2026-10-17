"""Run the Kernel scenarios, then check every trace in Respan.

    python run_all.py                        # every scenario, on real Kernel browsers
    python run_all.py --fake                 # on the local fake Kernel API + local Chrome
    python run_all.py sample_app replays     # some of them (by name)
    python run_all.py --list
    python run_all.py --no-content --probe-passwords

Prints PASS/FAIL per scenario with its trace id and a link to its root span in
Respan, and writes the same report to runs/<run id>.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENARIO_DIR = HERE / "scenarios"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("names", nargs="*", help="scenario names (default: all)")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="the package's local fake Kernel API with a local Chrome (no Kernel account)",
    )
    parser.add_argument("--headed", action="store_true", help="show the browser windows")
    parser.add_argument(
        "--no-content", action="store_true", help="RESPAN_TRACE_CONTENT=false (structure only)"
    )
    parser.add_argument(
        "--probe-passwords",
        action="store_true",
        help="KernelInstrumentor(probe_password_fields=True)",
    )
    return parser.parse_args()


def load_scenarios():
    sys.path.insert(0, str(SCENARIO_DIR))
    scenarios = []
    for path in sorted(SCENARIO_DIR.glob("[0-9][0-9]_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scenarios.append(module.SCENARIO)
    return scenarios


def write_report(results, harness) -> Path:
    mode = "fake Kernel API + local Chrome" if harness.FAKE else "Kernel cloud browsers"
    lines = [
        f"# Kernel scenarios - run {harness.RUN_ID} ({mode})",
        "",
        f"All spans: customer `{harness.CUSTOMER}`, metadata `run_id={harness.RUN_ID}`.",
        "",
        "| | Scenario | Spans | What to look for | Traces | Problems |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        status = "PASS" if not result.problems else "FAIL"
        spans = sum(len(rows) for rows in result.traces.values())
        links = " ".join(
            f"[{t[:8]}]({link})"
            for t, link in zip(sorted(result.traces), result.links, strict=False)
        )
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
    # Read by the harness at import, before the scenarios register their actions.
    if args.fake:
        os.environ["KERNEL_FAKE"] = "1"
    if args.headed:
        os.environ["HEADED"] = "1"
    if args.no_content:
        os.environ["RESPAN_TRACE_CONTENT"] = "false"
    if args.probe_passwords:
        os.environ["KERNEL_PROBE_PASSWORDS"] = "1"
    scenarios = load_scenarios()
    import _harness as harness  # loaded by load_scenarios()

    if args.list:
        for scenario in scenarios:
            print(f"{scenario.name:<20} {scenario.title:<50} {scenario.source}")
        harness.shutdown()
        return
    if args.names:
        unknown = set(args.names) - {s.name for s in scenarios}
        if unknown:
            raise SystemExit(f"unknown scenarios: {sorted(unknown)}")
        scenarios = [s for s in scenarios if s.name in args.names]

    try:
        for scenario in scenarios:
            print(f"\n== {scenario.name}: {scenario.title}")
            try:
                harness.execute(scenario)
            except Exception as error:  # noqa: BLE001 - keep going; verify() reports it
                print(f"  run raised {type(error).__name__}: {error}")
    finally:
        harness.shutdown()
    print("\nChecking traces in Respan (ingestion takes a moment)...\n")
    results = [harness.verify(scenario) for scenario in scenarios]
    for result in results:
        harness.print_result(result)
    passed = sum(not r.problems for r in results)
    print(f"\n{passed}/{len(results)} passed. Report: {write_report(results, harness)}")
    print(
        f"All runs: {harness.PLATFORM_URL}/platform/requests?customer_identifier={harness.CUSTOMER}"
    )


if __name__ == "__main__":
    main()
