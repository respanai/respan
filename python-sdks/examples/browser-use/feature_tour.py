"""browser-use feature tour - one scenario per instrumentation feature.

    python feature_tour.py                 # list scenarios
    python feature_tour.py structured      # run one
    python feature_tour.py all             # run every scenario, one trace each
    python feature_tour.py secrets --no-content --no-screenshots

Every scenario prints its trace id; paste it into Respan's Traces search.
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv

load_dotenv(override=True)

RESPAN_API_KEY = os.getenv("RESPAN_API_KEY")
RESPAN_BASE_URL = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
RESPAN_MODEL = os.getenv("RESPAN_MODEL", "gpt-4.1-mini")
if not RESPAN_API_KEY:
    raise RuntimeError("Set RESPAN_API_KEY (copy .env.example to .env).")

from browser_use import ActionResult, Agent, Browser, ChatOpenAI, Tools
from opentelemetry import trace
from pydantic import BaseModel
from respan import Respan
from respan_instrumentation_browser_use import BrowserUseInstrumentor

# A public practice site built for automation demos; these are its published
# demo credentials, not real secrets.
LOGIN_SITE = "the-internet.herokuapp.com"
DEMO_USERNAME = "tomsmith"
DEMO_PASSWORD = "SuperSecretPassword!"


class Story(BaseModel):
    title: str
    points: int


class TopStories(BaseModel):
    stories: list[Story]


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=RESPAN_MODEL, api_key=RESPAN_API_KEY, base_url=RESPAN_BASE_URL)


def make_browser(args: argparse.Namespace, **kwargs) -> Browser:
    return Browser(headless=args.headless, **kwargs)


async def print_trace_id(agent: Agent) -> None:
    # on_step_start runs inside the agent span, so the current span's trace id
    # is this run's trace id.
    if agent.state.n_steps == 1:
        trace_id = trace.get_current_span().get_span_context().trace_id
        print(f"  trace id: {trace_id:032x}")


async def structured(args: argparse.Namespace) -> None:
    """Structured output: the agent must return a typed object (see the `done` tool span)."""
    agent = Agent(
        task="Open news.ycombinator.com and return the top 3 stories with their points.",
        llm=make_llm(),
        browser=make_browser(args),
        output_model_schema=TopStories,
    )
    history = await agent.run(max_steps=8, on_step_start=print_trace_id)
    result = history.get_structured_output(TopStories)
    for story in result.stories if result else []:
        print(f"  {story.points:>5}  {story.title}")


async def custom_tool(args: argparse.Namespace) -> None:
    """Your own Python function as a browser-use action: it shows up as its own `tool` span."""
    tools = Tools()
    notes: list[str] = []

    @tools.action("Save a short note for the user. Call once per fact worth keeping.")
    async def save_note(text: str) -> ActionResult:
        notes.append(text)
        # long_term_memory is what the agent remembers about this call in later
        # steps; without it the model cannot tell the note was already saved.
        return ActionResult(
            extracted_content=f"Saved note #{len(notes)}",
            long_term_memory=f"Saved note #{len(notes)}: {text}",
        )

    agent = Agent(
        task=(
            "Open https://en.wikipedia.org/wiki/Web_browser. Call save_note exactly twice: "
            "once for who made the first web browser, once for the year. Then finish."
        ),
        llm=make_llm(),
        browser=make_browser(args),
        tools=tools,
    )
    await agent.run(max_steps=8, on_step_start=print_trace_id)
    for note in notes:
        print(f"  note: {note}")


async def secrets(args: argparse.Namespace) -> None:
    """sensitive_data: the model and the trace only ever see <secret>name</secret> placeholders."""
    agent = Agent(
        task=(
            f"Go to https://{LOGIN_SITE}/login and log in with username demo_user "
            "and password demo_pass. Report the message shown after login."
        ),
        llm=make_llm(),
        # browser-use refuses sensitive_data without a domain allowlist.
        browser=make_browser(args, allowed_domains=[LOGIN_SITE]),
        sensitive_data={"demo_user": DEMO_USERNAME, "demo_pass": DEMO_PASSWORD},
    )
    history = await agent.run(max_steps=8, on_step_start=print_trace_id)
    print(f"  result: {history.final_result()}")


async def failure(args: argparse.Namespace) -> None:
    """Action errors: browser-use returns them instead of raising; their tool spans turn red."""
    agent = Agent(
        task=(
            "Open https://this-domain-does-not-exist.invalid and read its title. "
            "If it cannot load, say so and finish."
        ),
        llm=make_llm(),
        browser=make_browser(args),
        max_failures=2,
    )
    history = await agent.run(max_steps=4, on_step_start=print_trace_id)
    print(f"  errors: {[e for e in history.errors() if e]}")


SCENARIOS = {
    "structured": structured,
    "custom-tool": custom_tool,
    "secrets": secrets,
    "failure": failure,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", nargs="?", choices=[*SCENARIOS, "all"])
    parser.add_argument("--headless", action="store_true", help="hide the browser window")
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="capture_content=False: keep span structure, timings and tokens; drop prompts, results, URLs",
    )
    parser.add_argument(
        "--no-screenshots",
        action="store_true",
        help="capture_screenshots=False: no step screenshots, image placeholders in prompts",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        print(f"\n== {name}: {SCENARIOS[name].__doc__}")
        await SCENARIOS[name](args)


if __name__ == "__main__":
    args = parse_args()
    if args.scenario is None:
        for name, scenario in SCENARIOS.items():
            print(f"{name:<12} {scenario.__doc__}")
        raise SystemExit(0)

    respan = Respan(
        api_key=RESPAN_API_KEY,
        base_url=RESPAN_BASE_URL,
        app_name="browser-use-feature-tour",
        metadata={"example": "browser-use-feature-tour"},
        instrumentations=[
            BrowserUseInstrumentor(
                capture_content=not args.no_content,
                capture_screenshots=not args.no_screenshots,
            )
        ],
    )
    try:
        asyncio.run(main(args))
    finally:
        respan.flush()
