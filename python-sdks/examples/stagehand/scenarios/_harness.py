"""Shared setup for the scenarios: env, Respan, the model, the browser, local
fixture pages, and live trace checks.

Every scenario runs under `customer_identifier=stagehand-py-scenarios` with
`metadata.scenario=<name>` and `metadata.run_id=<run>`. After a run, `verify()`
reads each trace back from the Respan logs API and checks it against the
scenario's `Expect` - the live half of the package's baseline test
(`tests/test_baseline.py`).
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parents[1]
load_dotenv(HERE / ".env", override=True)

from openai import AsyncOpenAI  # noqa: E402
from respan import Respan  # noqa: E402
from respan_instrumentation_stagehand import StagehandInstrumentor, with_model_name  # noqa: E402
from respan_tracing.testing.live import (  # noqa: E402
    RespanLogsClient,
    is_error,
    json_field,
    trace_problems,
)
from stagehand import (  # noqa: E402
    LLMMessageGenerateResult,
    LLMStructuredGenerateResult,
    Stagehand,
    local_browser,
)

API_KEY = os.getenv("RESPAN_API_KEY")
BASE_URL = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api").rstrip("/")
PLATFORM_URL = os.getenv("RESPAN_PLATFORM_URL", "https://platform.respan.ai").rstrip("/")
MODEL = os.getenv("RESPAN_MODEL", "gpt-4.1-mini")
SMALL_MODEL = os.getenv("RESPAN_SMALL_MODEL", "gpt-4o-mini")
CUSTOMER = "stagehand-py-scenarios"
RUN_ID = os.getenv("SCENARIO_RUN_ID") or time.strftime("%Y%m%d-%H%M%S")
if not API_KEY:
    raise RuntimeError("Set RESPAN_API_KEY (copy .env.example to .env).")

SCENARIO_ATTR = "respan.metadata.scenario"
SESSION_SPAN = "agent.stagehand"
_traces: dict[str, set[str]] = defaultdict(set)
_exported: Counter[str] = Counter()  # spans the SDK exported, per trace id
_client: Respan | None = None


# -- the model: Stagehand v4's client-model callback, through the Respan gateway --


def _openai_content(content: Any) -> list[dict[str, Any]]:
    blocks = content if isinstance(content, list) else [content]
    parts = []
    for block in blocks:
        if block.get("type") == "text":
            parts.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image":
            url = f"data:{block['mime_type']};base64,{block['data']}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def respan_model(model: str | None = None) -> Callable[[Any], Awaitable[Any]]:
    """`Stagehand.create(model=respan_model())`: every inference Stagehand's
    browser extension needs comes back to this process as a call to this
    function. Here it goes to the Respan gateway (OpenAI-compatible), so one
    Respan key covers the traces and the model calls."""
    model = model or MODEL
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    async def generate(params: Any) -> Any:
        request = params.model_dump(mode="json", by_alias=True, exclude_none=True)
        messages: list[dict[str, Any]] = []
        if request.get("system_prompt"):
            messages.append({"role": "system", "content": request["system_prompt"]})
        for message in request["messages"]:
            messages.append({"role": message["role"], "content": _openai_content(message["content"])})
        response_format = request.get("response_format") or {}
        structured = response_format.get("type") == "json_schema"
        options: dict[str, Any] = {}
        if structured:
            options["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": response_format["name"], "schema": response_format["schema"]},
            }
        if request.get("temperature") is not None:
            options["temperature"] = request["temperature"]
        if request.get("stop_sequences"):
            options["stop"] = request["stop_sequences"]
        if request.get("tools"):
            options["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description"),
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in request["tools"]
            ]
        response = await client.chat.completions.create(model=model, messages=messages, **options)
        choice = response.choices[0]
        text = choice.message.content or ""
        content: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        for call in choice.message.tool_calls or ():
            arguments = json.loads(call.function.arguments or "{}")
            content.append({"type": "tool_use", "id": call.id, "name": call.function.name, "input": arguments})
        usage = None
        if response.usage:
            details = response.usage.prompt_tokens_details
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cached_input_tokens": (details.cached_tokens if details else 0) or 0,
            }
        common = {"role": "assistant", "content": content, "stop_reason": choice.finish_reason, "usage": usage}
        if structured:
            return LLMStructuredGenerateResult(
                **common, output_format="json_schema", structured_content=json.loads(text)
            )
        return LLMMessageGenerateResult(**common, output_format="text")

    # Stagehand knows nothing about the model behind `generate`; this tells
    # Respan, so the LLM spans name it and get priced.
    return with_model_name(generate, model)


# -- Respan ---------------------------------------------------------------------


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
            app_name="stagehand-py-scenarios",
            customer_identifier=CUSTOMER,
            metadata={"run_id": RUN_ID},
            instrumentations=[
                StagehandInstrumentor(
                    capture_content=capture_content,
                    capture_screenshots=capture_screenshots,
                )
            ],
            span_postprocess_callback=_remember_trace,
            # Also trace the OpenAI SDK, as an app using both would: the model
            # call inside `generate` must still show up once, as Stagehand's
            # chat span (the token check in stagehand_problems() would catch a
            # second copy).
            is_auto_instrument=True,
        )
    return _client


# -- browser, fixtures ------------------------------------------------------------


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass


@functools.cache
def fixtures_url() -> str:
    """Serve ./fixtures on 127.0.0.1 for the rest of the process."""
    handler = functools.partial(_QuietHandler, directory=str(HERE / "fixtures"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


async def with_stagehand(
    run: Callable[[Stagehand, Any, Any], Awaitable[Any]],
    *,
    model: str | None = None,
    user_data_dir: str | None = None,
    system_prompt: str | None = None,
) -> Any:
    """Launch local Chrome, attach Stagehand with the Respan model, run, close."""
    options: dict[str, Any] = {"headless": os.getenv("HEADLESS") == "1"}
    if user_data_dir:
        options.update(user_data_dir=user_data_dir, preserve_user_data_dir=True)
    browser = await local_browser.launch(**options)
    try:
        stagehand = await Stagehand.create(
            browser=browser,
            model=respan_model(model),
            logging={"level": "error"},
            **({"system_prompt": system_prompt} if system_prompt else {}),
        )
        try:
            [page] = await browser.context.pages()
            return await run(stagehand, page, browser)
        finally:
            await stagehand.close()
    finally:
        await browser.close()


# -- scenarios -----------------------------------------------------------------------


@dataclass(frozen=True)
class Expect:
    traces: int = 1  # = Stagehand sessions
    spans_include: tuple[str, ...] = ()  # instrumentation span names, e.g. "page.screenshot"
    error_spans: tuple[str, ...] = ()  # spans that must be errors
    forbidden: tuple[str, ...] = ()  # strings no stored span may contain (secrets)
    prompt_images: bool | None = None  # True: some LLM prompt carried an image
    models: tuple[str, ...] = ()  # models that must have been called


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    source: str  # Stagehand's own example this mirrors
    look_for: str  # what to check in Respan
    run: Callable[[], Awaitable[Any]]
    expect: Expect = field(default_factory=Expect)


async def execute(scenario: Scenario) -> None:
    client = init()
    with client.propagate_attributes(metadata={"scenario": scenario.name}):
        await scenario.run()


# -- live verification --------------------------------------------------------------

LOGS = RespanLogsClient(BASE_URL, API_KEY)


def entity(row: dict[str, Any]) -> str | None:
    """The instrumentation's own span name (Respan shows task spans as plain `task`)."""
    return (row.get("metadata") or {}).get("traceloop.entity.name") or row.get("span_name")


def _span_id(row: dict[str, Any]) -> str | None:
    return LOGS.detail(row["unique_id"]).get("span_unique_id")


async def stagehand_problems(rows: list[dict[str, Any]]) -> list[str]:
    """Checks against Stagehand's own record of the session.

    - one session span per trace, and every span's thread id is its id;
    - the session's operation count matches the task spans in the trace;
    - each operation's LLM spans add up to the input tokens Stagehand itself
      reports for that operation (``metadata.usage`` on its result).
    """
    sessions = [r for r in rows if r.get("span_name") == SESSION_SPAN]
    if len(sessions) != 1:
        return [f"{len(sessions)} {SESSION_SPAN} spans, expected 1"]
    problems: list[str] = []
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
    ids = await asyncio.gather(*(asyncio.to_thread(_span_id, op) for op in operations))
    for op, span_id in zip(operations, ids, strict=True):
        if is_error(op):
            continue
        usage = (json_field(op, "output").get("metadata") or {}).get("usage") or {}
        chats = [c for c in children.get(span_id or "", []) if c.get("log_type") == "chat"]
        want = usage.get("input_tokens", 0)
        got = sum(c.get("prompt_tokens") or 0 for c in chats)
        if want and not chats:
            problems.append(f"{entity(op)}: inference ran but no LLM span")
        elif want and got != want:
            problems.append(f"{entity(op)}: Stagehand reports {want} input tokens, its LLM spans {got}")
    return problems


def _has_prompt_image(row: dict[str, Any]) -> bool:
    return "image_url" in json.dumps(row.get("prompt_messages") or row.get("input") or "")


@dataclass
class Result:
    scenario: Scenario
    traces: dict[str, list[dict[str, Any]]]
    problems: list[str]

    @property
    def links(self) -> list[str]:
        roots = [r for rows in self.traces.values() for r in rows if not r.get("span_parent_id")]
        return [f"{PLATFORM_URL}/platform/requests?unique_id={r['unique_id']}" for r in roots]


async def verify(scenario: Scenario) -> Result:
    expect = scenario.expect
    trace_ids = sorted(_traces.get(scenario.name, ()))
    traces = {t: await asyncio.to_thread(LOGS.wait_for_trace, t) for t in trace_ids}
    rows = [r for t in trace_ids for r in traces[t]]
    problems: list[str] = []
    if len(trace_ids) != expect.traces:
        problems.append(f"{len(trace_ids)} traces, expected {expect.traces}")
    for trace_id, trace_rows in traces.items():
        found = trace_problems(
            trace_rows,
            exported=_exported[trace_id],
            customer_identifier=CUSTOMER,
            metadata={"scenario": scenario.name, "run_id": RUN_ID},
            forbidden=expect.forbidden,
        )
        found += await stagehand_problems(trace_rows)
        problems.extend(f"trace {trace_id[:8]}: {problem}" for problem in found)
    names = {entity(r) for r in rows}
    problems += [f"no {wanted} span" for wanted in expect.spans_include if wanted not in names]
    failed = {entity(r) for r in rows if is_error(r)}
    problems += [f"{wanted} should be an error span" for wanted in expect.error_spans if wanted not in failed]
    stored_models = {r.get("model") or "" for r in rows if r.get("log_type") == "chat"}
    for model in expect.models:
        # Respan may store the provider-resolved name (e.g. "azure/<model>").
        if not any(m == model or m.endswith(f"/{model}") for m in stored_models):
            problems.append(f"no call to {model}")
    if expect.prompt_images is not None:
        has_image = any(_has_prompt_image(r) for r in rows if r.get("log_type") == "chat")
        if has_image != expect.prompt_images:
            problems.append(f"prompt images {has_image}, expected {expect.prompt_images}")
    return Result(scenario, traces, problems)


def print_result(result: Result) -> None:
    status = "PASS" if not result.problems else "FAIL"
    spans = sum(len(rows) for rows in result.traces.values())
    print(f"{status}  {result.scenario.name:<22} {spans:>3} spans  {result.scenario.title}")
    for problem in result.problems:
        print(f"        - {problem}")
    for trace_id in sorted(result.traces):
        print(f"        trace {trace_id}")
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
