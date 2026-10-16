from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import sys
import threading
from contextlib import ExitStack
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import litellm
import pytest
from opentelemetry import trace
from respan_instrumentation_skyvern import SkyvernInstrumentor
from respan_tracing.testing import UNREACHABLE_BASE_URL, capture_exported_spans

from respan import Respan

PAGES = {
    "index.html": (
        "<title>Respan Test</title><h1>Hello skyvern</h1>"
        "<a href='next.html'>Next page</a>"
        "<form action='next.html'><input name='token' placeholder='Token'>"
        "<button type='submit'>Send</button></form>"
    ),
    "next.html": "<title>Next</title><h1>Second page</h1><p>The secret number is 42.</p>",
}
# Skyvern names the prompt of each LLM call; the scripted model answers by it.
EXTRACT_ACTIONS = "extract-actions"
USAGE = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}


def _chromium_installed() -> bool:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        return Path(playwright.chromium.executable_path).exists()


@pytest.fixture(scope="session", autouse=True)
def chromium():
    """Skyvern drives Playwright's own Chromium; CI installs it, locally it must exist."""
    if _chromium_installed():
        return
    if not os.environ.get("CI"):
        pytest.skip("needs Playwright Chromium: python -m playwright install chromium")
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)


@pytest.fixture
def site(tmp_path):
    for name, html in PAGES.items():
        (tmp_path / name).write_text(html)
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/index.html"
    server.shutdown()


def action(action_type: str, target: str | None = None, **fields: Any) -> dict[str, Any]:
    """An action for the scripted model; `target` is the visible text of the element."""
    return {"action_type": action_type, "target": target, **fields}


class ScriptedModel:
    """Replaces `litellm.acompletion`: each extract-actions call takes the next turn.

    Turns are lists of actions; a turn ending in COMPLETE finishes the task. Every
    other prompt (goal checks, extraction) answers "goal achieved" once the page
    shows `goal_text`.
    Instrumentation wraps whatever `litellm.acompletion` is at activation, so this
    keeps Skyvern's handler code and the Respan wrapper on the real path.
    """

    def __init__(self, goal_text: str = "Second page"):
        self.goal_text = goal_text
        self.turns: dict[str, list[list[dict[str, Any]]]] = {}
        self.prompts: list[str] = []
        # Runs inside Skyvern's engine, before each answer (e.g. to register a secret).
        self.on_call: Any = None

    def script(self, *turns: list[dict[str, Any]], key: str = "") -> None:
        self.turns[key] = [list(turn) for turn in turns]

    def _turns_for(self, prompt: str) -> list[list[dict[str, Any]]]:
        # Parallel runs are told apart by a key in their prompt.
        for key, turns in self.turns.items():
            if key and key in prompt:
                return turns
        return self.turns.setdefault("", [])

    async def __call__(self, *args: Any, **kwargs: Any) -> litellm.ModelResponse:
        prompt = "\n".join(
            part.get("text", "")
            for message in kwargs.get("messages") or ()
            for part in (
                message["content"]
                if isinstance(message.get("content"), list)
                else [{"text": message.get("content") or ""}]
            )
        )
        name = _prompt_name()
        self.prompts.append(name or "")
        if self.on_call is not None:
            self.on_call()
        if name == EXTRACT_ACTIONS:
            turns = self._turns_for(prompt)
            turn = turns.pop(0) if turns else [action("COMPLETE")]
            actions = [_resolve(a, prompt) for a in turn]
            done = any(a["action_type"] == "COMPLETE" for a in actions)
            payload: dict[str, Any] = {
                "user_goal_stage": "scripted",
                "user_goal_achieved": done,
                "action_plan": "scripted",
                "actions": actions,
            }
        else:
            payload = {
                "page_info": "",
                "thoughts": "",
                "user_goal_achieved": self.goal_text in prompt,
            }
        return litellm.ModelResponse(
            model="scripted-1",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(payload)},
                }
            ],
            usage=dict(USAGE),
        )


def _prompt_name() -> str | None:
    """The prompt name Skyvern (native span) or Respan (chat span) put on the current span."""
    attributes = getattr(trace.get_current_span(), "attributes", None) or {}
    return attributes.get("skyvern.llm.prompt_name") or attributes.get("prompt_name")


def _resolve(scripted: dict[str, Any], prompt: str) -> dict[str, Any]:
    data = {
        "reasoning": "scripted",
        "confidence_float": 1.0,
        "id": None,
        "text": None,
        **{k: v for k, v in scripted.items() if k != "target"},
    }
    if target := scripted.get("target"):
        found = re.search(rf'id="(\w+)"[^>]*>\s*{re.escape(target)}', prompt) or re.search(
            rf'id="(\w+)"[^>]*placeholder="{re.escape(target)}"', prompt
        )
        data["id"] = found.group(1) if found else target  # unknown target -> action fails
    return data


@pytest.fixture
def model(monkeypatch):
    scripted = ScriptedModel()
    monkeypatch.setattr(litellm, "acompletion", scripted)
    return scripted


@pytest.fixture
def pipeline(model):
    """Build `Respan(...)` with the Skyvern instrumentor; return (client, exported spans).

    Depends on `model`, so the instrumentation wraps the scripted litellm call.
    """
    with ExitStack() as stack:

        def make(instrumentor: SkyvernInstrumentor | None = None, **kwargs: Any):
            return stack.enter_context(
                capture_exported_spans(
                    lambda on_export: Respan(
                        api_key="test-key",
                        base_url=UNREACHABLE_BASE_URL,
                        instrumentations=[instrumentor or SkyvernInstrumentor()],
                        span_postprocess_callback=on_export,
                        **kwargs,
                    )
                )
            )

        yield make
