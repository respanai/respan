"""Shared fixtures: a local HTTP page, a scripted Stagehand client model, headless Chrome.

Nothing here touches the network beyond 127.0.0.1.
"""

from __future__ import annotations

import functools
import json
import re
import shutil
import threading
from collections.abc import Callable
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from stagehand import LLMMessageGenerateResult, LLMStructuredGenerateResult

PAGE = """<!doctype html><title>Respan Test</title>
<h1>Hello browser agent</h1>
<form onsubmit="event.preventDefault(); document.title = 'Signed in';">
  <label>Email <input id="email" type="email"></label>
  <label>Password <input id="password" type="password"></label>
  <button id="go" type="submit">Sign in</button>
</form>"""
SECOND_PAGE = "<title>Second</title><h1>Second page heading</h1>"

USAGE = {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150, "cached_input_tokens": 20}


def chrome_path() -> str | None:
    mac = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if mac.exists():
        return str(mac)
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        if found := shutil.which(name):
            return found
    return None


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture(scope="session")
def site(tmp_path_factory):
    """PAGE at <site>/index.html and SECOND_PAGE at <site>/second.html."""
    root = tmp_path_factory.mktemp("site")
    (root / "index.html").write_text(PAGE)
    (root / "second.html").write_text(SECOND_PAGE)
    handler = functools.partial(_QuietHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _element_id(params: Any, pattern: str) -> str:
    """Accessibility-tree id of the first line matching `pattern`, e.g. "button: Sign in"."""
    text = json.dumps(params.model_dump(mode="json")["messages"])
    for line in text.split("\\n"):
        if re.search(pattern, line) and (match := re.search(r"\[(\d+-\d+)\]", line)):
            return match.group(1)
    raise AssertionError(f"scripted model: no element for {pattern}")


class ScriptedModel:
    """A Stagehand v4 client model (``model=<callable>``) that answers every
    structured inference deterministically. act and observe target the element
    whose accessibility-tree line matches ``target``."""

    def __init__(
        self,
        *,
        target: str = "button: Sign in",
        method: str = "click",
        arguments: list[str] | None = None,
        extraction: str | Callable[[Any], str] = "Hello browser agent",
        on_call: Callable[[Any], None] | None = None,
    ) -> None:
        self.target = target
        self.method = method
        self.arguments = arguments or []
        self.extraction = extraction
        self.on_call = on_call
        self.calls: list[Any] = []

    async def __call__(self, params: Any) -> Any:
        self.calls.append(params)
        if self.on_call:
            self.on_call(params)
        response_format = getattr(params, "response_format", None)
        name = getattr(response_format, "name", None)

        def structured(content: dict[str, Any]) -> LLMStructuredGenerateResult:
            return LLMStructuredGenerateResult(
                role="assistant",
                content=[{"type": "text", "text": json.dumps(content)}],
                output_format="json_schema",
                structured_content=content,
                usage=USAGE,
            )

        action = lambda: {  # noqa: E731
            "elementId": _element_id(params, self.target),
            "description": "the target",
            "method": self.method,
            "arguments": self.arguments,
        }
        if name == "Extraction":
            extraction = self.extraction(params) if callable(self.extraction) else self.extraction
            return structured({"extraction": extraction})
        if name == "Metadata":
            return structured({"progress": "done", "completed": True})
        if name == "Observation":
            return structured({"elements": [action()]})
        if name == "Act":
            return structured({"action": action(), "twoStep": False})
        return LLMMessageGenerateResult(
            role="assistant",
            content=[{"type": "text", "text": "ok"}],
            output_format="text",
            usage=USAGE,
        )
