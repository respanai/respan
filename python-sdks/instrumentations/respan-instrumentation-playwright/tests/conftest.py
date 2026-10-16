from __future__ import annotations

import functools
import shutil
import threading
from contextlib import ExitStack
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from respan_instrumentation_playwright import PlaywrightInstrumentor
from respan_instrumentation_playwright._instrumentation import _SCOPE
from respan_tracing.testing import UNREACHABLE_BASE_URL, capture_exported_spans

from respan import Respan

LOGIN_PAGE = """<!doctype html><title>Sign in</title>
<form action="/done.html">
  <label>User <input id="user" name="user"></label>
  <label>Password <input id="password" name="password" type="password"></label>
  <button id="submit">Sign in</button>
</form>"""
DONE_PAGE = "<!doctype html><title>Signed in</title><h1>Welcome</h1>"


def chrome_path() -> str | None:
    mac = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if mac.exists():
        return str(mac)
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        if found := shutil.which(name):
            return found
    return None


@pytest.fixture(autouse=True)
def fresh_learned_secrets():
    """Learned passwords last for the caller's context, which for sync tests is
    the whole pytest main thread: start every test without them."""
    token = _SCOPE._learned.set(())
    yield
    _SCOPE._learned.reset(token)


@pytest.fixture
def site(tmp_path):
    """A local site: /index.html is a sign-in form that submits to /done.html."""
    (tmp_path / "index.html").write_text(LOGIN_PAGE)
    (tmp_path / "done.html").write_text(DONE_PAGE)
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def pipeline():
    """Build `Respan(...)` with the Playwright instrumentor; return the exported spans."""
    with ExitStack() as stack:

        def make(instrumentor: PlaywrightInstrumentor | None = None, **kwargs: Any):
            return stack.enter_context(
                capture_exported_spans(
                    lambda on_export: Respan(
                        api_key="test-key",
                        base_url=UNREACHABLE_BASE_URL,
                        instrumentations=[instrumentor or PlaywrightInstrumentor()],
                        span_postprocess_callback=on_export,
                        **kwargs,
                    )
                )
            )

        yield make


def launch_options() -> dict[str, Any]:
    return {"executable_path": chrome_path(), "headless": True}
