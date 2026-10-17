from __future__ import annotations

import functools
import threading
from contextlib import ExitStack
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from respan_instrumentation_kernel import KernelInstrumentor
from respan_tracing.testing import UNREACHABLE_BASE_URL, capture_exported_spans

from respan import Respan

from .fake_kernel import FakeKernel, LocalChrome, chrome_path

# Absolutely positioned so a computer-use click lands on a known field. The
# noisy background makes screenshots photo-like (a blank page is a tiny PNG
# that JPEG cannot beat, unlike a real one).
LOGIN_PAGE = """<!doctype html><title>Sign in</title>
<style>input{position:absolute;left:20px;width:300px;height:30px}</style>
<canvas id="bg" width="1024" height="768" style="position:fixed;inset:0;z-index:-1"></canvas>
<script>const c=document.getElementById("bg").getContext("2d");
for(let x=0;x<1024;x+=4)for(let y=0;y<768;y+=4){
c.fillStyle=`hsl(${(x*7+y*13)%360},60%,${50+((x*y)%17)}%)`;c.fillRect(x,y,4,4);}</script>
<input id="user" name="user" style="top:20px">
<input id="password" name="password" type="password" style="top:80px">"""
USER_FIELD = (100, 35)
PASSWORD_FIELD = (100, 95)

needs_chrome = pytest.mark.skipif(chrome_path() is None, reason="needs a local Chrome/Chromium")


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture(scope="session")
def chrome():
    if chrome_path() is None:
        pytest.skip("needs a local Chrome/Chromium")
    browser = LocalChrome()
    yield browser
    browser.close()


@pytest.fixture
def fake_kernel(chrome):
    with FakeKernel(chrome) as fake:
        yield fake


@pytest.fixture
def site(tmp_path):
    (tmp_path / "index.html").write_text(LOGIN_PAGE)
    handler = functools.partial(_QuietHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/index.html"
    server.shutdown()
    server.server_close()


@pytest.fixture
def pipeline():
    """Build `Respan(...)` with the Kernel instrumentor; return the exported spans."""
    with ExitStack() as stack:

        def make(instrumentor: KernelInstrumentor | None = None, **kwargs: Any):
            return stack.enter_context(
                capture_exported_spans(
                    lambda on_export: Respan(
                        api_key="test-key",
                        base_url=UNREACHABLE_BASE_URL,
                        instrumentations=[instrumentor or KernelInstrumentor()],
                        span_postprocess_callback=on_export,
                        **kwargs,
                    )
                )
            )

        yield make
