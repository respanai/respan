from __future__ import annotations

import functools
import shutil
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respan_instrumentation_browser_use import BrowserUseInstrumentor


@pytest.fixture
def exporter(monkeypatch):
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    yield memory
    provider.shutdown()


@pytest.fixture
def instrumentor(exporter):
    instance = BrowserUseInstrumentor()
    instance.activate()
    yield instance
    instance.deactivate()


def usage() -> ChatInvokeUsage:
    return ChatInvokeUsage(
        prompt_tokens=120,
        prompt_cached_tokens=20,
        prompt_cache_creation_tokens=None,
        prompt_image_tokens=None,
        completion_tokens=30,
        total_tokens=150,
    )


class ScriptedLLM:
    """A browser-use chat model that replays canned structured outputs."""

    def __init__(self, outputs: list[dict[str, Any]], model: str = "scripted-1"):
        self.model = model
        self.outputs = list(outputs)
        self.calls = 0

    @property
    def provider(self) -> str:
        return "scripted"

    @property
    def name(self) -> str:
        return self.model

    @property
    def model_name(self) -> str:
        return self.model

    async def ainvoke(self, messages, output_format=None, **kwargs):
        self.calls += 1
        payload = self.outputs.pop(0)
        completion = (
            output_format.model_validate(payload) if output_format else str(payload)
        )
        return ChatInvokeCompletion(completion=completion, usage=usage())


def chrome_path() -> str | None:
    mac = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if mac.exists():
        return str(mac)
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        if found := shutil.which(name):
            return found
    return None


@pytest.fixture
def test_page(tmp_path):
    # browser-use only screenshots http(s) pages, so serve the page for real.
    (tmp_path / "index.html").write_text(
        "<title>Respan Test</title><h1>Hello browser agent</h1>"
    )
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/index.html"
    server.shutdown()
