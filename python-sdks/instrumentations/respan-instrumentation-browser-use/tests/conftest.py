from __future__ import annotations

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
