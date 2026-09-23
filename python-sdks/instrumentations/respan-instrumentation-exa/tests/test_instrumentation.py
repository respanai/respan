from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_exa import ExaInstrumentor, _instrumentation
from respan_instrumentation_exa._constants import (
    EXA_METADATA_NAMESPACE,
    METADATA_CITATIONS,
    METADATA_OPERATION,
    METADATA_REQUEST_ID,
    METADATA_RESEARCH_LEGACY,
    METADATA_STREAM_COMPLETED,
    OFF_CONTRACT_ALIASES,
    OperationConfig,
)
from respan_instrumentation_exa._serialization import json_dumps
from respan_instrumentation_exa._translator import (
    build_start_attributes,
    build_success_attributes,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.processors.base import FilteringSpanProcessor
from respan_tracing.utils.span_factory import propagate_attributes

_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(
    FilteringSpanProcessor(exporter=_EXPORTER, is_batching_enabled=False)
)
trace.set_tracer_provider(_PROVIDER)


class _AsyncChunksWithSyncClose:
    """Matches exa-py 2.20's async stream lifecycle surface."""

    def __init__(self, *values):
        self._values = iter(values)
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._values)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    def close(self):
        self.close_calls += 1


def _async_chunks(*values):
    return _AsyncChunksWithSyncClose(*values)


class Exa:
    def search(self, query, **kwargs):
        if query == "fail":
            raise RuntimeError("deterministic Exa failure")
        if query == "rate-limit":
            raise ValueError(
                'Request failed with status code 429: {"error":"rate limit"}'
            )
        return {
            "results": [{"url": "https://example.com", "text": "result text"}],
            "requestId": "req-search",
            "resolvedSearchType": "auto",
            "costDollars": {"total": 0.007},
        }

    def stream_search(self, query, **kwargs):
        return iter(
            [
                SimpleNamespace(content="fresh ", citations=[]),
                SimpleNamespace(
                    content="result",
                    citations=[{"url": "https://example.com"}],
                ),
            ]
        )

    def search_and_contents(self, query, **kwargs):
        return self.search(query, **kwargs)

    def get_contents(self, urls, **kwargs):
        return {"results": [{"url": urls[0], "text": "page body"}]}

    def find_similar(self, url, **kwargs):
        return {"results": [{"url": f"{url}/similar"}]}

    def find_similar_and_contents(self, url, **kwargs):
        return self.find_similar(url, **kwargs)

    def answer(self, query, **kwargs):
        return SimpleNamespace(
            answer="Grounded answer",
            citations=[{"url": "https://example.com/source"}],
            cost_dollars=SimpleNamespace(total=0.005),
        )

    def stream_answer(self, query, **kwargs):
        return iter(
            [
                SimpleNamespace(content="Grounded ", citations=[]),
                SimpleNamespace(content="stream", citations=[]),
            ]
        )


class AsyncExa:
    async def search(self, query, **kwargs):
        return {"results": [{"url": "https://example.com/async"}]}

    async def stream_search(self, query, **kwargs):
        return _async_chunks(
            SimpleNamespace(content="async ", citations=[]),
            SimpleNamespace(content="search", citations=[]),
        )

    async def search_and_contents(self, query, **kwargs):
        return await self.search(query, **kwargs)

    async def get_contents(self, urls, **kwargs):
        return {"results": [{"url": urls[0], "text": "async page"}]}

    async def find_similar(self, url, **kwargs):
        return {"results": [{"url": f"{url}/async-similar"}]}

    async def find_similar_and_contents(self, url, **kwargs):
        return await self.find_similar(url, **kwargs)

    async def answer(self, query, **kwargs):
        return SimpleNamespace(answer="Async answer", citations=[])

    async def stream_answer(self, query, **kwargs):
        return _async_chunks(
            SimpleNamespace(content="async ", citations=[]),
            SimpleNamespace(content="answer", citations=[]),
        )


class AgentRunEventsClient:
    def list(self, run_id, **kwargs):
        return {"data": [{"type": "run.completed", "runId": run_id}]}


class AgentBetaRunEventsClient(AgentRunEventsClient):
    def list(self, run_id, **kwargs):
        return super().list(run_id, **kwargs)


class AgentRunsClient:
    def create(self, *, query, stream=False, **kwargs):
        if stream:
            return iter(
                [
                    {"type": "run.started", "query": query},
                    {"type": "run.completed", "output": "agent result"},
                ]
            )
        return {"id": "run-1", "status": "queued", "query": query}

    def get(self, run_id):
        return {"id": run_id, "status": "completed", "output": "agent result"}

    def list(self, **kwargs):
        return {"data": [{"id": "run-1"}]}

    def cancel(self, run_id):
        return {"id": run_id, "status": "cancelled"}

    def delete(self, run_id):
        return {"id": run_id, "deleted": True}

    def poll_until_finished(self, run_id, **kwargs):
        return self.get(run_id)

    def create_and_wait(self, *, query, **kwargs):
        created = self.create(query=query)
        return self.get(created["id"])


class AgentBetaRunsClient(AgentRunsClient):
    def create(self, *, query, stream=False, **kwargs):
        return super().create(query=query, stream=stream, **kwargs)

    def get(self, run_id, **kwargs):
        return super().get(run_id)

    def list(self, **kwargs):
        return super().list(**kwargs)

    def cancel(self, run_id, **kwargs):
        return super().cancel(run_id)

    def stop(self, run_id, **kwargs):
        return {"id": run_id, "status": "stopped"}

    def delete(self, run_id, **kwargs):
        return super().delete(run_id)

    def poll_until_finished(self, run_id, **kwargs):
        return super().poll_until_finished(run_id, **kwargs)

    def create_and_wait(self, *, query, **kwargs):
        return super().create_and_wait(query=query, **kwargs)


class AsyncAgentRunEventsClient:
    async def list(self, run_id, **kwargs):
        return {"data": [{"type": "run.completed", "runId": run_id}]}


class AsyncAgentBetaRunEventsClient(AsyncAgentRunEventsClient):
    async def list(self, run_id, **kwargs):
        return await super().list(run_id, **kwargs)


class AsyncAgentRunsClient:
    async def create(self, *, query, stream=False, **kwargs):
        if stream:
            return _async_chunks(
                {"type": "run.started", "query": query},
                {"type": "run.completed", "output": "async agent result"},
            )
        return {"id": "async-run-1", "status": "queued", "query": query}

    async def get(self, run_id):
        return {"id": run_id, "status": "completed", "output": "async agent"}

    async def list(self, **kwargs):
        return {"data": [{"id": "async-run-1"}]}

    async def cancel(self, run_id):
        return {"id": run_id, "status": "cancelled"}

    async def delete(self, run_id):
        return {"id": run_id, "deleted": True}

    async def poll_until_finished(self, run_id, **kwargs):
        return await self.get(run_id)

    async def create_and_wait(self, *, query, **kwargs):
        created = await self.create(query=query)
        return await self.get(created["id"])


class AsyncAgentBetaRunsClient(AsyncAgentRunsClient):
    async def create(self, *, query, stream=False, **kwargs):
        return await super().create(query=query, stream=stream, **kwargs)

    async def get(self, run_id, **kwargs):
        return await super().get(run_id)

    async def list(self, **kwargs):
        return await super().list(**kwargs)

    async def cancel(self, run_id, **kwargs):
        return await super().cancel(run_id)

    async def stop(self, run_id, **kwargs):
        return {"id": run_id, "status": "stopped"}

    async def delete(self, run_id, **kwargs):
        return await super().delete(run_id)

    async def poll_until_finished(self, run_id, **kwargs):
        return await super().poll_until_finished(run_id, **kwargs)

    async def create_and_wait(self, *, query, **kwargs):
        return await super().create_and_wait(query=query, **kwargs)


class ResearchClient:
    def create(self, *, instructions, model="exa-research-fast", **kwargs):
        return {"researchId": "research-1", "status": "pending"}

    def get(self, research_id, *, stream=False, **kwargs):
        if stream:
            return iter(
                [
                    {"type": "research.started"},
                    {"type": "research.completed", "output": "report"},
                ]
            )
        return {"researchId": research_id, "status": "completed", "output": "report"}

    def list(self, **kwargs):
        return {"data": [{"researchId": "research-1"}]}

    def poll_until_finished(self, research_id, **kwargs):
        return self.get(research_id)


class AsyncResearchClient:
    async def create(self, *, instructions, model="exa-research-fast", **kwargs):
        return {"researchId": "async-research-1", "status": "pending"}

    async def get(self, research_id, *, stream=False, **kwargs):
        if stream:
            return _async_chunks(
                {"type": "research.started"},
                {"type": "research.completed", "output": "async report"},
            )
        return {"researchId": research_id, "status": "completed", "output": "report"}

    async def list(self, **kwargs):
        return {"data": [{"researchId": "async-research-1"}]}

    async def poll_until_finished(self, research_id, **kwargs):
        return await self.get(research_id)


def _modules():
    return {
        "exa_py.api": SimpleNamespace(Exa=Exa, AsyncExa=AsyncExa),
        "exa_py.agent.client": SimpleNamespace(
            AgentRunsClient=AgentRunsClient,
            AgentBetaRunsClient=AgentBetaRunsClient,
            AgentRunEventsClient=AgentRunEventsClient,
            AgentBetaRunEventsClient=AgentBetaRunEventsClient,
        ),
        "exa_py.agent.async_client": SimpleNamespace(
            AsyncAgentRunsClient=AsyncAgentRunsClient,
            AsyncAgentBetaRunsClient=AsyncAgentBetaRunsClient,
            AsyncAgentRunEventsClient=AsyncAgentRunEventsClient,
            AsyncAgentBetaRunEventsClient=AsyncAgentBetaRunEventsClient,
        ),
        "exa_py.research.sync_client": SimpleNamespace(ResearchClient=ResearchClient),
        "exa_py.research.async_client": SimpleNamespace(
            AsyncResearchClient=AsyncResearchClient
        ),
    }


@pytest.fixture(autouse=True)
def reset_instrumentation():
    _EXPORTER.clear()
    RespanTracer.reset_instance()
    _instrumentation._ENABLED = False
    _instrumentation._REFCOUNT = 0
    _instrumentation._CAPTURE_CONTENT = None
    _instrumentation._PATCHES.clear()
    yield
    for span in _EXPORTER.get_finished_spans():
        assert not any(key.startswith("exa.") for key in span.attributes)
    _instrumentation._ENABLED = False
    _instrumentation._REFCOUNT = 0
    for patch in reversed(_instrumentation._PATCHES):
        if getattr(patch.owner, patch.name, None) is patch.replacement:
            setattr(patch.owner, patch.name, patch.original)
    _instrumentation._PATCHES.clear()
    _instrumentation._CAPTURE_CONTENT = None
    RespanTracer.reset_instance()


def _activate(*, capture_content=True):
    instrumentor = ExaInstrumentor(
        capture_content=capture_content,
        module_overrides=_modules(),
    )
    instrumentor.activate()
    return instrumentor


def _span(name):
    return next(span for span in _EXPORTER.get_finished_spans() if span.name == name)


def _exa_metadata(attributes):
    return json.loads(attributes[RESPAN_METADATA])[EXA_METADATA_NAMESPACE]


def test_pure_tool_translation_is_canonical_and_redacts_secrets():
    config = OperationConfig("search", "tool", "search")
    attrs = build_start_attributes(
        config=config,
        call_input={"query": "fresh AI", "api_key": "secret"},
        capture_content=True,
        streaming=False,
        has_parent=False,
    )

    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
    payload = json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert payload == {
        "arguments": {"api_key": "<redacted>", "query": "fresh AI"},
        "name": "search",
    }
    assert _exa_metadata(attrs) == {
        "language": "python",
        "operation": "search",
        "stream": False,
    }
    assert not any(key.startswith("exa.") for key in attrs)
    for alias in OFF_CONTRACT_ALIASES:
        assert alias not in attrs


def test_pure_answer_translation_emits_chat_contract():
    config = OperationConfig("answer", "chat", "answer")
    default_start = build_start_attributes(
        config=config,
        call_input={"query": "What changed?"},
        capture_content=True,
        streaming=False,
        has_parent=False,
    )
    start = build_start_attributes(
        config=config,
        call_input={
            "query": "What changed?",
            "system_prompt": "Cite sources",
            "model": "exa-pro",
        },
        capture_content=True,
        streaming=False,
        has_parent=True,
    )
    finish = build_success_attributes(
        config=config,
        call_input={"query": "What changed?", "model": "exa-pro"},
        result={
            "answer": "A grounded answer",
            "citations": [{"url": "https://example.com/source"}],
            "model": "exa-pro",
        },
        capture_content=True,
        streaming=False,
    )

    assert start[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert start[SpanAttributes.LLM_SYSTEM] == "exa"
    assert SpanAttributes.LLM_REQUEST_MODEL not in default_start
    assert start[SpanAttributes.LLM_REQUEST_MODEL] == "exa-pro"
    assert finish[SpanAttributes.LLM_REQUEST_MODEL] == "exa-pro"
    assert start[SpanAttributes.TRACELOOP_ENTITY_NAME] == "answer"
    assert start[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert start[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == "What changed?"
    assert finish[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "A grounded answer"
    assert _exa_metadata(finish)[METADATA_CITATIONS] == [
        {"url": "https://example.com/source"}
    ]
    assert not any(key.startswith("exa.") for key in default_start | start | finish)


def test_sync_core_mapping_error_and_privacy():
    instrumentor = _activate()
    try:
        with propagate_attributes(metadata={"run_id": "exa-sop-run"}):
            result = Exa().search("fresh AI", contents={"highlights": True})
        assert result["results"]
        with pytest.raises(RuntimeError, match="deterministic Exa failure"):
            Exa().search("fail")
        with pytest.raises(ValueError, match="status code 429"):
            Exa().search("rate-limit")
    finally:
        instrumentor.deactivate()

    spans = [span for span in _EXPORTER.get_finished_spans() if span.name == "search"]
    assert len(spans) == 3
    success = next(span for span in spans if span.status.status_code.name == "OK")
    errors = [span for span in spans if span.status.status_code.name == "ERROR"]
    assert json.loads(success.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[
        "results"
    ]
    assert _exa_metadata(success.attributes)[METADATA_REQUEST_ID] == "req-search"
    assert json.loads(success.attributes[RESPAN_METADATA])["run_id"] == "exa-sop-run"
    assert success.attributes["status_code"] == 200
    assert {span.attributes["status_code"] for span in errors} == {429, 500}
    assert any(
        "deterministic Exa failure" in span.attributes["error.message"]
        for span in errors
    )
    assert all(
        not any(key.startswith("exa.") for key in span.attributes)
        for span in _EXPORTER.get_finished_spans()
    )

    _EXPORTER.clear()
    private = _activate(capture_content=False)
    try:
        Exa().answer("private question", system_prompt="private prompt")
    finally:
        private.deactivate()
    attrs = _span("answer").attributes
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    assert not any(key.startswith(f"{SpanAttributes.LLM_PROMPTS}.") for key in attrs)
    assert not any(
        key.startswith(f"{SpanAttributes.LLM_COMPLETIONS}.") for key in attrs
    )


def test_sync_and_async_streams_finish_on_exhaustion_and_close():
    instrumentor = _activate()
    try:
        stream = Exa().stream_search("stream query")
        assert _EXPORTER.get_finished_spans() == ()
        assert "fresh result" == "".join(chunk.content for chunk in stream)

        partial = Exa().stream_answer("partial")
        assert next(partial).content == "Grounded "
        partial.close()

        async def run_async():
            response = await AsyncExa().stream_answer("async")
            complete = "".join([chunk.content async for chunk in response])
            partial = await AsyncExa().stream_answer("async-partial")
            first = await partial.__anext__()
            partial.close()
            await partial.aclose()
            return complete, first.content, partial.close_calls

        assert asyncio.run(run_async()) == ("async answer", "async ", 1)
    finally:
        instrumentor.deactivate()

    search = _span("search")
    assert _exa_metadata(search.attributes)[METADATA_STREAM_COMPLETED] is True
    assert (
        json.loads(search.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])["content"]
        == "fresh result"
    )
    answers = [span for span in _EXPORTER.get_finished_spans() if span.name == "answer"]
    assert {
        _exa_metadata(span.attributes)[METADATA_STREAM_COMPLETED] for span in answers
    } == {False, True}


def test_agent_and_legacy_research_surfaces_preserve_hierarchy():
    instrumentor = _activate()
    try:
        run = AgentRunsClient().create_and_wait(query="research a company")
        assert run["status"] == "completed"
        events = list(AgentRunsClient().create(query="stream agent", stream=True))
        assert events[-1]["type"] == "run.completed"
        research = ResearchClient().poll_until_finished("research-1")
        assert research["status"] == "completed"
        assert AgentRunsClient().get("standalone-run")["status"] == "completed"
        assert ResearchClient().get("standalone-research")["status"] == "completed"

        async def run_async():
            async_run = await AsyncAgentRunsClient().create_and_wait(
                query="async research"
            )
            async_research = await AsyncResearchClient().poll_until_finished(
                "async-research-1"
            )
            return async_run, async_research

        async_run, async_research = asyncio.run(run_async())
        assert async_run["status"] == "completed"
        assert async_research["status"] == "completed"
    finally:
        instrumentor.deactivate()

    spans = _EXPORTER.get_finished_spans()
    operations = [_exa_metadata(span.attributes)[METADATA_OPERATION] for span in spans]
    assert operations.count("agent.runs.create_and_wait") == 2
    assert operations.count("agent.runs.create") == 1
    assert operations.count("agent.runs.get") == 1
    assert operations.count("research.poll_until_finished") == 2
    assert operations.count("research.get") == 1
    research_spans = [
        span
        for span in spans
        if _exa_metadata(span.attributes).get(METADATA_RESEARCH_LEGACY)
    ]
    assert research_spans
    assert any(
        _exa_metadata(span.attributes)[METADATA_OPERATION]
        == "agent.runs.create_and_wait"
        and span.name == "run"
        for span in spans
    )
    assert any(
        _exa_metadata(span.attributes)[METADATA_OPERATION]
        == "research.poll_until_finished"
        and span.name == "research"
        for span in spans
    )
    assert any(
        _exa_metadata(span.attributes)[METADATA_OPERATION] == "agent.runs.get"
        and span.name == "run.get"
        for span in spans
    )
    assert any(
        _exa_metadata(span.attributes)[METADATA_OPERATION] == "research.get"
        and span.name == "research.get"
        for span in spans
    )


def test_reference_counted_lifecycle_restores_owned_methods():
    original = Exa.search
    first = ExaInstrumentor(module_overrides=_modules())
    second = ExaInstrumentor(module_overrides=_modules())

    first.activate()
    wrapped = Exa.search
    second.activate()
    assert Exa.search is wrapped
    first.deactivate()
    assert Exa.search is wrapped
    second.deactivate()
    assert Exa.search is original


def test_released_exa_py_surface_is_present():
    import importlib.metadata

    from exa_py import AsyncExa as ReleasedAsyncExa
    from exa_py import Exa as ReleasedExa
    from exa_py.api import AsyncStreamAnswerResponse

    version = tuple(
        int(part) for part in importlib.metadata.version("exa-py").split(".")[:2]
    )
    assert (2, 20) <= version < (3, 0)
    assert all(
        hasattr(ReleasedExa, method)
        for method in (
            "search",
            "stream_search",
            "get_contents",
            "answer",
            "stream_answer",
        )
    )
    assert hasattr(ReleasedAsyncExa, "search")
    client = ReleasedExa(api_key="not-used")
    assert hasattr(client.tools, "web_search")
    assert hasattr(client.tools, "get_contents")
    assert hasattr(client.agent.runs, "create_and_wait")
    assert hasattr(client.research, "poll_until_finished")
    assert callable(AsyncStreamAnswerResponse.close)
    assert not hasattr(AsyncStreamAnswerResponse, "aclose")

    compatibility = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from respan_tracing.utils import span_factory; "
                "span_factory.__dict__.pop('merge_metadata_attributes', None); "
                "from respan_instrumentation_exa import ExaInstrumentor; "
                "assert ExaInstrumentor.__name__ == 'ExaInstrumentor'"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compatibility.returncode == 0, compatibility.stderr


def test_serialization_redacts_nested_keys():
    payload = json.loads(
        json_dumps(
            {
                "query": "safe",
                "headers": {"Authorization": "Bearer secret", "x-api-key": "secret"},
            }
        )
    )
    assert payload["query"] == "safe"
    assert payload["headers"] == {
        "Authorization": "<redacted>",
        "x-api-key": "<redacted>",
    }
