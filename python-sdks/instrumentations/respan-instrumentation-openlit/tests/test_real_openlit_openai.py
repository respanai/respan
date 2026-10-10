from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from anthropic import Anthropic
from openai import AsyncOpenAI, OpenAI, RateLimitError
from openai.resources.chat.completions import Completions
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_instrumentation_openlit import OpenLITInstrumentor
from respan_instrumentation_openlit._constants import OFF_CONTRACT_ALIASES
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from wrapt import FunctionWrapper

_MODEL = "gpt-4.1-mini"
_TOOL_NAME = "lookup_weather"
_PRIVATE_INPUT = "openlit-private-input-do-not-export"
_PRIVATE_OUTPUT = "openlit-private-output-do-not-export"


def _chat_response(
    *,
    content: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    response: dict[str, Any] = {
        "id": f"chatcmpl-openlit-{time.time_ns()}",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
    if prompt_tokens is not None and completion_tokens is not None:
        response["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    return response


def _responses_response(
    prompt: str,
    *,
    status: str = "completed",
    include_usage: bool = True,
) -> dict[str, Any]:
    text = f"Responses API: {prompt}"
    response: dict[str, Any] = {
        "id": f"resp-openlit-{time.time_ns()}",
        "object": "response",
        "created_at": 1_700_000_000.0,
        "model": _MODEL,
        "output": (
            [
                {
                    "id": "msg_openlit_response",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                        }
                    ],
                }
            ]
            if status == "completed"
            else []
        ),
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
    }
    if include_usage:
        response["usage"] = {
            "input_tokens": 6,
            "input_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "output_tokens": 4,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 10,
        }
    return response


class _OpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _request(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or "0")
        return json.loads(self.rfile.read(length)) if length else {}

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self) -> None:
        chunks = [
            {
                "id": "chatcmpl-openlit-stream",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000,
                "model": _MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "OpenLIT "},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-openlit-stream",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000,
                "model": _MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "stream works."},
                        "finish_reason": "stop",
                    }
                ],
            },
            {
                "id": "chatcmpl-openlit-stream",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000,
                "model": _MODEL,
                "choices": [],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 11,
                    "total_tokens": 18,
                },
            },
        ]
        body = (
            "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            + "data: [DONE]\n\n"
        )
        encoded = body.encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_responses_stream(self, prompt: str) -> None:
        created = {
            "type": "response.created",
            "sequence_number": 0,
            "response": _responses_response(
                prompt, status="in_progress", include_usage=False
            ),
        }
        delta = {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_openlit_response",
            "output_index": 0,
            "content_index": 0,
            "delta": f"Responses API: {prompt}",
            "logprobs": [],
        }
        completed = {
            "type": "response.completed",
            "sequence_number": 2,
            "response": _responses_response(prompt),
        }
        body = (
            "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in (created, delta, completed)
            )
            + "data: [DONE]\n\n"
        )
        encoded = body.encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:
        request = self._request()
        if self.path.endswith("/messages"):
            messages = request.get("messages") or []
            prompt = str(messages[0].get("content") if messages else "")
            self._send_json(
                200,
                {
                    "id": f"msg-openlit-{time.time_ns()}",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [{"type": "text", "text": f"Anthropic API: {prompt}"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                },
            )
            return
        if self.path.endswith("/responses"):
            prompt = str(request.get("input") or "")
            if request.get("stream"):
                self._send_responses_stream(prompt)
            else:
                self._send_json(200, _responses_response(prompt))
            return
        if self.path.endswith("/embeddings"):
            self._send_json(
                200,
                {
                    "object": "list",
                    "model": _MODEL,
                    "data": [
                        {
                            "object": "embedding",
                            "index": 0,
                            "embedding": [0.1, 0.2, 0.3],
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "total_tokens": 4},
                },
            )
            return
        if not self.path.endswith("/chat/completions"):
            self._send_json(404, {"error": {"message": "not found"}})
            return
        messages = request.get("messages") or []
        prompt = str(messages[0].get("content") if messages else "")
        if prompt == "rate-limit":
            self._send_json(
                429,
                {
                    "error": {
                        "message": "deterministic rate limit",
                        "type": "rate_limit_error",
                    }
                },
            )
            return
        if request.get("stream"):
            self._send_stream()
            return
        if prompt == "tool":
            self._send_json(
                200,
                _chat_response(
                    content=None,
                    prompt_tokens=12,
                    completion_tokens=7,
                    tool_calls=[
                        {
                            "id": "call_openlit_weather",
                            "type": "function",
                            "function": {
                                "name": _TOOL_NAME,
                                "arguments": json.dumps({"city": "Tokyo"}),
                            },
                        }
                    ],
                ),
            )
            return
        if prompt == "without-provider-usage":
            self._send_json(
                200,
                _chat_response(
                    content="No provider usage was returned.",
                    prompt_tokens=None,
                    completion_tokens=None,
                ),
            )
            return
        if prompt == _PRIVATE_INPUT:
            content, prompt_tokens, completion_tokens = _PRIVATE_OUTPUT, 3, 2
        else:
            content, prompt_tokens, completion_tokens = "Async OpenLIT works.", 5, 3
        self._send_json(
            200,
            _chat_response(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
        )


def _tool_definition(secret: bool = False) -> list[dict[str, Any]]:
    description = _PRIVATE_INPUT if secret else "Look up weather for a city"
    return [
        {
            "type": "function",
            "function": {
                "name": _TOOL_NAME,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def _scope_name(span: Any) -> str:
    return str(getattr(span.instrumentation_scope, "name", "") or "")


def _chat_with_prompt(spans: list[Any], prompt: str) -> Any:
    for span in spans:
        attrs = span.attributes
        if attrs.get(f"{SpanAttributes.LLM_PROMPTS}.0.content") == prompt:
            return span
    raise AssertionError(f"missing OpenLIT chat span for {prompt!r}")


def _assert_no_aliases(attrs: dict[str, Any]) -> None:
    assert OFF_CONTRACT_ALIASES.isdisjoint(attrs)


def test_real_current_openlit_openai_export_contract_and_privacy(
    monkeypatch,
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    instrumentor = OpenLITInstrumentor(max_content_length=4_096)
    client = OpenAI(
        api_key="local-openlit-key",
        base_url=base_url,
        max_retries=0,
        timeout=3,
    )
    anthropic_client = Anthropic(
        api_key="local-anthropic-key",
        base_url=base_url,
        max_retries=0,
        timeout=3,
    )
    foreign_calls = 0
    foreign_wrapper = None

    async def run_async_chat() -> None:
        async with AsyncOpenAI(
            api_key="local-openlit-key",
            base_url=base_url,
            max_retries=0,
            timeout=3,
        ) as async_client:
            await async_client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": "async"}],
            )
            cancelled_stream = await async_client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": "async-cancel"}],
                stream=True,
                stream_options={"include_usage": True},
            )
            first_chunk = await cancelled_stream.__anext__()
            assert first_chunk.choices[0].delta.content == "OpenLIT "
            source_response = cancelled_stream.response
            await cancelled_stream.close()
            await cancelled_stream.close()
            assert source_response.is_closed
            response = await async_client.responses.create(
                model=_MODEL,
                input="responses-async",
            )
            assert response.output_text == "Responses API: responses-async"
            response_stream = await async_client.responses.create(
                model=_MODEL,
                input="responses-async-stream",
                stream=True,
            )
            response_text = ""
            async for event in response_stream:
                if event.type == "response.output_text.delta":
                    response_text += event.delta
            assert response_text == "Responses API: responses-async-stream"
            early_response_stream = await async_client.responses.create(
                model=_MODEL,
                input="responses-async-early",
                stream=True,
            )
            first_event = await early_response_stream.__anext__()
            assert first_event.type == "response.created"
            source_response = early_response_stream.response
            await early_response_stream.close()
            await early_response_stream.close()
            assert source_response.is_closed

    try:
        instrumentor.activate()
        client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": "tool"}],
            tools=_tool_definition(),
            tool_choice={"type": "function", "function": {"name": _TOOL_NAME}},
        )
        stream = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": "stream"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        assert (
            "".join(
                chunk.choices[0].delta.content or ""
                for chunk in stream
                if chunk.choices
            )
            == "OpenLIT stream works."
        )
        early_stream = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": "early-close"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        assert next(early_stream).choices[0].delta.content == "OpenLIT "
        source_response = early_stream.response
        early_stream.close()
        early_stream.close()
        assert source_response.is_closed
        response = client.responses.create(
            model=_MODEL,
            input="responses-sync",
        )
        assert response.output_text == "Responses API: responses-sync"
        response_stream = client.responses.create(
            model=_MODEL,
            input="responses-sync-stream",
            stream=True,
        )
        assert (
            "".join(
                event.delta
                for event in response_stream
                if event.type == "response.output_text.delta"
            )
            == "Responses API: responses-sync-stream"
        )
        early_response_stream = client.responses.create(
            model=_MODEL,
            input="responses-sync-early",
            stream=True,
        )
        assert next(early_response_stream).type == "response.created"
        source_response = early_response_stream.response
        early_response_stream.close()
        early_response_stream.close()
        assert source_response.is_closed
        asyncio.run(run_async_chat())
        client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": "without-provider-usage"}],
        )
        with trace.get_tracer("openlit-real-contract-test").start_as_current_span(
            "parent"
        ):
            client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": "nested"}],
            )
        with pytest.raises(RateLimitError) as raised:
            client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": "rate-limit"}],
            )
        assert raised.value.status_code == 429
        client.embeddings.create(model=_MODEL, input=["bounded embedding input"])
        anthropic_response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16,
            messages=[{"role": "user", "content": "anthropic-active"}],
        )
        assert anthropic_response.content[0].text == ("Anthropic API: anthropic-active")

        installed_create = vars(Completions)["create"]

        def foreign_openai_wrapper(wrapped, instance, args, kwargs):
            del instance
            nonlocal foreign_calls
            foreign_calls += 1
            return wrapped(*args, **kwargs)

        foreign_wrapper = FunctionWrapper(installed_create, foreign_openai_wrapper)
        monkeypatch.setattr(Completions, "create", foreign_wrapper)
        provider.force_flush()
    finally:
        client.close()
        anthropic_client.close()
        instrumentor.deactivate()

    assert foreign_wrapper is not None
    assert vars(Completions)["create"] is foreign_wrapper
    finished_before_uninstrumented_calls = len(exporter.get_finished_spans())
    uninstrumented_client = OpenAI(
        api_key="local-openlit-key",
        base_url=base_url,
        max_retries=0,
        timeout=3,
    )
    uninstrumented_anthropic = Anthropic(
        api_key="local-anthropic-key",
        base_url=base_url,
        max_retries=0,
        timeout=3,
    )
    try:
        uninstrumented_client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": "after-deactivate"}],
        )
        uninstrumented_anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16,
            messages=[{"role": "user", "content": "anthropic-deactivated"}],
        )
        provider.force_flush()
    finally:
        uninstrumented_client.close()
        uninstrumented_anthropic.close()
    assert len(exporter.get_finished_spans()) == finished_before_uninstrumented_calls
    assert foreign_calls == 1

    spans = [
        span
        for span in exporter.get_finished_spans()
        if _scope_name(span).startswith("openlit.instrumentation.openai")
    ]
    assert len(spans) == 15
    assert all(not _scope_name(span).endswith(("httpx", "requests")) for span in spans)
    anthropic_spans = [
        span
        for span in exporter.get_finished_spans()
        if _scope_name(span).startswith("openlit.instrumentation.anthropic")
    ]
    assert len(anthropic_spans) == 1
    assert anthropic_spans[0].attributes[SpanAttributes.LLM_SYSTEM] == "anthropic"

    tool_span = _chat_with_prompt(spans, "tool")
    tool_attrs = dict(tool_span.attributes)
    definitions = json.loads(tool_attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])
    calls = json.loads(tool_attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert definitions == [
        {
            "type": "function",
            "function": {
                "name": _TOOL_NAME,
                "description": "Look up weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    assert calls == [
        {
            "id": "call_openlit_weather",
            "type": "function",
            "function": {
                "name": _TOOL_NAME,
                "arguments": '{"city": "Tokyo"}',
            },
        }
    ]
    assert tool_attrs["gen_ai.provider.name"] == "openai"
    assert tool_attrs[SpanAttributes.LLM_SYSTEM] == "openai"
    assert tool_attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 12
    assert tool_attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 7
    assert not any(key.startswith("gen_ai.tool.") for key in tool_attrs)
    _assert_no_aliases(tool_attrs)

    stream_span = _chat_with_prompt(spans, "stream")
    stream_attrs = dict(stream_span.attributes)
    assert stream_attrs["gen_ai.request.stream"] is True
    assert stream_attrs[SpanAttributes.LLM_IS_STREAMING] is True
    assert stream_attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "OpenLIT stream works."
    )
    assert stream_attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 7
    assert stream_attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 11
    assert stream_attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 18
    _assert_no_aliases(stream_attrs)

    for prompt in ("early-close", "async-cancel"):
        partial_spans = [
            span
            for span in spans
            if span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content") == prompt
        ]
        assert len(partial_spans) == 1
        partial_attrs = partial_spans[0].attributes
        assert partial_attrs["gen_ai.request.stream"] is True
        assert partial_attrs[SpanAttributes.LLM_IS_STREAMING] is True
        assert partial_attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
            "OpenLIT "
        )
        assert SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in partial_attrs

    async_span = _chat_with_prompt(spans, "async")
    assert async_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 5
    assert async_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 3

    for prompt in (
        "responses-sync",
        "responses-async",
        "responses-sync-stream",
        "responses-async-stream",
    ):
        response_span = _chat_with_prompt(spans, prompt)
        response_attrs = response_span.attributes
        assert response_attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 6
        assert response_attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 4
        assert response_attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 10
        assert response_attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
            f"Responses API: {prompt}"
        )
        assert response_attrs["gen_ai.request.stream"] is prompt.endswith("stream")
        if prompt.endswith("stream"):
            assert response_attrs[SpanAttributes.LLM_IS_STREAMING] is True
        else:
            assert SpanAttributes.LLM_IS_STREAMING not in response_attrs

    for prompt in ("responses-sync-early", "responses-async-early"):
        response_span = _chat_with_prompt(spans, prompt)
        assert response_span.attributes["gen_ai.request.stream"] is True
        assert response_span.attributes[SpanAttributes.LLM_IS_STREAMING] is True
        assert SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in response_span.attributes

    no_usage_span = _chat_with_prompt(spans, "without-provider-usage")
    for key in (
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
    ):
        assert key not in no_usage_span.attributes

    assert all(
        span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
        for span in spans
        if span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content") != "nested"
    )
    nested_span = _chat_with_prompt(spans, "nested")
    assert nested_span.parent is not None
    assert (
        nested_span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH]
        == (nested_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME])
    )

    error_span = _chat_with_prompt(spans, "rate-limit")
    error_attrs = dict(error_span.attributes)
    assert error_span.status.status_code is StatusCode.ERROR
    assert error_attrs["http.response.status_code"] == 429
    assert "status_code" not in error_attrs
    assert "error.message" not in error_attrs
    assert "deterministic rate limit" in error_span.status.description
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in error_attrs
    assert (
        json.loads(error_attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT])[0]["parts"][0][
            "content"
        ]
        == "rate-limit"
    )
    assert not any(
        key.startswith(f"{SpanAttributes.LLM_COMPLETIONS}.") for key in error_attrs
    )
    for key in (
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
    ):
        assert key not in error_attrs
    _assert_no_aliases(error_attrs)

    embedding_span = next(
        span for span in spans if span.attributes.get(RESPAN_LOG_TYPE) == "embedding"
    )
    assert embedding_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "embedding"
    assert embedding_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 4
    assert json.loads(
        embedding_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    ) == ["bounded embedding input"]
    assert json.loads(
        embedding_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    ) == [[0.1, 0.2, 0.3]]

    exporter.clear()
    private_instrumentor = OpenLITInstrumentor(
        capture_content=False,
        max_content_length=128,
    )
    private_client = OpenAI(
        api_key="private-local-key",
        base_url=base_url,
        max_retries=0,
        timeout=3,
    )
    private_anthropic = Anthropic(
        api_key="private-local-anthropic-key",
        base_url=base_url,
        max_retries=0,
        timeout=3,
    )
    try:
        private_instrumentor.activate()
        private_client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": _PRIVATE_INPUT}],
            tools=_tool_definition(secret=True),
        )
        private_anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16,
            messages=[{"role": "user", "content": _PRIVATE_INPUT}],
        )
        provider.force_flush()
    finally:
        private_client.close()
        private_anthropic.close()
        private_instrumentor.deactivate()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)
        provider.shutdown()

    private_spans = [
        span
        for span in exporter.get_finished_spans()
        if _scope_name(span).startswith("openlit.instrumentation.openai")
    ]
    assert len(private_spans) == 1, [
        (span.name, _scope_name(span)) for span in private_spans
    ]
    private_attrs = dict(private_spans[0].attributes)
    serialized = json.dumps(private_attrs, default=str)
    assert _PRIVATE_INPUT not in serialized
    assert _PRIVATE_OUTPUT not in serialized
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in private_attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in private_attrs
    assert SpanAttributes.LLM_REQUEST_FUNCTIONS not in private_attrs
    assert private_attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 3
    assert private_attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 2
    assert private_attrs["gen_ai.provider.name"] == "openai"
    assert private_attrs[SpanAttributes.LLM_SYSTEM] == "openai"
    _assert_no_aliases(private_attrs)
    private_anthropic_spans = [
        span
        for span in exporter.get_finished_spans()
        if _scope_name(span).startswith("openlit.instrumentation.anthropic")
    ]
    assert len(private_anthropic_spans) == 1
    assert _PRIVATE_INPUT not in json.dumps(
        dict(private_anthropic_spans[0].attributes), default=str
    )
    assert foreign_calls == 2
