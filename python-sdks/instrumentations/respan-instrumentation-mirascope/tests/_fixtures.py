"""Controlled values through the released Mirascope SDK, without paid inference."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
from mirascope import llm
from openai import AsyncOpenAI, OpenAI
from opentelemetry import trace


class Provider:
    id = "controlled"
    default_scope = "controlled/"

    def __init__(self, *, usage=None, content="native result", chunks=None, error=None):
        self.usage = (
            usage if usage is not None else llm.Usage(input_tokens=9, output_tokens=3)
        )
        self.content, self.chunks, self.error = content, chunks, error
        self.last = None
        self.observed = []
        self.closed = 0

    def _response(
        self, cls, *, model_id, messages, toolkit, format=None, ctx=None, **params
    ):
        if self.error is not None:
            raise self.error
        self.last = cls(
            raw={"fixture": True},
            provider_id=self.id,
            model_id=model_id,
            provider_model_name=model_id.split("/", 1)[-1],
            params=params,
            tools=toolkit,
            format=format,
            input_messages=messages,
            assistant_message=llm.messages.assistant(
                self.content,
                provider_id=self.id,
                model_id=model_id,
                provider_model_name=model_id.split("/", 1)[-1],
            ),
            finish_reason=None,
            usage=self.usage,
        )
        return self.last

    def call(self, **kwargs):
        return self._response(llm.Response, **kwargs)

    def context_call(self, **kwargs):
        return self._response(llm.ContextResponse, **kwargs)

    async def call_async(self, **kwargs):
        return self._response(llm.AsyncResponse, **kwargs)

    async def context_call_async(self, **kwargs):
        return self._response(llm.AsyncContextResponse, **kwargs)

    def _chunks(self):
        try:
            for chunk in (
                self.chunks
                if self.chunks is not None
                else [
                    llm.TextStartChunk(),
                    llm.TextChunk(delta="native stream"),
                    llm.TextEndChunk(),
                    llm.UsageDeltaChunk(input_tokens=5, output_tokens=2),
                ]
            ):
                self.observed.append(trace.get_current_span().get_span_context())
                if isinstance(chunk, BaseException):
                    raise chunk
                yield chunk
        finally:
            self.closed += 1

    async def _async_chunks(self) -> AsyncIterator:
        for chunk in self._chunks():
            yield chunk

    def _stream(
        self,
        cls,
        *,
        model_id,
        messages,
        toolkit,
        format=None,
        ctx=None,
        asynchronous=False,
        **params,
    ):
        if self.error is not None:
            raise self.error
        self.last = cls(
            provider_id=self.id,
            model_id=model_id,
            provider_model_name=model_id.split("/", 1)[-1],
            params=params,
            tools=toolkit,
            format=format,
            input_messages=messages,
            chunk_iterator=self._async_chunks() if asynchronous else self._chunks(),
        )
        return self.last

    def stream(self, **kwargs):
        return self._stream(llm.StreamResponse, **kwargs)

    def context_stream(self, **kwargs):
        return self._stream(llm.ContextStreamResponse, **kwargs)

    async def stream_async(self, **kwargs):
        return self._stream(llm.AsyncStreamResponse, asynchronous=True, **kwargs)

    async def context_stream_async(self, **kwargs):
        return self._stream(llm.AsyncContextStreamResponse, asynchronous=True, **kwargs)


def model(provider=None, **params):
    provider = provider or Provider()
    llm.register_provider(provider, scope="controlled/")
    return llm.Model("controlled/native-2.5", **params), provider


@llm.tool
def vector_tool(values: list[float], api_key: str = "fixture-default") -> dict:
    """Return the complete dense and sparse vector supplied by the caller."""
    return {
        "embedding": values,
        "sparse_vector": {index: value for index, value in enumerate(values)},
    }


@llm.tool
async def async_vector_tool(values: list[float]) -> list[float]:
    """Return the complete vector."""
    return values


def openai_model(
    *, usage=None, mode="completions", stream=False, xai=False, status=200
):
    from mirascope.llm.providers import OpenAIProvider, XAIProvider

    requests = []
    body = {
        "id": "chat-controlled",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4.1-nano",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "actual provider result"},
            }
        ],
    }
    if mode == "responses" or xai:
        body = {
            "id": "resp-controlled",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "grok-4" if xai else "gpt-4.1-nano",
            "output": [
                {
                    "id": "msg-controlled",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "actual provider result",
                            "annotations": [],
                        }
                    ],
                }
            ],
        }
    if usage is not None:
        body["usage"] = usage

    def handler(request):
        requests.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(
                status,
                json={
                    "error": {
                        "message": "api_key=fixture-secret provider failure",
                        "type": "server_error",
                    }
                },
            )
        if stream:
            events = [
                {
                    "id": "chat-controlled",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4.1-nano",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": "actual streamed result",
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chat-controlled",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4.1-nano",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
            if usage is not None:
                events.append(
                    {
                        "id": "chat-controlled",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "gpt-4.1-nano",
                        "choices": [],
                        "usage": usage,
                    }
                )
            text = (
                "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                + "data: [DONE]\n\n"
            )
            return httpx.Response(
                200, text=text, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=body)

    provider = (
        XAIProvider(api_key="controlled")
        if xai
        else OpenAIProvider(api_key="controlled")
    )
    owners = (
        [provider]
        if xai
        else [provider._completions_provider, provider._responses_provider]
    )
    for owner in owners:
        owner.client.close()
        owner.client = OpenAI(
            api_key="controlled",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        owner.async_client = AsyncOpenAI(
            api_key="controlled",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
    scope = "xai/" if xai else "openai/"
    llm.register_provider(provider, scope=scope)
    name = "xai/grok-4" if xai else f"openai/gpt-4.1-nano:{mode}"
    return llm.Model(name), provider, requests
