"""Controlled model/HTTP boundaries with real released LiveKit SDK machinery."""

from __future__ import annotations

import asyncio
import json

import httpx
from livekit.agents import function_tool, llm
from livekit.agents.types import APIConnectOptions
from livekit.plugins import openai as plugin
from openai import AsyncOpenAI

SCHEMA = {
    "name": "vector_tool",
    "description": "Actual controlled vector tool",
    "parameters": {
        "type": "object",
        "properties": {
            "dense": {"type": "array", "items": {"type": "integer"}},
            "sparse": {"type": "object"},
            "api_key": {
                "type": "string",
                "default": "PRIVATE_CREDENTIAL",
                "examples": ["PRIVATE_EXAMPLE"],
            },
            **{f"field{i}": {"type": "integer"} for i in range(120)},
        },
    },
}


@function_tool(raw_schema=SCHEMA)
async def vector_tool(raw_arguments: dict[str, object]):
    return {"dense": raw_arguments["dense"], "sparse": raw_arguments["sparse"]}


@function_tool
async def value_tool(value: str):
    return value


class Model(llm.LLM):
    def __init__(
        self,
        *,
        usage=True,
        tools=False,
        error=None,
        pause=None,
        finish=None,
        boundary=None,
    ):
        super().__init__()
        self.usage = usage
        self.tools_enabled = tools
        self.error = error
        self.pause = pause
        self.finish = finish
        self.boundary = boundary

    @property
    def model(self):
        return "fixture-model"

    @property
    def provider(self):
        return "openai"

    def chat(self, *, chat_ctx, tools=None, conn_options=None, **kwargs):
        return Stream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options or APIConnectOptions(max_retry=0),
        )

    async def aclose(self):
        pass


class Stream(llm.LLMStream):
    async def _run(self):
        if self._llm.pause:
            await self._llm.pause.wait()
        if self._llm.error is not None:
            raise self._llm.error
        if self._llm.boundary:
            self._llm.boundary()
        calls = (
            [
                llm.FunctionToolCall(
                    name="vector_tool",
                    arguments=json.dumps(
                        {
                            "dense": list(range(5000)),
                            "sparse": {i * 2: i / 256 for i in range(256)},
                            "api_key": "PRIVATE_CREDENTIAL",
                            "content": 'Bearer fixture-token"quoted" value',
                        }
                    ),
                    call_id=f"actual-call-{i}",
                )
                for i in range(2)
            ]
            if self._llm.tools_enabled
            else []
        )
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id="actual-request",
                delta=llm.ChoiceDelta(
                    role="assistant", content="PRIVATE_OUTPUT", tool_calls=calls
                ),
            )
        )
        await asyncio.sleep(0)
        if self._llm.finish:
            await self._llm.finish.wait()
        if self._llm.usage:
            values = {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "prompt_cached_tokens": 3,
            }
            fields = getattr(llm.CompletionUsage, "model_fields", {})
            if "reasoning_tokens" in fields:
                values["reasoning_tokens"] = 2
            if "cache_creation_tokens" in fields:
                values["cache_creation_tokens"] = 4
            if isinstance(self._llm.usage, dict):
                values = self._llm.usage
            self._event_ch.send_nowait(
                llm.ChatChunk(id="actual-request", usage=llm.CompletionUsage(**values))
            )


def chat_context(history=False):
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="system", content="Controlled native SDK instructions")
    if history:
        ctx.insert(
            llm.FunctionCall(
                call_id="history-only", name="previous", arguments='{"x":1}'
            )
        )
        ctx.insert(
            llm.FunctionCallOutput(
                call_id="history-only",
                name="previous",
                output=json.dumps({"vector": list(range(5000))}),
                is_error=False,
            )
        )
    ctx.add_message(role="user", content="PRIVATE_INPUT")
    return ctx


def provider_model(*, usage=True, error=False):
    def boundary(request):
        if error:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "controlled SDK error",
                        "type": "rate_limit_error",
                    }
                },
            )
        frames = [
            {
                "id": "actual-openai",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "controlled provider output",
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "actual-openai",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        if usage is not False:
            counts = {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            }
            if isinstance(usage, dict):
                counts = usage
            frames.append(
                {
                    "id": "actual-openai",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "fixture-model",
                    "choices": [],
                    "usage": counts,
                }
            )
        return httpx.Response(
            200,
            content="".join("data: " + json.dumps(f) + "\n\n" for f in frames)
            + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    client = AsyncOpenAI(
        api_key="synthetic",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(boundary)),
    )
    return plugin.LLM(model="fixture-model", client=client), client
