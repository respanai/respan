"""Controlled HTTP responses consumed by released LlamaIndex providers."""

from __future__ import annotations

import json

import httpx
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


def transport(*, fail: bool = False) -> httpx.MockTransport:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        if fail:
            raise httpx.ConnectError("controlled provider failure", request=request)
        if request.url.path.endswith("embeddings"):
            values = (
                payload["input"]
                if isinstance(payload["input"], list)
                else [payload["input"]]
            )
            vectors = [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": [float(n + i) for n in range(3072)],
                }
                for i, _ in enumerate(values)
            ]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": vectors,
                    "model": "text-embedding-3-large",
                    "usage": {"prompt_tokens": 7, "total_tokens": 7},
                },
            )
        messages = payload.get("messages", [])
        tools = payload.get("tools", [])
        tool_done = any(
            m.get("role") == "tool"
            or m.get("role") == "user"
            and str(m.get("content", "")).lstrip().startswith("Observation:")
            for m in messages
        )
        tool_call = tools and not tool_done
        answer = "Thought: I can answer.\nAnswer: controlled tracing answer."
        if (
            not tools
            and not tool_done
            and any("multiply_numbers" in str(m.get("content")) for m in messages)
        ):
            answer = 'Thought: use the tool.\nAction: multiply_numbers\nAction Input: {"a":7,"b":6}'
        tool_calls = (
            [
                {
                    "id": "fixture-tool-call-1",
                    "type": "function",
                    "function": {
                        "name": tools[0]["function"]["name"],
                        "arguments": json.dumps(
                            {"a": 7, "b": 6}
                            if "a"
                            in tools[0]["function"]
                            .get("parameters", {})
                            .get("properties", {})
                            else {
                                "title": "Trace tip",
                                "action": "Inspect the failed span",
                            }
                        ),
                    },
                }
            ]
            if tool_call
            else None
        )
        if payload.get("response_format"):
            answer = json.dumps(
                {"title": "Trace tip", "action": "Inspect the failed span"}
            )
        usage = {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2},
        }
        message = {
            "role": "assistant",
            "content": None if tool_call else answer,
            "tool_calls": tool_calls,
        }
        if payload.get("stream"):
            delta = dict(message)
            if tool_calls:
                delta["tool_calls"] = [{"index": 0, **tool_calls[0]}]
            chunk = {
                "id": "fixture-stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": "tool_calls" if tool_call else "stop",
                    }
                ],
                "usage": usage,
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                "id": f"fixture-chat-{calls}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_call else "stop",
                    }
                ],
                "usage": usage,
            },
        )

    return httpx.MockTransport(respond)


def build_fixture_llm(*, fail: bool = False) -> OpenAI:
    fixture = transport(fail=fail)
    return OpenAI(
        model="gpt-4o-mini",
        api_key="fixture-key",
        api_base="https://fixture.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=fixture),
        async_http_client=httpx.AsyncClient(transport=fixture),
    )


def build_fixture_embedding() -> OpenAIEmbedding:
    fixture = transport()
    return OpenAIEmbedding(
        model="text-embedding-3-large",
        api_key="fixture-key",
        api_base="https://fixture.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=fixture),
        async_http_client=httpx.AsyncClient(transport=fixture),
    )
