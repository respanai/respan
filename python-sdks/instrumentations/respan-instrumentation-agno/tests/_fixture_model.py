"""Synthetic OpenAI wire responses consumed by the released Agno model client."""

import json

import httpx
from agno.models.openai import OpenAIChat
from openai import AsyncOpenAI, OpenAI

MODEL = "fixture-agno-model"


def response(request):
    payload = json.loads(request.content)
    messages = payload["messages"]
    if any(m.get("content") == "expected failure" for m in messages):
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Expected fixture failure",
                    "type": "invalid_request_error",
                }
            },
        )
    tools = payload.get("tools") or []
    tool_calls = []
    if tools and not any(m.get("role") == "tool" for m in messages):
        function = tools[0]["function"]
        properties = function.get("parameters", {}).get("properties", {})
        arguments = {}
        for name, schema in properties.items():
            if name in function.get("parameters", {}).get("required", []) or name in {
                "city",
                "order_id",
                "member_id",
                "member_ids",
                "task",
            }:
                arguments[name] = (
                    ["member"]
                    if schema.get("type") == "array"
                    else (
                        "member"
                        if name == "member_id"
                        else "A-100"
                        if name == "order_id"
                        else "Paris"
                        if name == "city"
                        else "fixture task"
                    )
                )
        tool_calls = [
            {
                "id": "call-fixture",
                "type": "function",
                "function": {
                    "name": function["name"],
                    "arguments": json.dumps(arguments),
                },
            }
        ]
    content = "" if tool_calls else "Fixture Agno response."
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    common = {"id": "fixture-completion", "model": MODEL, "created": 1}
    if payload.get("stream"):
        delta = dict(message)
        if tool_calls:
            delta["tool_calls"] = [{"index": 0, **call} for call in tool_calls]
        chunks = [
            {
                **common,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                **common,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }
                ],
                "usage": usage,
            },
        ]
        return httpx.Response(
            200,
            content="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(
        200,
        json={
            **common,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": usage,
        },
    )


def model():
    return OpenAIChat(
        id=MODEL,
        client=OpenAI(
            api_key="fixture-provider-secret",
            http_client=httpx.Client(transport=httpx.MockTransport(response)),
        ),
        async_client=AsyncOpenAI(
            api_key="fixture-provider-secret",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(response)),
        ),
    )
