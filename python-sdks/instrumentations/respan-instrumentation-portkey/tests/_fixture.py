"""Actual released SDK HTTP/JSON/SSE boundary fixtures; no public method patches."""

from __future__ import annotations

import json

import httpx


def chat_body(model="fixture-model", tools=False, content="Portkey response."):
    message = {"role": "assistant", "content": content}
    if tools:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "current-call",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"Tokyo"}'},
                }
            ],
        }
    return {
        "id": "chat-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tools else "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 9,
            "total_tokens": 21,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }


def response_body(model="response-model"):
    return {
        "id": "resp-1",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "metadata": {},
        "model": model,
        "output": [
            {
                "id": "message-1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Responses result.",
                        "annotations": [],
                    }
                ],
            },
            {
                "id": "fc-1",
                "type": "function_call",
                "status": "completed",
                "call_id": "response-call",
                "name": "weather",
                "arguments": '{"city":"Tokyo"}',
            },
        ],
        "parallel_tool_calls": True,
        "temperature": 1,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1,
        "usage": {
            "input_tokens": 13,
            "output_tokens": 8,
            "total_tokens": 21,
            "input_tokens_details": {"cached_tokens": 5},
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    }


def events(body, *, responses=False):
    if responses:
        response = response_body(body.get("model", "response-model"))
        return [
            {
                "type": "response.created",
                "response": {
                    **response,
                    "status": "in_progress",
                    "output": [],
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "message-1",
                "output_index": 0,
                "content_index": 0,
                "delta": "Responses result.",
                "sequence_number": 1,
            },
            {"type": "response.completed", "response": response, "sequence_number": 2},
        ]
    model = body.get("model", "fixture-model")
    common = {
        "id": "stream-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
    }
    if body.get("tools"):
        delta1 = {
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "stream-call",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":'},
                }
            ],
        }
        delta2 = {"tool_calls": [{"index": 0, "function": {"arguments": '"Tokyo"}'}}]}
    else:
        delta1 = {"role": "assistant", "content": "Portkey "}
        delta2 = {"content": "stream."}
    return [
        {**common, "choices": [{"index": 0, "delta": delta1, "finish_reason": None}]},
        {
            **common,
            "choices": [
                {
                    "index": 0,
                    "delta": delta2,
                    "finish_reason": "tool_calls" if body.get("tools") else "stop",
                }
            ],
        },
        {
            **common,
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        },
    ]


class Transport:
    def __init__(self, checkpoint=None):
        self.requests = []
        self.checkpoint = checkpoint

    def __call__(self, request):
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if self.checkpoint:
            self.checkpoint(request, body)
        model = body.get("model", "fixture-model")
        if model == "error-401":
            return httpx.Response(
                401,
                json={"error": {"message": "controlled authorization error"}},
                request=request,
            )
        if model == "connection-error":
            raise httpx.ConnectError("controlled connection error", request=request)
        if body.get("stream"):
            ev = events(body, responses=request.url.path.endswith("/responses"))
            text = (
                "".join("data: " + json.dumps(e) + "\n\n" for e in ev)
                + "data: [DONE]\n\n"
            )
            return httpx.Response(
                200,
                text=text,
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": model,
                    "data": [
                        {
                            "object": "embedding",
                            "index": 0,
                            "embedding": [float(i) / 3072 for i in range(3072)],
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                },
                request=request,
            )
        if request.url.path.endswith("/responses"):
            data = response_body(model)
        elif (
            request.url.path.endswith("/completions")
            and not request.url.path.endswith("/chat/completions")
            and "/prompts/" not in request.url.path
        ):
            data = {
                "id": "text-1",
                "object": "text_completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "text": "Text completion.",
                        "index": 0,
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                },
            }
        else:
            data = chat_body(
                model,
                bool(body.get("tools")),
                '{"city":"Tokyo"}'
                if body.get("response_format")
                else "Portkey response.",
            )
        if body.get("tools") and "choices" in data:
            previous = (
                bool(body.get("messages"))
                and body["messages"][-1].get("role") == "tool"
            )
            if previous:
                data = chat_body(model, False, "Tokyo is sunny and 72F.")
            else:
                first = body["tools"][0]
                function = first.get("function", first)
                if isinstance(function, dict) and isinstance(function.get("name"), str):
                    data["choices"][0]["message"]["tool_calls"][0]["function"][
                        "name"
                    ] = function["name"]
        return httpx.Response(200, json=data, request=request)
