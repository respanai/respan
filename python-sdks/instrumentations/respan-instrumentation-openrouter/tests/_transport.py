"""Controlled HTTP fixtures consumed by the released SDKs' real parsers."""

from __future__ import annotations

import json

import httpx

MODEL = "openai/gpt-4.1-mini"
TEXT = "Trace data flows clearly."
VECTOR = [index / 4096 for index in range(3072)]
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Return weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
CALL = {
    "id": "call_weather_p9",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
}


def chat_payload(*, tools=False, structured=False):
    content = '{"summary":"Trace data flows clearly."}' if structured else TEXT
    message = {
        "role": "assistant",
        "content": None if tools else content,
        "refusal": None,
    }
    if tools:
        message["tool_calls"] = [CALL]
    return {
        "id": "chatcmpl-p9-openrouter",
        "object": "chat.completion",
        "system_fingerprint": None,
        "created": 1,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tools else "stop",
                "message": message,
            }
        ],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 5,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    }


def response_payload(*, tools=False):
    output = (
        [
            {
                "id": "fc-p9",
                "type": "function_call",
                "call_id": CALL["id"],
                "name": "get_weather",
                "arguments": CALL["function"]["arguments"],
                "status": "completed",
            }
        ]
        if tools
        else [
            {
                "id": "msg-p9",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": TEXT,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        ]
    )
    return {
        "id": "resp-p9-openrouter",
        "object": "response",
        "created_at": 1,
        "model": MODEL,
        "status": "completed",
        "completed_at": 2,
        "frequency_penalty": 0,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "metadata": {},
        "output": output,
        "parallel_tool_calls": True,
        "presence_penalty": 0,
        "temperature": 1,
        "top_p": 1,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 7,
            "output_tokens": 5,
            "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 1},
        },
    }


def events(values):
    return (
        "".join("data: " + json.dumps(value) + "\n\n" for value in values)
        + "data: [DONE]\n\n"
    ).encode()


def reply(request):
    body = json.loads(request.content or b"{}")
    if body.get("model") == "fixture/error":
        return httpx.Response(
            429,
            json={"error": {"message": "Controlled provider rate limit", "code": 429}},
            request=request,
        )
    if request.url.path.endswith("/embeddings"):
        return httpx.Response(
            200,
            json={
                "id": "embed-p9",
                "object": "list",
                "model": "openai/text-embedding-3-small",
                "data": [{"object": "embedding", "index": 0, "embedding": VECTOR}],
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
            request=request,
        )
    if request.url.path.endswith("/responses"):
        data = response_payload(tools=bool(body.get("tools")))
        if body.get("stream"):
            values = [
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {
                        **data,
                        "status": "in_progress",
                        "output": [],
                        "usage": None,
                    },
                },
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "id": "msg-p9",
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                },
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 2,
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": "msg-p9",
                    "delta": TEXT,
                    "logprobs": [],
                },
                {"type": "response.completed", "sequence_number": 3, "response": data},
            ]
            return httpx.Response(
                200,
                content=events(values),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        return httpx.Response(200, json=data, request=request)
    if request.url.path.endswith("/completions") and not request.url.path.endswith(
        "/chat/completions"
    ):
        return httpx.Response(
            200,
            json={
                "id": "cmpl-p9",
                "object": "text_completion",
                "created": 1,
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "text": TEXT,
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 5,
                    "total_tokens": 12,
                },
            },
            request=request,
        )
    data = chat_payload(
        tools=bool(body.get("tools")), structured=bool(body.get("response_format"))
    )
    if body.get("stream"):
        values = []
        for index, text in enumerate(("Trace data ", "flows clearly.")):
            delta = {"role": "assistant", "content": text}
            if body.get("tools"):
                delta = {
                    "tool_calls": [
                        {
                            "index": 0,
                            **(
                                {"id": CALL["id"], "type": "function"}
                                if index == 0
                                else {}
                            ),
                            "function": {
                                "name": "get_weather" if index == 0 else "",
                                "arguments": '{"city":' if index == 0 else '"Tokyo"}',
                            },
                        }
                    ]
                }
            values.append(
                {
                    "id": data["id"],
                    "object": "chat.completion.chunk",
                    "system_fingerprint": None,
                    "created": 1,
                    "model": MODEL,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
            )
        values.append(
            {
                "id": data["id"],
                "object": "chat.completion.chunk",
                "system_fingerprint": None,
                "created": 1,
                "model": MODEL,
                "choices": [],
                "usage": data["usage"],
            }
        )
        return httpx.Response(
            200,
            content=events(values),
            headers={"content-type": "text/event-stream"},
            request=request,
        )
    return httpx.Response(200, json=data, request=request)
