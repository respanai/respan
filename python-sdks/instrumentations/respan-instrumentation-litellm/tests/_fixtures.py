"""Real released LiteLLM/OpenAI SDKs with controlled provider HTTP bodies."""

import json
import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
import httpx
import openai
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

USAGE = {
    "prompt_tokens": 11,
    "completion_tokens": 7,
    "total_tokens": 18,
    "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2},
    "completion_tokens_details": {"reasoning_tokens": 2},
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {f"field_{i}": {"type": "string"} for i in range(100)},
            },
        },
    }
]
HISTORY = [
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "historical-id",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"value":1}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "historical-id", "content": "historical result"},
    {"role": "user", "content": "controlled question"},
]


def transport(
    *,
    stream=False,
    tools=False,
    usage=USAGE,
    error=False,
    on_request=None,
    vector_size=5000,
    responses=False,
):
    def handle(request):
        if on_request:
            on_request()
        if error:
            return httpx.Response(
                401,
                json={
                    "error": {
                        "message": "controlled native failure",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
            )
        if request.url.path.endswith("/embeddings"):
            result = {
                "object": "list",
                "model": "fixture-embed",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [float(i) for i in range(vector_size)],
                    }
                ],
            }
            if usage is not None:
                result["usage"] = {"prompt_tokens": 9, "total_tokens": 9}
            return httpx.Response(200, json=result)
        if responses or request.url.path.endswith("/responses"):
            output = (
                [
                    {
                        "id": "item-current-id",
                        "type": "function_call",
                        "call_id": "current-response-call-id",
                        "name": "lookup",
                        "arguments": '{"value":2}',
                        "status": "completed",
                    }
                ]
                if tools
                else [
                    {
                        "id": "message-id",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "controlled response",
                                "annotations": [],
                            }
                        ],
                    }
                ]
            )
            response = {
                "id": "resp-fixture",
                "object": "response",
                "created_at": 1,
                "model": "fixture-model",
                "status": "completed",
                "output": output,
            }
            if usage is not None:
                response["usage"] = {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                    "input_tokens_details": {"cached_tokens": 3},
                    "output_tokens_details": {"reasoning_tokens": 2},
                }
            if stream:
                events = [
                    {
                        "type": "response.created",
                        "response": {**response, "output": []},
                    },
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {**output[0], "arguments": ""} if tools else output[0],
                    },
                ]
                if tools:
                    events.extend(
                        [
                            {
                                "type": "response.function_call_arguments.delta",
                                "item_id": "item-current-id",
                                "output_index": 0,
                                "delta": '{"value":',
                            },
                            {
                                "type": "response.function_call_arguments.delta",
                                "item_id": "item-current-id",
                                "output_index": 0,
                                "delta": "2}",
                            },
                        ]
                    )
                else:
                    events.append(
                        {
                            "type": "response.output_text.delta",
                            "item_id": "message-id",
                            "output_index": 0,
                            "content_index": 0,
                            "delta": "controlled response",
                        }
                    )
                events.append({"type": "response.completed", "response": response})
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content="".join(
                        "data: " + json.dumps(e) + "\n\n" for e in events
                    ).encode(),
                )
            return httpx.Response(200, json=response)
        message = {
            "role": "assistant",
            "content": None if tools else "controlled response",
        }
        if tools:
            message["tool_calls"] = [
                {
                    "id": "current-source-id",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"value":2}'},
                }
            ]
        if stream:
            delta1 = {"role": "assistant", "content": "controlled "}
            delta2 = {"content": "response"}
            if tools:
                delta1 = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "current-source-id",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"value":'},
                        }
                    ],
                }
                delta2 = {"tool_calls": [{"index": 0, "function": {"arguments": "2}"}}]}
            chunks = [
                {
                    "id": "stream-fixture",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "fixture-model",
                    "choices": [{"index": 0, "delta": delta1, "finish_reason": None}],
                },
                {
                    "id": "stream-fixture",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "fixture-model",
                    "choices": [{"index": 0, "delta": delta2, "finish_reason": None}],
                },
                {
                    "id": "stream-fixture",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "fixture-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls" if tools else "stop",
                        }
                    ],
                },
            ]
            if usage is not None:
                chunks.append(
                    {
                        "id": "stream-fixture",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "fixture-model",
                        "choices": [],
                        "usage": usage,
                    }
                )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                    + "data: [DONE]\n\n"
                ).encode(),
            )
        response = {
            "id": "chat-fixture",
            "object": "chat.completion",
            "created": 1,
            "model": "fixture-model",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tools else "stop",
                }
            ],
        }
        if usage is not None:
            response["usage"] = usage
        return httpx.Response(200, json=response)

    return httpx.MockTransport(handle)


def client(*, async_=False, **kwargs):
    if async_:
        return openai.AsyncOpenAI(
            api_key="fixture-only",
            base_url="https://fixture.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=transport(**kwargs)),
        )
    return openai.OpenAI(
        api_key="fixture-only",
        base_url="https://fixture.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=transport(**kwargs)),
    )


def response_client(*, async_=False, **kwargs):
    if async_:
        native = AsyncHTTPHandler()
        native.client = httpx.AsyncClient(transport=transport(responses=True, **kwargs))
        return native
    return HTTPHandler(
        client=httpx.Client(transport=transport(responses=True, **kwargs))
    )
