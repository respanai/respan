"""Actual OpenAI HTTP/model responses through released Braintrust wrappers."""

import json
from contextlib import contextmanager

import braintrust
import httpx
from openai import OpenAI


def response(request):
    body = json.loads(request.content)
    if request.url.path.endswith("/embeddings"):
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "fixture-embed",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [float(i) for i in range(5000)],
                    }
                ],
                "usage": {"prompt_tokens": 9, "total_tokens": 9},
            },
        )
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "current-source-id",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"value":2}'},
            }
        ],
    }
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "prompt_tokens_details": {"cached_tokens": 3},
        "completion_tokens_details": {"reasoning_tokens": 2},
    }
    if body.get("stream"):
        chunks = [
            {
                "id": "fixture-response",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": None,
                        "delta": {"role": "assistant", "content": "controlled stream"},
                    }
                ],
            },
            {
                "id": "fixture-response",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture-model",
                "choices": [],
                "usage": usage,
            },
        ]
        return httpx.Response(
            200,
            text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
            + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(
        200,
        json={
            "id": "fixture-response",
            "object": "chat.completion",
            "created": 1,
            "model": "fixture-model",
            "choices": [
                {"index": 0, "finish_reason": "tool_calls", "message": message}
            ],
            "usage": usage,
        },
    )


@contextmanager
def client_context():
    native = OpenAI(
        api_key="controlled-fixture",
        base_url="https://fixture.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(response)),
    )
    try:
        yield braintrust.wrap_openai(native)
    finally:
        native.close()
