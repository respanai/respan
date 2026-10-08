"""Real released OpenAI transport fixtures for CrewAI's native LLM connector."""

import json

import httpx
from crewai import LLM
from openai import AsyncOpenAI, OpenAI


class FixtureServer:
    def __init__(self):
        self.requests = []
        self.before_response = None

    def handle(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        if self.before_response:
            self.before_response()
        text = json.dumps(body)
        if "fixture-failure" in text:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "controlled fixture failure",
                        "type": "invalid_request_error",
                        "code": "fixture_failure",
                    }
                },
            )
        usage = {
            "prompt_tokens": 9,
            "completion_tokens": 3,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 2},
        }
        if request.url.path.endswith("/responses"):
            return httpx.Response(
                200,
                json={
                    "id": "fixture-response-id",
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "model": "gpt-4o-mini",
                    "output": [
                        {
                            "type": "message",
                            "id": "fixture-message-id",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "fixture answer",
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 9,
                        "output_tokens": 3,
                        "total_tokens": 12,
                        "input_tokens_details": {"cached_tokens": 4},
                        "output_tokens_details": {"reasoning_tokens": 2},
                    },
                },
            )
        if body.get("stream"):
            base = {
                "id": "fixture-stream-id",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-4o-mini",
            }
            chunks = [
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "fixture "},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "stream"},
                            "finish_reason": "stop",
                        }
                    ],
                },
                {
                    **base,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                },
            ]
            return httpx.Response(
                200,
                text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                + "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        message = {"role": "assistant", "content": "fixture answer"}
        if body.get("tools") and not any(
            m.get("role") == "tool" for m in body.get("messages", [])
        ):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "fixture-tool-id",
                        "type": "function",
                        "function": {
                            "name": body["tools"][0]["function"]["name"],
                            "arguments": '{"city":"Tokyo"}',
                        },
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "id": "fixture-chat-id",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls"
                        if message.get("tool_calls")
                        else "stop",
                    }
                ],
                "usage": usage,
            },
        )

    def llm(self, **kwargs):
        llm = LLM(
            model="openai/gpt-4o-mini",
            api_key="fixture-only",
            base_url="https://fixture.invalid/v1",
            max_retries=0,
            **kwargs,
        )
        sync = OpenAI(
            api_key="fixture-only",
            base_url="https://fixture.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(self.handle)),
            max_retries=0,
        )
        async_client = AsyncOpenAI(
            api_key="fixture-only",
            base_url="https://fixture.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handle)),
            max_retries=0,
        )
        if hasattr(llm, "_get_sync_client"):
            llm._client = sync
            llm._async_client = async_client
        else:
            llm.client = sync
            llm.async_client = async_client
        return llm
