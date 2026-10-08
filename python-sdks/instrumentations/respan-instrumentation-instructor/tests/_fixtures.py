"""Released Instructor/OpenAI clients with controlled HTTP responses."""

import json
from contextlib import contextmanager

import httpx
import instructor
from openai import AsyncOpenAI, OpenAI


class Boundary:
    def __init__(
        self,
        *,
        payload=None,
        usage=True,
        invalid_attempts=0,
        invalid_field="age",
        status=200,
        during=None,
    ):
        self.payload = payload or {"name": "Ada", "age": 36}
        self.usage = usage
        self.invalid_attempts = invalid_attempts
        self.invalid_field = invalid_field
        self.status = status
        self.during = during
        self.requests = []

    def handle(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        if self.during:
            self.during()
        if self.status != 200:
            return httpx.Response(
                self.status, json={"error": {"message": "controlled provider failure"}}
            )
        payload = dict(self.payload)
        if len(self.requests) <= self.invalid_attempts:
            payload[self.invalid_field] = -1
        if request.url.path.endswith("/responses"):
            tools = body.get("tools") or []
            name = next(
                (tool.get("name") for tool in tools if tool.get("type") == "function"),
                "User",
            )
            result = {
                "id": "resp_fixture",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "fixture-model",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_fixture",
                        "call_id": "call_responses_fixture",
                        "name": name,
                        "arguments": json.dumps(payload),
                        "status": "completed",
                    }
                ],
            }
            if self.usage:
                result["usage"] = {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                    "input_tokens_details": {"cached_tokens": 3},
                    "output_tokens_details": {"reasoning_tokens": 2},
                }
            return httpx.Response(200, json=result)
        tools = body.get("tools") or []
        name = tools[0]["function"]["name"] if tools else "User"
        if body.get("stream"):
            data = json.dumps(
                {"tasks": [payload, dict(payload, name="Grace")]}
                if name.startswith("Iterable")
                else payload
            )
            events = []
            for index, fragment in enumerate(
                [data[: len(data) // 2], data[len(data) // 2 :]]
            ):
                delta = (
                    {"tool_calls": [{"index": 0, "function": {"arguments": fragment}}]}
                    if tools
                    else {"content": fragment}
                )
                if tools and index == 0:
                    delta["tool_calls"][0].update(
                        id="call_stream_fixture", type="function"
                    )
                    delta["tool_calls"][0]["function"]["name"] = name
                events.append(
                    {
                        "id": "chat_stream_fixture",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "fixture-model",
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": None}
                        ],
                    }
                )
            if self.usage:
                events.append(
                    {
                        "id": "chat_stream_fixture",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "fixture-model",
                        "choices": [],
                        "usage": self.chat_usage(),
                    }
                )
            text = (
                "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                + "data: [DONE]\n\n"
            )
            return httpx.Response(
                200, text=text, headers={"content-type": "text/event-stream"}
            )
        message = {"role": "assistant", "content": json.dumps(payload)}
        if tools:
            message.update(
                content=None,
                tool_calls=[
                    {
                        "id": "call_fixture",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(payload)},
                    }
                ],
            )
        result = {
            "id": "chat_fixture",
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
        if self.usage:
            result["usage"] = self.chat_usage()
        return httpx.Response(200, json=result)

    @staticmethod
    def chat_usage():
        return {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2},
        }


@contextmanager
def client_context(
    *, async_client=False, responses=False, from_provider=False, mode=None, **options
):
    boundary = Boundary(**options)
    cls = AsyncOpenAI if async_client else OpenAI
    http_cls = httpx.AsyncClient if async_client else httpx.Client
    native = cls(
        api_key="fixture-not-a-secret",
        base_url="https://fixture.invalid/v1",
        max_retries=0,
        http_client=http_cls(transport=httpx.MockTransport(boundary.handle)),
    )
    selected = mode or (
        instructor.Mode.RESPONSES_TOOLS if responses else instructor.Mode.TOOLS
    )
    if from_provider:
        client = instructor.from_provider(
            "openai/fixture-model",
            async_client=async_client,
            mode=selected,
            api_key="fixture-not-a-secret",
            base_url="https://fixture.invalid/v1",
            http_client=http_cls(transport=httpx.MockTransport(boundary.handle)),
        )
    else:
        client = instructor.from_openai(native, mode=selected)
    try:
        yield client, boundary
    finally:
        if not async_client:
            native.close()
