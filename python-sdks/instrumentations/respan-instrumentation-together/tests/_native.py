"""Real released Together clients, HTTP parsing and SSE bytes; no vendor stubs."""

import json

import httpx
from together import AsyncTogether, Together


class Body(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, parts):
        self.parts = parts
        self.reads = 0
        self.closes = 0

    def __iter__(self):
        for part in self.parts:
            self.reads += 1
            yield part

    async def __aiter__(self):
        for part in self.parts:
            self.reads += 1
            yield part

    def close(self):
        self.closes += 1

    async def aclose(self):
        self.closes += 1


class NativeRuntime:
    def __init__(self):
        self.requests = []
        self.bodies = []
        self.retry = False
        self.attempts = 0
        self.callback_count = 0

    def usage(self):
        return {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}

    def chat(self, payload):
        last = (payload.get("messages") or [{}])[-1].get("content", "")
        if last == "empty":
            return {
                "id": "empty-native",
                "object": "chat.completion",
                "model": payload["model"],
                "created": 1,
                "choices": [],
                "native_feedback": {"blocked": True, "reason": "controlled"},
                "zero": 0,
                "flag": False,
            }
        tools = payload.get("tools")
        has_result = any(m.get("role") == "tool" for m in payload.get("messages", []))
        message = {
            "role": "assistant",
            "content": "native response",
            "reasoning": "native reasoning",
            "reasoning_content": "full thought",
        }
        if tools and not has_result:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Tokyo","zero":0,"flag":false}',
                        },
                    }
                ],
            }
        return {
            "id": "native-chat",
            "object": "chat.completion",
            "created": 1,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls"
                    if tools and not has_result
                    else "stop",
                    "seed": 0,
                }
            ],
            "usage": self.usage(),
            "warnings": [{"message": "controlled warning"}],
        }

    def stream(self, payload, text=False):
        parts = []
        for i in range(70):
            choice = {"index": 0, "finish_reason": "stop" if i == 69 else None}
            choice.update(
                {"text": str(i) + ","}
                if text
                else {"delta": {"role": "assistant", "content": str(i) + ","}}
            )
            chunk = {
                "id": "native-stream",
                "object": "text_completion" if text else "chat.completion.chunk",
                "created": 1,
                "model": payload["model"],
                "choices": [choice],
            }
            if i == 69:
                chunk["usage"] = self.usage()
            parts.append(("data: " + json.dumps(chunk) + "\n\n").encode())
        if (payload.get("messages") or [{}])[-1].get("content") == "stream-error":
            parts = parts[:2] + [
                b'data: {"error":{"message":"controlled SSE error"}}\n\n'
            ]
        parts.append(b"data: [DONE]\n\n")
        body = Body(parts)
        self.bodies.append(body)
        return body

    def respond(self, request):
        self.attempts += 1
        payload = json.loads(request.content)
        self.requests.append(payload)
        if self.retry and self.attempts == 1:
            return httpx.Response(
                429,
                json={"error": {"message": "controlled retry"}},
                request=request,
                headers={"retry-after": "0"},
            )
        if (payload.get("messages") or [{}])[-1].get("content") == "failure":
            return httpx.Response(
                429,
                json={"error": {"message": "controlled provider failure"}},
                request=request,
            )
        path = request.url.path
        if payload.get("stream") is True:
            return httpx.Response(
                200,
                stream=self.stream(
                    payload,
                    text=path.endswith("/completions")
                    and not path.endswith("/chat/completions"),
                ),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        if path.endswith("/chat/completions"):
            value = self.chat(payload)
        elif path.endswith("/completions"):
            value = {
                "id": "native-text",
                "object": "text_completion",
                "created": 1,
                "model": payload["model"],
                "choices": [
                    {"index": 0, "text": "native completion", "finish_reason": "stop"}
                ],
                "usage": self.usage(),
            }
        elif path.endswith("/embeddings"):
            value = {
                "object": "list",
                "model": payload["model"],
                "data": [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": [float(j) for j in range(5001)],
                    }
                    for i, _ in enumerate(
                        payload["input"]
                        if isinstance(payload["input"], list)
                        else [payload["input"]]
                    )
                ],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            }
        elif path.endswith("/rerank"):
            value = {
                "id": "native-rerank",
                "model": payload["model"],
                "results": [
                    {
                        "index": 1,
                        "relevance_score": 0.9,
                        "document": {"text": "native doc", "flag": False, "zero": 0},
                    }
                ],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            }
        elif path.endswith("/images/generations"):
            value = {
                "id": "native-image",
                "data": [
                    {
                        "index": 0,
                        "b64_json": "Y29udHJvbGxlZA==",
                        "type": "b64_json",
                        "revised_prompt": "native revised",
                    }
                ],
            }
        else:
            raise AssertionError("unhandled local endpoint")
        return httpx.Response(200, json=value, request=request)

    def callback(self, response):
        self.callback_count += 1

    async def async_callback(self, response):
        self.callback_count += 1

    def client(self, retries=0):
        return Together(
            api_key="controlled-fixture",
            base_url="https://native.invalid/v1",
            max_retries=retries,
            http_client=httpx.Client(
                transport=httpx.MockTransport(self.respond),
                event_hooks={"response": [self.callback]},
            ),
        )

    def async_client(self, retries=0):
        return AsyncTogether(
            api_key="controlled-fixture",
            base_url="https://native.invalid/v1",
            max_retries=retries,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(self.respond),
                event_hooks={"response": [self.async_callback]},
            ),
        )
