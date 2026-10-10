"""Real Pipecat OpenAI service over its released native HTTP client boundary."""

import json

from openai import AsyncOpenAI, DefaultAsyncHttpxClient, _exceptions
from pipecat.services.openai.llm import OpenAILLMService

http = getattr(_exceptions, "httpx", None) or _exceptions.httpx2


class ProviderService(OpenAILLMService):
    def __init__(self, *, raw_usage=None, **kwargs):
        self.raw_usage = raw_usage
        super().__init__(**kwargs)

    def create_client(self, **kwargs):
        def response(request):
            payload = json.loads(request.content)
            assert payload["stream"] is True
            usage = {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 2},
            }
            usage = self.raw_usage if self.raw_usage is not None else usage
            chunks = [
                {
                    "id": "native-chat",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "fixture-native-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": "Actual HTTP text.",
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "native-chat",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "fixture-native-model",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
                {
                    "id": "native-chat",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "fixture-native-model",
                    "choices": [],
                    "usage": usage,
                },
            ]
            body = (
                "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                + "data: [DONE]\n\n"
            )
            return http.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                content=body.encode(),
            )

        return AsyncOpenAI(
            api_key="fixture",
            base_url="https://provider.test",
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(transport=http.MockTransport(response)),
        )
