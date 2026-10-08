"""Controlled model boundaries; BeeAI's released run/agent/tool APIs remain real."""

import json
from collections.abc import AsyncGenerator

from beeai_framework.backend import (
    AssistantMessage,
    ChatModel,
    EmbeddingModel,
    MessageToolCallContent,
)
from beeai_framework.backend.types import (
    ChatModelOutput,
    ChatModelUsage,
    EmbeddingModelOutput,
    EmbeddingModelUsage,
)


class FixtureChatModel(ChatModel):
    def __init__(self, *, mode: str = "text", error: bool = False) -> None:
        super().__init__()
        self.mode = mode
        self.error = error
        self.calls = 0

    @property
    def model_id(self):
        return "beeai-fixture-model"

    @property
    def provider_id(self):
        return "openai"

    async def _create(self, input, run):
        self.calls += 1
        if self.error:
            raise RuntimeError("Controlled BeeAI boundary failure")
        if self.mode == "agent":
            final = next(tool for tool in input.tools if tool.name == "final_answer")
            content = MessageToolCallContent(
                id="call_beeai_final",
                tool_name=final.name,
                args=json.dumps(
                    {"response": "Tracing connects model calls and tools."}
                ),
            )
        elif self.mode == "tool":
            content = [
                MessageToolCallContent(
                    id="call_beeai_city",
                    tool_name="city_summary",
                    args='{"city":"Paris"}',
                ),
                MessageToolCallContent(
                    id="call_beeai_city_second",
                    tool_name="city_summary",
                    args='{"city":"Lyon"}',
                ),
            ]
        elif self.mode == "structured":
            content = '{"checks":["parents","usage"]}'
        else:
            content = "Check parent links and provider usage."
        return ChatModelOutput(
            output=[AssistantMessage(content)],
            usage=ChatModelUsage(
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
                cached_prompt_tokens=3,
            ),
            finish_reason="tool_calls" if self.mode in {"agent", "tool"} else "stop",
        )

    async def _create_stream(self, input, run) -> AsyncGenerator[ChatModelOutput, None]:
        yield ChatModelOutput(
            output=[AssistantMessage("Check parent ", meta={"id": "fixture-stream"})]
        )
        yield ChatModelOutput(
            output=[
                AssistantMessage(
                    "links and provider usage.", meta={"id": "fixture-stream"}
                )
            ],
            usage=ChatModelUsage(
                prompt_tokens=11, completion_tokens=7, total_tokens=18
            ),
            finish_reason="stop",
        )


class FixtureEmbeddingModel(EmbeddingModel):
    @property
    def model_id(self):
        return "beeai-fixture-embedding"

    @property
    def provider_id(self):
        return "openai"

    async def _create(self, input, run):
        return EmbeddingModelOutput(
            values=input.values,
            embeddings=[[index / 128 for index in range(128)] for _ in input.values],
            usage=EmbeddingModelUsage(
                prompt_tokens=5, completion_tokens=0, total_tokens=5
            ),
        )
