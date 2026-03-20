"""
Respan Microsoft Agents Exporter - AutoGen Quick Start Example

This example shows how to instrument AutoGen with Respan
to capture traces from your agent runs.

AutoGen natively emits OpenTelemetry spans. The Respan instrumentor
intercepts those spans and forwards them to Respan for observability.

Prerequisites:
    pip install respan-exporter-microsoft-agents autogen-agentchat autogen-ext
    export RESPAN_API_KEY=your-api-key
    export OPENAI_API_KEY=your-openai-key
"""

import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from respan_exporter_microsoft_agents import RespanMicrosoftAgentsInstrumentor

# 1. Set up OpenTelemetry (AutoGen uses it internally)
provider = TracerProvider()
trace.set_tracer_provider(provider)

# 2. Activate Respan instrumentation to intercept AutoGen spans
RespanMicrosoftAgentsInstrumentor().instrument(
    api_key=os.getenv("RESPAN_API_KEY"),
    environment="development",
)


async def main():
    # 3. Create an AutoGen agent with an OpenAI model
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful assistant. Keep answers concise.",
    )

    # 4. Run a simple query - the trace is automatically captured by Respan
    response = await agent.on_messages(
        [TextMessage(content="What is the capital of France?", source="user")],
        cancellation_token=CancellationToken(),
    )
    print(f"Agent response: {response.chat_message.content}")
    print("\nTrace has been sent to Respan!")


if __name__ == "__main__":
    asyncio.run(main())
