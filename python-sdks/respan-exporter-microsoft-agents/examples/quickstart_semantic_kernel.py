"""
Respan Microsoft Agents Exporter - Semantic Kernel Quick Start Example

This example shows how to instrument Semantic Kernel with Respan
to capture traces from your AI service calls.

Semantic Kernel natively emits OpenTelemetry spans. The Respan instrumentor
intercepts those spans and forwards them to Respan for observability.

Prerequisites:
    pip install respan-exporter-microsoft-agents semantic-kernel
    export RESPAN_API_KEY=your-api-key
    export OPENAI_API_KEY=your-openai-key
"""

import asyncio
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from respan_exporter_microsoft_agents import RespanMicrosoftAgentsInstrumentor

# 1. Set up OpenTelemetry (Semantic Kernel uses it internally)
provider = TracerProvider()
trace.set_tracer_provider(provider)

# 2. Activate Respan instrumentation to intercept Semantic Kernel spans
RespanMicrosoftAgentsInstrumentor().instrument(
    api_key=os.getenv("RESPAN_API_KEY"),
    environment="development",
)


async def main():
    # 3. Create a Semantic Kernel instance with an OpenAI chat service
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id="gpt-4o-mini"))

    # 4. Invoke a prompt - the trace is automatically captured by Respan
    result = await kernel.invoke_prompt("What is the capital of France?")
    print(f"Response: {result}")
    print("\nTrace has been sent to Respan!")


if __name__ == "__main__":
    asyncio.run(main())
