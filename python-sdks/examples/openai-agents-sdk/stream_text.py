"""Stream Text — Streaming text output from an agent."""

import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, RawResponsesStreamEvent, trace
from openai.types.responses import ResponseTextDeltaEvent

respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])

agent = Agent(
    name="Storyteller",
    instructions="You are a creative storyteller. Tell short, engaging stories.",
)


async def main():
    with trace("Stream text example"):
        result = Runner.run_streamed(agent, "Tell me a short story about a robot.")
        async for event in result.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                if isinstance(event.data, ResponseTextDeltaEvent):
                    print(event.data.delta, end="", flush=True)
        print()
    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
