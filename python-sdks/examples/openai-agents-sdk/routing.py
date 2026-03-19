"""Routing — Triage agent routes to specialized language agents."""

import asyncio
import uuid
from dotenv import load_dotenv

load_dotenv(override=True)

from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, RawResponsesStreamEvent, TResponseInputItem, trace
from openai.types.responses import ResponseTextDeltaEvent, ResponseContentPartDoneEvent

respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])

french_agent = Agent(name="french_agent", instructions="You only speak French")
spanish_agent = Agent(name="spanish_agent", instructions="You only speak Spanish")
english_agent = Agent(name="english_agent", instructions="You only speak English")

triage_agent = Agent(
    name="triage_agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[french_agent, spanish_agent, english_agent],
)


async def main():
    conversation_id = uuid.uuid4().hex[:16]
    agent = triage_agent
    inputs: list[TResponseInputItem] = []
    questions = [
        "Can you help me with my math homework?",
        "Yeah, how to solve for x: 2x + 5 = 11?",
        "What's the capital of France?",
    ]

    with trace("Routing example", group_id=conversation_id):
        for question in questions:
            inputs.append({"content": question, "role": "user"})
            result = Runner.run_streamed(agent, input=inputs)
            async for event in result.stream_events():
                if isinstance(event, RawResponsesStreamEvent):
                    if isinstance(event.data, ResponseTextDeltaEvent):
                        print(event.data.delta, end="", flush=True)
                    elif isinstance(event.data, ResponseContentPartDoneEvent):
                        print()
            inputs = result.to_input_list()
            agent = result.current_agent
            print()

    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
