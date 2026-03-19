"""Customer Service — Multi-agent airline support with handoffs and context."""

from __future__ import annotations

import asyncio
import random
import uuid
from typing import Union
from dotenv import load_dotenv

load_dotenv(override=True)

from pydantic import BaseModel
from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import (
    Agent, HandoffOutputItem, ItemHelpers, MessageOutputItem, RunContextWrapper,
    Runner, ToolCallItem, ToolCallOutputItem, TResponseInputItem, function_tool,
    handoff, trace,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])


class AirlineAgentContext(BaseModel):
    passenger_name: Union[str, None] = None
    confirmation_number: Union[str, None] = None
    seat_number: Union[str, None] = None
    flight_number: Union[str, None] = None


@function_tool(name_override="faq_lookup_tool", description_override="Lookup frequently asked questions.")
async def faq_lookup_tool(question: str) -> str:
    if "bag" in question or "baggage" in question:
        return "You are allowed to bring one bag on the plane. It must be under 50 pounds and 22x14x9 inches."
    elif "seats" in question or "plane" in question:
        return "There are 120 seats on the plane. 22 business class and 98 economy. Exit rows are 4 and 16."
    elif "wifi" in question:
        return "We have free wifi on the plane, join Airline-Wifi"
    return "I'm sorry, I don't know the answer to that question."


@function_tool
async def update_seat(
    context: RunContextWrapper[AirlineAgentContext], confirmation_number: str, new_seat: str
) -> str:
    """Update the seat for a given confirmation number."""
    context.context.confirmation_number = confirmation_number
    context.context.seat_number = new_seat
    assert context.context.flight_number is not None, "Flight number is required"
    return f"Updated seat to {new_seat} for confirmation number {confirmation_number}"


async def on_seat_booking_handoff(context: RunContextWrapper[AirlineAgentContext]) -> None:
    context.context.flight_number = f"FLT-{random.randint(100, 999)}"


faq_agent = Agent[AirlineAgentContext](
    name="FAQ Agent",
    handoff_description="A helpful agent that can answer questions about the airline.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an FAQ agent. Use the faq lookup tool to answer questions. Do not rely on your own knowledge.
    If you cannot answer, transfer back to the triage agent.""",
    tools=[faq_lookup_tool],
)

seat_booking_agent = Agent[AirlineAgentContext](
    name="Seat Booking Agent",
    handoff_description="A helpful agent that can update a seat on a flight.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a seat booking agent. Ask for the confirmation number and desired seat, then use the tool.
    If the customer asks something unrelated, transfer back to the triage agent.""",
    tools=[update_seat],
)

triage_agent = Agent[AirlineAgentContext](
    name="Triage Agent",
    handoff_description="A triage agent that delegates to the appropriate agent.",
    instructions=f"{RECOMMENDED_PROMPT_PREFIX} You are a helpful triaging agent. Delegate questions to other agents.",
    handoffs=[faq_agent, handoff(agent=seat_booking_agent, on_handoff=on_seat_booking_handoff)],
)

faq_agent.handoffs.append(triage_agent)
seat_booking_agent.handoffs.append(triage_agent)


async def main():
    current_agent: Agent[AirlineAgentContext] = triage_agent
    input_items: list[TResponseInputItem] = []
    context = AirlineAgentContext()
    conversation_id = uuid.uuid4().hex[:16]
    questions = ["I need to change my seat", "My confirmation is ABC123, I want seat 12A"]

    with trace("Customer service", group_id=conversation_id):
        for question in questions:
            input_items.append({"content": question, "role": "user"})
            result = await Runner.run(current_agent, input_items, context=context)
            for item in result.new_items:
                name = item.agent.name
                if isinstance(item, MessageOutputItem):
                    print(f"{name}: {ItemHelpers.text_message_output(item)}")
                elif isinstance(item, HandoffOutputItem):
                    print(f"Handed off: {item.source_agent.name} -> {item.target_agent.name}")
                elif isinstance(item, ToolCallItem):
                    print(f"{name}: Calling tool...")
                elif isinstance(item, ToolCallOutputItem):
                    print(f"{name}: Tool result: {item.output}")
            input_items = result.to_input_list()
            current_agent = result.last_agent

    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
