"""Deterministic Pipeline — Sequential agent chain: outline, check, write."""

import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from pydantic import BaseModel
from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, trace

respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])


class Outline(BaseModel):
    title: str
    sections: list[str]


class QualityCheck(BaseModel):
    score: int
    feedback: str
    is_good: bool


outline_agent = Agent(
    name="Outliner",
    instructions="Generate a short story outline with a title and 3-4 sections.",
    output_type=Outline,
)

checker_agent = Agent(
    name="Quality Checker",
    instructions="Rate the outline on a 1-10 scale. Set is_good=True if score >= 7.",
    output_type=QualityCheck,
)

writer_agent = Agent(
    name="Writer",
    instructions="Write a short story based on the provided outline. Keep it under 200 words.",
)


async def main():
    with trace("Deterministic pipeline"):
        outline_result = await Runner.run(outline_agent, "A story about a time-traveling cat")
        outline = outline_result.final_output
        print(f"Outline: {outline.title} ({len(outline.sections)} sections)")

        check_result = await Runner.run(checker_agent, f"Outline: {outline}")
        check = check_result.final_output
        print(f"Quality: {check.score}/10 - {check.feedback}")

        if check.is_good:
            story_result = await Runner.run(writer_agent, f"Write based on this outline: {outline}")
            print(f"\n{story_result.final_output}")
        else:
            print("Outline quality too low, skipping story generation.")

    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
