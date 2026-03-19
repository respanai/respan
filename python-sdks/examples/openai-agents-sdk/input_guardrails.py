"""Input Guardrails — Validate user input before processing."""

import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from pydantic import BaseModel
from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, InputGuardrail, GuardrailFunctionOutput, RunContextWrapper, trace

respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])


class HomeworkCheck(BaseModel):
    is_homework: bool
    reasoning: str


guardrail_agent = Agent(
    name="Homework Checker",
    instructions="Check if the user is asking you to do their homework. Respond with is_homework=True if so.",
    output_type=HomeworkCheck,
)


async def homework_guardrail(ctx: RunContextWrapper, agent: Agent, input: str) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_homework,
    )


agent = Agent(
    name="Assistant",
    instructions="You are a helpful math tutor. You help students understand concepts but don't solve homework.",
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)


async def main():
    with trace("Input guardrails"):
        result = await Runner.run(agent, "Can you explain what a derivative is?")
        print(f"Passed: {result.final_output[:100]}...")

        try:
            result = await Runner.run(agent, "Solve this homework: what is 2x + 5 = 11, find x")
            print(f"Passed: {result.final_output[:100]}...")
        except Exception as e:
            print(f"Guardrail tripped: {e}")

    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
