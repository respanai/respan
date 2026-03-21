"""Responses API Tool Calls — Function calling with the Responses API, auto-traced."""

import os
import json
from dotenv import load_dotenv

load_dotenv(override=True)

from openai import OpenAI
from respan import Respan, workflow, task
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(instrumentations=[OpenAIInstrumentor()])

client = OpenAI(
    api_key=os.getenv("RESPAN_API_KEY"),
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


@task(name="get_weather")
def get_weather(city: str) -> str:
    return f"Sunny, 72°F in {city}"


@workflow(name="weather_assistant")
def run(question: str):
    response = client.responses.create(
        model="gpt-4.1-nano",
        instructions="You are a weather assistant.",
        input=[{"role": "user", "content": question}],
        tools=tools,
    )

    # Check for function calls in output
    tool_call = next(
        (item for item in response.output if item.type == "function_call"),
        None,
    )

    if tool_call:
        args = json.loads(tool_call.arguments)
        result = get_weather(**args)
        print(f"Tool: {tool_call.name}({args}) -> {result}")

        # Send tool result back
        final = client.responses.create(
            model="gpt-4.1-nano",
            instructions="You are a weather assistant.",
            input=[
                {"role": "user", "content": question},
                *response.output,
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": result,
                },
            ],
            tools=tools,
        )
        print(f"Answer: {final.output_text}")
    else:
        print(f"Answer: {response.output_text}")


run("What's the weather in Paris?")
respan.flush()
