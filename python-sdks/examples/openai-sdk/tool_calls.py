"""Tool Calls — Function calling with OpenAI, auto-traced."""

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
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


@task(name="get_weather")
def get_weather(city: str) -> str:
    return f"Sunny, 72F in {city}"


@workflow(name="weather_assistant")
def run(question: str):
    messages = [{"role": "user", "content": question}]

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
        tools=tools,

    )
    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)
        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = get_weather(**args)
            print(f"Tool: {tc.function.name}({args}) -> {result}")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )

        final = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            tools=tools,
    
        )
        print(f"Answer: {final.choices[0].message.content}")
    else:
        print(f"Answer: {message.content}")


run("What's the weather in Paris?")
respan.flush()
