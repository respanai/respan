"""Responses API Streaming — Stream a response, auto-traced."""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

from openai import OpenAI
from respan import Respan
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(instrumentations=[OpenAIInstrumentor()])

client = OpenAI(
    api_key=os.getenv("RESPAN_API_KEY"),
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)

stream = client.responses.create(
    model="gpt-4.1-nano",
    instructions="You are a helpful assistant.",
    input="Write a haiku about Python.",
    stream=True,
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
print()

respan.flush()
