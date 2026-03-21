"""Streaming — Stream a chat completion, auto-traced."""

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

stream = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[{"role": "user", "content": "Write a haiku about Python."}],
    stream=True,

)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()

respan.flush()
