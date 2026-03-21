"""Responses API — Simplest possible: one call, auto-traced."""

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

response = client.responses.create(
    model="gpt-4.1-nano",
    instructions="You are a helpful assistant.",
    input="Say hello in three languages.",
)
print(response.output_text)
respan.flush()
