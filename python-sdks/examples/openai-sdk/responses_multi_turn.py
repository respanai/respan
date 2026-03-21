"""Responses API Multi-Turn — Continue a conversation using previous_response_id."""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

from openai import OpenAI
from respan import Respan, workflow
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(instrumentations=[OpenAIInstrumentor()])

client = OpenAI(
    api_key=os.getenv("RESPAN_API_KEY"),
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)


@workflow(name="conversation")
def chat():
    # Turn 1
    r1 = client.responses.create(
        model="gpt-4.1-nano",
        instructions="You are a helpful assistant. Be concise.",
        input="What is the capital of France?",
        store=True,
    )
    print(f"Turn 1: {r1.output_text}\n")

    # Turn 2 — uses previous_response_id to chain context
    r2 = client.responses.create(
        model="gpt-4.1-nano",
        input="And what is its population?",
        previous_response_id=r1.id,
        store=True,
    )
    print(f"Turn 2: {r2.output_text}\n")

    # Turn 3
    r3 = client.responses.create(
        model="gpt-4.1-nano",
        input="Name three famous landmarks there.",
        previous_response_id=r2.id,
        store=True,
    )
    print(f"Turn 3: {r3.output_text}")


chat()
respan.flush()
