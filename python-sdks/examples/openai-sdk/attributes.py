"""Attributes — Attach customer info and metadata to traces."""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

from openai import OpenAI
from respan import Respan, workflow, propagate_attributes
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(
    instrumentations=[OpenAIInstrumentor()],
    metadata={"service": "chat-api", "version": "1.0.0"},
)

client = OpenAI(
    api_key=os.getenv("RESPAN_API_KEY"),
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)


@workflow(name="handle_request")
def handle_request(user_id: str, question: str):
    with propagate_attributes(
        customer_identifier=user_id,
        thread_identifier="conv_001",
        metadata={"plan": "pro"},  # merged with default metadata
    ):
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": question}],

        )
        print(f"[{user_id}] {response.choices[0].message.content}")


handle_request("user_alice", "What is an API gateway?")
handle_request("user_bob", "Explain rate limiting.")
respan.flush()
