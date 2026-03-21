"""Prompt Multi-Turn — Continue a conversation using a managed prompt."""

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

PROMPT_ID = "d767498c1cbb4951bb122eef423b5f76"


@workflow(name="prompt_conversation")
def chat():
    # Turn 1: Initial request using the prompt template
    response = client.chat.completions.create(
        model="placeholder",
        messages=[],
        extra_body={
            "prompt": {
                "prompt_id": PROMPT_ID,
                "schema_version": 2,
                "variables": {
                    "feature_request": "Add dark mode support to the dashboard",
                },
            }
        },
    )
    plan = response.choices[0].message.content
    print(f"=== Initial Plan ===\n{plan}\n")

    # Turn 2: Follow-up without the prompt template (regular chat)
    follow_up = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are a Lead Product Engineer."},
            {"role": "assistant", "content": plan},
            {
                "role": "user",
                "content": "Can you estimate the effort for each milestone in story points?",
            },
        ],
    )
    print(f"=== Follow-up ===\n{follow_up.choices[0].message.content}")


chat()
respan.flush()
