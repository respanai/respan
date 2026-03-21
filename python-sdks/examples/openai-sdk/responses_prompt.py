"""Responses API Prompt — Use a Respan-managed prompt with the Responses API.

With schema_version 2, the prompt template becomes `instructions` and the
body `input` is preserved as the user turn.
"""

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

PROMPT_ID = "d767498c1cbb4951bb122eef423b5f76"

response = client.responses.create(
    model="gpt-4.1-nano",
    input=[
        {
            "role": "user",
            "content": "Add a real-time notification system for order status updates",
        }
    ],
    extra_body={
        "respan_params": {
            "prompt": {
                "prompt_id": PROMPT_ID,
                "schema_version": 2,
                "variables": {
                    "feature_request": "Add a real-time notification system for order status updates",
                },
            }
        }
    },
)

print(response.output_text)
respan.flush()
