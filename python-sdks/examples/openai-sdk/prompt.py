"""Prompt — Use a Respan-managed prompt template with variables."""

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

response = client.chat.completions.create(
    model="placeholder",  # model is defined in the prompt config
    messages=[],  # messages are defined in the prompt template
    extra_body={
        "prompt": {
            "prompt_id": PROMPT_ID,
            "schema_version": 2,
            "variables": {
                "feature_request": "Add a real-time notification system for order status updates",
            },
        }
    },
)

print(response.choices[0].message.content)
respan.flush()
