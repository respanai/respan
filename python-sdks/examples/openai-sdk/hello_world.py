"""Hello World — Simplest possible: one OpenAI call, auto-traced."""

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

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[{"role": "user", "content": "Say hello in three languages."}],

)
print(response.choices[0].message.content)
respan.flush()
