"""Multi-Turn — Conversational chat with message history, auto-traced."""

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
    messages = [
        {"role": "system", "content": "You are a concise cooking assistant."}
    ]
    questions = [
        "What can I make with eggs and cheese?",
        "How long does the omelette take?",
        "Any tips to make it fluffy?",
    ]

    for question in questions:
        messages.append({"role": "user", "content": question})
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,

        )
        answer = response.choices[0].message.content
        messages.append({"role": "assistant", "content": answer})
        print(f"User: {question}")
        print(f"Bot:  {answer}\n")


chat()
respan.flush()
