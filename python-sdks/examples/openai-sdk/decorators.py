"""Decorators — Use @workflow and @task to structure traces."""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

from openai import OpenAI
from respan import Respan, workflow, task
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(instrumentations=[OpenAIInstrumentor()])

client = OpenAI(
    api_key=os.getenv("RESPAN_API_KEY"),
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)


@task(name="generate_outline")
def generate_outline(topic: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "Generate a 3-point outline. Be concise."},
            {"role": "user", "content": topic},
        ],

    )
    return response.choices[0].message.content


@task(name="write_draft")
def write_draft(outline: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "Write a short paragraph from this outline."},
            {"role": "user", "content": outline},
        ],

    )
    return response.choices[0].message.content


@workflow(name="content_pipeline")
def run(topic: str) -> str:
    outline = generate_outline(topic)
    print(f"Outline:\n{outline}\n")

    draft = write_draft(outline)
    print(f"Draft:\n{draft}")
    return draft


run("Benefits of open-source software")
respan.flush()
