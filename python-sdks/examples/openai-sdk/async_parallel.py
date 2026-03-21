"""Async Parallel — Run multiple OpenAI calls concurrently, all auto-traced."""

import os
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from openai import AsyncOpenAI
from respan import Respan, workflow, task
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(instrumentations=[OpenAIInstrumentor()])

client = AsyncOpenAI(
    api_key=os.getenv("RESPAN_API_KEY"),
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)


@task(name="summarize")
async def summarize(topic: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "Summarize in one sentence."},
            {"role": "user", "content": f"What is {topic}?"},
        ],

    )
    return response.choices[0].message.content


@workflow(name="parallel_summaries")
async def run():
    topics = ["quantum computing", "blockchain", "edge computing"]
    results = await asyncio.gather(*[summarize(t) for t in topics])
    for topic, summary in zip(topics, results):
        print(f"{topic}: {summary}\n")


asyncio.run(run())
respan.flush()
