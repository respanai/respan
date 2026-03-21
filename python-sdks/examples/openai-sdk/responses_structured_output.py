"""Responses API Structured Output — Pydantic model parsing, auto-traced."""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

from pydantic import BaseModel
from openai import OpenAI
from respan import Respan, workflow
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(instrumentations=[OpenAIInstrumentor()])

client = OpenAI(
    api_key=os.getenv("RESPAN_API_KEY"),
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)


class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str
    pros: list[str]
    cons: list[str]


@workflow(name="movie_review")
def review(movie: str) -> MovieReview:
    response = client.responses.parse(
        model="gpt-4.1-nano",
        instructions="You are a film critic. Rate movies 1-10.",
        input=f"Review: {movie}",
        text_format=MovieReview,
    )
    return response.output_parsed


result = review("The Matrix")
print(f"{result.title} — {result.rating}/10")
print(f"Summary: {result.summary}")
print(f"Pros: {', '.join(result.pros)}")
print(f"Cons: {', '.join(result.cons)}")

respan.flush()
