"""Prompt Gateway — Use a Respan-managed prompt with the Agents SDK.

The Agents SDK calls the Responses API internally, so prompt config is
passed via the X-Data-Respan-Params header on the OpenAI client.
The gateway converts prompt messages into `instructions` for the response.

NOTE: The header applies to every request made by the client, so all agents
sharing this client will use the same prompt template.
"""

import os
import json
import base64
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from openai import AsyncOpenAI
from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, set_default_openai_client, trace

respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])

PROMPT_ID = "d767498c1cbb4951bb122eef423b5f76"
PROMPT_VARIABLES = {
    "feature_request": "Add dark mode support to the dashboard",
}

respan_params = {
    "prompt": {
        "prompt_id": PROMPT_ID,
        "schema_version": 2,
        "variables": PROMPT_VARIABLES,
    }
}

client = AsyncOpenAI(
    api_key=os.getenv("RESPAN_API_KEY"),
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
    default_headers={
        "X-Data-Respan-Params": base64.b64encode(
            json.dumps(respan_params).encode()
        ).decode(),
    },
)
set_default_openai_client(client)

agent = Agent(
    name="Planner",
    instructions="You are a helpful assistant.",
)


async def main():
    with trace("Prompt gateway"):
        with respan.propagate_attributes(
            prompt={
                "prompt_id": PROMPT_ID,
                "variables": PROMPT_VARIABLES,
            },
        ):
            result = await Runner.run(
                agent, "Add dark mode support to the dashboard"
            )
            print(result.final_output)
    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
