#!/usr/bin/env python3
"""
Run a real Pydantic AI agent with spans sent to Respan.

Uses Respan as the LLM gateway so only RESPAN_API_KEY is needed (no OpenAI key).
Trace is sent as a tree: workflow -> task -> agent/LLM spans.
Set RESPAN_API_KEY and run:
  RESPAN_API_KEY=your_key poetry run python scripts/run_real_gateway_test.py
"""
import asyncio
import os

from pydantic_ai import Agent
from respan_exporter_pydantic_ai import instrument_pydantic_ai
from respan_tracing import RespanTelemetry, workflow, task


def _respan_base_url() -> str:
    raw = (
        os.getenv("RESPAN_BASE_URL")
        or "https://api.respan.ai/api"
    )
    return raw.rstrip("/")


@task(name="agent_run")
async def run_agent(agent: Agent, prompt: str):
    return await agent.run(prompt)


@workflow(name="pydantic_ai_demo")
async def run_demo_workflow(agent: Agent, prompt: str):
    return await run_agent(agent=agent, prompt=prompt)


async def main() -> None:
    respan_api_key = os.getenv("RESPAN_API_KEY")
    if not respan_api_key:
        print("Error: Set RESPAN_API_KEY (used for both gateway and telemetry).")
        return

    base_url = _respan_base_url()
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = respan_api_key

    telemetry = RespanTelemetry(
        app_name="pydantic-ai-native-span-test",
        api_key=respan_api_key,
        base_url=base_url,
        is_enabled=True,
        is_batching_enabled=False,
    )

    agent = Agent(model=os.getenv("RESPAN_GATEWAY_MODEL", "openai:gpt-4o-mini"))
    instrument_pydantic_ai(agent=agent)

    print("Running agent via Respan gateway (workflow -> task -> LLM spans)...")
    result = await run_demo_workflow(agent=agent, prompt='Reply with exactly "hello native span".')
    print(f"Agent response: {result.output}")

    telemetry.flush()
    print("Done. Check your Respan dashboard for the trace tree.")


if __name__ == "__main__":
    asyncio.run(main())
