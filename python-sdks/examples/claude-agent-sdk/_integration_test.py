"""
Integration test — exercises the full instrumentation pipeline.

Simulates SDK messages to verify payloads reach Respan via the OTEL pipeline.
Does NOT require ANTHROPIC_API_KEY — only RESPAN_API_KEY.
"""

import asyncio
import uuid

from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env", override=True)

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
)
from claude_agent_sdk.types import TextBlock

from respan import Respan
from respan_instrumentation_claude_agent_sdk import ClaudeAgentSDKInstrumentor

instrumentor = ClaudeAgentSDKInstrumentor()
respan = Respan(instrumentations=[instrumentor])


async def main():
    import claude_agent_sdk

    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")

    # Build fake messages that mimic what the SDK would stream
    assistant_msg = AssistantMessage(
        content=[TextBlock(text="The answer is 4.")],
        model="claude-sonnet-4-5-20250514",
    )
    assistant_msg.id = f"msg-{uuid.uuid4().hex[:8]}"

    messages = [
        SystemMessage(subtype="init", data={"session_id": session_id}),
        assistant_msg,
        ResultMessage(
            subtype="success",
            duration_ms=1200,
            duration_api_ms=800,
            is_error=False,
            num_turns=1,
            session_id=session_id,
            total_cost_usd=0.003,
            usage={
                "input_tokens": 50,
                "output_tokens": 10,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 0,
            },
            result="The answer is 4.",
        ),
    ]

    # Patch query to yield our fake messages
    async def fake_query(prompt, options=None, **kwargs):
        for msg in messages:
            yield msg

    original_query = claude_agent_sdk.query
    claude_agent_sdk.query = fake_query

    # Re-activate to patch the fake query
    instrumentor.deactivate()
    instrumentor.activate()

    # Run the auto-instrumented query with propagated attributes
    with respan.propagate_attributes(
        customer_identifier="integration-test-user",
        metadata={"test": "claude-agent-sdk-instrumentation"},
    ):
        async for msg in claude_agent_sdk.query(prompt="What is 2 + 2?"):
            print(f"  {type(msg).__name__}: {getattr(msg, 'subtype', '')}")

    # Flush to ensure all payloads are sent

    # Restore
    claude_agent_sdk.query = original_query

    print("\nDone — check Respan for traces with:")
    print(f"  customer_identifier = integration-test-user")
    print(f"  session_id = {session_id}")


if __name__ == "__main__":
    asyncio.run(main())
