#!/usr/bin/env python3
"""
Multi-Turn Session — Multiple queries sharing a conversation session.

Demonstrates running sequential queries where the agent maintains context
across turns. Each turn is traced with its session ID.

Setup:
    pip install claude-agent-sdk respan-ai respan-instrumentation-claude-agent-sdk python-dotenv

Run:
    python sessions/multi_turn_test.py
"""

import asyncio

import pytest
import claude_agent_sdk
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage


@pytest.mark.asyncio
async def test_multi_turn():
    """Run multiple turns and verify session tracking."""

    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        max_turns=1,
    )

    prompts = [
        "My name is Alice and I'm a software engineer.",
        "What is my name? Reply in one sentence.",
    ]

    for i, prompt in enumerate(prompts, 1):
        result = None
        async for message in claude_agent_sdk.query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                result = message

        if result:
            print(f"Turn {i}: subtype={result.subtype}")
            print(f"  Session: {result.session_id}")

    print("\nView traces at: https://platform.respan.ai/platform/traces")
    print("Each turn appears as a separate trace with its session ID.")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)

    from respan import Respan
    from respan_instrumentation_claude_agent_sdk import ClaudeAgentSDKInstrumentor

    respan = Respan(instrumentations=[ClaudeAgentSDKInstrumentor()])
    asyncio.run(test_multi_turn())
