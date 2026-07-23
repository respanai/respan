"""
Hello World — Claude Agent SDK + Respan tracing.

The simplest possible example: ask Claude a question, see the trace in Respan.

Setup:
    pip install claude-agent-sdk respan-ai respan-instrumentation-claude-agent-sdk python-dotenv

Run:
    python basic/hello_world_test.py

    # or with pytest:
    pytest basic/hello_world_test.py -v
"""

import asyncio

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from _sdk_runtime import query_for_result


@pytest.mark.asyncio
async def test_hello_world():
    """Ask Claude a simple question and export the trace."""

    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        max_turns=1,
    )

    result_message = await query_for_result(
        prompt="What is 2 + 2? Reply in one word.",
        options=options,
    )

    print(f"Result: {result_message.subtype}")
    print(f"Session: {result_message.session_id}")
    print("View trace at: https://platform.respan.ai/platform/traces")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)

    from respan import Respan
    from respan_instrumentation_claude_agent_sdk import ClaudeAgentSDKInstrumentor

    respan = Respan(instrumentations=[ClaudeAgentSDKInstrumentor()])
    asyncio.run(test_hello_world())
