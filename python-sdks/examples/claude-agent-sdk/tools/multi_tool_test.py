#!/usr/bin/env python3
"""
Multi-Tool — Agent using multiple tools in sequence.

Demonstrates a multi-turn agent that uses several tools to accomplish
a task, with each tool call captured as a child span.

Setup:
    pip install claude-agent-sdk respan-ai respan-instrumentation-claude-agent-sdk python-dotenv

Run:
    python tools/multi_tool_test.py
"""

import asyncio
import sys
import os

import pytest
from claude_agent_sdk import ClaudeAgentOptions

# Add basic/ to path for _sdk_runtime (standalone only, conftest handles pytest)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "basic"))


@pytest.mark.asyncio
async def test_multi_tool():
    """Run a query that requires multiple tool calls in sequence."""
    from _sdk_runtime import query_for_result

    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        max_turns=5,
        allowed_tools=["Read", "Glob", "Grep"],
    )

    tool_count = 0

    def _on_message(message):
        nonlocal tool_count
        msg_type = type(message).__name__
        print(f"  {msg_type}")
        # AssistantMessage with tool_use blocks indicates tool calls
        if hasattr(message, "content") and hasattr(message.content, "__iter__"):
            for block in getattr(message, "content", []):
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_count += 1

    result = await query_for_result(
        prompt=(
            "Find all Python files in the current directory, "
            "then read the first one and tell me what it does. Be concise."
        ),
        options=options,
        on_message=_on_message,
    )

    print(f"\nResult: subtype={result.subtype}, turns={result.num_turns}")
    print(f"Session: {result.session_id}")
    print("Check Respan traces to see the full tool call sequence.")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)

    from respan import Respan
    from respan_instrumentation_claude_agent_sdk import ClaudeAgentSDKInstrumentor

    respan = Respan(instrumentations=[ClaudeAgentSDKInstrumentor()])
    asyncio.run(test_multi_tool())
