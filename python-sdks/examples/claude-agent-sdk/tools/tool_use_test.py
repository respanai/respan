#!/usr/bin/env python3
"""
Tool Use — Trace agent tool calls through Respan.

Runs a query that uses Claude Code's built-in tools (Read, Glob, Grep),
then exports the full trace including tool spans.

Setup:
    pip install claude-agent-sdk respan-ai respan-instrumentation-claude-agent-sdk python-dotenv

Run:
    python tools/tool_use_test.py
"""

import asyncio
import sys
import os

import pytest
from claude_agent_sdk import ClaudeAgentOptions

# Add basic/ to path for _sdk_runtime (standalone only, conftest handles pytest)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "basic"))


@pytest.mark.asyncio
async def test_tool_use():
    """Run a query that uses tools and verify tool spans are exported."""
    from _sdk_runtime import query_for_result

    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        max_turns=3,
        allowed_tools=["Read", "Glob", "Grep"],
    )

    def _on_message(message):
        print(f"  {type(message).__name__}")

    result = await query_for_result(
        prompt="List the Python files in the current directory. Just show filenames.",
        options=options,
        on_message=_on_message,
    )

    print(f"\nResult: subtype={result.subtype}, turns={result.num_turns}")
    print(f"Session: {result.session_id}")
    print("Check Respan traces to see tool spans (Read, Glob, etc.)")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)

    from respan import Respan
    from respan_instrumentation_claude_agent_sdk import ClaudeAgentSDKInstrumentor

    respan = Respan(instrumentations=[ClaudeAgentSDKInstrumentor()])
    asyncio.run(test_tool_use())
