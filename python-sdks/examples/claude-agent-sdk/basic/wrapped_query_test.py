"""
Auto-Instrumented Query — the simplest integration pattern.

Respan auto-patches query() via ClaudeAgentSDKInstrumentor.
One line to instrument, zero boilerplate per call.

Setup:
    pip install claude-agent-sdk respan-ai respan-instrumentation-claude-agent-sdk python-dotenv

Run:
    python basic/wrapped_query_test.py

    # or with pytest:
    pytest basic/wrapped_query_test.py -v
"""

import asyncio

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from _sdk_runtime import query_for_result


@pytest.mark.asyncio
async def test_wrapped_query():
    """Use auto-instrumented query() for automatic tracing — simplest pattern."""

    message_types = []

    def _on_message(message):
        msg_type = type(message).__name__
        message_types.append(msg_type)
        print(f"  {msg_type}")

    result = await query_for_result(
        prompt="Name three primary colors. One word each, comma separated.",
        options=ClaudeAgentOptions(
            permission_mode="bypassPermissions",
            max_turns=1,
        ),
        on_message=_on_message,
    )

    print(f"\nMessage flow: {' -> '.join(message_types)}")
    print(f"Result: subtype={result.subtype}, turns={result.num_turns}")
    print("All traces exported automatically via auto-instrumented query()")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)

    from respan import Respan
    from respan_instrumentation_claude_agent_sdk import ClaudeAgentSDKInstrumentor

    respan = Respan(instrumentations=[ClaudeAgentSDKInstrumentor()])
    asyncio.run(test_wrapped_query())
