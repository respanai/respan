"""
Run all examples with a fake Claude Agent SDK transport.

Does NOT require ANTHROPIC_API_KEY. The fake transport still goes through the
real claude_agent_sdk.query() path, so Respan instrumentation sees the same
InternalClient.process_query seam as a live SDK run.
"""

import asyncio
import json
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv


def _load_env_files() -> None:
    seen = set()
    for directory in [Path.cwd(), *Path(__file__).resolve().parents]:
        env_path = directory / ".env"
        if env_path.exists() and env_path not in seen:
            seen.add(env_path)
            load_dotenv(env_path, override=True)


_load_env_files()

import claude_agent_sdk
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage
from claude_agent_sdk._internal.transport import Transport

from respan import Respan
from respan_instrumentation_claude_agent_sdk import ClaudeAgentSDKInstrumentor

_current_session_id: str | None = None
_current_prompt_text = ""
_current_with_tools = False
_original_query = None


class FakeClaudeTransport(Transport):
    """Minimal Claude Agent SDK transport that emits valid protocol messages."""

    def __init__(self, *, session_id: str, prompt_text: str, with_tools: bool = False):
        self._session_id = session_id
        self._prompt_text = prompt_text
        self._with_tools = with_tools
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self._closed = False

    async def connect(self) -> None:
        return None

    async def write(self, data: str) -> None:
        message = json.loads(data)
        if message.get("type") == "control_request":
            await self._queue.put(
                {
                    "type": "control_response",
                    "response": {
                        "request_id": message["request_id"],
                        "response": {"ok": True},
                    },
                }
            )
            return

        if message.get("type") == "user":
            await self._enqueue_conversation()

    async def _enqueue_conversation(self) -> None:
        await self._queue.put(
            {
                "type": "system",
                "subtype": "init",
                "session_id": self._session_id,
                "data": {"session_id": self._session_id},
            }
        )

        content_blocks = [{"type": "text", "text": f"Response to: {self._prompt_text}"}]
        if self._with_tools:
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "Glob",
                    "input": {"pattern": "*.py"},
                }
            )

        await self._queue.put(
            {
                "type": "assistant",
                "session_id": self._session_id,
                "message": {
                    "id": f"msg-{uuid.uuid4().hex[:8]}",
                    "model": "claude-sonnet-4-5-20250514",
                    "role": "assistant",
                    "content": content_blocks,
                    "usage": {
                        "input_tokens": 50,
                        "output_tokens": 10,
                        "cache_read_input_tokens": 5,
                        "cache_creation_input_tokens": 0,
                    },
                },
            }
        )

        await self._queue.put(
            {
                "type": "result",
                "subtype": "success",
                "duration_ms": 800,
                "duration_api_ms": 500,
                "is_error": False,
                "num_turns": 1,
                "session_id": self._session_id,
                "total_cost_usd": 0.003,
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 0,
                },
                "result": f"Response to: {self._prompt_text}",
            }
        )
        await self._queue.put(None)

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._queue.put(None)

    async def end_input(self) -> None:
        return None

    def is_ready(self) -> bool:
        return True

    async def read_messages(self):
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item


def set_fake_query(session_id: str, prompt_text: str, with_tools: bool = False):
    """Set the fake transport payload for the next query call."""
    global _current_session_id, _current_prompt_text, _current_with_tools
    _current_session_id = session_id
    _current_prompt_text = prompt_text
    _current_with_tools = with_tools


async def _dispatch_query(*, prompt, options=None, transport=None):
    if _original_query is None:
        raise RuntimeError("Original Claude Agent SDK query has not been installed.")
    if transport is None:
        if _current_session_id is None:
            raise RuntimeError("Fake transport was not configured for this query.")
        transport = FakeClaudeTransport(
            session_id=_current_session_id,
            prompt_text=_current_prompt_text,
            with_tools=_current_with_tools,
        )
    async for message in _original_query(prompt=prompt, options=options, transport=transport):
        yield message


async def run_hello_world():
    """basic/hello_world_test.py"""
    session_id = str(uuid.uuid4())
    set_fake_query(session_id, "What is 2 + 2?")

    from basic._sdk_runtime import query_for_result

    result = await query_for_result(
        prompt="What is 2 + 2? Reply in one word.",
        options=ClaudeAgentOptions(),
    )
    print(f"  Result: {result.subtype}, Session: {result.session_id}")
    return session_id


async def run_wrapped_query():
    """basic/wrapped_query_test.py"""
    session_id = str(uuid.uuid4())
    set_fake_query(session_id, "Primary colors")

    from basic._sdk_runtime import query_for_result

    message_types = []

    def _on_message(message):
        message_types.append(type(message).__name__)

    result = await query_for_result(
        prompt="Name three primary colors.",
        options=ClaudeAgentOptions(),
        on_message=_on_message,
    )
    print(f"  Flow: {' -> '.join(message_types)}, Result: {result.subtype}")
    return session_id


async def run_multi_turn():
    """sessions/multi_turn_test.py"""
    prompts = ["My name is Alice.", "What is my name?"]
    session_ids = []

    for prompt in prompts:
        session_id = str(uuid.uuid4())
        session_ids.append(session_id)
        set_fake_query(session_id, prompt)

        result = None
        async for message in claude_agent_sdk.query(prompt=prompt, options=ClaudeAgentOptions()):
            if isinstance(message, ResultMessage):
                result = message

        if result:
            print(f"  Turn: {result.subtype}, Session: {result.session_id}")

    return session_ids[-1]


async def run_stream_messages():
    """streaming/stream_messages_test.py"""
    session_id = str(uuid.uuid4())
    set_fake_query(session_id, "Write a haiku")

    message_flow = []
    async for message in claude_agent_sdk.query(
        prompt="Write a haiku about programming.",
        options=ClaudeAgentOptions(),
    ):
        message_flow.append(type(message).__name__)

    print(f"  Flow: {' -> '.join(message_flow)}")
    return session_id


async def run_tool_use():
    """tools/tool_use_test.py"""
    session_id = str(uuid.uuid4())
    set_fake_query(session_id, "List Python files", with_tools=True)

    from basic._sdk_runtime import query_for_result

    result = await query_for_result(
        prompt="List the Python files in the current directory.",
        options=ClaudeAgentOptions(),
    )
    print(f"  Result: {result.subtype}, Session: {result.session_id}")
    return session_id


async def run_multi_tool():
    """tools/multi_tool_test.py"""
    session_id = str(uuid.uuid4())
    set_fake_query(session_id, "Find and read Python files", with_tools=True)

    from basic._sdk_runtime import query_for_result

    result = await query_for_result(
        prompt="Find all Python files, read the first one.",
        options=ClaudeAgentOptions(),
    )
    print(f"  Result: {result.subtype}, Session: {result.session_id}")
    return session_id


async def main():
    global _original_query

    instrumentor = ClaudeAgentSDKInstrumentor(capture_content=True)
    respan = Respan(instrumentations=[instrumentor])

    _original_query = claude_agent_sdk.query
    claude_agent_sdk.query = _dispatch_query

    examples = [
        ("basic/hello_world_test.py", run_hello_world),
        ("basic/wrapped_query_test.py", run_wrapped_query),
        ("sessions/multi_turn_test.py", run_multi_turn),
        ("streaming/stream_messages_test.py", run_stream_messages),
        ("tools/tool_use_test.py", run_tool_use),
        ("tools/multi_tool_test.py", run_multi_tool),
    ]

    results = {}
    for name, fn in examples:
        print(f"\n[{name}]")
        with respan.propagate_attributes(
            customer_identifier="example-test-run",
            metadata={"example": name},
        ):
            session_id = await fn()
            results[name] = session_id

    time.sleep(2)

    print(f"\n{'='*60}")
    print(f"All {len(examples)} examples ran successfully!")
    print("customer_identifier = example-test-run")
    for name, sid in results.items():
        print(f"  {name}: session={sid}")


if __name__ == "__main__":
    asyncio.run(main())
