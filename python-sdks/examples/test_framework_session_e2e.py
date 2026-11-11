"""End-to-end: a framework's own conversation id becomes the Respan session.

Each test runs two turns of a real framework under one native conversation id
(LangGraph ``configurable.thread_id``, Agno ``session_id``) with no Respan
session code, then reads the session back from the Respan API and expects both
turns in it.

Sends real data. Requires RESPAN_API_KEY and RESPAN_BASE_URL (point it at
staging, e.g. https://staging-api.respan.ai/api); with a superadmin key also set
RESPAN_E2E_PROJECT to the project the key writes to. The Agno test calls a model
through the Respan gateway at the same base URL (RESPAN_E2E_MODEL, default
gpt-4o-mini).
"""

import os
import time
import uuid

import httpx
import pytest

READ_BACK_TIMEOUT_S = 180
READ_BACK_INTERVAL_S = 5
TURNS = 2


@pytest.fixture
def respan_env():
    api_key = os.getenv("RESPAN_API_KEY")
    base_url = os.getenv("RESPAN_BASE_URL")
    if not api_key or not base_url:
        pytest.skip("RESPAN_API_KEY and RESPAN_BASE_URL not set")
    return api_key, base_url.rstrip("/")


def _session_trace_ids(api_key: str, base_url: str, session_id: str) -> set[str]:
    """The session's turns (one trace each), as the Sessions page reads them."""
    params = {"view": "turns"}
    # A superadmin key has no org of its own: name the project the key writes to.
    if project := os.getenv("RESPAN_E2E_PROJECT"):
        params["project"] = project
    response = httpx.get(
        f"{base_url}/sessions/{session_id}/",
        params=params,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    if response.status_code == 404:
        return set()
    response.raise_for_status()
    return {
        turn["trace_unique_id"]
        for turn in response.json().get("results", [])
        if turn.get("trace_unique_id")
    }


def _wait_for_session(api_key: str, base_url: str, session_id: str) -> set[str]:
    deadline = time.monotonic() + READ_BACK_TIMEOUT_S
    trace_ids: set[str] = set()
    while time.monotonic() < deadline:
        trace_ids = _session_trace_ids(api_key, base_url, session_id)
        if len(trace_ids) >= TURNS:
            break
        time.sleep(READ_BACK_INTERVAL_S)
    return trace_ids


def test_langgraph_thread_id_is_one_session(respan_env):
    api_key, base_url = respan_env
    pytest.importorskip("langgraph")
    from typing import TypedDict

    from langgraph.graph import StateGraph
    from respan import Respan
    from respan_instrumentation_langchain import LangChainInstrumentor

    class State(TypedDict):
        value: str

    graph = StateGraph(State)
    graph.add_node("echo", lambda state: {"value": state["value"] + "!"})
    graph.set_entry_point("echo")
    graph.set_finish_point("echo")
    app = graph.compile()

    thread_id = f"e2e-langgraph-{uuid.uuid4()}"
    respan = Respan(
        api_key=api_key,
        base_url=base_url,
        instrumentations=[LangChainInstrumentor()],
    )
    try:
        for turn in range(TURNS):
            app.invoke(
                {"value": f"turn {turn}"},
                config={"configurable": {"thread_id": thread_id}},
            )
    finally:
        respan.shutdown()

    assert len(_wait_for_session(api_key, base_url, thread_id)) == TURNS


def test_agno_session_id_is_one_session(respan_env):
    api_key, base_url = respan_env
    pytest.importorskip("agno")
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
    from respan import Respan
    from respan_instrumentation_agno import AgnoInstrumentor

    session_id = f"e2e-agno-{uuid.uuid4()}"
    respan = Respan(
        api_key=api_key,
        base_url=base_url,
        instrumentations=[AgnoInstrumentor()],
    )
    agent = Agent(
        name="Session E2E",
        model=OpenAIChat(
            id=os.getenv("RESPAN_E2E_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            base_url=base_url,
        ),
        session_id=session_id,
        telemetry=False,
    )
    try:
        for turn in range(TURNS):
            agent.run(f"Reply with the single word: turn{turn}")
    finally:
        respan.shutdown()

    assert len(_wait_for_session(api_key, base_url, session_id)) == TURNS
