"""Basic LangChain tracing test for Respan exporter."""

import os

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langchain_openai")

from langchain_openai import ChatOpenAI

from respan_exporter_langchain import RespanCallbackHandler


def test_langchain_tracing_exporter_basic():
    """Run a LangChain chain and send traces to Respan."""

    respan_api_key = os.getenv("RESPAN_API_KEY")
    if not respan_api_key:
        pytest.skip("RESPAN_API_KEY not set")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")

    handler = RespanCallbackHandler(
        api_key=respan_api_key,
        base_url=os.getenv("RESPAN_BASE_URL"),
    )

    model_id = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_id, api_key=openai_api_key)

    result = llm.invoke(
        "Hello from Respan LangChain exporter test",
        config={"callbacks": [handler]},
    )

    assert result is not None
    assert result.content
