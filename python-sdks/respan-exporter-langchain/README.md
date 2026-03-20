# Respan Exporter for LangChain

**[respan.ai](https://respan.ai)** | **[Documentation](https://docs.respan.ai)** | **[PyPI](https://pypi.org/project/respan-exporter-langchain/)**

Respan exporter for LangChain traces using the native `BaseCallbackHandler` API.

## Installation

```bash
pip install respan-exporter-langchain
```

## Usage

```python
from langchain_openai import ChatOpenAI
from respan_exporter_langchain import RespanCallbackHandler

# Create the callback handler
handler = RespanCallbackHandler(api_key="your-respan-api-key")

# Use it with any LangChain component
llm = ChatOpenAI(model="gpt-4o-mini")
result = llm.invoke("Hello!", config={"callbacks": [handler]})

# Or pass it to a chain
chain = prompt | llm | parser
chain.invoke({"input": "Hello!"}, config={"callbacks": [handler]})
```

### Additional Options

```python
handler = RespanCallbackHandler(
    api_key="your-respan-api-key",
    environment="staging",
    customer_identifier="user-123",
    session_identifier="session-abc",
    trace_name="my-chain",
)
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `RESPAN_API_KEY` | Your Respan API key | — |
| `RESPAN_BASE_URL` | Respan API base URL | `https://api.respan.ai/api` |
| `RESPAN_ENVIRONMENT` | Environment tag for traces | `production` |
| `RESPAN_CUSTOMER_IDENTIFIER` | Customer identifier for traces | — |

## How It Works

This exporter uses LangChain's native `BaseCallbackHandler` API to capture trace events — the stable, framework-endorsed approach used by Langfuse, LangSmith, and other observability tools. Unlike OTel-based patching, this approach is resilient across LangChain version upgrades.

### Captured Events

| LangChain Event | Respan Log Type |
|---|---|
| Chain (root) | `workflow` |
| Chain (nested) | `task` |
| Agent chain | `agent` |
| LLM / Chat model | `generation` |
| Tool | `tool` |
| Retriever | `tool` |

### Features

- Automatic span hierarchy from LangChain's `run_id` / `parent_run_id`
- Token usage extraction from LLM responses
- Prompt messages and completion message capture
- Retry with exponential backoff on transient server errors
- Thread-safe span accumulation with background export
- Support for chains, agents, tools, retrievers, and chat models
