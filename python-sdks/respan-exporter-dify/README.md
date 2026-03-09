# respan-exporter-dify

**[respan.ai](https://respan.ai)** | **[Documentation](https://docs.respan.ai)** | **[PyPI](https://pypi.org/project/respan-exporter-dify/)**

Respan exporter for the Dify Python SDK (`dify-client-python`). Wrap the official Dify client to send chat, completion, and workflow calls to Respan for observability and cost tracking.

---

## Installation

```bash
pip install respan-exporter-dify
```

---

## Configuration

Gateway + tracing can be configured via environment variables or by passing
credentials in code.

| Variable           | Description                                  |
|--------------------|----------------------------------------------|
| `RESPAN_API_KEY`   | Your Respan API key                          |
| `RESPAN_BASE_URL`  | Respan gateway base URL (optional)           |
| `RESPAN_ENDPOINT`  | Respan ingest endpoint (optional; default)   |

---

## Usage

### Gateway mode (sync, no Dify API key)

The local unpublished package can translate Dify request objects into
OpenAI-compatible gateway calls. This lets you run examples with only
`RESPAN_API_KEY`.

```python
from dify_client import models
from respan_exporter_dify import create_client

respan_client = create_client(
    api_key="your-respan-api-key",
    gateway_base_url="https://api.respan.ai/api",
    gateway_model="gpt-4o-mini",
)

req = models.ChatRequest(
    query="Hello!",
    user="user-123",
    response_mode=models.ResponseMode.BLOCKING,
    inputs={},
)
response = respan_client.chat_messages(req=req)
print(response.answer)
```

### Wrap an existing client (sync)

Create the official Dify client, then wrap it with `create_client`. All calls are logged to Respan.

```python
from dify_client import Client
from dify_client.models import ChatRequest, ResponseMode
from respan_exporter_dify import create_client

dify_client = Client(api_key="your-dify-api-key")
respan_client = create_client(
    client=dify_client,
    api_key="your-respan-api-key",
)

req = ChatRequest(
    query="Hello!",
    user="user-123",
    response_mode=ResponseMode.BLOCKING,
    inputs={},
)
response = respan_client.chat_messages(req=req)
```

**Sync methods:** `chat_messages`, `completion_messages`, `run_workflows`. Use the same request types as with the vanilla Dify client.

### Wrap an existing client (async)

```python
import asyncio
from dify_client import AsyncClient
from dify_client.models import ChatRequest, ResponseMode
from respan_exporter_dify import create_async_client

async def main():
    dify_client = AsyncClient(api_key="your-dify-api-key")
    respan_client = create_async_client(
        client=dify_client,
        api_key="your-respan-api-key",
    )
    req = ChatRequest(
        query="Hello!",
        user="user-123",
        response_mode=ResponseMode.BLOCKING,
        inputs={},
    )
    response = await respan_client.achat_messages(req=req)
    return response

asyncio.run(main())
```

**Async methods:** `achat_messages`, `acompletion_messages`, `arun_workflows`.

### Streaming

Use `ResponseMode.STREAMING` and iterate over the returned stream. Data is sent to Respan when the stream is fully consumed.

```python
from dify_client.models import ChatRequest, ResponseMode

req = ChatRequest(
    query="Hello!",
    user="user-123",
    response_mode=ResponseMode.STREAMING,
    inputs={},
)
for event in respan_client.chat_messages(req=req):
    print(event)
```

Async streaming: use `respan_client.achat_messages(req=req)` with `async for event in ...`.

### Trace and session IDs (`respan_params`)

Pass `respan_params` to associate logs with a trace or session in Respan (e.g. for grouping by conversation or run).

```python
from respan_sdk.respan_types import RespanParams

params = RespanParams(
    trace_unique_id="run-abc-123",
    trace_name="my-workflow",
    session_identifier="conversation-456",
    span_workflow_name="dify-chat",
)
response = respan_client.chat_messages(req=req, respan_params=params)
```

To disable logging for a single call:

```python
response = respan_client.chat_messages(
    req=req,
    respan_params=RespanParams(disable_log=True),
)
```

---

## Alternative: build the wrapper with a Dify API key

You can construct the wrapper with a Dify API key instead of an existing client instance:

```python
from respan_exporter_dify import RespanDifyClient, RespanAsyncDifyClient

# Sync
respan_client = RespanDifyClient(
    dify_api_key="your-dify-api-key",
    api_key="your-respan-api-key",
)

# Async
respan_client = RespanAsyncDifyClient(
    dify_api_key="your-dify-api-key",
    api_key="your-respan-api-key",
)
```

Then call `chat_messages`, `completion_messages`, `run_workflows` (or the `a*` async variants) as in the usage examples above.

---

## Further reading

| Resource | Description |
|----------|-------------|
| [Respan Documentation](https://docs.respan.ai) | Platform docs, observability, and cost tracking |
| [Dify Python SDK](https://github.com/langgen-ai/dify-client-python) | Official Dify client used under the hood |
| [respan-sdk](https://pypi.org/project/respan-sdk/) | Shared types (e.g. `RespanParams`) and utilities |

---

## Dev guide

For contribution and SDK development conventions (imports, kwargs, documentation standards, user-facing docs), see the **BE Conventions** in the backend boilerplates:

- **Path:** `boilerplates/keywordsai/dev_guides/BE_conventions/BE_conventions.md`
- **Highlights:** Golden Rules, Documentation (user perspective only), SDK Development (shared types and releases)
