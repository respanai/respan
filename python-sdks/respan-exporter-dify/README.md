# respan-exporter-dify

**[respan.ai](https://respan.ai)** | **[Documentation](https://docs.respan.ai)** | **[PyPI](https://pypi.org/project/respan-exporter-dify/)**

Respan exporter for the Dify Python SDK (`dify-client-python`). Wrap the official Dify client to automatically send chat, completion, and workflow call logs to Respan for observability and cost tracking.

## Installation

```bash
pip install respan-exporter-dify
```

## Environment variables

Configure Respan ingest (optional if you pass credentials in code):

| Variable           | Description                          |
|--------------------|--------------------------------------|
| `RESPAN_API_KEY`   | Your Respan API key                  |
| `RESPAN_ENDPOINT`  | Respan ingest endpoint (optional; has a default) |

## Usage

### Sync client with a real API call

```python
from dify_client import Client
from dify_client.models import ChatMessageRequest, ResponseMode
from respan_exporter_dify import create_client

# Create the official Dify client
dify_client = Client(api_key="your-dify-api-key")

# Wrap it so calls are logged to Respan
respan_client = create_client(
    client=dify_client,
    api_key="your-respan-api-key",
)

# Build a request (blocking = non-streaming)
req = ChatMessageRequest(
    query="Hello!",
    user="user-123",
    response_mode=ResponseMode.BLOCKING,
    inputs={},
)

# Call Dify as usual — requests are automatically logged to Respan
response = respan_client.chat_messages(req=req)
```

Available sync methods: `chat_messages`, `completion_messages`, `run_workflows`. Use the same request types as with the vanilla Dify client.

### Async client

```python
import asyncio
from dify_client import AsyncClient
from dify_client.models import ChatMessageRequest, ResponseMode
from respan_exporter_dify import create_async_client

async def main():
    dify_client = AsyncClient(api_key="your-dify-api-key")
    respan_client = create_async_client(
        client=dify_client,
        api_key="your-respan-api-key",
    )

    req = ChatMessageRequest(
        query="Hello!",
        user="user-123",
        response_mode=ResponseMode.BLOCKING,
        inputs={},
    )

    response = await respan_client.achat_messages(req=req)
    return response

asyncio.run(main())
```

Available async methods: `achat_messages`, `acompletion_messages`, `arun_workflows`.

### Streaming

Use `ResponseMode.STREAMING` and iterate over the returned stream. Export runs when the stream is consumed (in the generator’s `finally` block).

```python
from dify_client.models import ChatMessageRequest, ResponseMode

req = ChatMessageRequest(
    query="Hello!",
    user="user-123",
    response_mode=ResponseMode.STREAMING,
    inputs={},
)

for event in respan_client.chat_messages(req=req):
    print(event)
```

Async streaming: use `respan_client.achat_messages(req=req)` and `async for event in ...`.

### Custom trace and session IDs with `respan_params`

Pass `respan_params` to tie logs to a trace or session in Respan (e.g. for grouping by conversation or run).

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
response = respan_client.chat_messages(req=req, respan_params=RespanParams(disable_log=True))
```

## Alternative: construct the wrapper directly

You can also use the client classes and pass a Dify API key instead of a client instance:

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

Then call `chat_messages`, `completion_messages`, `run_workflows` (or the `a*` async variants) as above.
