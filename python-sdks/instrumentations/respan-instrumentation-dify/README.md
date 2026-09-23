# Respan Dify instrumentation

Trace the official `dify-client` Python package with Respan.

Requires Python 3.11 through 3.13 and `dify-client>=0.1.10,<0.2.0`.

This package patches Dify's Service API client methods and emits canonical
Respan spans for chat, completion, workflow, file, feedback, application, and
conversation requests. Streaming responses are traced when the returned stream
is consumed, closed early, or raises. It supports the released sync-only
`dify-client` 0.1.10 package and the refreshed 0.1.12 source API, including its
sync and async Chat, Completion, Workflow, Knowledge Base, and Workspace
clients.

## Install

```bash
pip install respan-ai respan-instrumentation-dify dify-client
```

## Usage

```python
import os

from dify_client import ChatClient
from respan import Respan
from respan_instrumentation_dify import DifyInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[DifyInstrumentor()],
)

client = ChatClient(os.environ["DIFY_CHAT_API_KEY"])
response = client.create_chat_message(
    inputs={},
    query="Reply with one concise sentence about tracing.",
    user="respan-example-user",
    response_mode="blocking",
)
response.raise_for_status()
print(response.json()["answer"])

respan.flush()
respan.shutdown()
```

### Async clients

When the installed Dify SDK exposes its async clients, the same instrumentor
patches them automatically:

```python
from dify_client import AsyncChatClient

async with AsyncChatClient(
    os.environ["DIFY_CHAT_API_KEY"],
    base_url=os.getenv("DIFY_BASE_URL", "https://api.dify.ai/v1"),
) as client:
    response = await client.create_chat_message(
        inputs={},
        query="Stream one concise sentence about tracing.",
        user="respan-example-user",
        response_mode="streaming",
    )
    async for line in response.aiter_lines():
        print(line)
```

The span stays open until a streaming response is exhausted, explicitly
closed, leaves its context manager, or fails. Blocking responses emit when the
SDK request resolves. `include_content=False` keeps request/response bodies and
prompt/completion content off spans while retaining operation, status, and
usage attributes. Credential-like fields are always recursively redacted,
including Workspace credential-validation requests and echoed responses. When
multiple instrumentor instances overlap, the first
active instance's content policy remains in effect until the last instance is
deactivated; conflicting instances log a warning instead of changing it.
Deactivation restores only wrappers still owned by this instrumentor, so a
later patch from another library is left intact.

Use `Respan.propagate_attributes(...)` to attach customer, thread, trace group,
metadata, or prompt context to Dify spans.
