# Respan Dify instrumentation

Trace the `dify-client` Python package with Respan.

This package patches Dify's Service API client methods and emits canonical
Respan spans for chat, completion, workflow, file, feedback, application, and
conversation requests. Streaming responses are traced when the returned stream
is consumed.

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

Use `Respan.propagate_attributes(...)` to attach customer, thread, trace group,
metadata, or prompt context to Dify spans.
