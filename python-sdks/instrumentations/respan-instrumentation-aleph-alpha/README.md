# respan-instrumentation-aleph-alpha

Respan instrumentation plugin for the official Aleph Alpha Python SDK.

The package patches `aleph_alpha_client.Client` and `AsyncClient` model calls
and emits canonical Respan spans through the active OTEL pipeline. It captures
sync and async chat, sync and async completion, async streaming chat and
completion, embeddings, semantic embeddings, instructable embeddings,
evaluation, explanation, prompt and completion content, token usage, tool
definitions, and model tool calls.

## Installation

```bash
pip install respan-ai respan-instrumentation-aleph-alpha aleph-alpha-client
```

## Usage

```python
import os

from aleph_alpha_client import ChatRequest, Client, Message
from aleph_alpha_client.chat import Role
from respan import Respan
from respan_instrumentation_aleph_alpha import AlephAlphaInstrumentor

respan = Respan(instrumentations=[AlephAlphaInstrumentor()])
client = Client(token=os.environ["ALEPH_ALPHA_API_KEY"])

response = client.chat(
    request=ChatRequest(
        model="llama-3.1-8b-instruct",
        messages=[Message(role=Role.User, content="Say hello in one sentence.")],
    ),
    model="llama-3.1-8b-instruct",
)
print(response.message.content)
respan.flush()
```
