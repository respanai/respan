# respan-exporter-dify

Send Dify chat, completion, and workflow calls to Respan for tracing, usage visibility, and debugging.

## Configuration

### 1. Install

```bash
pip install respan-exporter-dify
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. It authenticates both gateway calls and tracing export. |
| `RESPAN_BASE_URL` | No | Optional Respan API base URL override. Most users can leave it unset. |

Only these Respan environment variables are needed for the quickstart. Pass `gateway_model`, `endpoint`, or `dify_api_key` in code when you need a different runtime setup.

## Quickstart

### 3. Run Script

```python
import os

from dify_client.models import ChatRequest, ResponseMode
from respan_exporter_dify import create_client, flush_pending_exports

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

client = create_client(
    api_key=respan_api_key,
    gateway_base_url=respan_base_url,
    gateway_model="gpt-4o-mini",
)

request = ChatRequest(
    query="What is 2+2?",
    user="user-123",
    response_mode=ResponseMode.BLOCKING,
    inputs={},
)

response = client.chat_messages(req=request)
print(response.answer)
flush_pending_exports(timeout=10)
```

The same wrapper also supports:

- wrapping an existing Dify `Client` or `AsyncClient`
- `chat_messages`, `completion_messages`, and `run_workflows`
- streaming via `ResponseMode.STREAMING`
- extra trace fields through `respan_params=RespanLogParams(...)`

### 4. View Dashboard

Open the Respan dashboard and inspect the latest Dify span created by the script. The exported record includes the request input, output, model, usage, and trace metadata.

## Further Reading

- [Dify examples in `respan-example-projects`](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/dify)
- [Hello world example](https://github.com/respanai/respan-example-projects/blob/main/python/tracing/dify/hello_world.py)
- [Gateway example](https://github.com/respanai/respan-example-projects/blob/main/python/tracing/dify/gateway.py)
- [Tracing example](https://github.com/respanai/respan-example-projects/blob/main/python/tracing/dify/tracing.py)
- [Respan params example](https://github.com/respanai/respan-example-projects/blob/main/python/tracing/dify/respan_params.py)
- [Streaming example](https://github.com/respanai/respan-example-projects/blob/main/python/tracing/dify/streaming.py)
- [Respan documentation](https://docs.respan.ai)
- [Dify Python SDK](https://github.com/langgen-ai/dify-client-python)
