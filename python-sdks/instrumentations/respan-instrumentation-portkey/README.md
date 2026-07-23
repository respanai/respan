# respan-instrumentation-portkey

Respan instrumentation plugin for Portkey. Wraps OpenInference's Portkey instrumentor, translates spans into the Respan tracing shape, and strips off-contract helper aliases before export.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-portkey
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key for trace export. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |
| `PORTKEY_API_KEY` | Yes | Your Portkey API key for Portkey Gateway calls. |
| `PORTKEY_PROVIDER` or `PORTKEY_CONFIG` | Yes | Portkey provider slug or config slug. |

## Quickstart

```python
import os
from dotenv import load_dotenv

load_dotenv()

from portkey_ai import Portkey
from respan import Respan, workflow
from respan_instrumentation_portkey import PortkeyInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
    instrumentations=[PortkeyInstrumentor()],
)

client = Portkey(
    api_key=os.environ["PORTKEY_API_KEY"],
    provider=os.getenv("PORTKEY_PROVIDER"),
    config=os.getenv("PORTKEY_CONFIG"),
)


@workflow(name="portkey_chat_completion")
def run_chat() -> str:
    response = client.chat.completions.create(
        model=os.getenv("PORTKEY_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": "Say hello from Portkey."}],
    )
    return response.choices[0].message.content or ""


try:
    print(run_chat())
finally:
    respan.flush()
    respan.shutdown()
```

## Further Reading

See the [Respan example projects](https://github.com/respanai/respan-example-projects) for runnable scripts.
