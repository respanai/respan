# Respan Helicone instrumentation

Trace calls made through Helicone's Python manual logger with Respan. This
package instruments the published `helicone-helpers` `HeliconeManualLogger`
surface without replacing or bypassing Helicone's own logging behavior.

The instrumentation covers:

- `HeliconeManualLogger.log_request()` and direct `send_log()` calls
- `HeliconeLogBuilder`, including streamed chunks and error logs
- OpenAI-, Anthropic-, and custom-model request/response payloads
- Helicone `_type=tool`, `_type=vector_db`, and `_type=data` custom logs
- sync manual logging and the builder's async `send_log()` lifecycle

## Install

```bash
pip install respan-ai respan-instrumentation-helicone "helicone-helpers~=1.2.1"
```

## Usage

```python
import os

from helicone_helpers import HeliconeManualLogger
from respan import Respan, workflow
from respan_instrumentation_helicone import HeliconeInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[HeliconeInstrumentor()],
)
helicone = HeliconeManualLogger(api_key=os.environ["HELICONE_API_KEY"])


@workflow(name="helicone_manual_log")
def run(prompt: str) -> str:
    request = {
        "model": "custom-chat-model",
        "messages": [{"role": "user", "content": prompt}],
    }

    def operation(recorder):
        response = {
            "model": "custom-chat-model",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
        }
        recorder.append_results(response)
        return "Hello!"

    return helicone.log_request(request, operation, provider="openai")


try:
    print(run("Say hello."))
finally:
    respan.flush()
    respan.shutdown()
```

`HeliconeInstrumentor(capture_content=False)` keeps span identity, model,
provider, timing, usage, and errors while omitting request and response content.
Authentication and Helicone transport headers are never copied into spans.
Safe constructor/request correlation headers are normalized into canonical
customer/session fields and one JSON `respan.metadata` attribute.

## Activation and overlap policy

Helicone manual logging is explicit instrumentation. Activate
`HeliconeInstrumentor()` only in applications that use `HeliconeManualLogger`;
it is not a direct-provider auto-instrumentor. The package patches Helicone's
logger and builder methods only—it does not patch OpenAI, Anthropic, or another
provider SDK.

Do not combine this package with instrumentation for the provider operation
that the same manual log describes unless you intentionally want both records.
For example, wrapping an OpenAI call in `HeliconeManualLogger.log_request()`
while also activating `OpenAIInstrumentor()` produces one provider span and one
Helicone manual-log span for the same logical model call.

The instrumentor is reference-counted and restores the exact SDK methods it
patched when the final owner deactivates. It also exposes `instrument()` and
`uninstrument()` aliases for compatibility with OpenTelemetry-style lifecycle
code.

See the Helicone example set in `respan-example-projects/python/tracing/helicone`
for deterministic sync, async-builder, streaming, error, tool, vector database,
and custom data coverage.
