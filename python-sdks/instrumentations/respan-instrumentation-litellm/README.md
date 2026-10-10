# Respan LiteLLM instrumentation

This plugin observes LiteLLM's native `CustomLogger` events and public
`completion`, `acompletion`, `embedding`, `aembedding`, `responses`, and
`aresponses` calls. It starts real sampled OpenTelemetry spans while preserving
SDK responses, errors, chunks and caller context. It supports LiteLLM 1.80.10
through the tested current stable 1.104.0.

```python
import os

import litellm
from respan import Respan
from respan_instrumentation_litellm import LiteLLMInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[LiteLLMInstrumentor()],
)
try:
    response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.choices[0].message.content)
finally:
    respan.shutdown()
```

`include_content=False` hides messages, tools, vectors and arbitrary metadata.
The Respan content environment/context settings and LiteLLM message redaction
flags are additional capture bounds. A call that starts private stays private;
an observed later veto removes captured payloads, including delayed streams
whose parent finishes private. Counts and actual model/provider fields remain.
General instrumentation and language-model suppression and sampler decisions
prevent content serialization.

The plugin retains complete tool schemas, current/history call IDs, single JSON
argument strings, tool-result message bodies and embedding vectors.
Complete message/choice arrays remain in entity I/O; indexed prompt/completion
projections are limited to 8 each so the default OpenTelemetry attribute budget
retains identity, model/provider, usage and run metadata. Counts are
validated actual provider counters; missing totals are not synthesized. Scoped
OpenAI raw-response/SSE observations prevent DTO coercion or LiteLLM estimation
from becoming source usage. Other provider paths use their native raw callback
usage when available; provider-specific counters outside the mapped fields are
not inferred. HTTP status comes from retained OpenAI/httpx provider error
responses. Native exceptions remain unchanged and are never completion text.

Shared instrumentors use one callback and one set of owned observers. The last
owner removes its callbacks from LiteLLM's native callback lists and restores
only hooks it still owns. Incompatible shared configurations raise `ValueError`.
`tracer_provider` can be supplied for an existing OpenTelemetry pipeline.
The exported `RespanLiteLLMCallback` remains available for explicit native
callback registration; use the instrumentor for deterministic public-call and
stream lifecycle observation.

Native stream interfaces are preserved where the released SDK provides them.
LiteLLM 1.80.10 does not provide `CustomStreamWrapper.aclose`; the plugin does
not add that API. Its Responses path also needs the upstream optional FastAPI
and orjson dependencies. Cancellation/errors and partial observed output are
retained without creating empty completions for unadvanced streams.

Controlled examples cover all six public APIs, tool/history payloads, full
vectors, native errors, streaming and privacy. They do not require a provider
account by default. See the paired examples' README for explicit live mode and
its limits. LiteLLM proxy/router/MCP/audio/image/rerank endpoints are not
independently validated by this package's bounded fixture set.
