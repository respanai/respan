# Respan Exa instrumentation (Python)

Native OpenTelemetry instrumentation for the official [`exa-py`](https://pypi.org/project/exa-py/) SDK.

Requires Python 3.11 through 3.13 and `exa-py>=2.20.0,<3.0.0`.

The package traces the stable 2.20 API surface that carries AI work:

- `Exa` and `AsyncExa` search, contents, answer, and streaming calls
- deprecated search/similarity compatibility methods still shipped by Exa 2.x
- Agent run create, stream, wait, poll, stop, and lifecycle operations
- the legacy Research client for compatibility (Exa recommends deep search for new research flows)
- executable provider-neutral, OpenAI, and Anthropic search/contents helpers through the core SDK methods they invoke

Websets, Search Monitor CRUD, and beta Agent Monitor operations are
intentionally not patched in the first release. They are long-lived
control-plane APIs rather than in-process SDK operations.

## Install

```bash
pip install respan-ai respan-instrumentation-exa exa-py
```

`exa-py>=2.20.0,<3.0.0` is the declared and tested range. Python 2.20 is required because it is the first stable release with the `get_contents` tool helpers.

## Use

```python
from exa_py import Exa
from respan import Respan
from respan_instrumentation_exa import ExaInstrumentor

respan = Respan(
    api_key="respan-key",
    instrumentations=[ExaInstrumentor()],
)

exa = Exa(api_key="exa-key")
result = exa.search(
    "recent advances in retrieval",
    type="auto",
    contents={"highlights": True},
)
respan.shutdown()
```

The integration is explicit-only. Exa is a search/tool/agent SDK and can overlap with OpenAI or Anthropic instrumentation when its tool adapters are used, so it is not direct-LLM auto-instrumentation.

Set `capture_content=False` on `ExaInstrumentor`, or set `TRACELOOP_TRACE_CONTENT=false`, to omit query, result, prompt, completion, and stream payloads while retaining operation/status metadata. API keys and authorization-like fields are always redacted.

Entity names are provider-neutral: tools use names such as `search` and `get_contents`, answers use `answer`, Agent runs use `run`, and legacy Research uses `research`. Native SDK operation names, stream state, result counts, request IDs, cost, citations, and legacy markers are stored only in the canonical `respan.metadata` JSON object. Answer spans set `gen_ai.request.model` only when the SDK request or response supplies a model; otherwise semantic naming resolves to bare `llm`.

Streaming spans end when the iterator is exhausted, fails, or is explicitly closed. Applications that abandon a stream without closing it cannot produce an exact completion timestamp; use the stream as a context manager where supported or close it explicitly.
