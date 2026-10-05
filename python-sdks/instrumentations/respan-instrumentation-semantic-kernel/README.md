# respan-instrumentation-semantic-kernel

Respan instrumentation for [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel).
Tested with Semantic Kernel Python **1.36.0** and **1.44.1**, using released
Respan tracing 2.20.1 and OpenAI 3.24.0. The supported SDK range is `>=1.36,<2`.

The adapter enables Semantic Kernel's native diagnostics and normalizes kernel
functions, prompt tasks, chat/text completions, automatic function invocation,
and agent spans. It also adds OpenAI/Azure OpenAI embedding spans at the SDK's
provider-request boundary, including every vector in each batch and reported
input usage. Other embedding connectors are not instrumented by this adapter.

## Installation

```bash
pip install respan-ai respan-instrumentation-semantic-kernel
```

## Usage

```python
from respan import Respan
from respan_instrumentation_semantic_kernel import SemanticKernelInstrumentor

respan = Respan(instrumentations=[SemanticKernelInstrumentor()])
# Run your Semantic Kernel application, then flush and close it.
respan.shutdown()
```

Initialize Respan before running the kernel. Already imported agent classes are
supported. The adapter retains native parent/child spans, current completion
tool calls, tool-call IDs, and historical tool messages. Token, cached-input, and
reasoning usage come from provider responses, including reported zero counts;
agent/tool spans do not inherit model token usage. Failed model requests retain
error status and metadata without manufacturing a completion or HTTP status.

## Content and lifecycle

`SemanticKernelInstrumentor(capture_content=False)` disables prompt, completion,
function payload, agent message, and embedding input/vector capture. The global
`TRACELOOP_TRACE_CONTENT=false` switch and Respan's `ENABLE_CONTENT_TRACING_KEY`
context override also apply. A request that starts with capture disabled cannot
re-enable it while awaiting its response. Model metadata and reported usage
remain available; controlled error messages remain error metadata.

Multiple instrumentor instances share one installation on the same tracer
provider and must use the same capture setting. The last deactivation restores
owned hooks and diagnostics settings. Foreign wrappers and subsequent external
settings changes are preserved; retained adapter hooks become inactive.
OpenTelemetry instrumentation suppression is respected.

Semantic Kernel's native streaming decorators keep their current span active
across a yielded chunk. With both the bare SDK and this adapter, closing an
outer iterator from another task does not reliably restore the caller's context
or finish its nested model span. Consume streams to completion in the creating
task; the adapter does not replace the SDK's iterator or transport behavior.

## Examples

The [companion examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/semantic-kernel)
exercise real Semantic Kernel APIs with deterministic local HTTP fixtures,
including agents, tools, embedding batches, streams, privacy, and failures.
Their README includes installation against an unreleased adapter checkout.
