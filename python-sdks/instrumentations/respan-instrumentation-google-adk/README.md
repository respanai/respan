# respan-instrumentation-google-adk

Respan instrumentation plugin for [Google Agent Development Kit](https://adk.dev/).

This package wraps the upstream OpenInference Google ADK instrumentor and
registers a Google-ADK-specific span processor. The processor composes Respan's
generic OpenInference translation and applies ADK-only normalization in this
package, so ADK runner, agent, LLM, and tool spans are exported through the same
Respan OTEL pipeline as the rest of the Python SDK.

## Installation

```bash
pip install respan-ai respan-instrumentation-google-adk
```

Install ADK's LiteLLM extension if you want to route models through an
OpenAI-compatible gateway:

```bash
pip install "google-adk[extensions]"
```

## Usage

```python
from respan import Respan
from respan_instrumentation_google_adk import GoogleADKInstrumentor

respan = Respan(instrumentations=[GoogleADKInstrumentor()])
```

Any Google ADK runs started after initialization are traced and exported to
Respan.

## Compatibility

Python 3.11–3.13 is supported. Validation uses released core dependencies with
only this adapter installed from the local checkout.

| Google ADK | OpenInference Google ADK | Coverage |
| --- | --- | --- |
| 1.5.0 | 0.1.12 | Legacy runner, agents, tools, streaming and cleanup |
| 2.11.0 | 1.0.2 | Common APIs, Workflow nodes, confirmation, abort and ModelConsultTool |

OpenInference 1.x requires ADK 2.10 or newer. For a legacy ADK application,
install a compatible pair in one resolver operation:

```bash
pip install respan-instrumentation-google-adk "google-adk==1.5.0" "openinference-instrumentation-google-adk==0.1.12"
```

Supported paths include sync/async runners, session state and callbacks,
sequential/parallel/custom agents, sync/async tools, SSE events, errors and
suppression. The adapter preserves current-turn tool calls, tool-result history,
execution IDs and actual provider usage. An error or response without reported
usage does not gain fabricated usage fields.

ADK 2.11 Workflow nodes use the SDK's native node tracing scope, enriched as
workflow/task spans. Tool confirmation records the paused node without claiming
that its tool executed; the tool span starts after confirmation permits execution.
ModelConsultTool advisor attempts emit model spans from ADK's actual request,
responses and elapsed time, including adviser errors. Advisor usage is taken
from provider response metadata, not ADK's synthesized usage summary.

Runner/agent iterators advance on demand in their own task context. Yielding an
event does not leave the SDK span active in the caller, and close/cancellation
finishes the iterator in its original context. The legacy bridge also closes
suspended ADK 1.5 model iterators.

Multiple adapter instances with matching settings share one registration.
A conflicting configuration raises `ValueError`, including content/privacy changes.
The last owner restores patches and removes the processor from its original
provider. Later wrappers remain installed; retained inner method guards bypass
instrumentation after deactivation. An independently activated upstream ADK
instrumentor remains externally owned and is not replaced.

Content controls honor OpenInference `TraceConfig`, `TRACELOOP_TRACE_CONTENT`
and Respan's content-disable context. ModelConsult captures privacy and parent
context at call start, so changing policy during the model await cannot reveal
content that started hidden. Token usage remains observable with content disabled.

The companion [Python examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/google-adk)
run deterministic models through the actual SDK. They verify tracing behavior;
they do not verify live Google credentials, audio/live connections, hosted
services or the availability of specific model features.
