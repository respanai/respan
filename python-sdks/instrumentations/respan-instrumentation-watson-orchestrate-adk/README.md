# respan-instrumentation-watson-orchestrate-adk

Respan instrumentation plugin for the IBM watsonx Orchestrate ADK
(`ibm-watsonx-orchestrate`).

The package uses native patching because no mature upstream OpenTelemetry or
OpenInference Watson Orchestrate ADK instrumentor is available. It traces:

- local `PythonTool` execution as tool spans
- generated `RunClient` run submission/completion methods as agent spans
- ADK chat and watsonx.ai autodiscover client calls as chat spans when those
  client modules are installed

## Installation

```bash
pip install respan-ai respan-instrumentation-watson-orchestrate-adk ibm-watsonx-orchestrate
```

## Usage

```python
from respan import Respan
from respan_instrumentation_watson_orchestrate_adk import (
    WatsonOrchestrateADKInstrumentor,
)

respan = Respan(
    instrumentations=[WatsonOrchestrateADKInstrumentor()],
)
```

Initialize Respan before invoking Watson Orchestrate ADK tools, run clients, or
chat clients. The instrumentor does not require the ADK package at import time;
missing optional client surfaces are skipped during activation.

## Compatibility and captured behavior

Validated with `ibm-watsonx-orchestrate` **2.12.0** and **2.18.0**. The
supported range remains `>=2.12.0,<3.0.0`.

- Run submission (including files), HTTP completion polling, and WebSocket
  callbacks retain the SDK's return values and cleanup behavior. Returned
  failed/cancelled run states mark the span as an error.
- watsonx.ai, Groq, and AI Gateway autodiscover requests, agent architect chat,
  and CPE refinement capture the request model, response model, system
  instructions, current-turn tool calls, and reported usage (including zero).
- `TempusClient.run_flow` and `arun_flow` produce workflow spans. Both methods
  are synchronous SDK calls; `arun_flow` submits work for remote execution.
- Nested local tool calls keep their parent relationship. Provider run IDs
  remain in the payload; each invocation has its own span identity.

`TRACELOOP_TRACE_CONTENT=false` or Respan's runtime content-tracing context
excludes prompts, completions, function definitions, and entity input/output.
Policy and attribution are captured when the call begins. OpenTelemetry
suppression skips instrumentation. Optional clients are patched only when
installed; multiple instrumentor instances share ownership, and deactivation
preserves foreign wrappers while making retained Respan wrappers inactive.

The deterministic companion examples use the installed SDK with controlled
HTTP/WebSocket transports. They do not establish live IBM service acceptance.
