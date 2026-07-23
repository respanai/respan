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
