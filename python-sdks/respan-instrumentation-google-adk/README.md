# respan-instrumentation-google-adk

Respan instrumentation for [Google ADK](https://github.com/google/adk-python) traces.

## Installation

```bash
pip install respan-instrumentation-google-adk
```

## Usage

```python
from respan import Respan
from respan_instrumentation_google_adk import GoogleAdkInstrumentor

respan = Respan(instrumentations=[GoogleAdkInstrumentor(environment="development")])
```

## Environment Variables

| Variable | Description |
|---|---|
| `RESPAN_API_KEY` | Your Respan API key |
| `GOOGLE_API_KEY` | Your Google Gemini API key |
| `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS` | Set to `true` for input/output capture |

## ADK Span Mapping

| ADK Span Name | Respan Log Type |
|---|---|
| `invocation` | `workflow` |
| `agent_run` | `agent` |
| `call_llm` | `generation` |
| `execute_tool` | `tool` |

## Architecture

`GoogleAdkInstrumentor` implements the Respan Instrumentation protocol (`name`, `activate(exporter)`, `deactivate()`). It patches OpenTelemetry span processors to intercept ADK spans, converts them to Respan payload format via `AdkSpanConverter`, and exports them through the provided exporter. Transport (HTTP, auth, retries) is handled by the exporter, not this package.

Spans for a trace are buffered until the ADK root span (`invocation`) ends so the converter can propagate fields across the full trace. On `deactivate()`, any pending buffer is flushed. If a trace grows beyond `trace_buffer_max_spans` (default `8192`, or `None` to disable the cap), that trace is flushed early with a warning.
