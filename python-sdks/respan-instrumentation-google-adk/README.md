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

## Architecture

Google ADK produces standard OpenTelemetry spans. `GoogleAdkInstrumentor` is a no-op marker class that satisfies the Respan `Instrumentation` protocol. ADK spans flow through the standard OTel pipeline (`RespanSpanProcessor` -> `BatchSpanProcessor` -> `RespanSpanExporter`) without any custom interception or conversion.
