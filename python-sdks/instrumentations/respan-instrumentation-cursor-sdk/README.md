# Respan Cursor SDK instrumentation

Trace Cursor hook events through the Respan OpenTelemetry runtime.

Cursor sends structured JSON to configured hook commands. This package converts those hook events into canonical Respan spans and injects them into the active `respan-tracing` pipeline.

## Install

```bash
pip install respan-ai respan-instrumentation-cursor-sdk
```

## Cursor hook command

Set Respan configuration in the environment used by Cursor:

```bash
export RESPAN_API_KEY="..."
export RESPAN_BASE_URL="https://api.respan.ai/api"
```

Then configure Cursor hooks to call the packaged runner:

```json
{
  "version": 1,
  "hooks": {
    "beforeSubmitPrompt": [{ "command": "respan-cursor-hook" }],
    "afterAgentThought": [{ "command": "respan-cursor-hook" }],
    "afterShellExecution": [{ "command": "respan-cursor-hook" }],
    "afterFileEdit": [{ "command": "respan-cursor-hook" }],
    "afterMCPExecution": [{ "command": "respan-cursor-hook" }],
    "afterAgentResponse": [{ "command": "respan-cursor-hook" }],
    "stop": [{ "command": "respan-cursor-hook" }]
  }
}
```

The runner reads one Cursor hook JSON payload from stdin, stores prompt state between hook calls, emits child spans as they happen, and emits the root agent span when `afterAgentResponse` arrives.

## Programmatic usage

```python
from respan import Respan
from respan_instrumentation_cursor_sdk import CursorSDKInstrumentor

respan = Respan(
    instrumentations=[CursorSDKInstrumentor()],
    app_name="cursor-sdk",
)

processor = CursorSDKInstrumentor.create_processor()
processor.process_event(
    {
        "hook_event_name": "afterAgentThought",
        "conversation_id": "conv_123",
        "generation_id": "gen_456",
        "text": "I need to inspect the repository.",
        "duration_ms": 320,
    }
)
respan.telemetry.flush()
```

## Supported Cursor events

- `beforeSubmitPrompt`
- `afterAgentThought`
- `afterShellExecution`
- `afterFileEdit`
- `afterMCPExecution`
- `afterAgentResponse`
- `stop`

Cursor-specific raw field names stay inside this package. The emitted spans use canonical Respan attributes for log type, entity input/output, workflow name, thread identifier, trace group, and metadata.
