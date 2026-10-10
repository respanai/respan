# Respan Cursor instrumentation

Trace native Python Cursor SDK runs and Cursor version-one hook events through the Respan OpenTelemetry pipeline.

| Surface | Tested current | Tested minimum |
| --- | --- | --- |
| Native `cursor-sdk` | 1.0.35 | 1.0.24 |
| Hook JSON/configuration | Current [official hook reference](https://cursor.com/docs/hooks), configuration version 1 | Original seven-event layout also accepted |
| `respan-tracing` | 2.20.1 | 2.17.0 |
| `respan-sdk` | 2.7.6 | 2.6.26 |
| OTel SDK / standard conventions | 1.45.0 / 0.66b0 | 1.38.0 / 0.59b0 |
| AI conventions | 0.5.1 | 0.4.1 |

The previously declared `respan-sdk` 2.6.1 floor cannot import the required `constants.span_attributes` module. The repaired floor is verified against a released wheel. The native SDK is optional: hook-only installation does not install or launch the Cursor bridge.

## Native Python SDK

```bash
pip install 'respan-instrumentation-cursor-sdk[native]'
```

Initialize tracing before activating the instrumentor:

```python
from cursor_sdk import CursorClient, LocalAgentOptions
from respan_instrumentation_cursor_sdk import CursorSDKInstrumentor
from respan_tracing import RespanTelemetry

telemetry = RespanTelemetry(is_auto_instrument=False)
instrumentor = CursorSDKInstrumentor()
instrumentor.activate()
try:
    with CursorClient.connect(bridge_url, auth_token=bridge_token) as client:
        with client.agents.create(
            api_key=cursor_api_key,
            model="composer-2.5",
            local=LocalAgentOptions(cwd="."),
        ) as agent:
            run = agent.send("Summarize the repository")
            print(run.wait().result)
finally:
    telemetry.flush()
    instrumentor.deactivate()
```

Sync/async `Agent.send` returns its original `Run`/`AsyncRun`. The adapter observes already-consumed Connect frames and the original SDK callback dispatch; it does not replay callbacks, consume extra stream items, launch a bridge, close a user client, or fetch usage automatically. Native iteration, `stream`, `wait`, `text`, typed values and raised errors retain SDK behavior.

Agent spans finish at observed terminal results. Actual legacy SDK tool messages and modern completed tool steps produce correlated tool spans with complete arguments/results and source call IDs; matching duplicate messages/steps are one invocation. Explicit `agent.get_usage()` produces a task containing the actual returned billing payload on releases that expose it (absent in minimum 1.0.24). Agent/task spans do not invent LLM token fields, cost, HTTP status or tool execution from declarations. Model selection, per-send options and MCP configuration remain native; captured credentials/environment/header values are redacted.

Keep the adapter active while runs finish. An abandoned run or final deactivation discards retained content and finishes its telemetry without cancelling the native run or closing its client. Only owned hooks are restored; shared owners require the same provider and `capture_content` setting.

## Cursor hook command

```bash
pip install respan-instrumentation-cursor-sdk
export RESPAN_API_KEY="..."
export RESPAN_BASE_URL="https://api.respan.ai/api"
```

For example, `.cursor/hooks.json` can configure:

```json
{
  "version": 1,
  "hooks": {
    "beforeSubmitPrompt": [{ "command": "respan-cursor-hook" }],
    "afterAgentThought": [{ "command": "respan-cursor-hook" }],
    "postToolUse": [{ "command": "respan-cursor-hook" }],
    "postToolUseFailure": [{ "command": "respan-cursor-hook" }],
    "subagentStart": [{ "command": "respan-cursor-hook" }],
    "subagentStop": [{ "command": "respan-cursor-hook" }],
    "afterAgentResponse": [{ "command": "respan-cursor-hook" }],
    "preCompact": [{ "command": "respan-cursor-hook" }],
    "stop": [{ "command": "respan-cursor-hook" }]
  }
}
```

The runner reads one JSON object from stdin. Permission hooks return neutral `allow`/`continue` JSON, including when telemetry is disabled or fails; logging stays on stderr. The command observes events and does not enforce permissions, execute tools, add follow-up prompts or change inputs.

The processor accepts the original events plus `sessionStart`, `sessionEnd`, `preToolUse`, `postToolUse`, `postToolUseFailure`, `subagentStart`, `subagentStop`, `beforeShellExecution`, `beforeMCPExecution`, `beforeReadFile`, `beforeTabFileRead`, `afterTabFileEdit`, `preCompact` and `workspaceOpen`. Before-hooks are observations, not executed-tool spans. Session/Tab/workspace observations are independent roots. Generic completion hooks and dedicated `afterShellExecution`/`afterMCPExecution`/`afterFileEdit` hooks can describe the same work: configure one completion family, since dedicated events may omit a common invocation ID.

`beforeSubmitPrompt` starts a generation. `afterAgentResponse` records an assistant message without ending the loop; `stop` ends the agent generation using its actual status. Successful repeated messages are retained in the root output. Error/cancelled stops set OTel ERROR without fabricated HTTP499/500 or error-as-output. Completed generation tombstones suppress repeated terminal events. Subagent start/stop correlation uses `subagent_id` when available, falling back to the supplied type; overlapping same-type subagents require IDs for exact matching.

Generation state uses conversation plus generation identity, owner-only atomic files and process/thread locks. Completed state contains no prompt/response bodies. `RESPAN_CURSOR_STATE_FILE` overrides the default `~/.cursor/state/respan_cursor_sdk_state.json`. Closing a processor discards pending content; use its context manager for standalone processing:

```python
from respan_instrumentation_cursor_sdk import CursorHookProcessor

with CursorHookProcessor(state_path="/tmp/cursor-fixture-state.json") as processor:
    result = processor.process_event(hook_json)
```

## Content and validation limits

`capture_content=False`, `TRACELOOP_TRACE_CONTENT=false`, `RESPAN_TRACE_CONTENT=false`, and the tracing content context opt-out disable capture. Initial/observed ancestor/end vetoes remain irreversible for an in-flight run/generation, including delayed completion after a private parent ends. Sampling/suppression gates precede owned payload serialization. Full JSON arguments, current IDs, vectors, edits and schemas are retained when capture is allowed; unknown objects are not stringified or traversed.

Activate before starting local parent spans. A recording parent that started before the adapter's privacy policy is unobserved, so its child runs and hook generations retain structural telemetry with content hidden.

Controlled fixtures validate released native SDK HTTP/Connect parsing, hook replay and CLI JSON. They do not validate a real bridge launch, live paid Cursor agent/cloud execution, account permissions, local store/custom-tool registration, conversation/artifact/download/observe/reload/administrative RPC tracing, or changes to Cursor's permission logic. Hook payloads are observations rather than provider completion events, so absent provider usage is not estimated by this adapter. Backend projections and platform read availability remain separate acceptance gates.
