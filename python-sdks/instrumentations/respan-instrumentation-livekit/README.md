# Respan LiveKit instrumentation

Translate LiveKit Agents' native tracing into Respan spans while preserving the
SDK's stream, tool execution and Session APIs. This package supports
`livekit-agents>=1.6.0,<2`; controlled native SDK validation covers 1.6.0 and 1.8.4.
The optional OpenAI companion was tested at matching versions 1.6.0 and 1.8.4,
with its released OpenAI dependency bounds (the current companion requires<3).

```bash
pip install respan-tracing respan-instrumentation-livekit livekit-plugins-openai
```

```python
import os
from respan_tracing import RespanTelemetry
from respan_instrumentation_livekit import LiveKitInstrumentor

telemetry = RespanTelemetry(
    api_key=os.environ["RESPAN_API_KEY"],
    app_name="livekit-example",
    is_auto_instrument=False,
)
owner = LiveKitInstrumentor()
owner.activate()
# Run the normal LiveKit LLMStream or AgentSession APIs here.
# Close the SDK's streams/session before releasing the owner.
owner.deactivate()
```

Native `LLMStream` request spans become chat spans with model/provider fields,
messages, current tool calls, complete available tool schemas and actual usage.
OpenAI SSE usage is observed before DTO coercion. Missing, boolean, negative or
otherwise unproven token counts are omitted; complete actual input/output counts
can produce a total. Counts observed before a failed stream remain available.

`execute_function_call` emits one tool span with the actual result, invocation ID
and declaring request parent when its original call object is available. Existing
Session `function_tool` spans are translated in place, including native error
flags, without an additional tool span. Session's native string rendering of tool
results is retained; utility calls retain structured raw results. Complete dense
and sparse vectors are retained in known tool values. Errors do not invent output,
usage or HTTP status; an actual SDK HTTP status is preserved when provided.
Other native SDK spans retain their graph and use task common fields.

Content capture has an initial upper bound and an irreversible veto at later
checkpoints. `LiveKitInstrumentor(capture_content=False)`, Respan content context,
`TRACELOOP_TRACE_CONTENT=false`, OTel GenAI message-content opt-out and available
native LiveKit redaction/capture controls are respected. Private spans omit
messages, arguments, outputs, schemas and exception text while keeping structural
fields and valid actual usage. A parent veto applies to delayed descendants. A local recording parent created
before activation has an unobserved initial policy: its child content stays
hidden conservatively, even if capture is enabled later. Activate before creating
local parents to enroll their initial bounds. Remote/non-recording parents retain
context policy semantics.
Sampling and OTel suppression skip owned payload serialization. Credential values
are redacted while structural schema property names and benign URL parameters
remain intact.

`LiveKitInstrumentor(capture_audio=True)` (off by default) keeps each turn's
audio as two clips the log panel plays: the agent's speech on `tts_node` (log
type `speech`) and the user's speech on a `user_speech` span under the turn
(log type `transcription`, with the transcript). Clips are 16-bit WAV, sent as
data URLs that Respan stores as log media, and are capped at 60 s each (a longer
clip is cut there and marked `respan.metadata.voice.audio_truncated`). Audio is
content: it needs `capture_content` and follows the same privacy vetoes. Frames
are copied as they pass, never held; buffers are released when their span or
call ends. The hook points are LiveKit internals: when they are missing, audio
capture is off (a debug log) and the agent runs unchanged.

Owners with the same provider/settings share hooks; incompatible settings are
rejected. Teardown restores only still-owned hooks/provider fields and preserves
foreign replacements. In-flight owned streams finish under the retained privacy
policy; reactivate after those streams close. Native values, exceptions, caller
context and iterator protocols are preserved.

Declared floors are `respan-tracing>=2.17.0`, `respan-sdk>=2.7.0` (first tested
release with the required constants module), and AI semantic conventions>=0.5.1.
Native LiveKit supplies the compatible OTel bounds. No unpublished companion
package is required.

Coverage is bounded to native LLM streams, function utility execution and
room-free Session tool tracing. Live voice rooms, realtime audio/video, remote
sessions and every provider plugin are not validated by this suite. The paired
Python examples use controlled SDK/HTTP fixtures by default, with explicit export
and optional live OpenAI modes. Local canonical wire validation and stored trace
projections are separate checks; downstream projection limitations do not change
this package's span contract.
