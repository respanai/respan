# @respan/instrumentation-openai-agents

Export OpenAI Agents SDK traces and spans (agents, model calls, tools, handoffs,
guardrails) through the Respan tracing pipeline.

## Sessions

When you set no session id, the trace's `groupId` (`withTrace(name, fn, { groupId })`
or `RunConfig.groupId`) is sent as the Respan session id
(`respan.sessions.session_identifier`) on every span of that trace, so every turn
of one conversation lands in one session with no Respan-specific code.

A session id already on the span always wins. An empty `groupId` or one longer
than 255 characters is not sent; Respan then detects the session on the server.
Unchanged: the trace group (`respan.trace.trace_group_identifier`) is still sent,
and no other span changes.
