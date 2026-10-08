# Respan instrumentation for AgentOps

The adapter routes AgentOps' native decorator and manual trace spans into the
active OTel provider. It normalizes workflow/session, agent, task/operation,
tool, guardrail, chat/text and embedding fields for Respan. Native values,
exceptions and SDK iterators remain owned by AgentOps.

```bash
pip install respan-ai respan-instrumentation-agentops
```

```python
from agentops import task, trace
from respan import Respan
from respan_instrumentation_agentops import AgentOpsInstrumentor

respan = Respan(instrumentations=[AgentOpsInstrumentor()])


@task(name="prepare")
def prepare(value):
    return value.upper()


@trace(name="workflow")
def workflow(value):
    return prepare(value)


print(workflow("hello"))
respan.shutdown()
```

Set `RESPAN_API_KEY` for export and optionally `RESPAN_BASE_URL` (default
`https://api.respan.ai/api`). Activate a tracing provider before this instrumentor.
The adapter enables an uninitialized native AgentOps tracing core without calling
`agentops.init()` or creating an AgentOps exporter. An application-owned AgentOps
core/exporter is preserved.

## Released SDK coverage

AgentOps **0.4.21** is both the current stable release and supported minimum
(`>=0.4.21,<0.5`). Tests use that released SDK with current Respan dependencies
and exact floors: tracing 2.17.0, SDK 2.7.6, OTel API/SDK 1.38.0, OTel semantic
conventions 0.59b0, and AI semantic conventions 0.5.1. Python 3.12 is exercised.

Function decorators support sync/async execution, source input/output, nested
parent context, and actual exceptions. Public `start_trace`, `end_trace`, and
`update_trace_metadata` are exercised. Default trace metadata is mapped into
`respan.metadata.agentops`; guardrail input/output specifications are preserved.
An explicit native error end state produces OTel ERROR, with no invented HTTP
status or error output. Ambiguous states such as Unknown/Indeterminate remain
ambiguous.

To use native AgentOps OpenAI instrumentation, install OpenAI and configure its
`OpenaiInstrumentor` **after** this adapter is active. The adapter does not enable
provider/framework auto-instrumentation. Controlled fixtures use released
OpenAI 2.44.0 for sync/async HTTP, SSE, tools and embeddings. They validate actual
response usage, cache/reasoning details, current calls separately from prompt
history, schemas and full vectors. Model declarations stay on the model span;
they do not create synthetic tool executions. Actual decorated tools create tool
spans. Call IDs are retained when an actual source supplies them, including
canonical IDs explicitly attached by application tool code. No IDs are inferred.

The adapter retains SDK-native spans and iterator implementations. AgentOps
0.4.21's sync generator helper attaches context eagerly, leaves context attached,
consumes the generator return value, and may leave spans open on close/error.
These bare-SDK behaviors are unchanged. Successful exhaustion is exercised;
examples run in isolated processes. This package does not claim coverage for
every AgentOps provider/framework or its authenticated cloud client lifecycle.
Live provider access and AgentOps cloud exports are not validated by the fixtures.

## Privacy and ownership

`AgentOpsInstrumentor(capture_content=False)`, `TRACELOOP_TRACE_CONTENT=false`,
and the Respan content context opt-out exclude inputs, outputs, prompts, calls,
schemas, SDK metadata, raw source fields and error descriptions. Recording and
capture are checked before serialization. Start policy is an upper bound; an
observed veto clears content irreversibly for that span and its ancestors,
including before context-manager detachment. Actual usage and structural
identifiers can remain. Known tool/schema/vector payloads are complete; generic
large content is bounded. Credentials and URL userinfo are redacted.

Compatible owners share guards and a first-position native span processor.
Different privacy settings or providers raise an error. Partial activation rolls
back owned changes. Deactivation restores only owned hooks/core fields, preserves
foreign replacements and leaves retained wrappers inert. It does not shut down a
shared provider or remove application-owned AgentOps instrumentation.

See the [paired examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/agentops)
for local fixtures and explicit Respan export. Local tests, HTTP acceptance and
stored semantic acceptance are separate gates.
