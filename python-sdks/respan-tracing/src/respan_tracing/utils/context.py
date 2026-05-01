from contextlib import contextmanager
from typing import Iterator, Optional
from opentelemetry import context as context_api
from opentelemetry.context import Context
from opentelemetry.semconv_ai import SpanAttributes
from respan_tracing.constants.context_constants import WORKFLOW_NAME_KEY, ENTITY_PATH_KEY


@contextmanager
def suppressed_parent_context() -> Iterator[None]:
    """Suppress the active OTel parent context for spans created in this block.

    Spans created (via @workflow / @task / @agent / @tool decorators or
    client.start_span) while this block is active see no active parent —
    they start fresh root traces. The OUTER span itself is untouched; only
    spans created INSIDE the with-block get the empty context.

    Use at execution boundaries where the inner work is conceptually
    independent of the outer span — most commonly, a Pulsar / Kafka /
    Celery batch consumer dispatching independent per-message tasks from
    inside a batch-level @workflow span:

        @workflow(name="my_consumer_handle_batch")
        async def _handle_batch(consumer, messages):
            for message in messages:
                with suppressed_parent_context():
                    await asyncio.to_thread(task.run, **message.payload)

    Without this, every per-message dispatch inherits _handle_batch's
    trace_id and downstream `count(distinct trace_unique_id)` collapses N
    messages into one trace per batch.

    Sub-workflow composition (workflow → workflow → workflow as one trace)
    is unaffected — the @workflow decorator's standard child-of-context
    behavior continues to work everywhere except inside this block.
    """
    token = context_api.attach(Context())
    try:
        yield
    finally:
        context_api.detach(token)


def get_entity_path(ctx: Optional[Context] = None) -> Optional[str]:
    """
    Retrieves the current entity path from the active context.
    This builds the hierarchical path like "workflow.task.subtask".
    
    Args:
        ctx: The context to read from (defaults to current active context)
        
    Returns:
        The entity path string or None if not set
    """
    if ctx is None:
        ctx = context_api.get_current()
    
    # First check for full entity path (set by TOOL/TASK spans)
    entity_path = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_PATH, context=ctx)
    if entity_path:
        return entity_path
    
    # Fall back to workflow name (set by WORKFLOW/AGENT spans)  
    workflow_name = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME, context=ctx)
    return workflow_name 