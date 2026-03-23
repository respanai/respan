from typing import Optional
from opentelemetry import context as context_api
from opentelemetry.context import Context
from opentelemetry.semconv_ai import SpanAttributes
from respan_tracing.constants.context_constants import WORKFLOW_NAME_KEY, ENTITY_PATH_KEY


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