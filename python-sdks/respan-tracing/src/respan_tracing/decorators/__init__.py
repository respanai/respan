from typing import Optional, Union, List, Dict, Any
from opentelemetry.semconv_ai import TraceloopSpanKindValues
from .base import create_entity_method


def workflow(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    processors: Optional[Union[str, List[str]]] = None,
    export_filter: Optional[Dict[str, Any]] = None,
):
    """Respan workflow decorator

    Args:
        name: Optional name for the workflow
        version: Optional version number
        method_name: Optional method name for class decorators
        processors: Optional processor name(s) to route this workflow's spans to.
                   Can be a single string or list of strings (e.g., "debug" or ["debug", "analytics"])
        export_filter: Optional filter dict to control which spans are exported.
                      Uses AND logic — all conditions must match. Each key is a span attribute
                      name mapping to {"operator": str, "value": any}.
                      Example: {"status_code": {"operator": "", "value": "ERROR"}}
    """
    return create_entity_method(
        name=name,
        version=version,
        method_name=method_name,
        span_kind=TraceloopSpanKindValues.WORKFLOW,
        processors=processors,
        export_filter=export_filter,
    )


def task(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    processors: Optional[Union[str, List[str]]] = None,
    export_filter: Optional[Dict[str, Any]] = None,
):
    """Respan task decorator

    Args:
        name: Optional name for the task
        version: Optional version number
        method_name: Optional method name for class decorators
        processors: Optional processor name(s) to route this task's spans to.
                   Can be a single string or list of strings (e.g., "debug" or ["debug", "analytics"])
        export_filter: Optional filter dict to control which spans are exported.
                      Uses AND logic — all conditions must match. Each key is a span attribute
                      name mapping to {"operator": str, "value": any}.
                      Example: {"status_code": {"operator": "", "value": "ERROR"}}
    """
    return create_entity_method(
        name=name,
        version=version,
        method_name=method_name,
        span_kind=TraceloopSpanKindValues.TASK,
        processors=processors,
        export_filter=export_filter,
    )


def agent(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    processors: Optional[Union[str, List[str]]] = None,
    export_filter: Optional[Dict[str, Any]] = None,
):
    """Respan agent decorator

    Args:
        name: Optional name for the agent
        version: Optional version number
        method_name: Optional method name for class decorators
        processors: Optional processor name(s) to route this agent's spans to.
                   Can be a single string or list of strings (e.g., "debug" or ["debug", "analytics"])
        export_filter: Optional filter dict to control which spans are exported.
                      Uses AND logic — all conditions must match. Each key is a span attribute
                      name mapping to {"operator": str, "value": any}.
                      Example: {"status_code": {"operator": "", "value": "ERROR"}}
    """
    return create_entity_method(
        name=name,
        version=version,
        method_name=method_name,
        span_kind=TraceloopSpanKindValues.AGENT,
        processors=processors,
        export_filter=export_filter,
    )


def tool(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    processors: Optional[Union[str, List[str]]] = None,
    export_filter: Optional[Dict[str, Any]] = None,
):
    """Respan tool decorator

    Args:
        name: Optional name for the tool
        version: Optional version number
        method_name: Optional method name for class decorators
        processors: Optional processor name(s) to route this tool's spans to.
                   Can be a single string or list of strings (e.g., "debug" or ["debug", "analytics"])
        export_filter: Optional filter dict to control which spans are exported.
                      Uses AND logic — all conditions must match. Each key is a span attribute
                      name mapping to {"operator": str, "value": any}.
                      Example: {"status_code": {"operator": "", "value": "ERROR"}}
    """
    return create_entity_method(
        name=name,
        version=version,
        method_name=method_name,
        span_kind=TraceloopSpanKindValues.TOOL,
        processors=processors,
        export_filter=export_filter,
    )
