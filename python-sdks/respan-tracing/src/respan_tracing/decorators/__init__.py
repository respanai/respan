from typing import Optional, Union, List, Callable
from opentelemetry.semconv_ai import TraceloopSpanKindValues
from respan_sdk import FilterParamDict
from respan_sdk.respan_types.span_types import SpanLink
from respan_tracing.decorators.base import create_entity_method, LinksParam


def workflow(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    processors: Optional[Union[str, List[str]]] = None,
    export_filter: Optional[FilterParamDict] = None,
    links: LinksParam = None,
    sample_rate: Optional[float] = None,
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
        links: Optional span links. Can be a list of SpanLink objects (static) or a
               callable returning a list of SpanLink objects (resolved at call time).
        sample_rate: Optional float between 0.0 and 1.0 controlling what fraction of
                    spans are exported. 1.0 = export all (default), 0.01 = export 1%.
                    When None, all spans are exported.
    """
    return create_entity_method(
        name=name,
        version=version,
        method_name=method_name,
        span_kind=TraceloopSpanKindValues.WORKFLOW,
        processors=processors,
        export_filter=export_filter,
        links=links,
        sample_rate=sample_rate,
    )


def task(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    processors: Optional[Union[str, List[str]]] = None,
    export_filter: Optional[FilterParamDict] = None,
    links: LinksParam = None,
    sample_rate: Optional[float] = None,
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
        links: Optional span links. Can be a list of SpanLink objects (static) or a
               callable returning a list of SpanLink objects (resolved at call time).
        sample_rate: Optional float between 0.0 and 1.0 controlling what fraction of
                    spans are exported. 1.0 = export all (default), 0.01 = export 1%.
                    When None, all spans are exported.
    """
    return create_entity_method(
        name=name,
        version=version,
        method_name=method_name,
        span_kind=TraceloopSpanKindValues.TASK,
        processors=processors,
        export_filter=export_filter,
        links=links,
        sample_rate=sample_rate,
    )


def agent(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    processors: Optional[Union[str, List[str]]] = None,
    export_filter: Optional[FilterParamDict] = None,
    links: LinksParam = None,
    sample_rate: Optional[float] = None,
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
        links: Optional span links. Can be a list of SpanLink objects (static) or a
               callable returning a list of SpanLink objects (resolved at call time).
        sample_rate: Optional float between 0.0 and 1.0 controlling what fraction of
                    spans are exported. 1.0 = export all (default), 0.01 = export 1%.
                    When None, all spans are exported.
    """
    return create_entity_method(
        name=name,
        version=version,
        method_name=method_name,
        span_kind=TraceloopSpanKindValues.AGENT,
        processors=processors,
        export_filter=export_filter,
        links=links,
        sample_rate=sample_rate,
    )


def tool(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    processors: Optional[Union[str, List[str]]] = None,
    export_filter: Optional[FilterParamDict] = None,
    links: LinksParam = None,
    sample_rate: Optional[float] = None,
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
        links: Optional span links. Can be a list of SpanLink objects (static) or a
               callable returning a list of SpanLink objects (resolved at call time).
        sample_rate: Optional float between 0.0 and 1.0 controlling what fraction of
                    spans are exported. 1.0 = export all (default), 0.01 = export 1%.
                    When None, all spans are exported.
    """
    return create_entity_method(
        name=name,
        version=version,
        method_name=method_name,
        span_kind=TraceloopSpanKindValues.TOOL,
        processors=processors,
        export_filter=export_filter,
        links=links,
        sample_rate=sample_rate,
    )
