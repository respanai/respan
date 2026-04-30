from typing import Optional, Union, List, Callable
from opentelemetry.semconv_ai import TraceloopSpanKindValues
from respan_sdk import FilterParamDict
from respan_sdk.respan_types.span_types import SpanLink
from respan_tracing.decorators.base import create_entity_method, LinksParam


def workflow(
    name: Optional[Union[str, Callable]] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    processors: Optional[Union[str, List[str]]] = None,
    export_filter: Optional[FilterParamDict] = None,
    links: LinksParam = None,
    sample_rate: Optional[float] = None,
    has_parent_trace: bool = False,
):
    """Respan workflow decorator.

    Trace-root behavior (BREAKING CHANGE in v3.0):

        @workflow spans default to **fresh root**. The decorator detaches
        any inherited OTel context before creating the span, so OTel
        allocates a new trace_id with no parent. This matches reality at
        every entry point in our system (Celery tasks, Pulsar consumer
        batch handlers, gunicorn views, signal receivers) — these are
        independent units of work whose lifecycles are not children of
        whatever happened to be the active OTel span when the function ran.

        Pass has_parent_trace=True to opt back into inheritance for
        the rare case where a @workflow is genuinely a sub-step of an
        outer workflow span and should share its trace_id.

        SpanBuffer continuation/injection (parent_trace_id / trace_id on
        client.get_span_buffer) is auto-detected and always respected —
        no flag needed.

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
                Use links for cross-trace correlation (e.g., "this trace was
                triggered by that one") instead of trace inheritance.
        sample_rate: Optional float between 0.0 and 1.0 controlling what fraction of
                    spans are exported. 1.0 = export all (default), 0.01 = export 1%.
                    When None, all spans are exported.
        has_parent_trace: Opt back into the v2 behavior — inherit the
                    active OTel trace_id as a child span. Default False.
                    Use only when the decorated function is conceptually a
                    sub-step of the active span's workflow; otherwise the
                    fresh-root default is correct.
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
        has_parent_trace=has_parent_trace,
    )


def task(
    name: Optional[Union[str, Callable]] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    processors: Optional[Union[str, List[str]]] = None,
    export_filter: Optional[FilterParamDict] = None,
    links: LinksParam = None,
    sample_rate: Optional[float] = None,
):
    """Respan task decorator

    Args:
        name: Name for the task. Can be a string (static) or a callable
              that receives (*args, **kwargs) and returns a string (dynamic).
              Dynamic names are resolved at each invocation.
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
    name: Optional[Union[str, Callable]] = None,
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
    name: Optional[Union[str, Callable]] = None,
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
