from typing import Optional, Union, List
from opentelemetry.semconv_ai import TraceloopSpanKindValues
from .base import create_entity_method


def workflow(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    processors: Optional[Union[str, List[str]]] = None,
    errors_only: bool = False,
    sample_rate: Optional[float] = None,
):
    """Respan workflow decorator

    Args:
        name: Optional name for the workflow
        version: Optional version number
        method_name: Optional method name for class decorators
        processors: Optional processor name(s) to route this workflow's spans to.
                   Can be a single string or list of strings (e.g., "debug" or ["debug", "analytics"])
        errors_only: If True, only export spans where the function raised an exception.
                    Useful for high-volume paths where you only care about failures.
        sample_rate: Sample rate between 0.0 and 1.0. If set, only this fraction of
                    calls will produce spans. None means 100% (default behavior).
    """
    return create_entity_method(
        name=name,
        version=version,
        method_name=method_name,
        span_kind=TraceloopSpanKindValues.WORKFLOW,
        processors=processors,
        errors_only=errors_only,
        sample_rate=sample_rate,
    )


def task(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    processors: Optional[Union[str, List[str]]] = None,
    errors_only: bool = False,
    sample_rate: Optional[float] = None,
):
    """Respan task decorator

    Args:
        name: Optional name for the task
        version: Optional version number
        method_name: Optional method name for class decorators
        processors: Optional processor name(s) to route this task's spans to.
                   Can be a single string or list of strings (e.g., "debug" or ["debug", "analytics"])
        errors_only: If True, only export spans where the function raised an exception.
                    Useful for high-volume paths where you only care about failures.
        sample_rate: Sample rate between 0.0 and 1.0. If set, only this fraction of
                    calls will produce spans. None means 100% (default behavior).
    """
    return create_entity_method(
        name=name,
        version=version,
        method_name=method_name,
        span_kind=TraceloopSpanKindValues.TASK,
        processors=processors,
        errors_only=errors_only,
        sample_rate=sample_rate,
    )


def agent(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    processors: Optional[Union[str, List[str]]] = None,
    errors_only: bool = False,
    sample_rate: Optional[float] = None,
):
    """Respan agent decorator

    Args:
        name: Optional name for the agent
        version: Optional version number
        method_name: Optional method name for class decorators
        processors: Optional processor name(s) to route this agent's spans to.
                   Can be a single string or list of strings (e.g., "debug" or ["debug", "analytics"])
        errors_only: If True, only export spans where the function raised an exception.
                    Useful for high-volume paths where you only care about failures.
        sample_rate: Sample rate between 0.0 and 1.0. If set, only this fraction of
                    calls will produce spans. None means 100% (default behavior).
    """
    return create_entity_method(
        name=name,
        version=version,
        method_name=method_name,
        span_kind=TraceloopSpanKindValues.AGENT,
        processors=processors,
        errors_only=errors_only,
        sample_rate=sample_rate,
    )


def tool(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    processors: Optional[Union[str, List[str]]] = None,
    errors_only: bool = False,
    sample_rate: Optional[float] = None,
):
    """Respan tool decorator

    Args:
        name: Optional name for the tool
        version: Optional version number
        method_name: Optional method name for class decorators
        processors: Optional processor name(s) to route this tool's spans to.
                   Can be a single string or list of strings (e.g., "debug" or ["debug", "analytics"])
        errors_only: If True, only export spans where the function raised an exception.
                    Useful for high-volume paths where you only care about failures.
        sample_rate: Sample rate between 0.0 and 1.0. If set, only this fraction of
                    calls will produce spans. None means 100% (default behavior).
    """
    return create_entity_method(
        name=name,
        version=version,
        method_name=method_name,
        span_kind=TraceloopSpanKindValues.TOOL,
        processors=processors,
        errors_only=errors_only,
        sample_rate=sample_rate,
    )
