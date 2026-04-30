import json
import inspect
from functools import wraps
from typing import Optional, TypeVar, Callable, Any, ParamSpec, Awaitable
from opentelemetry import context as context_api
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk import FilterParamDict
from respan_tracing.constants.context_constants import (
    ENABLE_CONTENT_TRACING_KEY
)
from respan_tracing.constants.generic_constants import LOGGER_NAME_DECORATORS
from respan_tracing.utils.logging import get_respan_logger
from respan_tracing.utils.span_setup import setup_span, cleanup_span, LinksParam

logger = get_respan_logger(LOGGER_NAME_DECORATORS)


P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[P, R | Awaitable[R]])


def _is_json_size_valid(json_str: str) -> bool:
    """Check if JSON string size is less than 1MB"""
    return len(json_str) < 1_000_000


def _should_send_prompts() -> bool:
    """Check if we should send prompt content in traces"""
    return context_api.get_value(ENABLE_CONTENT_TRACING_KEY) is not False


def _is_async_method(fn):
    """Check if function is async or async generator"""
    return inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)


def _setup_span(
    entity_name: str,
    span_kind: str,
    version: Optional[int] = None,
    processors=None,
    export_filter: Optional[FilterParamDict] = None,
    links: LinksParam = None,
    sample_rate: Optional[float] = None,
):
    """Setup OpenTelemetry span and context.

    Delegates to the shared setup_span() utility.
    Returns (span, ctx_token) for backward compatibility with existing callers.
    Context tokens for entity_name/entity_path/root are tracked internally and
    cleaned up in _cleanup_span().
    """
    span, ctx_token, entity_name_token, entity_path_token, root_ctx_token = setup_span(
        entity_name=entity_name,
        span_kind=span_kind,
        version=version,
        processors=processors,
        export_filter=export_filter,
        links=links,
        sample_rate=sample_rate,
    )
    # Store extra tokens on the span object for _cleanup_span to detach
    span._entity_name_token = entity_name_token
    span._entity_path_token = entity_path_token
    span._root_ctx_token = root_ctx_token
    return span, ctx_token


def _handle_span_input(span, args, kwargs):
    """Handle entity input logging"""
    try:
        if _should_send_prompts():
            json_input = json.dumps({"args": list(args), "kwargs": kwargs})
            if _is_json_size_valid(json_input):
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_input)
    except (TypeError, ValueError) as e:
        # Skip if serialization fails
        pass

def _handle_span_output(span, result):
    """Handle entity output logging"""
    try:
        if _should_send_prompts():
            json_output = json.dumps(result)
            if _is_json_size_valid(json_output):
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_output)
    except (TypeError, ValueError) as e:
        # Skip if serialization fails
        pass


def _cleanup_span(span, ctx_token):
    """End span and detach all context tokens."""
    cleanup_span(
        span,
        ctx_token,
        entity_name_token=getattr(span, '_entity_name_token', None),
        entity_path_token=getattr(span, '_entity_path_token', None),
        root_ctx_token=getattr(span, '_root_ctx_token', None),
    )


def _handle_generator(span, ctx_token, generator):
    """Handle generator functions"""
    try:
        for item in generator:
            yield item
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
    finally:
        _cleanup_span(span, ctx_token)


async def _ahandle_generator(span, ctx_token, async_generator):
    """Handle async generator functions"""
    try:
        async for item in async_generator:
            yield item
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
    finally:
        _cleanup_span(span, ctx_token)


def create_entity_method(
    name=None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    span_kind: str = "task",
    processors=None,
    export_filter: Optional[FilterParamDict] = None,
    links: LinksParam = None,
    sample_rate: Optional[float] = None,
) -> Callable[[F], F]:
    """Create entity decorator for methods or classes"""

    if method_name is not None:
        # Class decorator
        return _create_entity_class(
            name=name,
            version=version,
            method_name=method_name,
            span_kind=span_kind,
            processors=processors,
            export_filter=export_filter,
            links=links,
            sample_rate=sample_rate,
        )
    else:
        # Method decorator
        return _create_entity_method_decorator(
            name=name,
            version=version,
            span_kind=span_kind,
            processors=processors,
            export_filter=export_filter,
            links=links,
            sample_rate=sample_rate,
        )


def _create_entity_method_decorator(
    name=None,
    version: Optional[int] = None,
    span_kind: str = "task",
    processors=None,
    export_filter: Optional[FilterParamDict] = None,
    links: LinksParam = None,
    sample_rate: Optional[float] = None,
) -> Callable[[F], F]:
    """Create method decorator.

    Args:
        name: Span name. Can be:
            - str: static name used for all invocations
            - callable: called with (*args, **kwargs) at each invocation,
              must return a str. Useful for dynamic span names based on
              runtime arguments (e.g., evaluator config).
            - None: defaults to the decorated function's __name__
    """

    def decorator(fn: F) -> F:
        # Static name resolved once; callable resolved per-call below
        is_name_callable = callable(name)
        static_name = None if is_name_callable else (name or fn.__name__)

        def _resolve_name(*args, **kwargs) -> str:
            if is_name_callable:
                try:
                    return name(*args, **kwargs)
                except Exception as e:
                    logger.warning(
                        f"Dynamic span name callable failed for {fn.__name__}: {e}. "
                        f"Falling back to function name."
                    )
                    return fn.__name__
            return static_name

        if _is_async_method(fn):
            if inspect.isasyncgenfunction(fn):
                # Async generator
                @wraps(fn)
                async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                    entity_name = _resolve_name(*args, **kwargs)
                    span, ctx_token = _setup_span(
                        entity_name=entity_name,
                        span_kind=span_kind,
                        version=version,
                        processors=processors,
                        export_filter=export_filter,
                        links=links,
                        sample_rate=sample_rate,
                    )
                    _handle_span_input(span, args, kwargs)

                    try:
                        result = fn(*args, **kwargs)
                        async for item in _ahandle_generator(span, ctx_token, result):
                            yield item
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        _cleanup_span(span, ctx_token)
                        raise

                return async_gen_wrapper
            else:
                # Regular async function
                @wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    entity_name = _resolve_name(*args, **kwargs)
                    span, ctx_token = _setup_span(
                        entity_name=entity_name,
                        span_kind=span_kind,
                        version=version,
                        processors=processors,
                        export_filter=export_filter,
                        links=links,
                        sample_rate=sample_rate,
                    )
                    _handle_span_input(span, args, kwargs)

                    try:
                        result = await fn(*args, **kwargs)
                        _handle_span_output(span, result)
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
                    finally:
                        _cleanup_span(span, ctx_token)

                return async_wrapper
        else:
            # Sync function
            @wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                entity_name = _resolve_name(*args, **kwargs)
                span, ctx_token = _setup_span(
                    entity_name=entity_name,
                    span_kind=span_kind,
                    version=version,
                    processors=processors,
                    export_filter=export_filter,
                    links=links,
                    sample_rate=sample_rate,
                )
                _handle_span_input(span, args, kwargs)

                try:
                    result = fn(*args, **kwargs)

                    # Handle generators
                    if inspect.isgeneratorfunction(fn):
                        return _handle_generator(span, ctx_token, result)
                    else:
                        _handle_span_output(span, result)
                        return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                finally:
                    if not inspect.isgeneratorfunction(fn):
                        _cleanup_span(span, ctx_token)

            return sync_wrapper

    return decorator


def _create_entity_class(
    name: Optional[str],
    version: Optional[int],
    method_name: str,
    span_kind: str = "task",
    processors=None,
    export_filter: Optional[FilterParamDict] = None,
    links: LinksParam = None,
    sample_rate: Optional[float] = None,
):
    """Create class decorator"""

    def decorator(cls):
        entity_name = name or cls.__name__

        # Get the original method
        original_method = getattr(cls, method_name)

        # Create decorated method
        decorated_method = _create_entity_method_decorator(
            name=entity_name,
            version=version,
            span_kind=span_kind,
            processors=processors,
            export_filter=export_filter,
            links=links,
            sample_rate=sample_rate,
        )(original_method)

        # Replace the method
        setattr(cls, method_name, decorated_method)

        return cls

    return decorator
